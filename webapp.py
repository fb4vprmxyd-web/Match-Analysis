"""Web UI - the first real server this project has had.

    python3 webapp.py                # http://localhost:5050

One link for everything: "/" now serves templates/dashboard.html (the
existing season-progress/report-browser page, previously a bare static
file with nowhere to be hosted from) - its sidebar got two more panes,
"Coach's Brief" and "Pre-Match Selector", each an iframe onto this app's
own pages, the same way it already iframes season-tracker.html. So the
whole platform lives at one URL; the sidebar switches what's showing,
nothing ever navigates away from it. Its `reports/...` relative hrefs
still resolve correctly since /reports/<file> is served here too (see
serve_report()).

The Pre-Match Analysis and Set Piece Analysis panes are no longer
hand-maintained: _prematch_cards() builds them from whatever reports
actually exist in REPORTS_ROOT, captioned from prematch_builds (see
match_selector.py) when a fixture went through build_upcoming.py or
/selector. Post-Match Reports stays hand-written - it carries per-match
detail (scores, Coach's Brief status) this app doesn't compute.

Everything before this ran as CLI scripts against local cache/reports; this
is the same underlying pipeline (brief_interpreter.py, selector_interpreter.py,
match_selector.py, coach_brief_engine.py, db.py) behind pages people can
actually use directly:

  /brief       - list of matches, which ones have a saved brief.
  /brief/<id>  - paste the coach's notes, see the plain-English readback,
                 confirm or edit, save to Postgres (coach_brief_checks).
  /selector    - describe a pre-match sample in free text ("last 5 cup
                 games each team"), see the readback, resolve it against
                 the live fixture list, and get a finished PDF back - not
                 a command to run. The coach's assistant using this has no
                 reason to ever open a terminal.

These pages have no header/nav of their own (see base.html) - when shown
inside dashboard.html's iframe panes the sidebar already provides that;
when opened directly (a shared link) the per-page <title> and content
carry enough context on their own.

Deliberately no server-side session/state between requests beyond who's
signed in - the interpreted result travels through the page itself (a
hidden JSON field) so "confirm" only ever acts on exactly what was just
shown, never a stale re-interpretation. Login is real (see auth.py): every
route except /login requires a session. Behind a reverse proxy (Caddy on
the droplet, terminating TLS - this app never speaks TLS itself), set
FORCE_HTTPS=1 so the session cookie is marked Secure; ProxyFix below makes
Flask trust the proxy's X-Forwarded-Proto so redirects/url_for still
resolve to https:// rather than quietly downgrading to http://.

Data collection (run.py's incremental pull) runs on its own schedule now
(see scheduler.py / docker-compose.yml's `scheduler` service) - nobody
needs to trigger that by hand either. The one thing that used to still
need a person: saving a brief only wrote to Postgres, so someone had to
separately rerun `python3 run.py --force` for the PDF to actually reflect
it. /brief/<id>/save now rebuilds that one match's report itself (same
subprocess pattern /selector/resolve already used for pre-match) and hands
back a working link - so the whole loop, pull through finished PDF, no
longer needs anyone but the coach's assistant.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import anthropic
from flask import (Flask, render_template, request, abort, send_from_directory,
                   send_file, session, redirect, url_for)
from werkzeug.middleware.proxy_fix import ProxyFix

import auth
import db as DB
from envconfig import load_match, ROOT, REPORTS_ROOT
from match_lookup import load_state, match_date
from brief_interpreter import interpret_brief, readback as brief_readback
from selector_interpreter import interpret_selection, readback as selector_readback
from match_selector import (resolve_selection, find_upcoming_fixture,
                            record_prematch_build, load_prematch_builds,
                            sample_caption)
from run import discover, build_env, OPP_COLOURS, DEFAULT_OPP_COLOUR

app = Flask(__name__)
# Required for session cookies to survive a restart - and to mean the same
# thing across every gunicorn worker, which each import this module
# separately. A random fallback keeps `python3 webapp.py` working locally
# without any setup; production must set FLASK_SECRET_KEY itself, or every
# worker restart (and every request landing on a different worker) logs
# everyone out.
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(32)

# Trust exactly one hop of X-Forwarded-* - the droplet's Caddy container,
# which terminates TLS and forwards plain HTTP to gunicorn internally over
# their shared Docker network (see docker-compose.prod.yml). Without this,
# Flask sees every request as scheme=http (Caddy's own connection to it),
# so url_for(..., _external=True) and the Secure cookie flag below would
# both silently misbehave even though the browser is really on https://. A
# no-op when there's no proxy in front (nothing sets these headers), so
# this is safe for local `python3 webapp.py` too.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Only mark the session cookie Secure once there's actually a TLS-terminating
# proxy in front - set FORCE_HTTPS=1 in the droplet's .env once Caddy is
# live. Doing this unconditionally would break login over plain HTTP (every
# local/Docker test so far), since browsers refuse to send a Secure cookie
# back over an insecure connection.
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("FORCE_HTTPS", "") == "1"
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# Routes reachable without being signed in - the login page itself, plus
# Flask's own static file handler if one's ever added. Everything else,
# including every PDF under /reports/, requires a session.
_PUBLIC_ROUTES = {"login", "static"}


@app.before_request
def _require_login():
    if request.endpoint in _PUBLIC_ROUTES or request.endpoint is None:
        return
    if not session.get("user"):
        return redirect(url_for("login", next=request.full_path))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html", next=request.args.get("next", ""))
    username = request.form.get("username", "")
    password = request.form.get("password", "")
    if auth.verify(username, password):
        session.clear()
        session["user"] = username
        session.permanent = True
        dest = request.form.get("next") or "/"
        # never redirect off-site using a value an attacker could plant in
        # the query string - only ever back into this same app.
        if not dest.startswith("/"):
            dest = "/"
        return redirect(dest)
    return render_template("login.html", next=request.form.get("next", ""),
                          error="Wrong username or password.")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.context_processor
def _inject_session_user():
    return {"session_user": session.get("user")}


def _cfg_team():
    cfg = load_match()
    return cfg, cfg.get("team", "Leyton Orient")


def _prematch_cards():
    """One card per opponent that has a pre-match and/or set piece report on
    disk. Caption text (fixture_text, sample_note, auto/manual) comes from
    prematch_builds (see match_selector.py) - written automatically by
    build_upcoming.py for whatever's next, and overwritten with 'manual'
    the moment the coach's assistant resolves a custom sample for that
    same fixture through /selector. There is only ever one file per
    opponent (Pre-Match Analysis - <opp>.pdf), so a manual rebuild already
    replaces the automatic one on disk - this just reflects whichever file
    is actually there. Reports that predate prematch_builds tracking (or
    were dropped in by hand) still show up via the filename scan below,
    just without those captions."""
    cards = {}
    for rec in load_prematch_builds().values():
        opp = rec.get("opponent")
        if not opp:
            continue
        cards[opp] = {
            "opponent": opp,
            "accent": OPP_COLOURS.get(opp, DEFAULT_OPP_COLOUR),
            "fixture_text": rec.get("fixtureText") or "",
            "sample_note": rec.get("sampleNote") or "",
            "auto": rec.get("source") == "auto",
            "built_at": rec.get("builtAt") or "",
        }

    for prefix in ("Pre-Match Analysis - ", "Set Piece Report - "):
        for path in REPORTS_ROOT.glob(f"{prefix}*.pdf"):
            opp = path.stem[len(prefix):]
            if opp not in cards:
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                cards[opp] = {
                    "opponent": opp, "accent": OPP_COLOURS.get(opp, DEFAULT_OPP_COLOUR),
                    "fixture_text": "", "sample_note": "", "auto": False,
                    "built_at": mtime.isoformat(timespec="seconds"),
                }

    out = []
    for opp, c in cards.items():
        pre_path = REPORTS_ROOT / f"Pre-Match Analysis - {opp}.pdf"
        corner_path = REPORTS_ROOT / f"Set Piece Report - {opp}.pdf"
        c["prematch_url"] = f"/reports/Pre-Match Analysis - {opp}.pdf" if pre_path.exists() else None
        c["corner_url"] = f"/reports/Set Piece Report - {opp}.pdf" if corner_path.exists() else None
        if c["prematch_url"] or c["corner_url"]:
            out.append(c)
    # Same ordering convention as Post-Match Reports: most recent date
    # first. Most of these cards are now historical snapshots (pre-match
    # state "as of" an already-played kickoff, see build_prematch.py's
    # before_date sampling), not previews of what's still ahead, so a past
    # date here is normal and expected, not suspicious - sorting newest
    # date first naturally surfaces the latest game's snapshot above the
    # early-season ones, same as Post-Match does. fixture_text ends in a
    # date formatted "%A %-d %B %Y" (see match_selector._format_fixture) -
    # parsed back out rather than stored separately, since the display
    # string already has it.
    out.sort(key=lambda c: c["built_at"], reverse=True)

    played_opponents = {
        path.stem[len("Post Match Report - Leyton Orient v "):].rsplit(" (", 1)[0]
        for path in REPORTS_ROOT.glob("Post Match Report - Leyton Orient v *.pdf")
    }

    def _fixture_sort_key(c):
        text = c.get("fixture_text") or ""
        date_part = text.rsplit(" · ", 1)[-1] if text else ""
        try:
            d = datetime.strptime(date_part, "%A %d %B %Y")
        except ValueError:
            return (1, 0)
        # A scheduled date already in the past with no post-match report to
        # show for it means the match was postponed, not actually played on
        # that date (confirmed on Mansfield Town) - comparing to "today"
        # alone can't tell a postponement apart from a genuine historical
        # snapshot once most of these dates are in the past by design, but
        # "was a post-match report ever built for this opponent" can.
        if c["opponent"] not in played_opponents and d.date() < datetime.now(timezone.utc).date():
            return (1, 0)
        return (0, -d.toordinal())

    out.sort(key=_fixture_sort_key)
    return out


@app.route("/")
def dashboard():
    cards = _prematch_cards()
    return render_template("dashboard.html",
                          prematch_cards=[c for c in cards if c["prematch_url"]],
                          corner_cards=[c for c in cards if c["corner_url"]])


@app.route("/brief")
def brief_list():
    state = load_state()
    matches = []
    for mid, rec in sorted(state.items(), key=lambda kv: kv[0], reverse=True):
        has_brief = DB.available() and DB.load_coach_brief_notes(int(mid)) is not None
        matches.append({
            "id": mid,
            "opponent": rec.get("opponent", "?"),
            "output": rec.get("output", ""),
            "has_brief": has_brief,
        })
    return render_template("index.html", matches=matches, db_available=DB.available())


@app.route("/brief/<int:match_id>")
def brief_form(match_id):
    if not DB.available():
        abort(503, "DATABASE_URL not set - Coach's Brief needs Postgres, see db.py.")
    state = load_state()
    rec = state.get(str(match_id))
    if rec is None:
        abort(404, f"No match {match_id} on record.")
    notes = DB.load_coach_brief_notes(match_id) or ""
    return render_template("brief.html", match_id=match_id, opponent=rec.get("opponent"),
                          notes=notes, checks=None, readback_lines=None)


@app.route("/brief/<int:match_id>/interpret", methods=["POST"])
def brief_interpret(match_id):
    if not DB.available():
        abort(503, "DATABASE_URL not set - Coach's Brief needs Postgres, see db.py.")
    state = load_state()
    rec = state.get(str(match_id))
    if rec is None:
        abort(404, f"No match {match_id} on record.")
    notes = request.form["notes"]
    cfg, team = _cfg_team()
    opp = rec.get("opponent")
    when = match_date(cfg, team, match_id)
    try:
        checks = interpret_brief(notes, cfg, team, opp, match_date=when)
    except SystemExit as exc:
        # envconfig's ANTHROPIC_API_KEY check calls sys.exit() - fine for a
        # CLI script, but SystemExit isn't a normal exception and shouldn't
        # be allowed to propagate out of a request handler. Caught here and
        # shown as a plain error instead.
        return render_template("brief.html", match_id=match_id, opponent=opp,
                              notes=notes, checks=None, readback_lines=None,
                              error=str(exc))
    except anthropic.APIError as exc:
        return render_template("brief.html", match_id=match_id, opponent=opp,
                              notes=notes, checks=None, readback_lines=None,
                              error=f"Claude API error: {exc}")
    return render_template("brief.html", match_id=match_id, opponent=opp, notes=notes,
                          checks=checks, checks_json=json.dumps(checks),
                          readback_lines=brief_readback(checks))


@app.route("/brief/<int:match_id>/save", methods=["POST"])
def brief_save(match_id):
    if not DB.available():
        abort(503, "DATABASE_URL not set - Coach's Brief needs Postgres, see db.py.")
    notes = request.form["notes"]
    checks = json.loads(request.form["checks_json"])
    DB.save_coach_brief_notes(match_id, notes)
    DB.save_coach_brief_checks(match_id, checks)
    state = load_state()
    rec = state.get(str(match_id), {})
    opp = rec.get("opponent")

    # Saving a brief is the point where the coach's assistant actually
    # wants a finished report, not just a database write - same reasoning
    # as /selector/resolve for pre-match. Rebuilds just THIS match (not
    # run.py's whole incremental pass) using the cache/output names
    # already on record for it from the scheduled pull, so re-saving a
    # brief a few times while refining it just re-runs this one quick
    # build, not a full re-pull.
    cfg, team = _cfg_team()
    report_url = None
    build_error = None
    if rec.get("cache") and rec.get("output"):
        matches, apis, problems, reached = discover(cfg, team)
        row = matches[matches.id == match_id]
        if len(row):
            env = build_env(row.iloc[0], team, cfg, rec["cache"], rec["output"])
            r = subprocess.run([sys.executable, str(ROOT / "build_report.py")],
                               cwd=str(ROOT), env=env, capture_output=True, text=True)
            if r.returncode == 0:
                report_url = f"/reports/{rec['output']}"
            else:
                build_error = (f"Report build failed:\n{r.stdout[-800:]}\n"
                              f"{r.stderr[-800:]}")
        else:
            build_error = "Couldn't find this match's fixture to rebuild the report."
    else:
        build_error = "No cache/output on record for this match yet."

    return render_template("brief_saved.html", match_id=match_id, opponent=opp,
                          count=len(checks), report_url=report_url,
                          build_error=build_error)


def _known_opponents(cfg, team):
    """Every opponent name on the live fixture list, played or upcoming -
    what /selector's dropdown is populated from, so a coach's assistant
    never has to guess Impect's exact spelling of a club name again
    ("Stevenage" vs the real "FC Stevenage"). Upcoming fixtures are
    already in discover()'s output (see build_upcoming.py's own use of
    it), so next week's opponent shows up here just as soon as the
    fixture is published - same requirement /selector already has via
    find_upcoming_fixture() for the report's own game-info banner."""
    matches, apis, problems, reached = discover(cfg, team)
    if len(matches) == 0:
        return []
    names = set(matches.homeSquadName) | set(matches.awaySquadName)
    names.discard(team)
    return sorted(names)


def _selector_page(**kwargs):
    cfg, us = _cfg_team()
    kwargs.setdefault("opponents", _known_opponents(cfg, us))
    return render_template("selector.html", **kwargs)


@app.route("/selector")
def selector_form():
    return _selector_page(opponent="", notes="", specs=None)


@app.route("/selector/interpret", methods=["POST"])
def selector_interpret():
    opp = request.form["opponent"].strip()
    notes = request.form["notes"]
    if not opp:
        return _selector_page(opponent=opp, notes=notes,
                              specs=None, error="Enter the opponent's name first.")
    cfg, us = _cfg_team()
    try:
        specs = interpret_selection(notes, us, opp)
    except SystemExit as exc:
        # see brief_interpret()'s identical catch - envconfig's key check
        # calls sys.exit(), which must not be allowed to escape a request.
        return _selector_page(opponent=opp, notes=notes, specs=None, error=str(exc))
    except anthropic.APIError as exc:
        return _selector_page(opponent=opp, notes=notes, specs=None,
                              error=f"Claude API error: {exc}")
    return _selector_page(opponent=opp, notes=notes, specs=specs,
                          specs_json=json.dumps(specs),
                          readback_lines=selector_readback(specs))


@app.route("/selector/resolve", methods=["POST"])
def selector_resolve():
    opp = request.form["opponent"]
    specs = json.loads(request.form["specs_json"])
    cfg, us = _cfg_team()
    try:
        opp_caches = resolve_selection(cfg, specs["opponent"])
        our_caches = resolve_selection(cfg, specs["us"])
    except Exception as exc:
        return _selector_page(opponent=opp, notes="",
                              specs=specs, specs_json=json.dumps(specs),
                              readback_lines=selector_readback(specs),
                              error=f"Couldn't resolve against the live fixture "
                                    f"list: {exc}")
    if not opp_caches or not our_caches:
        return _selector_page(opponent=opp, notes="",
                              specs=specs, specs_json=json.dumps(specs),
                              readback_lines=selector_readback(specs),
                              error="Resolved zero matches for one side - check "
                                    "the opponent name and that fixtures are "
                                    "published yet.")

    # The coach's assistant needs a finished report, not a shell command to
    # run themselves - so this builds it directly rather than handing back
    # env-var exports. fixture is looked up from the real, not-yet-played
    # fixture between us and this opponent so the report's game-info banner
    # is accurate; if there isn't one on record yet (friendly, not
    # published, wrong opponent name), that's surfaced rather than silently
    # falling back to build_prematch.py's stale hardcoded example fixture.
    fixture_id, fixture = find_upcoming_fixture(cfg, us, opp)
    env = dict(os.environ)
    env.update({
        "PRE_OPP": opp,
        "PRE_OPP_CACHES": ",".join(opp_caches),
        "PRE_OUR_CACHES": ",".join(our_caches),
    })
    if fixture:
        env["PRE_FIXTURE"] = fixture
    r = subprocess.run([sys.executable, str(ROOT / "build_prematch.py")],
                       cwd=str(ROOT), env=env, capture_output=True, text=True)
    if r.returncode != 0:
        return _selector_page(opponent=opp, notes="",
                              specs=specs, specs_json=json.dumps(specs),
                              readback_lines=selector_readback(specs),
                              error=f"Report build failed:\n{r.stdout[-800:]}\n{r.stderr[-800:]}")

    out_name = f"Pre-Match Analysis - {opp}.pdf"
    out_path = REPORTS_ROOT / out_name

    # The set piece report has always used the same sample as the matching
    # pre-match report (see dashboard.html's own captions - "same sample as
    # the pre-match report"), and build_corners.py literally imports its
    # OPP_CACHES/OUR_CACHES from build_prematch.py, so there's no separate
    # sample to describe here - the same resolved caches and env just
    # builds the second report too, one action instead of making the coach
    # repeat the same "last N games" description a second time.
    corner_name = f"Set Piece Report - {opp}.pdf"
    corner_error = None
    rc = subprocess.run([sys.executable, str(ROOT / "build_corners.py")],
                        cwd=str(ROOT), env=env, capture_output=True, text=True)
    if rc.returncode != 0:
        corner_error = f"{rc.stdout[-800:]}\n{rc.stderr[-800:]}"
    corner_path = REPORTS_ROOT / corner_name

    # Marks this fixture as built so build_upcoming.py's automatic pass
    # never overwrites this deliberate choice later - see
    # match_selector.record_prematch_build().
    if fixture_id is not None:
        note = sample_caption(specs["us"], specs["opponent"], our_caches, opp_caches)
        record_prematch_build(fixture_id, opp, "manual",
                              fixture_text=fixture or "", sample_note=note)

    return render_template("selector_resolved.html", opponent=opp,
                          opp_caches=opp_caches, our_caches=our_caches,
                          report_url=f"/reports/{out_name}",
                          report_exists=out_path.exists(),
                          corner_url=f"/reports/{corner_name}",
                          corner_exists=corner_path.exists() and not corner_error,
                          corner_error=corner_error,
                          fixture_found=fixture is not None)


@app.route("/reports/<path:filename>")
def serve_report(filename):
    return send_from_directory(str(REPORTS_ROOT), filename)


if __name__ == "__main__":
    # NOT 5000 - macOS's own AirPlay Receiver (ControlCenter) listens there
    # on both IPv4 and IPv6, and since "localhost" resolves to ::1 first in
    # most browsers, http://localhost:5000 silently hits AirPlay Receiver
    # instead of this app (a near-blank page, not an error) even while this
    # server is running and answering fine on 127.0.0.1:5000. 5050 avoids
    # the collision entirely rather than asking every dev to go disable a
    # macOS system service.
    #
    # debug=True only if asked for: Flask's debugger lets anyone who can
    # trigger an unhandled exception run arbitrary code on the box, so the
    # default has to be off. This whole block never runs in production
    # anyway - the webapp container starts gunicorn directly (see
    # Dockerfile/docker-compose.yml), which imports `app` as a WSGI
    # callable and never executes this file as __main__.
    debug = os.environ.get("FLASK_DEBUG", "") == "1"
    app.run(debug=debug, host="127.0.0.1", port=5050)
