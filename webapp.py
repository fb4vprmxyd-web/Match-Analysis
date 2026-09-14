"""Web UI - the first real server this project has had.

    python3 webapp.py                # http://localhost:5050

Everything before this ran as CLI scripts against local cache/reports; this
is the same underlying pipeline (brief_interpreter.py, selector_interpreter.py,
match_selector.py, coach_brief_engine.py, db.py) behind pages people can
actually use directly:

  /brief/<id>  - paste the coach's notes, see the plain-English readback,
                 confirm or edit, save to Postgres (coach_brief_checks).
  /selector    - describe a pre-match sample in free text ("last 5 cup
                 games each team"), see the readback, resolve it against
                 the live fixture list, and get a finished PDF back - not
                 a command to run. The coach's assistant using this has no
                 reason to ever open a terminal.

Deliberately no server-side session/state between requests - the
interpreted result travels through the page itself (a hidden JSON field)
so "confirm" only ever acts on exactly what was just shown, never a stale
re-interpretation. No auth yet either - see the module docstring in
match_lookup.py; this is local/LAN-only until it's closer to deployment.

Post-match report building (run.py / build_report.py) still stays on the
CLI - those run on a schedule against freshly-pulled data, not on demand
from a page. Pre-match reports are different: there's no fixed schedule,
just "build one for the next opponent whenever", so /selector triggers
build_prematch.py directly (subprocess, same way run.py already shells out
to build_report.py) rather than leaving that as a manual step.
"""

import json
import os
import subprocess
import sys

import anthropic
from flask import Flask, render_template, request, abort, send_from_directory

import db as DB
from envconfig import load_match, ROOT, REPORTS_ROOT
from match_lookup import load_state, match_date
from brief_interpreter import interpret_brief, readback as brief_readback
from selector_interpreter import interpret_selection, readback as selector_readback
from match_selector import resolve_selection, find_upcoming_fixture

app = Flask(__name__)


def _cfg_team():
    cfg = load_match()
    return cfg, cfg.get("team", "Leyton Orient")


@app.route("/")
def index():
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
    return render_template("brief_saved.html", match_id=match_id,
                          opponent=rec.get("opponent"), count=len(checks))


@app.route("/selector")
def selector_form():
    return render_template("selector.html", opponent="", notes="", specs=None)


@app.route("/selector/interpret", methods=["POST"])
def selector_interpret():
    opp = request.form["opponent"].strip()
    notes = request.form["notes"]
    if not opp:
        return render_template("selector.html", opponent=opp, notes=notes,
                              specs=None, error="Enter the opponent's name first.")
    cfg, us = _cfg_team()
    try:
        specs = interpret_selection(notes, us, opp)
    except SystemExit as exc:
        # see brief_interpret()'s identical catch - envconfig's key check
        # calls sys.exit(), which must not be allowed to escape a request.
        return render_template("selector.html", opponent=opp, notes=notes,
                              specs=None, error=str(exc))
    except anthropic.APIError as exc:
        return render_template("selector.html", opponent=opp, notes=notes,
                              specs=None, error=f"Claude API error: {exc}")
    return render_template("selector.html", opponent=opp, notes=notes, specs=specs,
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
        return render_template("selector.html", opponent=opp, notes="",
                              specs=specs, specs_json=json.dumps(specs),
                              readback_lines=selector_readback(specs),
                              error=f"Couldn't resolve against the live fixture "
                                    f"list: {exc}")
    if not opp_caches or not our_caches:
        return render_template("selector.html", opponent=opp, notes="",
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
    fixture = find_upcoming_fixture(cfg, us, opp)
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
        return render_template("selector.html", opponent=opp, notes="",
                              specs=specs, specs_json=json.dumps(specs),
                              readback_lines=selector_readback(specs),
                              error=f"Report build failed:\n{r.stdout[-800:]}\n{r.stderr[-800:]}")

    out_name = f"Pre-Match Analysis - {opp}.pdf"
    out_path = REPORTS_ROOT / out_name
    return render_template("selector_resolved.html", opponent=opp,
                          opp_caches=opp_caches, our_caches=our_caches,
                          report_url=f"/reports/{out_name}",
                          report_exists=out_path.exists(),
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
    app.run(debug=True, port=5050)
