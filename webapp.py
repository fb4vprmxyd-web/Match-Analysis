"""Coach's Brief web UI - the first real server this project has had.

    python3 webapp.py                # http://localhost:5000

Everything before this ran as CLI scripts against local cache/reports; this
is the same underlying pipeline (brief_interpreter.py, coach_brief_engine.py,
db.py) behind a page the coach's assistant can actually use: pick a match,
paste notes, see the plain-English readback, confirm or edit, save.

Deliberately no server-side session/state between requests - the
interpreted checks travel through the page itself (a hidden JSON field) so
"confirm" only ever saves exactly what was just shown, never a stale
re-interpretation. No auth yet either - see the module docstring in
match_lookup.py; this is local/LAN-only until it's closer to deployment.

Scope for this first slice is the Coach's Brief flow only (notes -> saved
checks in Postgres). Report building and the pre-match match-selector UI
stay on the CLI for now - see run.py / build_report.py / interpret_selection.py.
"""

import json

import anthropic
from flask import Flask, render_template, request, abort

import db as DB
from envconfig import load_match
from match_lookup import load_state, match_date
from brief_interpreter import interpret_brief, readback

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
                          readback_lines=readback(checks))


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


if __name__ == "__main__":
    # NOT 5000 - macOS's own AirPlay Receiver (ControlCenter) listens there
    # on both IPv4 and IPv6, and since "localhost" resolves to ::1 first in
    # most browsers, http://localhost:5000 silently hits AirPlay Receiver
    # instead of this app (a near-blank page, not an error) even while this
    # server is running and answering fine on 127.0.0.1:5000. 5050 avoids
    # the collision entirely rather than asking every dev to go disable a
    # macOS system service.
    app.run(debug=True, port=5050)
