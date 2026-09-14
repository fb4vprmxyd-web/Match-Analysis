"""CLI: turn a coach's free-text notes into a saved Coach's Brief.

    python3 interpret_brief.py --match-id 302115
    python3 interpret_brief.py --match-id 302115 --notes-file notes.txt
    cat notes.txt | python3 interpret_brief.py --match-id 302115 --opp "Bradford City"

Interprets via brief_interpreter.interpret_brief(), prints a plain-English
readback of every check for the coach to sanity-check BEFORE anything is
written, and only calls db.save_coach_brief_checks()/save_coach_brief_notes()
once that's confirmed. Re-running for the same --match-id (e.g. after the
coach edits their notes) fully replaces whatever checks were there before -
that's how "regenerate" works, see db.save_coach_brief_checks().

Requires Postgres (DATABASE_URL) - Coach's Brief checks have nowhere else
to live; see db.py.
"""

import argparse
import os
import subprocess
import sys
import tempfile

import db as DB
from envconfig import load_match
from match_lookup import resolve_opponent, match_date
from brief_interpreter import interpret_brief, readback


def _resolve_opponent(match_id: int, given: str):
    opp = resolve_opponent(match_id, given)
    if not opp:
        sys.exit(f"No opponent on record for match {match_id} - pass --opp explicitly.")
    return opp


def _read_notes(args) -> str:
    if args.notes:
        return args.notes
    if args.notes_file:
        return open(args.notes_file).read()
    print("Paste the coach's notes, then Ctrl-D:", file=sys.stderr)
    return sys.stdin.read()


def _edit(checks: list) -> list:
    import json
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(path, "w") as fh:
        json.dump(checks, fh, indent=2)
    editor = os.environ.get("EDITOR", "vi")
    subprocess.call([editor, path])
    with open(path) as fh:
        edited = json.load(fh)
    os.unlink(path)
    return edited


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match-id", type=int, required=True)
    ap.add_argument("--opp", help="Opponent name (defaults to the ledger entry for this match)")
    ap.add_argument("--notes", help="Notes as a single argument")
    ap.add_argument("--notes-file", help="Path to a text file with the notes")
    args = ap.parse_args()

    if not DB.available():
        sys.exit("DATABASE_URL not set - Coach's Brief checks require Postgres, see db.py.")

    cfg = load_match()
    team = cfg.get("team", "Leyton Orient")
    opp = _resolve_opponent(args.match_id, args.opp)
    when = match_date(cfg, team, args.match_id)
    raw_text = _read_notes(args)

    checks = interpret_brief(raw_text, cfg, team, opp, match_date=when)

    while True:
        print(f"\n--- {len(checks)} checks for match {args.match_id} ({team} v {opp}) ---")
        for line in readback(checks):
            print(" ", line)
        choice = input("\nSave these checks? [y]es / [n]o / [e]dit raw JSON: ").strip().lower()
        if choice == "y":
            break
        if choice == "n":
            print("Aborted - nothing saved.")
            return
        if choice == "e":
            checks = _edit(checks)
            continue
        print("Please answer y, n, or e.")

    DB.save_coach_brief_notes(args.match_id, raw_text)
    DB.save_coach_brief_checks(args.match_id, checks)
    print(f"Saved {len(checks)} checks for match {args.match_id}.")


if __name__ == "__main__":
    main()
