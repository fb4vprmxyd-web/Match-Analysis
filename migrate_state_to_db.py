"""One-off: import state/processed.json into Postgres.

Run this once, after DATABASE_URL is set and pointed at a real database,
to carry over the existing processed-matches history instead of starting
that ledger from empty. Safe to re-run - it's an upsert, matching
db.save_state()'s own semantics.

    python3 migrate_state_to_db.py
"""

import json
import sys

from envconfig import ROOT
import db as DB

STATE_FILE = ROOT / "state" / "processed.json"


def main():
    if not DB.available():
        sys.exit("DATABASE_URL is not set - nothing to migrate into.")
    if not STATE_FILE.exists():
        sys.exit(f"{STATE_FILE} does not exist - nothing to migrate from.")

    state = json.loads(STATE_FILE.read_text())
    if not state:
        print("state/processed.json is empty - nothing to do.")
        return

    DB.save_state(state)
    print(f"migrated {len(state)} match(es) into Postgres:")
    for match_id, rec in sorted(state.items(), key=lambda kv: int(kv[0])):
        print(f"  {match_id}  {rec.get('opponent')}")


if __name__ == "__main__":
    main()
