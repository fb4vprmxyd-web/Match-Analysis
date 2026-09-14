"""Small shared helpers for resolving a match's opponent/date from the
processed-matches ledger or the live fixture list. Used by both
interpret_brief.py's CLI and webapp.py's UI so the two don't drift into
separate, slightly different lookup logic."""

import json

import db as DB
from envconfig import DATA_ROOT
from run import discover


def local_state() -> dict:
    path = DATA_ROOT / "state" / "processed.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_state() -> dict:
    return DB.load_state() if DB.available() else local_state()


def resolve_opponent(match_id: int, given: str = None):
    if given:
        return given
    rec = load_state().get(str(match_id))
    return rec.get("opponent") if rec else None


def match_date(cfg, team, match_id: int):
    matches, apis, problems, reached = discover(cfg, team)
    row = matches[matches.id == match_id]
    return row.scheduledDate.iloc[0] if len(row) else None
