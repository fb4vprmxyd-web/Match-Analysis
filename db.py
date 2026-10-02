"""Postgres-backed state storage.

The processed-matches ledger (`state/processed.json` today) is genuinely
relational - one row per match, a handful of columns, queried by id - which
makes it a poor fit for a Docker volume and a good fit for the Postgres
instance already running alongside the recruiting platform on the shared
droplet. This module is the only place that knows how to talk to it.

Deliberately optional: if DATABASE_URL isn't set, `available()` returns
False and every caller (run.py, build_season.py) falls back to the local
JSON file exactly as before. That keeps local development working without
Docker or Postgres installed, and makes the Postgres path something you opt
into by setting one environment variable, not a hard new dependency.

    DATABASE_URL=postgresql://user:pass@host:5432/dbname

Large per-match data (the cached parquet files, generated PDFs and HTML)
does NOT belong in Postgres - see docker-compose.yml's own note and the
README - it stays as files, on a Docker volume once deployed, on local disk
today.
"""

import os
from datetime import datetime, timezone

from envconfig import load_env

load_env()

DATABASE_URL = os.environ.get("DATABASE_URL")

SCHEMA = """
CREATE TABLE IF NOT EXISTS processed_matches (
    match_id                BIGINT PRIMARY KEY,
    last_calculation_date   TEXT,
    built_at                TEXT,
    cache                   TEXT,
    opponent                TEXT,
    output                  TEXT
);

-- One match's Coach's Brief, as structured checks rather than hand-written
-- Python - see metric_catalog.py and coach_brief_engine.py. Replaces the
-- old single-active-coach_brief.py-file architecture: every match's real
-- Brief now lives here permanently, independent of every other match's, so
-- rebuilding one match's report never risks overwriting another's notes.
CREATE TABLE IF NOT EXISTS coach_brief_checks (
    id              SERIAL PRIMARY KEY,
    match_id        BIGINT NOT NULL,
    section         TEXT NOT NULL,   -- 'OUT OF POSSESSION' / 'IN POSSESSION' / 'DID IT WORK'
    sort_order      INTEGER NOT NULL DEFAULT 0,
    instruction     TEXT NOT NULL,   -- the coach's own label, e.g. "Stop crosses"
    metric_key      TEXT NOT NULL,   -- metric_catalog.CATALOG key
    subject         TEXT NOT NULL DEFAULT 'us',  -- whose data: 'us' or 'opp'
    params          JSONB NOT NULL DEFAULT '{}',
    basis           TEXT NOT NULL,   -- 'beat_opponent' / 'absolute' / 'vs_baseline' / 'info'
    lower_better    BOOLEAN NOT NULL DEFAULT FALSE,
    target          DOUBLE PRECISION,   -- for 'absolute'
    margin          DOUBLE PRECISION,   -- for 'beat_opponent' / 'vs_baseline'
    -- for 'vs_baseline': {"caches": ["<cache dir>", ...]} - resolved to
    -- concrete cache directories at check-creation time (by whatever turns
    -- "vs their previous game" into a check), not re-derived at grading
    -- time, so grading never needs fixture-list awareness.
    baseline        JSONB,
    source_note     TEXT,            -- the free-text sentence this came from, if any
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS coach_brief_checks_match_id_idx
    ON coach_brief_checks (match_id);

-- The coach's raw notes for a match, kept alongside the interpreted checks
-- above so "regenerate" can re-interpret from the original wording rather
-- than from whatever the last interpretation produced.
CREATE TABLE IF NOT EXISTS coach_brief_notes (
    match_id     BIGINT PRIMARY KEY,
    raw_text     TEXT NOT NULL,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per upcoming fixture that already has a pre-match/corner report -
-- see match_selector.record_prematch_build(). Existence alone is the gate:
-- build_upcoming.py's automatic pass only fills in a fixture with no row
-- here yet, whether the row it finds later came from that same automatic
-- pass or from the coach's assistant using /selector by hand.
CREATE TABLE IF NOT EXISTS prematch_builds (
    fixture_id   BIGINT PRIMARY KEY,
    opponent     TEXT,
    source       TEXT NOT NULL,   -- 'auto' or 'manual'
    built_at     TEXT
);
-- Cached display text for the dashboard's Pre-Match/Corner Analysis cards
-- (see match_selector.sample_caption()) - added after the table above, so
-- existing rows just get NULLs here until their fixture is next rebuilt.
ALTER TABLE IF EXISTS prematch_builds ADD COLUMN IF NOT EXISTS fixture_text TEXT;
ALTER TABLE IF EXISTS prematch_builds ADD COLUMN IF NOT EXISTS sample_note TEXT;
"""

_schema_ready = False


def available() -> bool:
    return bool(DATABASE_URL)


def _connect():
    import psycopg2
    return psycopg2.connect(DATABASE_URL)


def init_schema() -> None:
    """Idempotent - safe to call on every connection, only actually hits
    the database once per process."""
    global _schema_ready
    if _schema_ready:
        return
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(SCHEMA)
        conn.commit()
    _schema_ready = True


def load_state() -> dict:
    """Same shape run.py's JSON-file load_state() already returns:
    {str(matchId): {lastCalculationDate, builtAt, cache, opponent, output}}
    - so callers don't need to know which backend they're talking to."""
    init_schema()
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT match_id, last_calculation_date, built_at, cache, opponent, output
            FROM processed_matches
        """)
        rows = cur.fetchall()
    return {
        str(mid): {
            "lastCalculationDate": lcd,
            "builtAt": built_at,
            "cache": cache,
            "opponent": opponent,
            "output": output,
        }
        for mid, lcd, built_at, cache, opponent, output in rows
    }


def save_state(state: dict) -> None:
    """Upserts every entry in `state`. run.py always calls this with the
    full accumulated dict (matching its JSON-file save_state's contract),
    so a plain upsert-per-row keeps both backends behaving identically -
    at this scale (tens of matches) there's no reason to diff and send
    only what changed."""
    init_schema()
    with _connect() as conn, conn.cursor() as cur:
        for match_id, rec in state.items():
            cur.execute("""
                INSERT INTO processed_matches
                    (match_id, last_calculation_date, built_at, cache, opponent, output)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (match_id) DO UPDATE SET
                    last_calculation_date = EXCLUDED.last_calculation_date,
                    built_at              = EXCLUDED.built_at,
                    cache                 = EXCLUDED.cache,
                    opponent              = EXCLUDED.opponent,
                    output                = EXCLUDED.output
            """, (
                int(match_id),
                rec.get("lastCalculationDate"),
                rec.get("builtAt"),
                rec.get("cache"),
                rec.get("opponent"),
                rec.get("output"),
            ))
        conn.commit()


def load_prematch_builds() -> dict:
    """Same shape match_selector.py's JSON-file load_prematch_builds()
    returns: {str(fixtureId): {opponent, source, fixtureText, sampleNote,
    builtAt}}."""
    init_schema()
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT fixture_id, opponent, source, fixture_text, sample_note, built_at
            FROM prematch_builds
        """)
        rows = cur.fetchall()
    return {
        str(fid): {
            "opponent": opponent, "source": source,
            "fixtureText": fixture_text or "", "sampleNote": sample_note or "",
            "builtAt": built_at,
        }
        for fid, opponent, source, fixture_text, sample_note, built_at in rows
    }


def save_prematch_build(fixture_id: int, opponent: str, source: str,
                        fixture_text: str = "", sample_note: str = "") -> None:
    init_schema()
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO prematch_builds
                (fixture_id, opponent, source, fixture_text, sample_note, built_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (fixture_id) DO UPDATE SET
                opponent     = EXCLUDED.opponent,
                source       = EXCLUDED.source,
                fixture_text = EXCLUDED.fixture_text,
                sample_note  = EXCLUDED.sample_note,
                built_at     = EXCLUDED.built_at
        """, (int(fixture_id), opponent, source, fixture_text, sample_note,
              datetime.now(timezone.utc).isoformat(timespec="seconds")))
        conn.commit()


# ------------------------------------------------------- coach's brief checks
def save_coach_brief_checks(match_id: int, checks: list) -> None:
    """Replaces every check for this match - a full delete+insert, not a
    diff/merge, matching how 'regenerate' is meant to work: the coach edits
    his notes, re-interpretation produces a fresh full set of checks, and
    that fully replaces whatever was there before. Each check dict:
    {section, sort_order, instruction, metric_key, subject, params, basis,
    lower_better, target, margin, baseline, source_note}."""
    import json
    init_schema()
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM coach_brief_checks WHERE match_id = %s", (match_id,))
        for c in checks:
            cur.execute("""
                INSERT INTO coach_brief_checks
                    (match_id, section, sort_order, instruction, metric_key,
                     subject, params, basis, lower_better, target, margin,
                     baseline, source_note)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                match_id, c["section"], c.get("sort_order", 0), c["instruction"],
                c["metric_key"], c.get("subject", "us"),
                json.dumps(c.get("params", {})), c["basis"],
                c.get("lower_better", False), c.get("target"), c.get("margin"),
                json.dumps(c["baseline"]) if c.get("baseline") else None,
                c.get("source_note"),
            ))
        conn.commit()


# The page always reads out of-possession first, in-possession second, then
# the outcome section last - "ORDER BY section" alone sorts those three
# strings alphabetically instead ('DID IT WORK' < 'IN POSSESSION' <
# 'OUT OF POSSESSION'), which silently put the outcome section first. Sort
# in Python against this fixed order rather than trust string comparison.
SECTION_ORDER = ["OUT OF POSSESSION", "IN POSSESSION", "DID IT WORK"]


def load_coach_brief_checks(match_id: int) -> list:
    """Returns checks in the page's fixed section order, then sort_order,
    ready for coach_brief_engine.evaluate_from_db()."""
    init_schema()
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT section, sort_order, instruction, metric_key, subject,
                   params, basis, lower_better, target, margin, baseline, source_note
            FROM coach_brief_checks
            WHERE match_id = %s
            ORDER BY sort_order, id
        """, (match_id,))
        rows = cur.fetchall()
    cols = ("section", "sort_order", "instruction", "metric_key", "subject",
            "params", "basis", "lower_better", "target", "margin", "baseline",
            "source_note")
    checks = [dict(zip(cols, r)) for r in rows]

    def _key(c):
        sec = c["section"]
        idx = SECTION_ORDER.index(sec) if sec in SECTION_ORDER else len(SECTION_ORDER)
        return (idx, c["sort_order"])

    return sorted(checks, key=_key)


def save_coach_brief_notes(match_id: int, raw_text: str) -> None:
    init_schema()
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO coach_brief_notes (match_id, raw_text, updated_at)
            VALUES (%s, %s, now())
            ON CONFLICT (match_id) DO UPDATE SET
                raw_text = EXCLUDED.raw_text, updated_at = now()
        """, (match_id, raw_text))
        conn.commit()


def load_coach_brief_notes(match_id: int):
    init_schema()
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT raw_text FROM coach_brief_notes WHERE match_id = %s",
                    (match_id,))
        row = cur.fetchone()
    return row[0] if row else None
