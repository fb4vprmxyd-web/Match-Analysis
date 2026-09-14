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
