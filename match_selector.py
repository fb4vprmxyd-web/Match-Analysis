"""Resolves a match-selection spec ("last 10 games each team", "last 5 cup
games each team") into actual pulled cache directories, ready to feed
straight into build_prematch.py's OPP_CACHES/OUR_CACHES.

Like metric_catalog.py on the Coach's Brief side, this is the safety rail
behind letting the coach's assistant describe a sample in free text:
whatever turns his sentence into a spec (today, done by hand each time
this session; later, an LLM) only ever produces a {team, count,
competition, before_date} dict - a query filter, not an open-ended
instruction - which this module then executes deterministically against
the real fixture list. There's much less room for that mapping to be
wrong than there is on the Coach's Brief side (see the design discussion
this was built from): "last 5 cup games" has exactly one correct answer.

    spec = {"team": "Wycombe Wanderers", "count": 5, "competition": "league"}
    caches = resolve_selection(cfg, spec, api)
    # -> ["wycombe_wanderers_270224", "wycombe_wanderers_270174", ...]
"""

import glob
import os

import pandas as pd

from envconfig import CACHE_ROOT
from run import slug, discover
from fetch_data import pull_match


def find_upcoming_fixture(cfg: dict, team: str, opp: str):
    """The next not-yet-played match between `team` and `opp`, formatted as
    build_prematch.py's PRE_FIXTURE expects ("<competition> · <venue> ·
    <date>"). Returns None if there's no scheduled fixture on record - a
    pre-match report built without one falls back to whatever stale example
    string is hardcoded in build_prematch.py, so callers should surface
    that rather than silently shipping the wrong fixture details."""
    matches, apis, problems, reached = discover(cfg, team)
    if len(matches) == 0:
        return None
    involved = (matches.homeSquadName == opp) | (matches.awaySquadName == opp)
    upcoming = matches[involved & (matches.available == False)].sort_values("scheduledDate")
    if len(upcoming) == 0:
        return None
    row = upcoming.iloc[0]
    home = row.homeSquadName == team
    when = pd.to_datetime(row.scheduledDate, errors="coerce", utc=True)
    date_txt = when.strftime("%A %-d %B %Y") if pd.notna(when) else ""
    venue = cfg.get("home_venue", "Brisbane Road") if home else f"Away at {opp}"
    competition = str(row.get("competitionName", "")).strip()
    return " · ".join(p for p in (competition, venue, date_txt) if p)


def _find_existing_cache(match_id: int):
    """A cache already exists for this match if any directory under
    CACHE_ROOT ends in _<matchId> - robust to the several different naming
    conventions this project has used over time (data_<slug>_<id>,
    <slug>_<id>), rather than assuming today's exact pattern."""
    hits = glob.glob(str(CACHE_ROOT / f"*_{match_id}"))
    return hits[0] if hits else None


def _current_season(matches: pd.DataFrame):
    """The season of a team's most recently PLAYED match - there's no
    single 'current season' config for an arbitrary team the way there is
    for Orient's own match.toml fallback, so this is inferred from their
    own results instead."""
    played = matches[matches.available == True].sort_values("scheduledDate")
    if len(played) == 0:
        return None
    return played.season.iloc[-1]


def _filter_pool(matches: pd.DataFrame, count: int, competition: str = "all",
                 before_date=None) -> pd.DataFrame:
    """Pure filtering over an already-fetched match list - no API call, so
    it can be reused without re-discovering fixtures every time."""
    if len(matches) == 0:
        return matches

    season = _current_season(matches)
    pool = matches[matches.available == True]
    if season is not None:
        pool = pool[pool.season == season]
    if before_date is not None:
        pool = pool[pd.to_datetime(pool.scheduledDate, utc=True) < pd.Timestamp(before_date, tz="UTC")]

    if competition in ("league", "cup"):
        is_cup = pool.competitionName.astype(str).str.lower().str.contains("cup")
        pool = pool[is_cup] if competition == "cup" else pool[~is_cup]

    pool = pool.sort_values("scheduledDate", ascending=False).head(count)
    return pool.sort_values("scheduledDate")  # oldest-first for a stable cache list


def select_recent_matches(cfg: dict, team: str, count: int,
                          competition: str = "all", before_date=None) -> pd.DataFrame:
    """Up to `count` most recent played matches for `team`, filtered by
    competition scope ('all' / 'league' / 'cup') and capped to their
    current season - deliberately never reaches back into an earlier
    season to make up the count, matching every other sample-size
    convention already in this project (uneven counts between two sides
    are normal, not an error).
    """
    matches, apis, problems, reached = discover(cfg, team)
    return _filter_pool(matches, count, competition, before_date)


def resolve_selection(cfg: dict, spec: dict, apis: dict = None) -> list:
    """Runs the same selection as select_recent_matches() (one discover()
    call, not two) and ensures every match it names is actually pulled,
    reusing whatever's already cached rather than re-pulling. Returns
    cache directory NAMES (not full paths), the same shape
    build_prematch.py's OPP_CACHES/OUR_CACHES already expects."""
    matches, discovered_apis, problems, reached = discover(cfg, spec["team"])
    pool = _filter_pool(matches, spec["count"], spec.get("competition", "all"),
                        spec.get("before_date"))
    caches = []
    for _, row in pool.iterrows():
        mid = int(row.id)
        existing = _find_existing_cache(mid)
        if existing:
            caches.append(os.path.basename(existing))
            continue
        api = discovered_apis.get(row._account)
        if api is None:
            continue
        cache_name = f"{slug(spec['team'])}_{mid}"
        pull_match(api, mid, str(CACHE_ROOT / cache_name), season_ids=None)
        caches.append(cache_name)
    return caches
