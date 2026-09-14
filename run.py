"""One-command runner: pull any new or revised match, rebuild its report.

    python3 run.py                 # do the work
    python3 run.py --dry-run       # decide, print, touch nothing
    python3 run.py --list          # every fixture the accounts can see
    python3 run.py --match 285457  # one specific match
    python3 run.py --force         # rebuild even if nothing changed

Run it whenever you know data has landed. Nothing runs on a timer.

Why it re-checks matches it has already built: Impect *revises* data after first
publication - there is a verification pass that moves the packing numbers by
roughly a fifth. The runner records `lastCalculationDate` per match and rebuilds
whenever it moves, so a report is never left carrying pre-verification figures.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import unicodedata
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

from envconfig import load_env, load_match, ROOT, DATA_ROOT, CACHE_ROOT, REPORTS_ROOT
from fetch_data import pull_match
import db as DB

STATE = DATA_ROOT / "state" / "processed.json"

# Only the opponent's colour cannot be derived from the match record.
OPP_COLOURS = {
    "Oxford United": "#FFDD00", "Burton Albion": "#F5C518",
    "Sheffield Wednesday": "#1C6BC1", "Wigan Athletic": "#1D5BA4",
    "FC Barnsley": "#E4022D", "FC Bromley": "#E8B21E",
    "Luton Town": "#F76B1C", "Wycombe Wanderers": "#4EB6E3",
    "FC Stevenage": "#E4002B", "Mansfield Town": "#F5B317",
    "Plymouth Argyle": "#08533B", "Bradford City": "#8E1F2F",
    "FC Blackpool": "#F68B1F", "FC Reading": "#1B4C9C",
    "Cambridge United": "#F6C519", "Doncaster Rovers": "#C8102E",
    "Rotherham United": "#DD1C2E", "Lincoln City": "#C8102E",
    "Exeter City": "#B01B2E", "Huddersfield Town": "#0072CE",
    "Northampton Town": "#8B1B3F", "AFC Wimbledon": "#1B458F",
    "FC Port Vale": "#D9D9D9",
}
DEFAULT_OPP_COLOUR = "#F5C518"


def slug(text: str) -> str:
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")


def pretty_matchday(name: str) -> str:
    """Impect labels rounds in German ('1. Runde', '12. Spieltag')."""
    if not isinstance(name, str):
        return ""
    m = re.match(r"\s*(\d+)\.\s*Spieltag", name)
    if m:
        return f"Matchday {m.group(1)}"
    m = re.match(r"\s*(\d+)\.\s*Runde", name)
    if m:
        n = int(m.group(1))
        return {1: "First Round", 2: "Second Round", 3: "Third Round",
                4: "Fourth Round", 5: "Fifth Round"}.get(n, f"Round {n}")
    return name


def load_state() -> dict:
    # Postgres is the source of truth once DATABASE_URL is set (see db.py) -
    # the JSON file is the local-dev fallback, not a second copy to keep in
    # sync, so this never reads both.
    if DB.available():
        return DB.load_state()
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {}


def save_state(state: dict) -> None:
    if DB.available():
        DB.save_state(state)
        return
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2, sort_keys=True))


def accounts_from_config(cfg: dict) -> list:
    """Each account: a credential prefix plus the iterations it can reach."""
    accts = cfg.get("accounts")
    if accts:
        return accts
    return [{"name": "default", "env_prefix": "IMPECT",
             "iterations": cfg.get("iterations", [])}]


def connect(prefix: str):
    from impectPy import Impect
    user = os.environ.get(f"{prefix}_USER")
    pw = os.environ.get(f"{prefix}_PASS")
    if not user or not pw:
        return None, f"{prefix}_USER / {prefix}_PASS not set in .env"
    api = Impect()
    api.login(user, pw)
    return api, None


def discover(cfg: dict, team: str):
    """Every fixture involving `team`, across all configured accounts."""
    rows, apis, problems, reached = [], {}, [], []
    for acct in accounts_from_config(cfg):
        prefix = acct.get("env_prefix", "IMPECT")
        api, err = connect(prefix)
        if err:
            problems.append(f"{acct.get('name', prefix)}: {err}")
            continue
        apis[acct.get("name", prefix)] = api
        try:
            its = api.getIterations().set_index("id")
        except Exception:
            its = None
        for it in acct.get("iterations", []):
            try:
                m = api.getMatches(iteration=int(it))
            except Exception as exc:
                # 403 here means the iteration is outside this account's licence.
                # Impect returns 403 for anything unlicensed, so we cannot tell
                # "not covered" from "does not exist" - report it and move on.
                why = ("not covered by this licence"
                       if type(exc).__name__ == "ForbiddenError"
                       else type(exc).__name__)
                problems.append(f"iteration {it}: {why}")
                continue
            reached.append(int(it))
            m = m[m.homeSquadName.eq(team) | m.awaySquadName.eq(team)].copy()
            if len(m):
                m["_account"] = acct.get("name", prefix)
                m["_iteration"] = int(it)
                if its is not None and int(it) in its.index:
                    m["competitionName"] = its.loc[int(it), "competitionName"]
                    m["season"] = its.loc[int(it), "season"]
                rows.append(m)
    matches = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return matches, apis, problems, reached


def build_env(row, team: str, cfg: dict, cache: str, out: str) -> dict:
    home = row.homeSquadName == team
    opp = row.awaySquadName if home else row.homeSquadName
    when = pd.to_datetime(row.scheduledDate, errors="coerce", utc=True)
    date_txt = when.strftime("%A %-d %B %Y") if pd.notna(when) else ""
    venue = cfg.get("home_venue", "Brisbane Road") if home else f"Away at {opp}"
    env = dict(os.environ)
    env.update({
        "PMR_MATCH_ID": str(int(row.id)),
        "PMR_TEAM": team,
        "PMR_OPP": str(opp),
        "PMR_COMP": f"{row.competitionName} {row.season}".strip(),
        "PMR_MATCHDAY": pretty_matchday(row.matchDayName),
        "PMR_VENUE": f"{venue} · {date_txt}".strip(" ·"),
        "PMR_OPP_C": OPP_COLOURS.get(str(opp), DEFAULT_OPP_COLOUR),
        "PMR_CACHE": cache,
        "PMR_OUT": out,
    })
    return env


def season_pool(api, iteration: int, team: str, limit: int):
    """Own matches if there are enough for a baseline, else the wider pool.

    Mirrors how the summary page chooses its comparison basis, so we never pull
    hundreds of matches we are not going to use.
    """
    try:
        m = api.getMatches(iteration=iteration)
    except Exception:
        return []
    avail = m[m.available == True]
    own = avail[avail.homeSquadName.eq(team) | avail.awaySquadName.eq(team)]
    if len(own) >= 5:
        return [int(i) for i in own.id.tolist()]
    return [int(i) for i in avail.id.tolist()[:limit]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--match", type=int)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    load_env()
    cfg = load_match()
    team = cfg.get("team", "Leyton Orient")
    limit = int(cfg.get("season_pool_limit", 150))

    matches, apis, problems, reached = discover(cfg, team)
    for p in problems:
        print(f"  ! {p}")
    if not len(matches):
        sys.exit("No fixtures visible. Check .env credentials and match.toml "
                 "iterations.")

    matches = matches.sort_values("scheduledDate")

    if args.list:
        keys = [c for c in ("competitionName", "season") if c in matches.columns]
        comps = (matches.groupby(keys or ["_iteration"]).size()
                 .rename("fixtures").reset_index())
        print(f"iterations reachable: {reached}")
        print("competitions visible for this club:")
        print(comps.to_string(index=False))
        print()
        cols = ["id", "_account", "competitionName", "season", "matchDayName",
                "homeSquadName", "awaySquadName", "scheduledDate",
                "homeSquadGoals", "awaySquadGoals", "available"]
        print(matches[[c for c in cols if c in matches.columns]]
              .to_string(index=False))
        return

    state = load_state()
    playable = matches[matches.available == True]

    # Scope routine runs to the club's current season - otherwise a season
    # that was never fully backfilled (any match not yet in state.json) gets
    # treated as "new work" and silently built alongside this week's report.
    # --match is an explicit override, so it is exempt from this filter.
    fb_row = matches[matches.id == int(cfg.get("id", 0))]
    current_season = fb_row.season.iloc[0] if len(fb_row) else None

    if args.match:
        playable = playable[playable.id == args.match]
        if not len(playable):
            sys.exit(f"Match {args.match} is not available (or not {team}'s).")
    elif current_season is not None:
        before = len(playable)
        playable = playable[playable.season == current_season]
        skipped = before - len(playable)
        if skipped:
            print(f"  (scoped to season {current_season} - {skipped} available "
                  f"match(es) from other seasons ignored; use --match <id> to "
                  f"target one directly)")

    todo = []
    for _, row in playable.iterrows():
        key = str(int(row.id))
        calc = str(row.get("lastCalculationDate"))
        prev = state.get(key)
        if prev is None:
            todo.append((row, "new"))
        elif prev.get("lastCalculationDate") != calc:
            todo.append((row, "revised by Impect"))
        elif args.force:
            todo.append((row, "forced"))

    if not todo:
        print(f"Up to date — {len(playable)} available {team} match(es), "
              f"nothing new or revised.")
        return

    print(f"{len(todo)} match(es) to build:")
    for row, why in todo:
        opp = row.awaySquadName if row.homeSquadName == team else row.homeSquadName
        print(f"  · {row.homeSquadName} v {row.awaySquadName}  "
              f"({str(row.scheduledDate)[:10]})  [{why}]")
    if args.dry_run:
        print("\n--dry-run: nothing pulled or built.")
        return

    built = []
    for row, why in todo:
        home = row.homeSquadName == team
        opp = row.awaySquadName if home else row.homeSquadName
        # Opponent-only naming collides the moment two matches share an
        # opponent - a same-season home/away double header, or an old
        # season never cleared out. The match id makes both the cache dir
        # and the output filename unique and stable across rebuilds.
        mid = int(row.id)
        when = pd.to_datetime(row.scheduledDate, errors="coerce", utc=True)
        date_tag = when.strftime("%Y-%m-%d") if pd.notna(when) else str(mid)
        cache = f"{slug(opp)}_{mid}"
        out = f"Post Match Report - {team} v {opp} ({date_tag}).pdf"
        api = apis.get(row._account)
        if api is None:
            print(f"  ! no session for account {row._account}; skipping")
            continue

        print(f"\n=== {team} v {opp} ===")
        ids = season_pool(api, int(row._iteration), team, limit)
        pull_match(api, int(row.id), str(CACHE_ROOT / cache), season_ids=ids)

        env = build_env(row, team, cfg, cache, out)
        print(f"  building {out}")
        r = subprocess.run([sys.executable, str(ROOT / "build_report.py")],
                           cwd=str(ROOT), env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  ! build failed:\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
            continue

        state[str(int(row.id))] = {
            "lastCalculationDate": str(row.get("lastCalculationDate")),
            "builtAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "output": out, "opponent": str(opp), "cache": cache,
        }
        save_state(state)
        built.append(out)

    print("\n" + "=" * 60)
    for o in built:
        print(f"  built: {o}")
    state_desc = "Postgres" if DB.available() else str(STATE.relative_to(DATA_ROOT))
    print(f"{len(built)} report(s) written. State: {state_desc}")

    if built:
        try:
            import build_season
            build_season.main()
        except Exception as exc:
            print(f"  ! season tracker rebuild failed ({type(exc).__name__}: {exc}) "
                  "- post-match report(s) above are unaffected")


if __name__ == "__main__":
    main()
