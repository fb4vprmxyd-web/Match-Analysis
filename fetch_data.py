"""Pull one match (plus season context) from the Impect API into ./data.

Credentials are read from the .env file next to this script - never from the
command line, so they cannot leak into shell history or a terminal log:

    cp .env.example .env && chmod 600 .env    # then edit .env

    python3 fetch_data.py                 # match from match.toml
    python3 fetch_data.py --match 216038  # any other match id

`--list-matches` prints every available Leyton Orient fixture with its id, so
you can re-point the report at a different game.
"""

import argparse
import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from impectPy import Impect
from report_config import MATCH_ID, TEAM

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Iterations the account can see that contain the club. Extend if you move club.
# Iterations the account can see that contain the club. The EFL Cup sits on a
# separate licence from the league competitions, so both are listed here.
from envconfig import load_match as _lm, credentials as _creds
ITERATIONS = _lm().get("iterations", [1465, 2117, 1514, 2227])
# 1465/2117 League One 25/26 & 26/27 · 1514/2227 EFL Cup 25/26 & 26/27


def connect() -> Impect:
    user, pw = _creds()          # from .env; never passed on the command line
    api = Impect()
    api.login(user, pw)
    return api


def club_matches(api, team=TEAM) -> pd.DataFrame:
    frames = []
    for it in ITERATIONS:
        try:
            m = api.getMatches(iteration=it)
        except Exception as exc:                      # 403 on unlicensed comps
            print(f"  iteration {it}: skipped ({type(exc).__name__})")
            continue
        frames.append(m)
    m = pd.concat(frames, ignore_index=True)
    return m[m.homeSquadName.eq(team) | m.awaySquadName.eq(team)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", type=int, default=MATCH_ID)
    ap.add_argument("--list-matches", action="store_true")
    ap.add_argument("--no-season", action="store_true",
                    help="skip the season pull used for the summary page")
    args = ap.parse_args()

    api = connect()
    os.makedirs(DATA, exist_ok=True)

    m = club_matches(api)
    if args.list_matches:
        cols = ["id", "matchDayName", "homeSquadName", "awaySquadName",
                "scheduledDate", "homeSquadGoals", "awaySquadGoals", "available"]
        print(m.sort_values("scheduledDate")[cols].to_string(index=False))
        return

    mid = args.match
    row = m[m.id == mid]
    if len(row) and not bool(row.available.iloc[0]):
        sys.exit(f"Match {mid} has no data yet (available=False).")
    pull_match(api, mid, DATA, season_ids=None if args.no_season else
               [int(i) for i in m[m.available == True].id.tolist()])
    print(f"done -> {DATA}")


def pull_match(api, mid: int, dest: str, season_ids=None) -> None:
    """Pull every endpoint for one match into `dest`, plus optional season pool.

    Shared by the CLI and by run.py so both stay in step.
    """
    os.makedirs(dest, exist_ok=True)
    print(f"  pulling match {mid} -> {dest}")
    api.getEvents(matches=[mid]).to_parquet(f"{dest}/ev.parquet")
    api.getPlayerMatchsums(matches=[mid]).to_parquet(f"{dest}/ps.parquet")
    api.getSquadMatchsums(matches=[mid]).to_parquet(f"{dest}/ss.parquet")
    for name, fn in (("sp", api.getStartingPositions),
                     ("su", api.getSubstitutions),
                     ("setpieces", api.getSetPieces),
                     ("squad_scores", api.getSquadMatchScores),
                     ("fm", api.getFormations)):
        try:
            fn(matches=[mid]).to_parquet(f"{dest}/{name}.parquet")
        except Exception as exc:
            print(f"    {name}: unavailable ({type(exc).__name__})")

    if season_ids:
        print(f"    season context: {len(season_ids)} matches")
        api.getSquadMatchsums(matches=season_ids).to_parquet(
            f"{dest}/season_squad.parquet")
        api.getPlayerMatchsums(matches=season_ids).to_parquet(
            f"{dest}/season_player.parquet")


if __name__ == "__main__":
    main()
