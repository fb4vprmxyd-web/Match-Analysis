"""Automatic pre-match + set piece report generation for whatever fixture is
next on the calendar - the mirror image of run.py: that rebuilds the match
just played, this builds the one coming up next.

    python3 build_upcoming.py

Run by scheduler.py right after run.py, on the same Mon/Thu trigger - once
this is deployed nobody needs to open /selector just to get a first draft
ready for the next opponent.

Builds only the first time a given fixture has no report at all - see
match_selector.record_prematch_build(). The moment anything exists for that
fixture (this script's own first pass, or the coach's assistant using
/selector by hand to pick a specific sample) it is left alone from then on;
a deliberate /selector choice always sticks until the fixture is played and
a new one becomes next. Re-running /selector by hand always still works -
the automatic skip only applies to this script.

Default sample when there's no coach input to size it: every match each
side has played so far this season, all competitions - the widest sample
that still makes sense without someone describing what they want.
"""

import os
import subprocess
import sys

from envconfig import load_env, load_match, ROOT
from match_selector import (next_upcoming_row, find_upcoming_fixture,
                            resolve_selection, load_prematch_builds,
                            record_prematch_build)


def main():
    load_env()
    cfg = load_match()
    team = cfg.get("team", "Leyton Orient")

    row = next_upcoming_row(cfg, team)
    if row is None:
        print("No upcoming fixture on record - nothing to build.")
        return

    fixture_id = int(row.id)
    opp = str(row.awaySquadName if row.homeSquadName == team else row.homeSquadName)

    built = load_prematch_builds()
    prev = built.get(str(fixture_id))
    if prev:
        print(f"Already have a report for {opp} (fixture {fixture_id}) - "
              f"built {prev.get('source', '?')} at {prev.get('builtAt', '?')}. "
              f"Skipping.")
        return

    print(f"Next fixture: {team} v {opp} (id {fixture_id}) - no report yet, "
          f"building one.")

    opp_caches = resolve_selection(cfg, {"team": opp, "count": None, "competition": "all"})
    our_caches = resolve_selection(cfg, {"team": team, "count": None, "competition": "all"})
    if not opp_caches or not our_caches:
        print(f"  ! resolved zero matches for one side (opp={len(opp_caches)}, "
              f"us={len(our_caches)}) - skipping for now, will retry next run "
              f"once there's data.")
        return

    _, fixture_text = find_upcoming_fixture(cfg, team, opp)
    env = dict(os.environ)
    env.update({
        "PRE_OPP": opp,
        "PRE_OPP_CACHES": ",".join(opp_caches),
        "PRE_OUR_CACHES": ",".join(our_caches),
    })
    if fixture_text:
        env["PRE_FIXTURE"] = fixture_text

    r = subprocess.run([sys.executable, str(ROOT / "build_prematch.py")],
                       cwd=str(ROOT), env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ! pre-match build failed:\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
        return
    print(f"  built Pre-Match Analysis - {opp}.pdf")

    rc = subprocess.run([sys.executable, str(ROOT / "build_corners.py")],
                        cwd=str(ROOT), env=env, capture_output=True, text=True)
    if rc.returncode != 0:
        print(f"  ! set piece build failed:\n{rc.stdout[-800:]}\n{rc.stderr[-800:]}")
    else:
        print(f"  built Set Piece Report - {opp}.pdf")

    record_prematch_build(fixture_id, opp, "auto", fixture_text=fixture_text or "",
                          sample_note="Every game each side has played this season")


if __name__ == "__main__":
    main()
