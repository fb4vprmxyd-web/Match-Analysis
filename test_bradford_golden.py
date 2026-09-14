"""Golden-value regression test, pinned to the Bradford City post-match
report (data_bradford_city_302115, matchId 302115).

Companion to test_luton_golden.py, covering the one path Luton's match
can't: Luton finished 1-1 with no own goal, so a regression in own-goal
crediting (build_report.py's match_score() - a real bug found and fixed
this session, where a scorer's own-goal row is credited to THEIR squad in
the raw data but belongs on the OTHER side's tally) would sail straight
through it undetected. Bradford's match has exactly one own goal
(Gillesphey, credited to Orient's score), so this is the tripwire for that.

Deliberately NOT pinning coach_brief.py's CB.evaluate() output here, even
though it was verified number-by-number for this match too: coach_brief.py
is a single rotating config (see its own module docstring) that gets
rewritten for whichever match currently has real dictated notes - Bradford
is just the resting state at the time this test was written, not a
permanent fixture. Pinning its output would make this test fail the moment
a future match's real Brief replaces Bradford's, for no actual regression.
team_stats() and the scoreline don't have that problem - they're pure
functions of this match's own cached data, so they're safe to pin.

See _golden_runner.py's docstring for why the actual data loading happens
in an isolated subprocess rather than importing build_report here directly.

Run with:  pytest test_bradford_golden.py -v
       or: python3 test_bradford_golden.py
"""

import os

import pytest

from _golden_helpers import load_golden

from envconfig import CACHE_ROOT
ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(str(CACHE_ROOT), "data_bradford_city_302115")

if not os.path.exists(CACHE_DIR):
    pytest.skip(f"{CACHE_DIR} not pulled - nothing to check against",
                allow_module_level=True)

TEAM, OPP = "Leyton Orient", "Bradford City"


@pytest.fixture(scope="module")
def data():
    return load_golden("302115", TEAM, OPP, CACHE_DIR, "EFL Cup 26/27",
                       "Third Round", "Brisbane Road - Tuesday 8 September 2026")


# ------------------------------------------------------- the own-goal score
def test_score_credits_own_goal_to_the_right_side(data):
    """The actual bug this file exists to catch: an own goal's row is
    tagged to the SCORER's own squad in the raw data (GOALS is NaN there),
    not the side it benefits - match_score() has to add it to the other
    side by hand. Before the fix this match rendered as 0-3, not 1-3.

    Reads data["hg"]/data["ag"] straight from BR.match_score() as run by
    _golden_runner.py (the real function main() uses), not a
    reimplementation of it - otherwise this test would only ever check its
    own copy of the logic and never notice if the real function regressed.
    """
    assert len(data["own_goal_squads"]) == 1, (
        f"expected exactly one own goal in this match's data, found "
        f"{len(data['own_goal_squads'])} - if Impect revised this match, "
        "the pinned score below may need updating")
    assert data["own_goal_squads"][0] == "Bradford City", (
        "own-scorer's squad changed - re-check which side the goal should "
        "now be credited to before touching the pinned score")

    assert (data["home"], data["away"]) == ("Leyton Orient", "Bradford City")
    assert (data["hg"], data["ag"]) == (1, 3), (
        f"expected Leyton Orient 1 - 3 Bradford City (the own goal must land "
        f"on Orient's side of the score), got {data['hg']} - {data['ag']}")


# --------------------------------------------------------------- team_stats
EXPECTED_TEAM_STATS = [
    ["xG", "1.83", "2.25"],
    ["Post-Shot xG", "1.30", "4.16"],
    ["xT", "1.26", "3.71"],
    ["Bypassed Defenders", "60.1", "70.4"],
    ["Possession", "52%", "48%"],
    ["Field Tilt", "49%", "51%"],
    ["Box Touches", "26", "23"],
    ["Shots", "16", "17"],
    ["Shots on Target", "5", "8"],
    ["Successful Passes", "319", "270"],
    ["Pass Accuracy", "70%", "64%"],
    ["Open-Play Crosses (Accuracy)", "(18%) 17", "17 (35%)"],
    ["Aerial Duels", "(34%) 47", "47 (66%)"],
    ["Ground Duels", "(45%) 73", "73 (55%)"],
    ["Interceptions", "49", "35"],
    ["Fouls", "9", "10"],
    ["Corners", "8", "12"],
]


def test_team_stats(data):
    assert data["team_stats"] == EXPECTED_TEAM_STATS


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
