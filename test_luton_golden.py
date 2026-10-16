"""Golden-value regression test, pinned to the Luton Town post-match report.

That report (data_luton_town_270182, matchId 270182) is the one match this
project has verified number-by-number end to end: every row of team_stats(),
every player in player_table(), and every ranking claim in season_context()
was independently recomputed from raw parquet and confirmed correct - see the
"Luton post-match report" verification pass. It also happens to be the
report that caught the self-inclusion ranking bug (season_context() had a
match "beat itself" and claim "6th highest of 5 matches") - bullets [0] and
[1] below exist specifically to catch that bug coming back.

This does NOT exercise own-goal crediting (Luton's match has no own goal) -
see test_bradford_golden.py for that path.

The actual data loading/computation happens in an isolated subprocess (via
_golden_helpers.load_golden -> _golden_runner.py), not in this process -
build_report.py resolves its match config from the environment only on
first import, so importing it directly here would make this file's result
depend on whether pytest happened to collect it before or after
test_bradford_golden.py. See _golden_runner.py's docstring.

Run with:  pytest test_luton_golden.py -v
       or: python3 test_luton_golden.py

If Impect revises this match's data, these numbers will legitimately need
re-pinning - look at what load_golden() actually returns before updating the
expected values, rather than just pasting in whatever the new output is.
"""

import os

import pytest

from _golden_helpers import load_golden

from envconfig import CACHE_ROOT
ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(str(CACHE_ROOT), "data_luton_town_270182")

if not os.path.exists(CACHE_DIR):
    pytest.skip(f"{CACHE_DIR} not pulled - nothing to check against",
                allow_module_level=True)

TEAM, OPP = "Leyton Orient", "Luton Town"


@pytest.fixture(scope="module")
def data():
    return load_golden("270182", TEAM, OPP, CACHE_DIR, "League One 26/27",
                       "5. Spieltag", "away - Saturday 5 September 2026")


# ------------------------------------------------------------- the scoreline
def test_score(data):
    assert (data["home"], data["away"]) == ("Luton Town", "Leyton Orient")
    assert (data["hg"], data["ag"]) == (1, 1), (
        f"expected Luton 1 - 1 Orient, got {data['hg']} - {data['ag']}")


# --------------------------------------------------------------- team_stats
# (label, team_string, opp_string) exactly as they render on the match-stats
# page - pinned as the formatted strings, not the raw floats, since a
# formatting regression (wrong rounding, swapped side) is exactly what this
# should catch and a raw-float comparison could miss.
EXPECTED_TEAM_STATS = [
    ["xG", "0.60", "2.40"],
    ["Post-Shot xG", "0.98", "1.05"],
    ["xT", "0.86", "0.78"],
    ["Bypassed Defenders", "36.1", "66.1"],
    ["Possession", "47%", "53%"],
    ["Field Tilt", "55%", "45%"],
    ["Box Touches", "22", "61"],
    ["Shots", "6", "20"],
    ["Shots on Target", "3", "6"],
    ["Successful Passes", "377", "438"],
    ["Pass Accuracy", "72%", "76%"],
    ["Open-Play Crosses (Accuracy)", "(40%) 10", "10 (50%)"],
    ["Aerial Duels", "(41%) 32", "32 (59%)"],
    ["Ground Duels", "(52%) 48", "48 (48%)"],
    ["Interceptions", "38", "29"],
    ["Fouls", "12", "9"],
    ["Corners", "2", "7"],
]


def test_team_stats(data):
    assert data["team_stats"] == EXPECTED_TEAM_STATS


# ------------------------------------------------------------- player_table
# Three players verified this session: Olowu (the specific "aerial duels are
# wrong" flag - checked out correct, 1/1), Clare (cross-checked alongside
# him), Kabia (the receiver Orient's progression actually ran through).
EXPECTED_PLAYER_ROWS = {
    "Joseph Olowu": dict(BPD=6.99, BPDR=0.0, xG=0.0, succ=46.0, unsucc=11.0,
                         aw=1.0, al=0.0, gw=5.0, gl=1.0, pressures=10.0),
    "Sean Clare": dict(BPD=3.0, BPDR=0.0, xG=0.0, succ=9.0, unsucc=2.0,
                       aw=1.0, al=0.0, gw=0.0, gl=1.0, pressures=3.0),
    "Jaze Kabia": dict(BPD=pytest.approx(1.3), BPDR=7.8, xG=0.055, succ=7.0,
                       unsucc=2.0, aw=0.0, al=2.0, gw=1.0, gl=4.0, pressures=33.0),
}


def test_player_table(data):
    players = data["players"]
    for name, expected in EXPECTED_PLAYER_ROWS.items():
        assert name in players, f"{name} missing from player_table()"
        row = players[name]
        for col, exp_val in expected.items():
            got = float(row[col])
            assert got == pytest.approx(exp_val, abs=0.01), (
                f"{name}.{col}: expected {exp_val}, got {got}")


# --------------------------------------------------------- season_context()
# Full bullet list, pinned verbatim. Bullets [0] and [1] are the ones that
# actually caught the self-inclusion ranking bug ("6th highest of 5
# matches" before the fix in build_report.py's season_context/others_of) -
# if a future change to that ranking logic regresses, these two are the
# tripwire.
EXPECTED_BULLETS = [
    "Orient generated 0.60 xG to Luton Town's 2.40, the 5th highest xG total "
    "they recorded in 5 matches this season (average 1.09).",
    "Luton Town bypassed more Orient defenders than Orient bypassed theirs "
    "(36.1 to 66.1). Orient's 36.1 BPD is the 5th highest of their 5 matches "
    "this season.",
    "Orient's ball win removed defenders totalled 6.2 and was concentrated "
    "in the second half (4.4 against 1.8). That is the 5th highest figure "
    "of their 5 matches (average 11.1).",
    "Orient completed 14 of 50 open-play passes into the final third (28%). "
    "Of players attempting more than one, Mitchell was the most accurate "
    "(2/4 - 50%).",
    "Olowu registered the most BPD in the match (7.0). That is his highest "
    "total in 5 appearances this season. The most BPDR was 7.8 by Kabia, "
    "reflecting how much of Orient's progression ran through him as the "
    "receiver rather than the passer.",
    "Orient won 41% of aerial duels (13/32) and 52% of ground duels "
    "(25/48). The physical contest was a clear point of difference between "
    "the sides.",
    "Set pieces went against Orient: 0.03 xG generated from 50 phases "
    "against 0.80 from 49. Their most dangerous type was corners (0.71 xG "
    "from 8 phases).",
    "Orient pressed more often than Luton Town (168 pressures to 139) but "
    "got less out of it: their ball wins removed 6.2 opposition defenders "
    "to 14.8. Luton Town's progression came mainly through the center "
    "(19.5 of their 66.1 bypassed defenders).",
    "On the press profile Orient posted a PPDA of 6.8 against Luton Town's "
    "9.0, an average press height of 71m against 63m, and pressed 40% of "
    "Luton Town's build-up to their 35%.",
    "On Impect's expected-points model Orient were worth 0.38 points to "
    "Luton Town's 2.47 on xG, and 1.29 to 1.40 on post-shot xG, with a 7% "
    "win probability against 77%.",
]


def test_season_context_bullets(data):
    assert data["season_context"] == EXPECTED_BULLETS


def test_season_context_no_self_inclusion(data):
    """Narrower tripwire for the exact bug that was found: a rank phrase
    like '6th highest of 5 matches' is impossible (the pool only has 5
    matches in it) and means the current match is being counted against
    itself. Matches on "5 matches" loosely, not "of 5 matches", since the
    real bullets phrase it various ways ("of their 5 matches this
    season") - the first version of this check used a too-strict substring
    and would have missed the very bug it was written to catch."""
    for b in data["season_context"]:
        if "5 matches" in b or "5 appearances" in b:
            assert "6th" not in b and "7th" not in b, (
                f"rank exceeds pool size - self-inclusion bug is back: {b!r}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
