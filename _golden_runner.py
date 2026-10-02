"""Loads one match's report data and dumps the numbers golden tests pin, as
one line of JSON on stdout. Not a test file itself - always invoked as an
isolated subprocess (see _golden_helpers.run_isolated), never imported.

Isolation is the whole point: build_report.py resolves TEAM/OPP/CACHE from
the environment through report_config, but only the FIRST time it's
imported in a process - Python caches the module after that. Two golden
test files both doing `import build_report` in the same pytest process
would silently share whichever one imported it first, so test B would
quietly run against test A's match. Running each in a fresh subprocess
sidesteps that instead of trying to out-clever the module cache.

    python3 _golden_runner.py <match_id> <team> <opp> <cache> <comp> <matchday> <venue>
"""

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")


def main():
    match_id, team, opp, cache, comp, matchday, venue = sys.argv[1:8]
    os.environ["PMR_MATCH_ID"] = match_id
    os.environ["PMR_TEAM"] = team
    os.environ["PMR_OPP"] = opp
    os.environ["PMR_CACHE"] = cache
    os.environ["PMR_COMP"] = comp
    os.environ["PMR_MATCHDAY"] = matchday
    os.environ["PMR_VENUE"] = venue

    import build_report as BR
    import report_metrics as M

    (ev, ps, season_sq, season_pl, shirts, setp, squad_sums,
     scores) = BR.load()

    home, away, hg, ag = BR.match_score(ev)
    og = ev[ev.actionType == "OWN_GOAL"]

    tbl = M.player_table(ev, ps, team)
    players = {name: {k: float(v) for k, v in row.items()}
              for name, row in tbl.set_index("playerName").to_dict(orient="index").items()}

    out = {
        "home": home, "away": away, "hg": hg, "ag": ag,
        "own_goal_squads": og.squadName.tolist(),
        "team_stats": [[r[0], r[1], r[2]]
                       for r in M.team_stats(ev, ps, team, opp, scores)],
        "players": players,
        "season_context": BR.season_context(ev, ps, season_sq, season_pl,
                                            tbl, setp, squad_sums, scores),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
