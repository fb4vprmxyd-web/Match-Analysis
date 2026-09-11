"""Build the Leyton Orient post-match debrief.

Recreates the Wycombe Wanderers report template
(`Post Match Report v Leyton Orient.pdf`) from Impect match-level API data.

    python3 build_report.py

Data is read from the cached parquet files written by `fetch_data.py`, so this
runs offline once the pull has been done.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

from report_config import (apply_style, TEAM, OPP, COMPETITION, MATCHDAY,
                           VENUE_DATE, surname)
import report_metrics as M
import report_pages as P
import coach_brief as CB
import manual_corrections as MC

from envconfig import setting as _s
CACHE = _s("PMR_CACHE", "cache", "data")
if not os.path.isabs(CACHE):
    CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), CACHE)
# Derived from the match config so it can never be left pointing at the
# previous fixture (it silently overwrote the wrong report once).
OUT = os.environ.get("PMR_OUT", f"Post Match Report - {TEAM} v {OPP}.pdf")


def load():
    ev = pd.read_parquet(os.path.join(CACHE, "ev.parquet"))
    ps = pd.read_parquet(os.path.join(CACHE, "ps.parquet"))
    season_sq = pd.read_parquet(os.path.join(CACHE, "season_squad.parquet"))
    season_pl = pd.read_parquet(os.path.join(CACHE, "season_player.parquet"))
    sp_path = os.path.join(CACHE, "setpieces.parquet")
    setp = pd.read_parquet(sp_path) if os.path.exists(sp_path) else None
    if setp is not None and len(setp):
        setp = MC.apply(setp, int(setp.matchId.iloc[0]))
    ss_path = os.path.join(CACHE, "ss.parquet")
    squad_sums = pd.read_parquet(ss_path) if os.path.exists(ss_path) else None
    fm_path = os.path.join(CACHE, "fm.parquet")
    formations = pd.read_parquet(fm_path) if os.path.exists(fm_path) else None
    sc_path = os.path.join(CACHE, "squad_scores.parquet")
    scores = pd.read_parquet(sc_path) if os.path.exists(sc_path) else None
    shirts = {}
    for fn in ("sp.parquet", "su.parquet"):
        p = os.path.join(CACHE, fn)
        if os.path.exists(p):
            d = pd.read_parquet(p)
            for _, r in d[d.squadName == TEAM].iterrows():
                if pd.notna(r.get("shirtNumber")):
                    shirts[r.playerName] = int(r.shirtNumber)
    return (ev, ps, season_sq, season_pl, shirts, setp, squad_sums,
            scores, formations)


def ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{ {1:'st',2:'nd',3:'rd'}.get(n % 10, 'th') }"


def _split_footnote(bullets):
    """Pull the comparison-basis note out of the bullet list into a footer.

    It is a caveat about how the page was built, not a finding about the match,
    and as an eleventh bullet it overflowed the page.
    """
    note = None
    keep = []
    for b in bullets:
        if b.startswith("Comparisons here are drawn"):
            note = b
        else:
            keep.append(b)
    return keep, "KEY STATS & SUMMARY", note


def rank_phrase(n: int) -> str:
    """'highest' reads better than '1st highest'."""
    return "highest" if n == 1 else f"{ordinal(n)} highest"


def season_context(ev, ps, season_sq, season_pl, tbl, setp=None,
                   squad_sums=None, scores=None, season_scores=None):
    """Factual bullets for the summary page.

    The comparison basis adapts to the sample available. With a full league
    season behind them, claims are ranked against the club's own campaign. In a
    cup tie early in a season there is no such baseline, so the club's own
    ranking is suppressed and the wider competition pool is used instead - a
    same-season reference rather than a stale cross-season one.
    """
    b = []
    ours = season_sq[season_sq.squadName == TEAM].copy() if season_sq is not None \
        else pd.DataFrame()
    n_own = ours.matchId.nunique() if len(ours) else 0
    have_own = n_own >= 5
    pool = season_sq if season_sq is not None else pd.DataFrame()
    n_pool = pool.matchId.nunique() if len(pool) else 0

    # The season pool is pulled once a match is available, which includes
    # THIS match's own row - and that row is computed via a different
    # pipeline (squad matchsums) than the this-match figures below (summed
    # fresh from the event feed), so the two "same" values can differ in the
    # third decimal place. Left in, this match beats itself by a rounding
    # hair and inflates its own rank past the size of the pool ("6th highest
    # of 5 matches" - a real bug this produced for both BPD and BWRD).
    # `others` is the pool with this exact match excluded by id, for ranking
    # only; n_own/n_pool stay as the full count, since "of N matches this
    # season" correctly includes this one - just not as its own competitor.
    cur_match_id = int(ev.matchId.iloc[0]) if "matchId" in ev.columns and len(ev) else None

    def others_of(df):
        if cur_match_id is None or df is None or len(df) == 0 or "matchId" not in df.columns:
            return df
        return df[df.matchId != cur_match_id]

    def pool_pct(col, value, higher_better=True):
        """Percentile of a value against every team performance in the pool."""
        if col not in pool.columns:
            return None
        srs = others_of(pool)[col].dropna()
        if len(srs) < 12:
            return None
        share = (srs < value).mean() if higher_better else (srs > value).mean()
        return share * 100

    def pool_note(col, value, higher_better=True, fmt="{:.2f}"):
        p = pool_pct(col, value, higher_better)
        if p is None:
            return ""
        return (f" That ranks in the {ordinal(int(round(p)))} percentile of the "
                f"{len(pool)} team performances in this season's competition.")

    t = ev[ev.squadName == TEAM]
    o = ev[ev.squadName == OPP]

    # --- 1. xG
    this_xg, opp_xg = float(t.SHOT_XG.sum()), float(o.SHOT_XG.sum())
    lead = (f"Orient generated {this_xg:.2f} xG to {OPP}'s {opp_xg:.2f}")
    if have_own:
        rk = int((others_of(ours).SHOT_XG > this_xg).sum()) + 1
        b.append(f"{lead}, the {rank_phrase(rk)} xG total they recorded in "
                 f"{n_own} matches this season (average "
                 f"{ours.SHOT_XG.mean():.2f}).")
    else:
        b.append(lead + "." + pool_note("SHOT_XG", this_xg))

    # --- 2. bypassed defenders
    bpd_t, bpd_o = float(t.BYPASSED_DEFENDERS.sum()), float(o.BYPASSED_DEFENDERS.sum())
    who = "Orient bypassed more defenders than they conceded" if bpd_t > bpd_o \
        else f"{OPP} bypassed more Orient defenders than Orient bypassed theirs"
    line = f"{who} ({bpd_t:.1f} to {bpd_o:.1f})."
    if have_own:
        rk = int((others_of(ours).BYPASSED_DEFENDERS > bpd_t).sum()) + 1
        line += (f" Orient's {bpd_t:.1f} BPD is the {rank_phrase(rk)} of their "
                 f"{n_own} matches this season.")
    else:
        line += pool_note("BYPASSED_DEFENDERS", bpd_t)
    b.append(line)

    # --- 3. press, split by half
    h1 = float(t[t.periodId == 1].BALL_WIN_REMOVED_OPPONENTS_DEFENDERS.sum())
    h2 = float(t[t.periodId == 2].BALL_WIN_REMOVED_OPPONENTS_DEFENDERS.sum())
    tot_bw = h1 + h2
    if tot_bw > 0 and abs(h1 - h2) / tot_bw < 0.15:
        shape = f"was split almost evenly across the halves ({h1:.1f} then {h2:.1f})"
    else:
        stronger = "first" if h1 > h2 else "second"
        shape = (f"was concentrated in the {stronger} half ({max(h1,h2):.1f} "
                 f"against {min(h1,h2):.1f})")
    line = f"Orient's ball win removed defenders totalled {tot_bw:.1f} and {shape}."
    if have_own:
        srs = ours.BALL_WIN_REMOVED_OPPONENTS_DEFENDERS
        rk = int((others_of(ours).BALL_WIN_REMOVED_OPPONENTS_DEFENDERS > tot_bw).sum()) + 1
        line += (f" That is the {rank_phrase(rk)} figure of their {n_own} matches "
                 f"(average {srs.mean():.1f}).")
    else:
        line += pool_note("BALL_WIN_REMOVED_OPPONENTS_DEFENDERS", tot_bw)
    b.append(line)

    # --- 4. passes into the final third
    pf = M.passes_into_final_third(ev, TEAM)
    ok = int((pf.result == "SUCCESS").sum())
    rate = 0 if len(pf) == 0 else ok / len(pf) * 100
    per = (pf.groupby("playerName")
           .agg(ok=("result", lambda s: (s == "SUCCESS").sum()),
                tot=("result", "size")))
    per = per[per.tot > 1]
    extra = ""
    if len(per):
        per["pc"] = per.ok / per.tot * 100
        r = per.sort_values("pc", ascending=False).iloc[0]
        extra = (f" Of players attempting more than one, "
                 f"{surname(per.sort_values('pc', ascending=False).index[0])} was "
                 f"the most accurate ({int(r.ok)}/{int(r.tot)} - {r.pc:.0f}%).")
    b.append(f"Orient completed {ok} of {len(pf)} open-play passes into the final "
             f"third ({rate:.0f}%).{extra}")

    # --- 5. individuals
    top = tbl.sort_values("BPD", ascending=False).iloc[0]
    topr = tbl.sort_values("BPDR", ascending=False).iloc[0]
    line = (f"{surname(top.playerName)} registered the most BPD in the match "
            f"({top.BPD:.1f}).")
    if season_pl is not None and len(season_pl):
        pl = season_pl[(season_pl.squadName == TEAM)
                       & (season_pl.playerName == top.playerName)]
        by_match = pl.groupby("matchId").BYPASSED_DEFENDERS.sum()
        if len(by_match) >= 5:
            by_match_others = (by_match[by_match.index != cur_match_id]
                               if cur_match_id is not None else by_match)
            rk = int((by_match_others > top.BPD).sum()) + 1
            line += (f" That is his {rank_phrase(rk)} total in {len(by_match)} "
                     f"appearances this season.")
    line += (f" The most BPDR was {topr.BPDR:.1f} by {surname(topr.playerName)}, "
             f"reflecting how much of Orient's progression ran through him as the "
             f"receiver rather than the passer.")
    b.append(line)

    # --- 6. duels
    d = ps[ps.squadName == TEAM][["WON_AERIAL_DUELS", "LOST_AERIAL_DUELS",
                                  "WON_GROUND_DUELS", "LOST_GROUND_DUELS"]].sum()
    aw, al = float(d.WON_AERIAL_DUELS), float(d.LOST_AERIAL_DUELS)
    gw, gl = float(d.WON_GROUND_DUELS), float(d.LOST_GROUND_DUELS)
    ar = aw / (aw + al) * 100 if (aw + al) else 0
    gr = gw / (gw + gl) * 100 if (gw + gl) else 0
    tail = ("Both contests were close to even, so the difference came from what "
            "each side did with the ball rather than from winning it back."
            if 45 <= ar <= 55 and 45 <= gr <= 55 else
            "The physical contest was a clear point of difference between the sides.")
    b.append(f"Orient won {ar:.0f}% of aerial duels ({aw:.0f}/{aw+al:.0f}) and "
             f"{gr:.0f}% of ground duels ({gw:.0f}/{gw+gl:.0f}). {tail}")

    # --- 7. set pieces
    if setp is not None and len(setp):
        tt = M.setpiece_team(setp)
        us = tt[tt.attackingSquadName == TEAM]
        them = tt[tt.attackingSquadName == OPP]
        our_xg, their_xg = us.xg.sum(), them.xg.sum()
        worst = them.sort_values("xg", ascending=False).head(1)
        detail = ""
        if len(worst) and worst.iloc[0].xg > 0:
            w = worst.iloc[0]
            detail = (f" Their most dangerous type was "
                      f"{w.grp.replace('_',' ').lower()}s "
                      f"({w.xg:.2f} xG from {int(w.phases)} phases).")
        verdict = "went Orient's way" if our_xg > their_xg else "went against Orient"
        b.append(f"Set pieces {verdict}: {our_xg:.2f} xG generated from "
                 f"{int(us.phases.sum())} phases against {their_xg:.2f} from "
                 f"{int(them.phases.sum())}.{detail}")

    # --- 8. out of possession
    if squad_sums is not None:
        S = squad_sums.set_index("squadName")
        opr, tpr = (float(S.loc[TEAM, "NUMBER_OF_PRESSES"]),
                    float(S.loc[OPP, "NUMBER_OF_PRESSES"]))
        obw, tbw = (float(S.loc[TEAM, "BALL_WIN_REMOVED_OPPONENTS_DEFENDERS"]),
                    float(S.loc[OPP, "BALL_WIN_REMOVED_OPPONENTS_DEFENDERS"]))
        lane = M.opponent_progression(ev, OPP)
        top_lane = lane.idxmax().replace("_", " ").lower() if lane.sum() else "n/a"
        cmp_ = "more" if opr > tpr else "less"
        pay = ("and got more out of it" if obw > tbw
               else "but got less out of it")
        b.append(f"Orient pressed {cmp_} often than {OPP} ({opr:.0f} pressures to "
                 f"{tpr:.0f}) {pay}: their ball wins removed {obw:.1f} opposition "
                 f"defenders to {tbw:.1f}. {OPP}'s progression came mainly through "
                 f"the {top_lane} ({lane.max():.1f} of their {lane.sum():.1f} "
                 f"bypassed defenders).")

    # --- 9. press profile
    if scores is not None and "PPDA" in scores.columns:
        C = scores.set_index("squadName")
        b.append(
            f"On the press profile Orient posted a PPDA of "
            f"{float(C.loc[TEAM,'PPDA']):.1f} against {OPP}'s "
            f"{float(C.loc[OPP,'PPDA']):.1f}, an average press height of "
            f"{float(C.loc[TEAM,'MEAN_PRESSURE_HEIGHT']):.0f}m against "
            f"{float(C.loc[OPP,'MEAN_PRESSURE_HEIGHT']):.0f}m, and pressed "
            f"{float(C.loc[TEAM,'PRESSURE_DURING_OPP_BUILD_UP_PERCENT'])*100:.0f}% "
            f"of {OPP}'s build-up to their "
            f"{float(C.loc[OPP,'PRESSURE_DURING_OPP_BUILD_UP_PERCENT'])*100:.0f}%.")

    # --- 10. expected points
    xp = M.expected_points(scores, TEAM, OPP)
    if xp:
        b.append(f"On Impect's expected-points model Orient were worth "
                 f"{xp['xg'][0]:.2f} points to {OPP}'s {xp['xg'][1]:.2f} on xG, and "
                 f"{xp['psxg'][0]:.2f} to {xp['psxg'][1]:.2f} on post-shot xG, with "
                 f"a {xp['win'][0]*100:.0f}% win probability against "
                 f"{xp['win'][1]*100:.0f}%.")

    if not have_own:
        b.append(f"Comparisons here are drawn against the {len(pool)} team "
                 f"performances in this season's competition, not against last "
                 f"season - the squad has changed too much for 25/26 to be a fair "
                 f"yardstick, and Orient have only {n_own} match of 26/27 data.")

    return b


def ours_conceded(ours: pd.DataFrame) -> pd.Series:
    """xG conceded per match, aligned to `ours` row order."""
    return ours.CONCEDED_SHOT_XG.fillna(0)


def match_score(ev):
    """(home, away, homeGoals, awayGoals) for this match.

    An own goal's row is credited to the scorer's own squad (GOALS is NaN
    there), not the side it benefits - has to be added to the other side by
    hand or a match with one shows the wrong scoreline. Factored out of
    main() so the golden-value regression test exercises this exact
    function rather than a reimplementation of it.
    """
    home = ev.homeSquadName.iloc[0]
    away = ev.awaySquadName.iloc[0]
    og = ev[ev.actionType == "OWN_GOAL"]
    hg = (int(ev[ev.squadName == home].GOALS.sum())
         + int((og.squadName == away).sum()))
    ag = (int(ev[ev.squadName == away].GOALS.sum())
         + int((og.squadName == home).sum()))
    return home, away, hg, ag


def main():
    apply_style()
    (ev, ps, season_sq, season_pl, shirts, setp, squad_sums, scores,
     formations) = load()

    home, away, hg, ag = match_score(ev)
    score = f"{hg} - {ag}"
    score_line = f"{home} ({hg}) vs ({ag}) {away}"

    rows = M.team_stats(ev, ps, TEAM, OPP, scores)
    tbl = M.player_table(ev, ps, TEAM)
    bench = M.season_benchmarks(season_pl, TEAM)
    dtbl = M.defensive_player_table(ev, ps, TEAM)
    # Impect's published pxT totals - the timeline is reconciled to these
    xt_totals = ps.groupby("squadName")["PXT_ATTACK"].sum().to_dict()
    mins, full_match = M.minutes_map(ps, TEAM)
    su_path = os.path.join(CACHE, 'su.parquet')
    su = pd.read_parquet(su_path) if os.path.exists(su_path) else None
    first_change, changes = M.substitution_summary(su, TEAM)
    review = M.performance_review(ev, ps, squad_sums, scores, setp, tbl,
                                  TEAM, OPP)
    # coach_brief.TARGETS is written against one match's specific pre-match
    # notes (see its own docstring) - grading a different match against them
    # produces a page that looks fine but is actually checking instructions
    # nobody gave for this game. Skip it until the real notes for this match
    # are in, rather than ship a page that quietly compares to the wrong brief.
    include_brief = not os.environ.get("PMR_SKIP_COACH_BRIEF")
    brief = (CB.evaluate(ev, ps, squad_sums, scores, setp, formations, TEAM, OPP)
            if include_brief else None)

    pages = [
        lambda f: P.page_cover(f, TEAM, OPP, COMPETITION, MATCHDAY,
                               VENUE_DATE, score),
        lambda f: P.page_match_stats(f, rows, TEAM, OPP),
        lambda f: P.page_team_charts_a(f, ev, TEAM, OPP, score_line,
                                       xt_totals=xt_totals),
        lambda f: P.page_team_charts_b(f, ev, TEAM, shirts,
                                      first_change, changes),
    ]
    if setp is not None:
        pages.append(lambda f: P.page_set_pieces(f, setp, ev, TEAM, OPP))
    if squad_sums is not None:
        pages.append(lambda f: P.page_out_of_possession(
            f, ev, ps, squad_sums, dtbl, TEAM, OPP, scores))
    pages += [
        lambda f: P.page_player_stats_a(f, ev, TEAM, mins, full_match, OPP),
        lambda f: P.page_player_stats_b(f, tbl, bench, mins, full_match),
        lambda f: P.page_summary(f, *_split_footnote(
            season_context(ev, ps, season_sq, season_pl, tbl, setp,
                           squad_sums, scores))),
        lambda f: P.page_performance_review(f, *review, TEAM, OPP, score_line),
    ]
    if include_brief:
        pages.append(lambda f: P.page_coach_brief(f, brief, TEAM, OPP, score_line))

    with PdfPages(OUT) as pdf:
        for builder in pages:
            fig = plt.figure(figsize=(10.67, 14.23), facecolor="#171E29")
            builder(fig)
            pdf.savefig(fig, facecolor="#171E29")
            plt.close(fig)

    print(f"wrote {OUT}")
    for label, lt, rt, *_ in rows:
        print(f"  {label:32s} {lt:>12s}  |  {rt}")


if __name__ == "__main__":
    main()
