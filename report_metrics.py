"""Metric definitions for the post-match debrief.

Every figure printed in the report is computed here from Impect match-level
data, so the mapping from template metric -> Impect field is auditable:

    xG                    SHOT_XG
    Post-Shot xG          POSTSHOT_XG
    xT                    PXT_ATTACK          (Impect packing xT)
    BPD                   BYPASSED_DEFENDERS
    BPDR                  BYPASSED_DEFENDERS_RECEIVING
    BLRT                  BALL_LOSS_REMOVED_TEAMMATES
    Ball recoveries       BALL_WIN_NUMBER
    Pressures             NUMBER_OF_PRESSES   (player matchsums only)

Coordinates: Impect ships `*AdjCoordinates*` already normalised so the acting
team always attacks towards +X. Raw coordinates flip at half-time; adjusted
ones do not, so all spatial charts use the adjusted pair.
"""

import numpy as np
import pandas as pd

from report_config import (surname, SETPIECE_XG_CONCEDED_CONCERN,
                           SETPIECE_XG_CREATED_GOOD,
                           SETPIECE_XG_CREATED_POOR)

PASS_ACTIONS = ("LOW_PASS", "CHIPPED_PASS", "DIAGONAL_PASS",
                "SHORT_AERIAL_PASS", "LOW_CROSS", "HIGH_CROSS")
CROSS_ACTIONS = ("LOW_CROSS", "HIGH_CROSS")
FINAL_THIRD = ("FINAL_THIRD", "OPPONENT_BOX")
ONBALL = ("PASS", "RECEPTION", "DRIBBLE")


# --------------------------------------------------------------- helpers
def minute(ev: pd.DataFrame) -> pd.Series:
    """Continuous match minute. gameTimeInSec carries a 10000s offset in P2."""
    sec = ev.gameTimeInSec.astype(float)
    return np.where(ev.periodId == 1, sec / 60.0, (sec - 10000.0) / 60.0 + 45.0)


def possession_share(ev: pd.DataFrame) -> dict:
    """Time-on-ball share.

    The raw `duration` column contains one event spanning the half-time break
    (~7119s), which would swamp the totals. Recomputing from gameTimeInSec
    within each period avoids that.
    """
    e = ev.sort_values("eventNumber").copy()
    e["_sec"] = e.gameTimeInSec.astype(float)
    e["_d"] = e.groupby("periodId")["_sec"].diff().shift(-1)
    e["_d"] = e["_d"].where((e["_d"] >= 0) & (e["_d"] < 120), 0.0)
    tot = e.groupby("attackingSquadName")["_d"].sum()
    return (tot / tot.sum() * 100).to_dict()


def _sum_by_squad(ev, col):
    return ev.groupby("squadName")[col].sum().to_dict()


def _count_by_squad(ev, mask):
    return ev[mask].groupby("squadName").size().to_dict()


def open_play(ev):
    return ev.phase.ne("SET_PIECE")


def _goals_by_squad(ev: pd.DataFrame) -> dict:
    """{squadName: goals}, own goals credited to the side that benefits -
    same logic as build_report.py's match_score(), generalised from
    home/away to any squad name so team_stats() doesn't need the match
    score passed in separately."""
    out = ev.groupby("squadName").GOALS.sum().to_dict()
    for _, og in ev[ev.actionType == "OWN_GOAL"].iterrows():
        scorer = og.squadName
        beneficiary = next((s for s in out if s != scorer), None)
        if beneficiary is not None:
            out[beneficiary] = out.get(beneficiary, 0) + 1
    return out


# ------------------------------------------------------------ team stats
def team_stats(ev: pd.DataFrame, psum: pd.DataFrame, team: str, opp: str,
               scores: pd.DataFrame = None) -> tuple:
    """The page-2 profile table, split into (attacking_rows, defending_rows) -
    coach-specified list and order, see git history for the handwritten spec
    this replaced the old single flat list with.

    Possession/Field Tilt/Bypassed Defenders/Pass Accuracy/Box Touches/
    Duels/Fouls/Corners from the old version are deliberately dropped, not
    carried forward - this is the new list, not an addition to the old one.
    """
    goals = _goals_by_squad(ev)

    cr_all = ev[ev.action.isin(CROSS_ACTIONS)]
    cr_tot = cr_all.groupby("squadName").size().to_dict()
    cr_ok = cr_all[cr_all.result == "SUCCESS"].groupby("squadName").size().to_dict()

    box_passes = ev[(ev.actionType == "PASS") & (ev.result == "SUCCESS")
                    & (ev.startPitchPosition != "OPPONENT_BOX")
                    & (ev.endPitchPosition == "OPPONENT_BOX")]
    box_passes_n = box_passes.groupby("squadName").size().to_dict()

    shots_in = _count_by_squad(ev, (ev.actionType == "SHOT")
                               & (ev.startPitchPosition == "OPPONENT_BOX"))
    shots_tot = _sum_by_squad(ev, "SHOT_AT_GOAL_NUMBER")
    sot = _sum_by_squad(ev, "SHOT_AT_GOAL_NUMBER_ON_TARGET")

    ints = _count_by_squad(ev, ev.actionType == "INTERCEPTION")
    wins = _sum_by_squad(ev, "BALL_WIN_NUMBER")
    bwrd = _sum_by_squad(ev, "BALL_WIN_REMOVED_OPPONENTS_DEFENDERS")
    xt = psum.groupby("squadName")["PXT_ATTACK"].sum().to_dict()
    press = psum.groupby("squadName")["NUMBER_OF_PRESSES"].sum().to_dict() \
        if "NUMBER_OF_PRESSES" in psum.columns else {}

    def g(d, k, default=0.0):
        return float(d.get(k, default) or 0.0)

    def pct(num, den):
        return 0.0 if den == 0 else num / den * 100

    def row(label, d_team, d_opp, fmt="{:.0f}"):
        vt, vo = g(d_team, team), g(d_opp, opp)
        return (label, fmt.format(vt), fmt.format(vo), vt, vo)

    attacking = [
        row("Goals", goals, goals),
        ("xG", f"{g(_sum_by_squad(ev,'SHOT_XG'),team):.2f}",
         f"{g(_sum_by_squad(ev,'SHOT_XG'),opp):.2f}",
         g(_sum_by_squad(ev, "SHOT_XG"), team), g(_sum_by_squad(ev, "SHOT_XG"), opp)),
    ]
    ps_ = _sum_by_squad(ev, "POSTSHOT_XG")
    attacking.append(("Post-Shot xG", f"{g(ps_,team):.2f}", f"{g(ps_,opp):.2f}",
                      g(ps_, team), g(ps_, opp)))
    attacking.append(("Shots / on target",
                      f"{g(shots_tot,team):.0f}/{g(sot,team):.0f}",
                      f"{g(shots_tot,opp):.0f}/{g(sot,opp):.0f}",
                      g(shots_tot, team), g(shots_tot, opp)))
    attacking.append(row("Inside Box", shots_in, shots_in))
    outside_in = {team: g(shots_tot, team) - g(shots_in, team),
                 opp: g(shots_tot, opp) - g(shots_in, opp)}
    attacking.append(row("Outside Box", outside_in, outside_in))
    attacking.append(("Packing xT", f"{g(xt,team):.2f}", f"{g(xt,opp):.2f}",
                      g(xt, team), g(xt, opp)))
    attacking.append(row("Passes Into Box", box_passes_n, box_passes_n))
    ct, co = g(cr_tot, team), g(cr_tot, opp)
    attacking.append(("Crosses (Accurate)",
                      f"({pct(g(cr_ok,team), ct):.0f}%) {ct:.0f}",
                      f"{co:.0f} ({pct(g(cr_ok,opp), co):.0f}%)", ct, co))

    defending = [
        row("Interceptions", ints, ints),
        row("Pressures", press, press),
        row("Ball Wins", wins, wins),
        ("Ball Win Removed Defenders", f"{g(bwrd,team):.1f}", f"{g(bwrd,opp):.1f}",
         g(bwrd, team), g(bwrd, opp)),
    ]
    return attacking, defending


# ------------------------------------------------------- event selections
def shots(ev: pd.DataFrame) -> pd.DataFrame:
    s = ev[ev.actionType == "SHOT"].copy()
    s["is_goal"] = s.SHOT_AT_GOAL_NUMBER_SUCCESS.fillna(0) > 0
    s["xg"] = s.SHOT_XG.fillna(0)
    return s


def passes(ev: pd.DataFrame, squad: str) -> pd.DataFrame:
    return ev[(ev.squadName == squad) & (ev.actionType == "PASS")].copy()


def passes_into_final_third(ev, squad):
    p = passes(ev, squad)
    return p[(~p.startPitchPosition.isin(FINAL_THIRD))
             & (p.endPitchPosition.isin(FINAL_THIRD)) & open_play(p)]


def passes_in_final_third(ev, squad):
    p = passes(ev, squad)
    return p[p.startPitchPosition.isin(FINAL_THIRD)
             & p.endPitchPosition.isin(FINAL_THIRD) & open_play(p)]


def crosses_and_cutbacks(ev, squad):
    p = passes(ev, squad)
    return p[p.action.isin(CROSS_ACTIONS) & open_play(p)]


def passes_into_box(ev, squad):
    p = passes(ev, squad)
    return p[(p.startPitchPosition != "OPPONENT_BOX")
             & (p.endPitchPosition == "OPPONENT_BOX") & open_play(p)]


def box_touches(ev, squad):
    """Every event that started with the ball already in the box - see the
    note on the team_stats() box_touch count for why this isn't filtered to
    OFFENSIVE_TOUCHES>0."""
    return ev[(ev.squadName == squad)
              & (ev.startPitchPosition == "OPPONENT_BOX")].copy()


def cross_target_share(ev, squad):
    """Share of open-play crosses reaching the central or back-post zone.

    'Back post' is lateral, not depth: a cross from the right finishing on the
    left of the goal. Shared with the Coach's Brief so both pages agree.
    """
    c = ev[(ev.squadName == squad) & ev.action.isin(CROSS_ACTIONS)
           & ev.phase.ne("SET_PIECE")].dropna(
        subset=["endAdjCoordinatesX", "endAdjCoordinatesY",
                "startAdjCoordinatesY"])
    c = c[c.endAdjCoordinatesX > 36]
    if len(c) == 0:
        return None, 0, {}
    side = np.sign(c.startAdjCoordinatesY)
    ey = c.endAdjCoordinatesY
    back = (np.sign(ey) == -side) & (ey.abs() > 4)
    central = ey.abs() <= 7
    zone = np.where(back, "back post", np.where(central, "central", "near post"))
    counts = pd.Series(zone).value_counts().to_dict()
    return float((back | central).sum()) / len(c), len(c), counts


def split_success(df):
    ok = df[df.result == "SUCCESS"]
    bad = df[df.result != "SUCCESS"]
    return ok, bad


# ------------------------------------------------------------ player level
def minutes_map(psum: pd.DataFrame, squad: str):
    """Minutes played per player, plus the full-match length for this game.

    Matchsums carry one row per player-position, so a player who changed role
    mid-match must have his durations summed before converting.
    """
    m = (psum[psum.squadName == squad].groupby("playerName")["playDuration"]
         .sum().div(60.0))
    full = float(m.max()) if len(m) else 90.0
    return m.to_dict(), full


def player_table(ev: pd.DataFrame, psum: pd.DataFrame, squad: str) -> pd.DataFrame:
    """Per-player totals for the page-6 grid."""
    e = ev[(ev.squadName == squad) & ev.playerName.notna()]
    agg = e.groupby("playerName").agg(
        BPD=("BYPASSED_DEFENDERS", "sum"),
        BPDR=("BYPASSED_DEFENDERS_RECEIVING", "sum"),
        xG=("SHOT_XG", "sum"),
        succ=("SUCCESSFUL_PASSES", "sum"),
        unsucc=("UNSUCCESSFUL_PASSES", "sum"),
        BLRT=("BALL_LOSS_REMOVED_TEAMMATES", "sum"),
        recoveries=("BALL_WIN_NUMBER", "sum"),
    )
    # Duels and pressures are only populated in the matchsums (see team_stats).
    ms = (psum[psum.squadName == squad].groupby("playerName")
          [["WON_AERIAL_DUELS", "LOST_AERIAL_DUELS", "WON_GROUND_DUELS",
            "LOST_GROUND_DUELS", "NUMBER_OF_PRESSES", "PXT_ATTACK"]].sum())
    ms = ms.reindex(agg.index).fillna(0)
    agg["xT"] = ms.PXT_ATTACK          # matchsum, not events - see team_stats
    agg["aw"] = ms.WON_AERIAL_DUELS
    agg["al"] = ms.LOST_AERIAL_DUELS
    agg["gw"] = ms.WON_GROUND_DUELS
    agg["gl"] = ms.LOST_GROUND_DUELS
    agg["pressures"] = ms.NUMBER_OF_PRESSES
    agg["passes"] = agg.succ + agg.unsucc
    agg["aerials"] = agg.aw + agg.al
    agg["grounds"] = agg.gw + agg.gl
    return agg.reset_index()


def substitution_summary(su: pd.DataFrame, squad: str):
    """First change (match minute) and the list of changes.

    Minutes come from the printed `gameTime` string; `gameTimeInSec` carries a
    10000s offset in the second period and reading it raw puts a 69th-minute
    substitution at minute 192.
    """
    if su is None or len(su) == 0:
        return None, []
    d = su[(su.squadName == squad)
           & su.substitutionType.isin(["SUB_ON", "SUB_OFF"])].copy()
    if len(d) == 0:
        return None, []
    d["min"] = d.gameTime.astype(str).str.split(":").str[0].astype(int)
    changes = []
    for m, grp in d.groupby("min"):
        off = [p for p in grp[grp.substitutionType == "SUB_OFF"].playerName]
        on = [p for p in grp[grp.substitutionType == "SUB_ON"].playerName]
        changes.append((int(m), off, on))
    return int(d["min"].min()), sorted(changes)


def passing_network(ev: pd.DataFrame, squad: str, start_min=None, end_min=None):
    """Average successful-pass positions + pair volumes, optionally windowed.

    Windowing matters: drawing one network over the whole match mixes a starting
    XI that played 69 minutes with substitutes who played 20, and plots their
    average positions as though comparable.
    """
    p = ev[(ev.squadName == squad) & (ev.actionType == "PASS")
           & (ev.result == "SUCCESS") & ev.playerName.notna()
           & ev.passReceiverPlayerName.notna()
           & (ev.passReceiverType == "TEAMMATE")].copy()
    if start_min is not None or end_min is not None:
        p["_min"] = minute(p)
        if start_min is not None:
            p = p[p["_min"] >= start_min]
        if end_min is not None:
            p = p[p["_min"] < end_min]
    nodes = p.groupby("playerName").agg(
        x=("startAdjCoordinatesX", "mean"),
        y=("startAdjCoordinatesY", "mean"),
        n=("eventId", "count")).reset_index()
    edges = (p.groupby(["playerName", "passReceiverPlayerName"])
             .size().reset_index(name="n"))
    return nodes, edges


def label_offsets(nodes: pd.DataFrame, ax, size_col="n", s_base=95,
                  s_extra=520, pad=1.5, push=7.5):
    """Where to print each node's label, WITHOUT moving the node.

    Average positions are the data - two players genuinely occupying the same
    ground is a finding, not a rendering fault, so the circles stay exactly
    where Impect puts them. Only the label is displaced, and only when it would
    otherwise be printed on top of another one.

    "Would otherwise be printed on top of another one" has to be judged
    against the REAL rendered circle, not a guessed one - `s_base`/`s_extra`
    must match the caller's own `ax.scatter(..., s=s_base + s_extra * n/max)`
    marker-size formula exactly, and `ax` must already have its final xlim/
    ylim/aspect set (i.e. called after draw_pitch), since the conversion from
    matplotlib's points-based marker size to a data-space radius goes through
    the axes' own current pixel scale. A fixed pitch-unit guess here
    previously overstated the true circle size, flagging labels as clashing
    - and pushing them outside the circle with a leader line - even when
    they were nowhere near touching.

    Returns a list of (label_x, label_y, needs_leader_line) per row.
    """
    if nodes is None or len(nodes) == 0:
        return []
    x = nodes.x.to_numpy(dtype=float)
    y = nodes.y.to_numpy(dtype=float)
    smax = float(nodes[size_col].max()) or 1.0
    s_vals = s_base + s_extra * (nodes[size_col].to_numpy(dtype=float) / smax)

    fig = ax.figure
    radius_pt = np.sqrt(s_vals / np.pi)                 # matplotlib size -> points
    radius_px = radius_pt * fig.dpi / 72.0               # points -> pixels
    inv = ax.transData.inverted()
    p0 = np.array(inv.transform((0, 0)))
    p1 = np.array(inv.transform((1, 0)))
    data_per_px = float(np.hypot(*(p1 - p0)))            # pixels -> data units
    rad = radius_px * data_per_px + pad

    out = []
    for i in range(len(nodes)):
        clash = [j for j in range(len(nodes))
                 if j != i and np.hypot(x[j] - x[i], y[j] - y[i]) < rad[i] + rad[j]]
        if not clash:
            out.append((x[i], y[i], False))
            continue
        # step away from the crowd, so two clashing labels go opposite ways
        cx, cy = float(np.mean(x[clash])), float(np.mean(y[clash]))
        dx, dy = x[i] - cx, y[i] - cy
        d = float(np.hypot(dx, dy))
        if d < 1e-6:
            dx, dy, d = 0.0, 1.0, 1.0
        lx = float(np.clip(x[i] + dx / d * push, -50, 50))
        ly = float(np.clip(y[i] + dy / d * push, -31, 31))
        out.append((lx, ly, True))
    return out


def recovery_grid(ev: pd.DataFrame, squad: str, ncols=6, nrows=5):
    """Share of a team's ball recoveries per pitch cell, attacking left->right."""
    r = ev[(ev.squadName == squad) & (ev.BALL_WIN_NUMBER > 0)]
    if len(r) == 0:
        return np.zeros((nrows, ncols)), {}
    xb = np.linspace(-52.5, 52.5, ncols + 1)
    yb = np.linspace(-34, 34, nrows + 1)
    H, _, _ = np.histogram2d(r.startAdjCoordinatesX, r.startAdjCoordinatesY,
                             bins=[xb, yb])
    grid = (H.T / len(r)) * 100
    grid = grid[::-1]                       # top row = high Y
    thirds = {
        "Own Third": int((r.startAdjCoordinatesX < -17.5).sum()),
        "Mid Third": int(r.startAdjCoordinatesX.between(-17.5, 17.5).sum()),
        "Final Third": int((r.startAdjCoordinatesX > 17.5).sum()),
    }
    return grid, thirds


# ========================================================= season benchmarks
# Each page-6 bar is a single-match total, which on its own says nothing about
# whether it was good. These medians give every bar a reference point.
BENCH_SPEC = {
    "BPD": ["BYPASSED_DEFENDERS"],
    "BPDR": ["BYPASSED_DEFENDERS_RECEIVING"],
    "xT": ["PXT_ATTACK"],
    "xG": ["SHOT_XG"],
    "passes": ["SUCCESSFUL_PASSES", "UNSUCCESSFUL_PASSES"],
    "aerials": ["WON_AERIAL_DUELS", "LOST_AERIAL_DUELS"],
    "grounds": ["WON_GROUND_DUELS", "LOST_GROUND_DUELS"],
    "BLRT": ["BALL_LOSS_REMOVED_TEAMMATES"],
    "pressures": ["NUMBER_OF_PRESSES"],
    "recoveries": ["BALL_WIN_NUMBER"],
}


def season_benchmarks(season_pl: pd.DataFrame, squad: str,
                      min_seconds: float = 3600, min_apps: int = 5):
    """Median per-match output, restricted to matches the player largely started.

    Restricting to 60+ minutes stops a run of cameo appearances from dragging a
    player's baseline down and making an ordinary match look exceptional.
    Matchsums carry one row per player-position, so totals are summed per
    (player, match) before the median is taken.
    """
    d = season_pl[season_pl.squadName == squad]
    if len(d) == 0:
        return pd.DataFrame()
    cols = sorted({c for v in BENCH_SPEC.values() for c in v})
    agg = {c: "sum" for c in cols}
    agg["playDuration"] = "sum"
    g = d.groupby(["playerName", "matchId"]).agg(agg).reset_index()
    g = g[g.playDuration >= min_seconds]
    for name, srcs in BENCH_SPEC.items():
        g[name] = g[srcs].sum(axis=1)
    out = g.groupby("playerName").agg(
        apps=("matchId", "nunique"),
        **{n: (n, "median") for n in BENCH_SPEC})
    return out[out.apps >= min_apps]


# ============================================================== set pieces
SP_ORDER = ["CORNER", "FREE_KICK", "THROW_IN", "GOAL_KICK"]


def _sp_group(cat: str) -> str:
    if isinstance(cat, str) and cat.startswith("CORNER"):
        return "CORNER"
    return cat


def setpiece_phase_concentration(sp: pd.DataFrame, squad: str):
    """Share of a team's set-piece xG produced by its single biggest phase.

    'X xG from set pieces' can mean two very different things: a genuine
    pattern spread across several deliveries, or one costly phase (often a
    penalty won during a corner/throw-in scramble, since Impect attributes the
    penalty's xG to the dead-ball phase it occurred in) sitting inside an
    otherwise-empty total. A rule that only checks the total cannot tell these
    apart, and calling the second one a "soft spot" implies a repeatable
    defensive issue that the data doesn't actually show.

    Returns (share, category label, whether that phase produced a goal).
    share is 0.0 if the team has no set-piece xG at all.
    """
    d = sp[sp.attackingSquadName == squad]
    total = d["setPieceSubPhase_SHOT_XG"].sum()
    if total <= 0 or len(d) == 0:
        return 0.0, None, False
    idx = d["setPieceSubPhase_SHOT_XG"].idxmax()
    row = d.loc[idx]
    share = float(row["setPieceSubPhase_SHOT_XG"] / total)
    goal = bool(row.get("setPieceSubPhase_GOALS", 0) > 0)
    cat = _sp_group(row["setPieceCategory"])
    label = cat.replace("_", " ").title() if isinstance(cat, str) else "phase"
    return share, label, goal


# A single phase carrying this much of the total is "one moment", not a
# pattern worth coaching as a set-piece weakness or weapon.
SETPIECE_CONCENTRATION_THRESHOLD = 0.60


def setpiece_team(sp: pd.DataFrame) -> pd.DataFrame:
    """Volume and value of every set-piece phase, by attacking team."""
    d = sp.copy()
    d["grp"] = d.setPieceCategory.map(_sp_group)
    out = d.groupby(["attackingSquadName", "grp"]).agg(
        phases=("setPieceId", "size"),
        shots=("setPieceSubPhase_SHOT_AT_GOAL_NUMBER", "sum"),
        xg=("setPieceSubPhase_SHOT_XG", "sum"),
        goals=("setPieceSubPhase_GOALS", "sum"),
        bpd=("setPieceSubPhase_BYPASSED_DEFENDERS", "sum"),
        pxt=("setPieceSubPhase_PXT_POSITIVE", "sum"),
    ).reset_index()
    return out


def setpiece_players(sp: pd.DataFrame, squad: str) -> pd.DataFrame:
    """Who took set pieces and what they were worth."""
    d = sp[sp.attackingSquadName == squad]
    d = d[d.setPieceSubPhaseMainEventPlayerName.notna()]
    return (d.groupby("setPieceSubPhaseMainEventPlayerName")
            .agg(taken=("setPieceId", "size"),
                 xg=("setPieceSubPhase_SHOT_XG", "sum"),
                 bpd=("setPieceSubPhase_BYPASSED_DEFENDERS", "sum"),
                 ok=("setPieceSubPhaseMainEventOutcome",
                     lambda s: (s == "SUCCESSFUL").sum()))
            .reset_index()
            .rename(columns={"setPieceSubPhaseMainEventPlayerName": "playerName"})
            .sort_values("taken", ascending=False))


def setpiece_first_contact(sp: pd.DataFrame) -> pd.DataFrame:
    """First-contact outcome on deliveries where it was recorded."""
    d = sp[sp.setPieceSubPhaseFirstTouchWon.notna()]
    return (d.groupby(["attackingSquadName", "setPieceSubPhaseFirstTouchWon"])
            .size().unstack(fill_value=0))


# ========================================================= out of possession
def defensive_team(squad_sums: pd.DataFrame, team: str, opp: str) -> list:
    """Team-level out-of-possession comparison rows."""
    s = squad_sums.set_index("squadName")

    def g(col, sq):
        return float(s.loc[sq, col]) if col in s.columns else 0.0

    rows = [
        ("Suffered Bypassed Defenders", "{:.1f}",
         g("SUFFERED_BYPASSED_DEFENDERS", team), g("SUFFERED_BYPASSED_DEFENDERS", opp)),
        ("Conceded xG", "{:.2f}", g("CONCEDED_SHOT_XG", team), g("CONCEDED_SHOT_XG", opp)),
        ("Conceded Post-Shot xG", "{:.2f}",
         g("CONCEDED_POSTSHOT_XG", team), g("CONCEDED_POSTSHOT_XG", opp)),
        ("Pressures", "{:.0f}", g("NUMBER_OF_PRESSES", team), g("NUMBER_OF_PRESSES", opp)),
        ("Counter-Pressures", "{:.0f}",
         g("NUMBER_OF_PRESSES_COUNTER_PRESS", team),
         g("NUMBER_OF_PRESSES_COUNTER_PRESS", opp)),
        ("Presses Between the Lines", "{:.0f}",
         g("NUMBER_OF_PRESSES_BETWEEN_THE_LINES", team),
         g("NUMBER_OF_PRESSES_BETWEEN_THE_LINES", opp)),
        ("Defensive Touches", "{:.0f}",
         g("DEFENSIVE_TOUCHES", team), g("DEFENSIVE_TOUCHES", opp)),
        ("Ball Win Removed Defenders", "{:.1f}",
         g("BALL_WIN_REMOVED_OPPONENTS_DEFENDERS", team),
         g("BALL_WIN_REMOVED_OPPONENTS_DEFENDERS", opp)),
    ]
    return rows


def press_profile(scores: pd.DataFrame, team: str, opp: str) -> list:
    """Press-quality metrics from `getSquadMatchScores`.

    PPDA is deliberately kept out of the two-sided comparison bars: lower is
    better for it, which inverts the meaning of bar length. These render as
    tiles instead, each carrying its own direction-of-good note.
    """
    if scores is None or "PPDA" not in scores.columns:
        return []
    C = scores.set_index("squadName")

    def g(col, sq, mult=1.0):
        return float(C.loc[sq, col]) * mult if col in C.columns else float("nan")

    spec = [
        ("PPDA", "{:.1f}", "PPDA", 1.0, "lower = more aggressive"),
        ("Mean Press Height", "{:.0f}m", "MEAN_PRESSURE_HEIGHT", 1.0,
         "metres upfield"),
        ("Opp Build-Up Pressed", "{:.0f}%",
         "PRESSURE_DURING_OPP_BUILD_UP_PERCENT", 100.0, "higher = better"),
        ("Opp Opening Balls Pressed", "{:.0f}%",
         "OPP_OPENING_BALLS_PRESSED_PERCENT", 100.0, "higher = better"),
        ("Forced High Passes", "{:.0f}%", "FORCED_HIGH_PASSES_PERCENT", 100.0,
         "higher = better"),
    ]
    out = []
    for label, fmt, col, mult, note in spec:
        out.append((label, fmt, g(col, team, mult), g(col, opp, mult), note))
    return out


def expected_points(scores: pd.DataFrame, team: str, opp: str):
    """Impect's xPts model - the cleanest one-line verdict on a match."""
    if scores is None or "EXPECTED_POINTS_SHOT_XG" not in scores.columns:
        return None
    C = scores.set_index("squadName")
    return {
        "xg": (float(C.loc[team, "EXPECTED_POINTS_SHOT_XG"]),
               float(C.loc[opp, "EXPECTED_POINTS_SHOT_XG"])),
        "psxg": (float(C.loc[team, "EXPECTED_POINTS_POSTSHOT_XG"]),
                 float(C.loc[opp, "EXPECTED_POINTS_POSTSHOT_XG"])),
        "win": (float(C.loc[team, "WIN_PROBABLITY_SHOT_XG"]),
                float(C.loc[opp, "WIN_PROBABLITY_SHOT_XG"])),
        "draw": float(C.loc[team, "DRAW_PROBABILITY_SHOT_XG"]),
    }


def performance_review(ev, psum, squad_sums, scores, setp, tbl, team, opp,
                       result_word, max_items=6):
    """Interpretation layer: what held up, what needs work.

    Every entry is gated on a condition computed from the match, so the verdict
    changes with the data instead of being prose pinned to one game. Kept light
    on figures deliberately — the numbers are on the eight pages before this.

    Also returns a `how_it_went` paragraph - a plain-language scene-setter
    built from the same signals this function already computes for pros/
    cons (territory, chance quality, duels, final-third fluency), so it
    stays a real read of the match rather than a single generic sentence.
    """
    S = squad_sums.set_index("squadName")
    C = scores.set_index("squadName") if scores is not None else None

    def s(col, sq=None):
        sq = sq or team
        return float(S.loc[sq, col]) if col in S.columns else 0.0

    def c(col, sq=None):
        sq = sq or team
        if C is None or col not in C.columns:
            return None
        return float(C.loc[sq, col])

    pros, cons = [], []

    # --- possession / territory
    poss, tilt = c("RATIO_BALL_POSSESSION"), c("FIELD_TILT")
    if tilt and tilt > 0.55:
        # Language scales with the margin: 64% is "far more", 55% is not.
        strength = "far more" if tilt >= 0.62 else "the greater share"
        if poss and poss > 0.53:
            detail = (f"Orient had both the ball and the territory, spending "
                      f"{strength} of the game in the opposition's half - a "
                      f"platform to attack from for long spells.")
        else:
            # possession near 50/50 - claiming "more of the ball" would be false
            detail = (f"Possession was shared almost evenly, but Orient still "
                      f"spent {strength} of the game in the opposition's half. "
                      f"It wasn't about having more of the ball, it was about "
                      f"where that ball spent its time.")
        pros.append(("Controlled the territory", detail))

    # --- did control become danger?
    xg, oxg = s("SHOT_XG"), s("SHOT_XG", opp)
    if oxg - xg >= 0.20:
        cons.append(("Control never became danger",
                     "Holding the territory still produced the poorer set of "
                     "chances. Turning field position into genuine threat is the "
                     "single biggest gap in this performance."))
    elif xg - oxg >= 0.20:
        detail = ("Orient created the better chances, though neither side made "
                  "many. The edge in quality was theirs, but the margin was slim."
                  if xg < 1.0 else
                  "Holding the territory translated into the better and more "
                  "dangerous openings, which is the point of controlling a game.")
        pros.append(("Created the better chances", detail))

    # --- press design
    ppda, oppda = c("PPDA"), c("PPDA", opp)
    ph, oph = c("MEAN_PRESSURE_HEIGHT"), c("MEAN_PRESSURE_HEIGHT", opp)
    if ppda and oppda and ph and oph and ppda < oppda and ph > oph:
        pros.append(("The press was well built",
                     "Orient pressed higher up the pitch and more aggressively "
                     "than the opposition on every measure. The out-of-possession "
                     "shape was not the problem."))

    # --- but did the press pay?
    bwrd, obwrd = s("BALL_WIN_REMOVED_OPPONENTS_DEFENDERS"), \
        s("BALL_WIN_REMOVED_OPPONENTS_DEFENDERS", opp)
    # A 6% gap is noise. Only flag this when the opposition's turnovers were
    # materially more damaging than ours.
    if obwrd > 0 and bwrd < obwrd * 0.8:
        cons.append(("Aggression without reward",
                     "The ball was won back often but rarely anywhere that hurt "
                     "the opposition, whose own turnovers were noticeably more "
                     "damaging. The answer is better traps and cover, not more "
                     "running."))

    # --- duels
    aw, al = s("WON_AERIAL_DUELS"), s("LOST_AERIAL_DUELS")
    gw, gl = s("WON_GROUND_DUELS"), s("LOST_GROUND_DUELS")
    ar = aw / (aw + al) if (aw + al) else 0
    gr = gw / (gw + gl) if (gw + gl) else 0
    if 0.45 <= ar <= 0.55 and 0.45 <= gr <= 0.55:
        pros.append(("Stood up physically",
                     "Both the aerial and ground contests finished all but even. "
                     "This was not a game lost to being out-fought."))
    elif ar < 0.45 and gr < 0.45:
        cons.append(("Second best in the duels",
                     "Beaten in the air and on the ground. Before anything "
                     "technical is addressed, the side has to compete better for "
                     "loose and contested ball."))
    else:
        # The joint check above only speaks when both duel types agree. When
        # they diverge it goes silent on both - so a side can win the aerial
        # battle outright while losing on the ground and get credit for
        # neither. >55%/<45% deliberately match the tied band above: anything
        # inside 45-55% is "roughly even" everywhere else in this function, so
        # a bare 51-49 split does not count as a "won" isolated duel type here
        # either - it takes a real margin, not a one-duel edge, to be singled
        # out.
        if aw + al >= 8:
            if ar > 0.55:
                pros.append(("Won the aerial battle",
                             "A clear edge in the air, regardless of how the "
                             "ground contest went."))
            elif ar < 0.45:
                cons.append(("Beaten in the air",
                             "Beaten in the air often enough for it to be a "
                             "pattern rather than a one-off. Worth addressing "
                             "on its own, separately from how the ground "
                             "duels went."))
        if gw + gl >= 8:
            if gr > 0.55:
                pros.append(("Won it on the ground",
                             "A clear edge in the tackles and contested "
                             "balls, regardless of the aerial picture."))
            elif gr < 0.45:
                cons.append(("Beaten on the ground",
                             "The side is losing the individual battles that "
                             "decide second balls and loose-ball recoveries."))

    # --- final third entry
    pf = passes_into_final_third(ev, team)
    rate = None
    if len(pf):
        rate = (pf.result == "SUCCESS").sum() / len(pf)
        if rate < 0.55:
            cons.append(("Entry into the final third breaks down",
                         "Well under half of the open-play passes into the final "
                         "third found a teammate. The attack stalls at exactly "
                         "the moment it needs to be sharpest."))
        elif rate > 0.65:
            pros.append(("Clean entry into the final third",
                         "Most attempts to play into the final third found a "
                         "teammate, so attacks arrived intact rather than "
                         "breaking down on the way in."))

    # --- where the opposition hurt them
    lane = opponent_progression(ev, opp)
    if len(lane) and lane.sum() > 0:
        top = lane.idxmax()
        centre_share = lane.get("CENTER", 0) / lane.sum()
        if top == "CENTER" and centre_share > 0.35:
            cons.append(("The middle was left open",
                         "The opposition did most of their damage straight "
                         "through the centre rather than around it. Protecting "
                         "the space between the lines is the clearest structural "
                         "fix."))
        elif "WING" in str(top):
            cons.append(("Opened up out wide",
                         "The opposition made most of their progress down the "
                         "flanks. The wide areas need better cover from the "
                         "full-back and the man in front of him."))
        elif centre_share <= 0.30:
            # The mirror of "The middle was left open" above - that con only
            # ever fires on failure, so a side that genuinely kept the centre
            # shut got no credit for it. 30% (vs the con's 35%) leaves a
            # deliberate gap between the two so a borderline 32-33% game stays
            # silent rather than being called a clean success.
            pros.append(("Kept the middle shut",
                         "Only a small share of the opposition's progression "
                         "came through the centre, and no single channel "
                         "dominated - their threat was spread thin across the "
                         "width of the pitch rather than punching a hole down "
                         "the middle."))

    # --- set pieces
    # Both branches need an ABSOLUTE floor as well as a ratio. A ratio alone
    # fires on negligible volumes: 0.32 xG conceded against 0.14 created is a
    # 2.3x ratio but no weakness at all, and it contradicted the Coach's Brief.
    if setp is not None and len(setp):
        t = setpiece_team(setp)
        us = t[t.attackingSquadName == team]
        them = t[t.attackingSquadName == opp]
        f, a = us.xg.sum(), them.xg.sum()
        our_phases = int(us.phases.sum())
        conceded_goals = int(them.goals.sum())

        if a >= SETPIECE_XG_CONCEDED_CONCERN and a > f * 1.5:
            share, cat, goal = setpiece_phase_concentration(setp, opp)
            if share >= SETPIECE_CONCENTRATION_THRESHOLD:
                goal_txt = " that produced a goal" if goal else ""
                cons.append(("A costly set-piece moment",
                             f"Dead balls conceded a real chance, but nearly "
                             f"all of it came from a single {cat.lower()} "
                             f"phase{goal_txt}. That is one costly moment, "
                             f"not a repeated pattern - a different problem "
                             f"from leaking chances across several "
                             f"deliveries."))
            else:
                cons.append(("Set pieces were a soft spot",
                             "Dead balls against were the richest source of "
                             "danger in the match, and produced more than the "
                             "opposition managed in open play."))
        elif f >= SETPIECE_XG_CREATED_GOOD and f > a * 1.5:
            share, cat, goal = setpiece_phase_concentration(setp, team)
            if share >= SETPIECE_CONCENTRATION_THRESHOLD:
                goal_txt = " that produced a goal" if goal else ""
                pros.append(("One big set-piece moment",
                             f"A real chance created from dead balls, but "
                             f"nearly all of it came from a single "
                             f"{cat.lower()} phase{goal_txt} - worth having, "
                             f"but not yet evidence of a repeatable set-piece "
                             f"routine."))
            else:
                pros.append(("Dangerous from dead balls",
                             "Set pieces were a genuine weapon, producing more "
                             "threat than the opposition managed from theirs."))
        elif a < SETPIECE_XG_CONCEDED_CONCERN and conceded_goals == 0:
            pros.append(("Dead balls defended well",
                         "Almost nothing was conceded from set pieces despite the "
                         "opposition having plenty of them - the defending of the "
                         "box on dead balls held up."))

        # The attacking return is a separate question from the defensive one.
        if our_phases >= 20 and f < SETPIECE_XG_CREATED_POOR:
            worst = us.sort_values("phases", ascending=False).head(1)
            detail = ""
            if len(worst):
                w = worst.iloc[0]
                detail = (f" Even the most-used delivery, "
                          f"{w.grp.replace('_',' ').lower()}s, barely "
                          f"produced anything.")
            cons.append(("Our own set pieces produced nothing",
                         f"Plenty of set-piece phases generated barely any "
                         f"threat at all.{detail} That is a lot of possession "
                         f"in good areas being handed back."))

    # --- crossing
    cr = crosses_and_cutbacks(ev, team)
    if len(cr) >= 5:
        ok = (cr.result == "SUCCESS").sum() / len(cr)
        if ok < 0.3:
            # Separate the delivery from the occupation. Saying "crossing was
            # wasteful" while the Coach's Brief passes the cross target zone
            # reads as a contradiction; both are true and the combination is the
            # actual finding.
            zone_share, _, _ = cross_target_share(ev, team)
            if zone_share is not None and zone_share >= 0.5:
                cons.append(("Crosses found no one in the box",
                             "The deliveries reached the right areas - most found "
                             "the central or back-post zone - yet almost none "
                             "found a teammate. The ball into the box is not the "
                             "problem; who is arriving on it is."))
            else:
                cons.append(("Crossing was wasteful",
                             "Almost every open-play cross failed to find a "
                             "teammate, and the deliveries were not reaching the "
                             "dangerous zones either."))

    # --- goalkeeping load
    cps, cg = s("CONCEDED_POSTSHOT_XG"), s("CONCEDED_GOALS")
    if cps > cg + 0.5:
        pros.append(("The goalkeeper kept them in it",
                     f"{opp} created more than they actually scored - good "
                     f"goalkeeping made up the difference."))

    # --- individual bright spots
    # Both were previously unconditional, so the column always gained two
    # positives however ordinary the individual numbers were. Each detail
    # names its own player rather than leaning on "he"/"him": these get
    # joined into one flowing paragraph with no title shown, and a bare
    # pronoun here used to borrow whatever subject the previous sentence
    # happened to have (reading as if the goalkeeper had done this too).
    if len(tbl):
        topp = tbl.sort_values("BPD", ascending=False).iloc[0]
        topr = tbl.sort_values("BPDR", ascending=False).iloc[0]
        bpd_tot, bpdr_tot = tbl.BPD.sum(), tbl.BPDR.sum()
        bpd_share = topp.BPD / bpd_tot if bpd_tot else 0
        bpdr_share = topr.BPDR / bpdr_tot if bpdr_tot else 0

        if topp.BPD >= 8 or bpd_share >= 0.25:
            name = surname(topp.playerName)
            pros.append((f"{name} broke lines",
                         f"{name} carried and passed through the opposition "
                         f"more than anyone else on the pitch - proof there's "
                         f"a route forward when the ball is used decisively."))
        if bpdr_share >= 0.25:
            name = surname(topr.playerName)
            pros.append((f"A clear focal point in {name}",
                         f"A big share of Orient's progression went through "
                         f"{name} as the man receiving the ball, which says a "
                         f"lot about where the side looks to get out."))

    # --- how it went: a plain-language scene-setter for the Performance
    # Review page, reusing the same signals already computed above
    # (territory, chance quality, duels, final-third fluency) rather than
    # a single generic result sentence - every branch below always
    # produces something, so the paragraph length doesn't depend on how
    # eventful the match was.
    verb = {"won": "beat", "lost": "lost to", "drew": "drew with"}[result_word]
    sentences = [f"{team} {verb} {opp}."]

    if tilt is not None:
        if tilt > 0.55:
            sentences.append(f"{team} had the better share of the game, "
                             f"spending long spells in the opposition's half.")
        elif tilt < 0.45:
            sentences.append(f"{opp} had the better share of the game, and "
                             f"{team} spent more time defending than attacking.")
        else:
            sentences.append("Territory was shared fairly evenly between "
                             "the two sides.")

    if xg - oxg >= 0.20:
        sentences.append("That control showed up in the chances too, with "
                         "the better and more dangerous openings falling to "
                         f"{team}.")
    elif oxg - xg >= 0.20:
        sentences.append(f"{opp} created the better chances of the two "
                         f"sides, even when they didn't have the run of play.")
    else:
        sentences.append("Chances were shared fairly evenly, without either "
                         "side stringing together a run of clear openings.")

    if 0.45 <= ar <= 0.55 and 0.45 <= gr <= 0.55:
        sentences.append("Physically it was an even contest, with neither "
                         "side dominating the challenges.")
    elif ar < 0.45 and gr < 0.45:
        sentences.append(f"{team} came off second best in the physical "
                         f"battles, beaten in the air and on the ground.")
    elif ar > 0.55 and gr > 0.55:
        sentences.append(f"{team} won the physical battle, coming out on "
                         f"top in the air and on the ground.")
    else:
        sentences.append("The physical side of the game was mixed, with "
                         "each side having their moments.")

    if rate is None or 0.55 <= rate <= 0.65:
        sentences.append("Moving the ball into the final third was a "
                         "routine, unremarkable part of the game for both "
                         "sides.")
    elif rate < 0.55:
        sentences.append("Getting the ball into dangerous areas was a "
                         "struggle, with a lot of promising moves breaking "
                         "down in the final third.")
    else:
        sentences.append("When the ball reached the final third it usually "
                         "stuck, rather than being given straight back.")

    sentences.append("Here's a closer look at what stood out.")
    how_it_went = " ".join(sentences)

    return pros[:max_items], cons[:max_items], how_it_went


def defensive_player_table(ev: pd.DataFrame, psum: pd.DataFrame,
                           squad: str) -> pd.DataFrame:
    """Per-player defensive volume.

    Note: SUFFERED_BYPASSED_DEFENDERS exists only at squad level (every player
    row is 0), so 'who got bypassed' cannot be shown per player. What follows
    is what the feed does attribute individually.
    """
    e = ev[(ev.squadName == squad) & ev.playerName.notna()]
    counts = (e[e.actionType.isin(["INTERCEPTION", "CLEARANCE", "BLOCK"])]
              .groupby(["playerName", "actionType"]).size()
              .unstack(fill_value=0))
    for c in ("INTERCEPTION", "CLEARANCE", "BLOCK"):
        if c not in counts.columns:
            counts[c] = 0
    agg = e.groupby("playerName").agg(
        recoveries=("BALL_WIN_NUMBER", "sum"),
        bwrd=("BALL_WIN_REMOVED_OPPONENTS_DEFENDERS", "sum"))
    agg = agg.join(counts, how="left").fillna(0)

    pcols = ["NUMBER_OF_PRESSES", "NUMBER_OF_PRESSES_BUILD_UP",
             "NUMBER_OF_PRESSES_BETWEEN_THE_LINES",
             "NUMBER_OF_PRESSES_COUNTER_PRESS", "DEFENSIVE_TOUCHES"]
    ms = (psum[psum.squadName == squad].groupby("playerName")[pcols].sum()
          .reindex(agg.index).fillna(0))
    agg = agg.join(ms)
    agg["press_other"] = (agg.NUMBER_OF_PRESSES - agg.NUMBER_OF_PRESSES_BUILD_UP
                          - agg.NUMBER_OF_PRESSES_BETWEEN_THE_LINES
                          - agg.NUMBER_OF_PRESSES_COUNTER_PRESS).clip(lower=0)
    agg["def_actions"] = agg.INTERCEPTION + agg.CLEARANCE + agg.BLOCK
    return agg.reset_index()


def opponent_progression(ev: pd.DataFrame, opp: str):
    """Where the opposition bypassed defenders, by lane and by pitch third."""
    o = ev[(ev.squadName == opp) & (ev.BYPASSED_DEFENDERS > 0)]
    by_lane = o.groupby("startLane").BYPASSED_DEFENDERS.sum()
    order = ["LEFT_WING", "LEFT_HALF_SPACE", "CENTER",
             "RIGHT_HALF_SPACE", "RIGHT_WING"]
    by_lane = by_lane.reindex(order).fillna(0)
    return by_lane


def dribbles_conceded(ev: pd.DataFrame, opp: str) -> pd.DataFrame:
    """The one place individual defensive responsibility is named."""
    d = ev[(ev.squadName == opp) & (ev.actionType == "DRIBBLE")
           & ev.dribbleOpponentPlayerName.notna()]
    if len(d) == 0:
        return pd.DataFrame(columns=["playerName", "faced", "beaten"])
    return (d.groupby("dribbleOpponentPlayerName")
            .agg(faced=("eventId", "size"),
                 beaten=("result", lambda s: (s == "SUCCESS").sum()))
            .reset_index().rename(columns={"dribbleOpponentPlayerName": "playerName"})
            .sort_values("faced", ascending=False))
