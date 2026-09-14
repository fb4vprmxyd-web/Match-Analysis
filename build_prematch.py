"""Pre-match analysis of an upcoming opponent, from recent league form.

    python3 build_prematch.py

Second-generation version. The first build_prematch.py (used for AFC Wimbledon,
Sheffield Wednesday, Wigan Athletic) worked from a single reference match on
each side - useful early on, but genuinely one data point standing in for a
pattern. This version aggregates each side's last N cached matches - not
necessarily the same competition, or the same count, on both sides. Walsall's
sample was their last 2 League Two games; Barnsley's is their last 2 League
One + 2 EFL Cup games (4 total), against Orient's last 2 League One + however
many EFL Cup games are actually published by Impect at build time (its data
lags kickoff by several days, and doesn't always land in order). The division
check and every match-count caption is computed per side, not assumed equal.

Every squad/player matchsum number is the MEAN across a side's cached matches
(so figures stay on a normal single-game scale); every event-derived number
(lanes, press profile, shot locations) is computed from the matches' events
concatenated together, so percentages are recomputed from real combined totals
rather than averaged as ratios-of-ratios. The one thing that cannot be
aggregated is a starting-XI passing network - a lineup is a lineup - so that
one visual is drawn from the single most recent match on each side and labelled
as such.

The old "THE PROFILE" page compared the opponent to whoever THEY had played in
their one reference match - a comparison that meant less as soon as there was
a second data point. It's replaced here with Orient's own recent form (also
aggregated across two matches) as the comparison side throughout: every chart
that used to read "them vs their prior opponent" now reads "them vs us."
"""

import os
import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

import report_metrics as RM
from report_config import (apply_style, BG, PANEL, INK, MUTED, GRID, GOOD, BAD,
                           BARC, TEAM_C, PAGE_W, PAGE_H, style_bar_axis, surname,
                           draw_pitch, contrast_ink)
from envconfig import CACHE_ROOT, REPORTS_ROOT


def _resolve_caches(names):
    """Bare cache names resolve under CACHE_ROOT; an absolute path (or one
    already containing a separator, for a relative override) passes through
    unchanged - matches build_report.py's own CACHE resolution."""
    out = []
    for n in names:
        n = n.strip()
        if os.path.isabs(n) or os.sep in n:
            out.append(n)
        else:
            out.append(os.path.join(str(CACHE_ROOT), n))
    return out


# ------------------------------------------------------------------ config
OPPONENT = os.environ.get("PRE_OPP", "FC Barnsley")
# each side's last cached matches, comma-separated so a future match day just
# adds one. Not necessarily the same competition or the same count on both
# sides - see the module docstring.
OPP_CACHES = _resolve_caches(os.environ.get(
    "PRE_OPP_CACHES",
    "data_barnsley_l1_1,data_barnsley_l1_2,data_barnsley_cup_1,data_barnsley_cup_2"
).split(","))
US = "Leyton Orient"
# Orient's Walsall EFL Cup tie isn't published by Impect yet (checked
# 2026-08-28, still unavailable days after kickoff) - our own sample is 3
# matches (2 League One + 1 EFL Cup) against Barnsley's 4, asymmetric on
# purpose rather than waiting. Add the Walsall cache here once it lands.
OUR_CACHES = _resolve_caches(os.environ.get(
    "PRE_OUR_CACHES",
    "data_sheffield_wednesday,data,data_oxford_united").split(","))
OUR_COMPETITION = os.environ.get("PRE_OUR_COMP", "League One")
FIXTURE = os.environ.get("PRE_FIXTURE",
                         "League One · Brisbane Road · Saturday 29 August 2026")
os.makedirs(REPORTS_ROOT, exist_ok=True)
OUT = os.path.join(str(REPORTS_ROOT), f"Pre-Match Analysis - {OPPONENT}.pdf")

# Colours per opponent - extend as needed; falls back to a neutral slate blue
# so a new opponent never crashes the build for want of a hex code. Walsall
# and Orient are both red clubs, so Walsall gets a darker, distinguishable
# shade rather than a colour that isn't really theirs.
OPP_COLOURS = {
    "Sheffield Wednesday": "#2E86DE",
    "AFC Wimbledon": "#FDB913",
    "Wigan Athletic": "#1D428A",
    "FC Walsall": "#7A1F2B",
    # also a red club - a third, deliberately distinct red so the profile
    # charts don't put two near-identical reds side by side
    "FC Barnsley": "#8B1414",
}
OPP_C = os.environ.get("PRE_OPP_C", OPP_COLOURS.get(OPPONENT, "#5B7A99"))
# "REF_C" now means us (Orient) on every comparison chart, not a reference
# opponent - kept as Orient's own brand red so it reads consistently with
# every other report in this suite.
REF_C = TEAM_C


def short_labels(names):
    """Surname, or initial+surname where two players share one.

    Max Lowe and Jamal Lowe both rendered as "Lowe", which is worse than
    useless on a chart the coach reads at a glance.
    """
    from collections import Counter
    sur = {n: surname(n) for n in names}
    dupes = {k for k, v in Counter(sur.values()).items() if v > 1}
    return {n: (f"{n.split()[0][0]}. {s}" if s in dupes else s)
            for n, s in sur.items()}


# ------------------------------------------------------------------- data
MIRROR = {"LEFT_WING": "RIGHT_WING", "RIGHT_WING": "LEFT_WING",
          "LEFT_HALF_SPACE": "RIGHT_HALF_SPACE",
          "RIGHT_HALF_SPACE": "LEFT_HALF_SPACE", "CENTER": "CENTER"}
LANES = ["LEFT_WING", "LEFT_HALF_SPACE", "CENTER", "RIGHT_HALF_SPACE",
         "RIGHT_WING"]
LANE_LABEL = ["Left\nwing", "Left\nhalf-space", "Centre", "Right\nhalf-space",
              "Right\nwing"]
ZONE_LABEL = {"FBL": "Left back", "FBR": "Right back", "CB": "Centre backs",
              "DM": "Defensive mid", "CM": "Centre mid", "AM": "Attacking mid",
              "WL": "Left wing", "WR": "Right wing", "IB": "Inside box",
              "IBWL": "Box wide left", "IBWR": "Box wide right"}


_ENDPOINTS = ("ss", "ps", "ev", "squad_scores", "setpieces", "fm", "su", "sp")


def _load_matches(caches):
    """Load and concatenate every cached endpoint across N matches.

    A squad matchsum (ss/squad_scores) frame concatenated this way carries one
    row per match, which is exactly what `_mean_row` below wants; an events
    frame concatenated this way just IS the combined event stream, since a raw
    event has no notion of "which match" beyond its own matchId column.
    """
    out = {n: [] for n in _ENDPOINTS}
    for c in caches:
        c = c.strip()
        for n in _ENDPOINTS:
            p = os.path.join(c, f"{n}.parquet")
            if os.path.exists(p):
                out[n].append(pd.read_parquet(p))
    return {n: (pd.concat(frames, ignore_index=True) if frames else None)
            for n, frames in out.items()}


def _most_recent_cache(caches):
    """Which cache holds the most recent match, by its own date - list order
    is a convention, not something to trust blindly."""
    best, best_dt = caches[0].strip(), None
    for c in caches:
        c = c.strip()
        p = os.path.join(c, "ev.parquet")
        if not os.path.exists(p):
            continue
        when = pd.to_datetime(pd.read_parquet(p).iloc[0].dateTime,
                              errors="coerce", utc=True)
        if best_dt is None or (pd.notna(when) and when > best_dt):
            best, best_dt = c, when
    return best


def load():
    opp = _load_matches(OPP_CACHES)
    ours = _load_matches(OUR_CACHES)
    opp["recent_cache"] = _most_recent_cache(OPP_CACHES)
    return opp, ours


def _mean_row(df, squad):
    """A squad's matchsum columns, averaged across however many cached
    matches it appears in - the per-game scale every page already expects,
    just no longer resting on a single match."""
    if df is None:
        return pd.Series(dtype=float)
    d = df[df.squadName == squad]
    return d.mean(numeric_only=True) if len(d) else pd.Series(dtype=float)


def facts(opp, ours):
    """Every number the document quotes, derived once.

    `opp` and `ours` are each a dict of concatenated endpoint frames covering
    two matches (see `_load_matches`). Every matchsum figure below is a mean
    across a side's matches; every event-derived figure is computed from the
    concatenated event stream so shares are real combined percentages, not an
    average of two ratios.
    """
    opp_row = _mean_row(opp["ss"], OPPONENT)
    our_row = _mean_row(ours["ss"], US)

    def g(col):
        return float(opp_row.get(col, 0.0))

    def g_us(col):
        return float(our_row.get(col, 0.0))

    f = {"ref": US}

    # ---- which matches this is built from, for both sides - replaces the
    # old single "reference match" metadata now that there isn't one match.
    def _match_list(evdf, squad):
        rows = []
        if evdf is None or len(evdf) == 0:
            return rows
        for _, g_ in evdf.groupby("matchId"):
            r0 = g_.iloc[0]
            home = r0.homeSquadName == squad
            opp_name = r0.awaySquadName if home else r0.homeSquadName
            when = pd.to_datetime(r0.dateTime, errors="coerce", utc=True)
            # GOALS credits the scoring player, so an own goal (charged to
            # the conceding side's own player) is invisible to a plain
            # GOALS.sum() - it has to be added to the OTHER side's tally
            # explicitly, or a match with one silently shows the wrong score.
            og_by_squad = len(g_[(g_.squadName == squad) & (g_.actionType == "OWN_GOAL")])
            og_by_opp = len(g_[(g_.squadName == opp_name) & (g_.actionType == "OWN_GOAL")])
            # a cup tie that finished level and went to penalties records the
            # shootout as PENALTY_KICK events, separate from GOALS - without
            # this a 0-0 (aet) sits in the list with no explanation of why a
            # drawn match was worth including at all. periodId 5 is the
            # shootout itself - periods 1/2 (and 3/4 for extra time) can
            # also contain a PENALTY_KICK from a normal in-play foul, which
            # is not a shootout and must not be reported as one.
            pk = g_[(g_.action == "PENALTY_KICK") & (g_.periodId == 5)]
            pens = None
            if len(pk):
                pens = (int((pk[pk.squadName == squad].result == "SUCCESS").sum()),
                        int((pk[pk.squadName == opp_name].result == "SUCCESS").sum()))
            rows.append(dict(
                opp=opp_name, venue="home" if home else "away",
                date=when.strftime("%-d %b %Y") if pd.notna(when) else "",
                sort=when, comp=f"{r0.competitionName} {r0.season}", pens=pens,
                gf=int(g_[g_.squadName == squad].GOALS.sum()) + og_by_opp,
                ga=int(g_[g_.squadName == opp_name].GOALS.sum()) + og_by_squad))
        rows.sort(key=lambda r: r["sort"])
        return rows

    f["opp_matches"] = _match_list(opp["ev"], OPPONENT)
    f["our_matches"] = _match_list(ours["ev"], US)
    # the sample can now mix competitions (league + cup) on both sides, so
    # "which division do they play in" is no longer a single string read off
    # one match - it's whether ANY of the opponent's sampled matches share
    # our own league. Barnsley's earliest sampled match happens to be an EFL
    # Cup tie, but they're still a League One side; reading only the first
    # match's competition would have called that a division gap incorrectly.
    f["opp_competition"] = (f["opp_matches"][0]["comp"] if f["opp_matches"] else "")
    f["ref_diff_division"] = bool(f["opp_matches"]) and not any(
        OUR_COMPETITION in mm["comp"] for mm in f["opp_matches"])
    # how each side's sample actually splits by competition, for captions
    # that need to say so rather than assume "league" throughout
    def _comp_mix(matches):
        counts = {}
        for mm in matches:
            k = mm["comp"].rsplit(" ", 1)[0]   # drop the season, e.g. "26/27"
            counts[k] = counts.get(k, 0) + 1
        return ", ".join(f"{v} {k}" for k, v in counts.items())
    f["opp_comp_mix"] = _comp_mix(f["opp_matches"])
    f["our_comp_mix"] = _comp_mix(f["our_matches"])

    for k, c in [("xg", "SHOT_XG"), ("psxg", "POSTSHOT_XG"),
                 ("bpd", "BYPASSED_DEFENDERS"), ("pxt", "PXT_ATTACK"),
                 ("succ", "SUCCESSFUL_PASSES"), ("unsucc", "UNSUCCESSFUL_PASSES"),
                 ("press", "NUMBER_OF_PRESSES"), ("wins", "BALL_WIN_NUMBER"),
                 ("bwrd", "BALL_WIN_REMOVED_OPPONENTS_DEFENDERS"),
                 ("gw", "WON_GROUND_DUELS"), ("gl", "LOST_GROUND_DUELS"),
                 ("aw", "WON_AERIAL_DUELS"), ("al", "LOST_AERIAL_DUELS"),
                 ("fouls", "NUMBER_OF_FOULS"), ("shots", "SHOT_AT_GOAL_NUMBER"),
                 ("goals", "GOALS")]:
        f[k], f[k + "_r"] = g(c), g_us(c)

    f["phase"] = {p: (g(f"BYPASSED_DEFENDERS_AT_PHASE_{p}"),
                      g_us(f"BYPASSED_DEFENDERS_AT_PHASE_{p}"))
                  for p in ("IN_POSSESSION", "ATTACKING_TRANSITION",
                            "SET_PIECE", "SECOND_BALL")}
    f["action"] = {a: (g(f"BYPASSED_DEFENDERS_BY_ACTION_{a}"),
                       g_us(f"BYPASSED_DEFENDERS_BY_ACTION_{a}"))
                   for a in ("LOW_PASS", "CHIPPED_PASS", "DIAGONAL_PASS",
                             "DRIBBLE", "HIGH_CROSS", "LOW_CROSS", "HEADER")}
    f["press_kind"] = {k: (g(f"NUMBER_OF_PRESSES_{k}"), g_us(f"NUMBER_OF_PRESSES_{k}"))
                       for k in ("BUILD_UP", "BETWEEN_THE_LINES", "COUNTER_PRESS")}
    f["wins_zone"] = {z: g(f"BALL_WIN_NUMBER_IN_PITCH_POSITION_{z}")
                      for z in ("FIRST_THIRD", "MIDDLE_THIRD", "FINAL_THIRD",
                                "OPPONENT_BOX")}
    f["passtype"] = {t: (g(f"SUCCESSFUL_PASSES_BY_ACTION_{t}"),
                         g(f"UNSUCCESSFUL_PASSES_BY_ACTION_{t}"))
                     for t in ("LOW_PASS", "CHIPPED_PASS", "DIAGONAL_PASS",
                               "HIGH_CROSS", "LOW_CROSS")}
    # every shot-action category Impect tracks - a chart that only lists the
    # open-play ones silently undercounts any match with a free kick or
    # open-goal effort.
    f["shot_type"] = {t: (g(f"SHOT_AT_GOAL_NUMBER_BY_ACTION_{t}"),
                          g(f"SHOT_XG_BY_ACTION_{t}"))
                      for t in ("LONG_RANGE_SHOT", "MID_RANGE_SHOT",
                                "CLOSE_RANGE_SHOT", "ONE_VS_ONE_AGAINST_GK",
                                "HEADER", "OPEN_GOAL_SHOT", "FREE_KICK",
                                "CORNER", "PENALTY_KICK")}
    f["sp_xg"] = g("SHOT_XG_BY_ACTION_CORNER") + g("SHOT_XG_BY_ACTION_FREE_KICK")
    f["xg_transition"] = g("SHOT_XG_AT_PHASE_ATTACKING_TRANSITION")

    # ---- scores endpoint: the press profile, mean across each side's matches
    opp_sc = _mean_row(opp["squad_scores"], OPPONENT)
    our_sc = _mean_row(ours["squad_scores"], US)
    f["scores"] = {}
    for c in ("PPDA", "MEAN_PRESSURE_HEIGHT",
              "PRESSURE_DURING_OPP_BUILD_UP_PERCENT",
              "OPP_OPENING_BALLS_PRESSED_PERCENT",
              "PRESSURE_OPP_GK_PERCENT", "RATIO_BALL_POSSESSION",
              "FIELD_TILT", "RATIO_SECOND_BALL_WINS",
              "EXPECTED_POINTS_SHOT_XG"):
        if c in opp_sc.index or c in our_sc.index:
            f["scores"][c] = (float(opp_sc.get(c, 0.0)), float(our_sc.get(c, 0.0)))

    fm = opp["fm"]
    f["formation"] = "unknown"
    f["formation_varied"] = False
    if fm is not None and len(fm):
        r = fm[fm.squadName == OPPONENT]
        if len(r):
            modes = r.formation.value_counts()
            f["formation"] = str(modes.index[0])
            f["formation_varied"] = len(modes) > 1

    # ---- lanes. Impect lanes are relative to the acting team's attack, so the
    # opponent's lanes must be mirrored to express which side of THIS team
    # was opened up.
    ev = opp["ev"]
    if ev is not None and len(ev):
        bu = ev[(ev.squadName == OPPONENT) & (ev.actionType == "PASS")
                & ev.startPitchPosition.isin(["FIRST_THIRD", "OWN_BOX"])]
        v = bu.startLane.value_counts()
        f["lane_build"] = [float(v.get(l, 0)) / max(v.sum(), 1) * 100 for l in LANES]

        pr = ev[(ev.squadName == OPPONENT) & (ev.BYPASSED_DEFENDERS > 0)]
        v = pr.groupby("startLane").BYPASSED_DEFENDERS.sum()
        f["lane_prog"] = [float(v.get(l, 0)) / max(v.sum(), 1) * 100 for l in LANES]

        # both of the opponent's own opponents across these two games, combined
        # - there is no longer one single "reference opponent" to read off
        op = ev[(ev.squadName != OPPONENT) & (ev.BYPASSED_DEFENDERS > 0)]
        v = op.groupby("startLane").BYPASSED_DEFENDERS.sum()
        tot = max(v.sum(), 1)
        f["lane_conceded"] = [float(v.get(MIRROR[l], 0)) / tot * 100 for l in LANES]

        f["recov"] = ev[(ev.squadName == OPPONENT) & (ev.BALL_WIN_NUMBER > 0)].copy()
    else:
        f["lane_build"] = f["lane_prog"] = f["lane_conceded"] = [0.0] * len(LANES)
        f["recov"] = None

    # which physical side is actually bigger, for every place the report
    # names a side ("attack their X", "they come down their X") - these used
    # to just assume right for conceded progression and left for their own,
    # true for the opponent that text was first written against and silently
    # unverified (sometimes backwards) for every one since.
    # `close` flags a near-even split (under 5 points apart) so the prose
    # can stop short of a confident directional claim a coin-flip doesn't
    # actually support - a real 37%-to-36% split (Bradford City) was getting
    # the same unqualified "that is the side to attack" wording as a genuine
    # 60/40 lean, which overstates what a 1-point gap actually shows.
    def _side_split(vals):
        left, right = vals[0] + vals[1], vals[3] + vals[4]
        close = abs(left - right) < 5
        return ("LEFT", left, close) if left >= right else ("RIGHT", right, close)

    f["prog_side"], f["prog_share"], f["prog_close"] = _side_split(f["lane_prog"])
    f["conceded_side"], f["conceded_share"], f["conceded_close"] = _side_split(f["lane_conceded"])

    # ---- starting XI passing network: cannot be aggregated (a lineup is a
    # lineup), so this comes from the single most recent match only.
    recent = opp["recent_cache"]

    def _rd_recent(name):
        p = os.path.join(recent, f"{name}.parquet")
        return pd.read_parquet(p) if os.path.exists(p) else None

    rev, rsu, rsp = _rd_recent("ev"), _rd_recent("su"), _rd_recent("sp")
    f["first_change"], f["changes"] = RM.substitution_summary(rsu, OPPONENT)
    f["shirts"] = {}
    if rsp is not None and len(rsp):
        for _, r in rsp[rsp.squadName == OPPONENT].iterrows():
            if pd.notna(r.get("shirtNumber")):
                f["shirts"][r.playerName] = int(r.shirtNumber)
    f["net"] = (None, None)
    f["net_passes"] = 0
    f["net_opp"] = f["net_date"] = ""
    if rev is not None and len(rev):
        f["net"] = RM.passing_network(rev, OPPONENT, end_min=f["first_change"])
        pp = rev[(rev.squadName == OPPONENT) & (rev.actionType == "PASS")
                & (rev.result == "SUCCESS") & (rev.passReceiverType == "TEAMMATE")].copy()
        pp["_m"] = RM.minute(pp)
        f["net_passes"] = int((pp["_m"] < (f["first_change"] or 999)).sum())
        r0 = rev.iloc[0]
        f["net_opp"] = r0.awaySquadName if r0.homeSquadName == OPPONENT else r0.homeSquadName
        when = pd.to_datetime(r0.dateTime, errors="coerce", utc=True)
        f["net_date"] = when.strftime("%-d %b %Y") if pd.notna(when) else ""

    # ---- build-up map: three views of "which channel should we use"
    # All three use the BUILDING team's own lanes, so they compare directly and
    # need no mirroring (unlike the defensive-shape chart on THE CHANNELS).
    f["their_press_lane"] = {}
    if ev is not None and len(ev):
        bu2 = ev[(ev.squadName != OPPONENT) & (ev.actionType == "PASS")
                 & ev.startPitchPosition.isin(["FIRST_THIRD", "OWN_BOX"])]
        for l in LANES:
            d = bu2[bu2.startLane == l]
            f["their_press_lane"][l] = (
                float(d.pressure.mean()) if len(d) else 0.0,
                float((d.result != "SUCCESS").mean() * 100) if len(d) else 0.0,
                int(len(d)))

    zc = [c for c in opp_row.index
          if c.startswith("BALL_WIN_REMOVED_OPPONENTS_FROM_OPPONENT_PACKING_ZONE_")]
    z = {c.split("ZONE_")[1]: float(opp_row[c]) for c in zc}
    f["zone_damage"] = {k: v for k, v in
                        sorted(z.items(), key=lambda kv: -kv[1]) if v > 0}

    f["our_build"] = {}
    our_ev = ours["ev"]
    f["our_match_count"] = our_ev.matchId.nunique() if our_ev is not None else 0
    if our_ev is not None and len(our_ev):
        ob = our_ev[(our_ev.squadName == US) & (our_ev.actionType == "PASS")
                    & our_ev.startPitchPosition.isin(["FIRST_THIRD", "OWN_BOX"])]
        for l in LANES:
            d = ob[ob.startLane == l]
            f["our_build"][l] = (
                float((d.result != "SUCCESS").mean() * 100) if len(d) else 0.0,
                float(d.pressure.mean()) if len(d) else 0.0,
                int(len(d)))

    stp = opp["setpieces"]
    f["corners"] = 0
    f["corner_xg"] = 0.0
    f["takers"] = {}
    f["corner_concentrated"] = False
    if stp is not None and len(stp):
        d = stp[stp.attackingSquadName == OPPONENT]
        c = d[d.setPieceCategory.str.contains("CORNER", na=False)]
        # a corner can span two setpiece subphases (the delivery, then a
        # separate first-touch event) - len(c) counts subphase rows, not
        # physical corners, and overcounts by however many corners had both.
        # setPieceId is the real per-corner key, matching how the corner
        # report itself counts (build_corners.py's corner_phases()).
        f["corners"] = c.setPieceId.nunique()
        f["corner_xg"] = float(c["setPieceSubPhase_SHOT_XG"].sum())
        # a corner total that reads as "a genuine pattern" can still be one
        # single costly phase inflating an otherwise-empty total - same
        # concentration check the Performance Review uses for set pieces.
        if f["corner_xg"] > 0:
            idx = c["setPieceSubPhase_SHOT_XG"].idxmax()
            biggest = float(c.loc[idx, "setPieceSubPhase_SHOT_XG"])
            f["corner_concentrated"] = (
                biggest / f["corner_xg"]) >= RM.SETPIECE_CONCENTRATION_THRESHOLD
        for cat in ("CORNER_LEFT", "CORNER_RIGHT", "FREE_KICK"):
            t = d[d.setPieceCategory == cat].setPieceSubPhaseMainEventPlayerName
            if len(t.dropna()):
                f["takers"][cat] = t.dropna().value_counts().index[0]
        th = d[d.setPieceCategory == "THROW_IN"].setPieceSubPhaseMainEventPlayerName.dropna()
        f["throwins"] = len(th)
        f["throwin_taker"] = None
        if len(th):
            vc = th.value_counts()
            f["throwin_taker"] = (vc.index[0], int(vc.iloc[0]))
    else:
        f["throwins"] = 0
        f["throwin_taker"] = None
    f["opp_match_count"] = ev.matchId.nunique() if ev is not None and len(ev) else 0

    pl = opp["ps"]
    pos = (pl.groupby("playerName")["position"]
           .agg(lambda x: x.value_counts().index[0]))
    p = pl.groupby("playerName").agg(
        mins=("playDuration", lambda s: s.sum() / 60),
        BPD=("BYPASSED_DEFENDERS", "sum"),
        BPDR=("BYPASSED_DEFENDERS_RECEIVING", "sum"),
        pxT=("PXT_ATTACK", "sum"), xG=("SHOT_XG", "sum"),
        press=("NUMBER_OF_PRESSES", "sum"), wins=("BALL_WIN_NUMBER", "sum"),
        loss=("BALL_LOSS_REMOVED_TEAMMATES", "sum"),
        aw=("WON_AERIAL_DUELS", "sum"), al=("LOST_AERIAL_DUELS", "sum"),
        gw=("WON_GROUND_DUELS", "sum"), gl=("LOST_GROUND_DUELS", "sum"),
    ).reset_index()
    p["position"] = p.playerName.map(pos)
    f["players"] = p
    f["labels"] = short_labels(p.playerName.tolist())
    return f


# ------------------------------------------------------------------ pieces
def title(fig, text, sub=None, y=0.955):
    fig.text(0.5, y, text, ha="center", va="center", color=INK,
             fontsize=29, weight="bold")
    if sub:
        fig.text(0.5, y - 0.032, sub, ha="center", va="center", color=MUTED,
                 fontsize=10.5)


def compare_rows(fig, rect, rows, left, right):
    ax = fig.add_axes(rect); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    n = len(rows); slot = 1.0 / n; bar_h = slot * 0.36
    for i, (label, fmt, lv, rv) in enumerate(rows):
        top = 1 - i * slot
        ax.text(.5, top - slot * 0.28, label, ha="center", va="center",
                color=INK, fontsize=9.5, weight="bold")
        y = top - slot * 0.72
        tot = (lv or 0) + (rv or 0)
        frac = 0.5 if tot == 0 else (lv or 0) / tot
        frac = min(max(frac, 0.06), 0.94)
        ax.add_patch(FancyBboxPatch((0, y - bar_h / 2), frac, bar_h,
                                    boxstyle="round,pad=0,rounding_size=0.012",
                                    fc=OPP_C, ec="none", mutation_aspect=0.05))
        ax.add_patch(FancyBboxPatch((frac, y - bar_h / 2), 1 - frac, bar_h,
                                    boxstyle="round,pad=0,rounding_size=0.012",
                                    fc=REF_C, ec="none", mutation_aspect=0.05))
        ax.text(frac - 0.012, y, fmt.format(lv), ha="right", va="center",
                color="#FFFFFF", fontsize=9, weight="bold")
        ax.text(frac + 0.012, y, fmt.format(rv), ha="left", va="center",
                color="#10161F", fontsize=9, weight="bold")


def grouped(fig, rect, labels, a, b, ylabel, names, note=None):
    ax = fig.add_axes(rect)
    style_bar_axis(ax, ylabel=ylabel, label_size=8.5, tick_size=7)
    x = np.arange(len(labels))
    ax.bar(x - .19, a, width=.36, color=OPP_C, zorder=3, label=names[0])
    ax.bar(x + .19, b, width=.36, color=REF_C, zorder=3, label=names[1])
    for i, (va, vb) in enumerate(zip(a, b)):
        top = max(a + b) if len(a) else 1
        ax.text(i - .19, va + top * .03, f"{va:.0f}" if va >= 10 else f"{va:.1f}",
                ha="center", va="bottom", fontsize=6.4, color=INK, weight="bold")
        ax.text(i + .19, vb + top * .03, f"{vb:.0f}" if vb >= 10 else f"{vb:.1f}",
                ha="center", va="bottom", fontsize=6.4, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=18, ha="right")
    ax.set_ylim(0, max(max(a + b) * 1.28, 1))
    ax.legend(frameon=False, fontsize=6.6, labelcolor=INK, loc="upper right",
              ncol=2, handlelength=1.1)
    if note:
        fig.text(rect[0], rect[1] - 0.050, note, ha="left", va="center",
                 color=MUTED, fontsize=6.8)


def player_bar(fig, rect, d, col, ylabel, colour=BARC, fmt="{:.1f}", n=10,
               ascending=False, labels=None):
    d = d[d[col] != 0].sort_values(col, ascending=ascending).head(n)
    ax = fig.add_axes(rect)
    style_bar_axis(ax, ylabel=ylabel, label_size=8.5, tick_size=7)
    x = np.arange(len(d))
    ax.bar(x, d[col].values, color=colour, width=.7, zorder=3)
    for i, v in enumerate(d[col].values):
        ax.text(i, v + d[col].max() * .04, fmt.format(v), ha="center",
                va="bottom", fontsize=6.4, color=INK, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([(labels or {}).get(p, surname(p)) for p in d.playerName],
                       rotation=40, ha="right", fontsize=6.4)
    ax.set_ylim(0, d[col].max() * 1.25)


def point_card(fig, rect, num, headline, stat, statlabel, body, accent):
    ax = fig.add_axes(rect); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1,
                                boxstyle="round,pad=0,rounding_size=0.02",
                                fc=PANEL, ec=GRID, lw=0.8, mutation_aspect=0.30))
    ax.add_patch(Rectangle((0, 0.06), 0.012, 0.88, fc=accent, ec="none"))
    ax.text(0.045, 0.87, f"{num}. {headline}", ha="left", va="top", color=INK,
            fontsize=9.6, weight="bold")
    ax.text(0.045, 0.52, stat, ha="left", va="center", color=accent,
            fontsize=19, weight="bold")
    ax.text(0.045, 0.30, statlabel, ha="left", va="center", color=MUTED,
            fontsize=6.8)
    ax.text(0.44, 0.60, "\n".join(textwrap.wrap(body, 42)), ha="left",
            va="top", color="#B9C4D2", fontsize=7.4, linespacing=1.5)


# ------------------------------------------------------------------- pages
def page_cover(fig, f):
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(Rectangle((0, .58), 1, .42, color=OPP_C, zorder=0))
    ax.add_patch(Rectangle((0, 0), 1, .58, color=BG, zorder=0))
    # text on the opponent band has to adapt to OPP_C's brightness - fixed
    # white read fine on Wigan's navy but would wash out on a light colour
    # like AFC Wimbledon's yellow.
    ink = contrast_ink(OPP_C, dark="#141414", light="#FFFFFF")
    sub_ink = "#3A3320" if ink == "#141414" else "#EAF3FF"
    ax.text(.5, .84, "PRE-MATCH ANALYSIS", ha="center", va="center",
            color=ink, fontsize=24, weight="bold", zorder=2)
    ax.text(.5, .73, OPPONENT.upper(), ha="center", va="center",
            color=ink, fontsize=40, weight="bold", zorder=2)
    ax.text(.5, .655, FIXTURE, ha="center", va="center", color=sub_ink,
            fontsize=12, zorder=2)

    n_opp, n_our = len(f["opp_matches"]), len(f["our_matches"])
    sample_note = (f"Built from their last {n_opp} matches, our last {n_our}"
                   if n_opp != n_our else
                   f"Built from each side's last {n_opp} matches")
    ax.text(.5, .535, sample_note, ha="center", va="center",
            color=MUTED, fontsize=11, zorder=2)

    def _match_col(cx, side_label, matches, colour):
        ax.text(cx, .465, side_label, ha="center", va="center", color=colour,
                fontsize=13, weight="bold", zorder=2)
        for i, m in enumerate(matches):
            y = .410 - i * 0.052
            score = f"{m['gf']}-{m['ga']}" if m['venue'] == "home" else f"{m['ga']}-{m['gf']}"
            pens_note = ""
            if m.get("pens"):
                us_p, opp_p = m["pens"]
                pens_note = (f" (won {us_p}-{opp_p} pens)" if us_p > opp_p
                            else f" (lost {us_p}-{opp_p} pens)")
            line = f"{m['opp']}  {score}{pens_note}  ({m['venue']}, {m['date']})"
            ax.text(cx, y, line, ha="center", va="center", color=INK,
                    fontsize=9.5, zorder=2)

    _match_col(0.27, OPPONENT, f["opp_matches"], ink)
    _match_col(0.73, "Leyton Orient", f["our_matches"], TEAM_C)
    bottom_row_y = .410 - (max(n_opp, n_our) - 1) * 0.052
    ax.plot([0.5, 0.5], [bottom_row_y - 0.028, .49], color=MUTED, lw=0.7,
            alpha=.4, zorder=1)

    if f["ref_diff_division"]:
        note = (f"{OPPONENT}'s sample is {f['opp_comp_mix']} - not "
                f"{OUR_COMPETITION}, unlike Orient's own. Read the comparison "
                f"charts with that gap in mind.")
    else:
        note = (f"{OPPONENT}: {f['opp_comp_mix']}. Orient: {f['our_comp_mix']}. "
                f"Both sides include {OUR_COMPETITION}, so the comparison "
                f"charts are a fair like-for-like.")
    # the match lists above grow with the sample size, so this has to clear
    # whichever side has more rows rather than sit at a position tuned for
    # the original two-matches-each case
    note_y = min(.245, bottom_row_y - 0.113)
    ax.text(.5, note_y, note, ha="center", va="center", color="#B9C4D2",
            fontsize=9.5, linespacing=1.7, zorder=2, wrap=True)
    match_note = (f"{n_opp} matches for them, {n_our} for us"
                 if n_opp != n_our else f"{n_opp} matches each")
    final_y = min(.105, note_y - 0.09)
    ax.text(.5, final_y,
            f"{match_note}. Enough to start telling a pattern from a\n"
            "one-off - not yet enough to call it settled.", ha="center",
            va="center", color=MUTED, fontsize=8.5, linespacing=1.6, zorder=2)


def page_profile(fig, f):
    n_opp, n_our = len(f["opp_matches"]), len(f["our_matches"])
    sample = (f"last {n_opp} for them, {n_our} for us" if n_opp != n_our
             else f"last {n_opp} matches each")
    title(fig, "THE PROFILE", f"{OPPONENT} vs {f['ref']} — {sample}, per game")
    fig.text(0.085, 0.888, OPPONENT, ha="left", va="center", color=OPP_C,
             fontsize=12, weight="bold")
    fig.text(0.915, 0.888, f["ref"], ha="right", va="center", color=REF_C,
             fontsize=12, weight="bold")
    rows = [
        ("xG", "{:.2f}", f["xg"], f["xg_r"]),
        ("Post-Shot xG", "{:.2f}", f["psxg"], f["psxg_r"]),
        ("Shots", "{:.0f}", f["shots"], f["shots_r"]),
        ("Bypassed Defenders", "{:.1f}", f["bpd"], f["bpd_r"]),
        ("Packing xT", "{:.2f}", f["pxt"], f["pxt_r"]),
        ("Successful Passes", "{:.0f}", f["succ"], f["succ_r"]),
        ("Pressures", "{:.0f}", f["press"], f["press_r"]),
        ("Ball Wins", "{:.0f}", f["wins"], f["wins_r"]),
        ("Ball Win Removed Defenders", "{:.1f}", f["bwrd"], f["bwrd_r"]),
        ("Ground Duels Won", "{:.0f}", f["gw"], f["gl"]),
        ("Aerial Duels Won", "{:.0f}", f["aw"], f["al"]),
        ("Fouls", "{:.0f}", f["fouls"], f["fouls_r"]),
    ]
    compare_rows(fig, [0.075, 0.075, 0.85, 0.78], rows, OPPONENT, f["ref"])
    poss = f["scores"].get("RATIO_BALL_POSSESSION")
    bits = []
    if poss:
        more_less = "Less" if poss[0] < poss[1] else "More"
        bits.append(f"{more_less} of the ball ({poss[0]*100:.0f}% to {poss[1]*100:.0f}%)")
    bpd_cmp = "more" if f["bpd"] > f["bpd_r"] else "less"
    bits.append(f"{bpd_cmp} progression ({f['bpd']:.1f} to {f['bpd_r']:.1f} "
               f"bypassed defenders)")
    if f["bwrd_r"] > 0:
        bits.append(f"turnovers that hurt {f['bwrd']/f['bwrd_r']:.1f}x as much")
    fig.text(0.5, 0.042, ", ".join(bits) + ".",
             ha="center", va="center", color=MUTED, fontsize=8)


def page_style(fig, f):
    title(fig, "HOW THEY PLAY")
    ph = ["IN_POSSESSION", "ATTACKING_TRANSITION", "SET_PIECE", "SECOND_BALL"]
    lab = ["Settled\npossession", "Attacking\ntransition", "Set piece", "Second ball"]
    a = [f["phase"][p][0] for p in ph]; b = [f["phase"][p][1] for p in ph]
    share = a[1] / sum(a) * 100
    fig.text(0.075, 0.885, "Where their progression comes from", ha="left",
             va="center", color=INK, fontsize=11, weight="bold")
    grouped(fig, [0.075, 0.665, 0.385, 0.175], lab, a, b, "Bypassed defenders",
            (OPPONENT, f["ref"]),
            note=f"{share:.0f}% of their progression came in transition.")

    kinds = ["BUILD_UP", "BETWEEN_THE_LINES", "COUNTER_PRESS"]
    klab = ["In their\nbuild-up", "Between\nthe lines", "Counter-\npress"]
    ka = [f["press_kind"][k][0] for k in kinds]
    kb = [f["press_kind"][k][1] for k in kinds]
    fig.text(0.545, 0.885, "How they press", ha="left", va="center", color=INK,
             fontsize=11, weight="bold")
    grouped(fig, [0.545, 0.665, 0.385, 0.175], klab, ka, kb, "Pressures",
            (OPPONENT, f["ref"]),
            note=f"Between the lines: {ka[1]:.0f} to {kb[1]:.0f}.")

    acts = ["LOW_PASS", "CHIPPED_PASS", "DIAGONAL_PASS", "DRIBBLE",
            "HIGH_CROSS", "LOW_CROSS"]
    alab = ["Low pass", "Chipped", "Diagonal", "Dribble", "High cross", "Low cross"]
    aa = [f["action"][x][0] for x in acts]
    ab = [f["action"][x][1] for x in acts]
    fig.text(0.075, 0.590, "How they move the ball forward", ha="left",
             va="center", color=INK, fontsize=11, weight="bold")
    grouped(fig, [0.075, 0.385, 0.385, 0.170], alab, aa, ab, "Bypassed defenders",
            (OPPONENT, f["ref"]))
    diag_o, diag_u = f["action"]["DIAGONAL_PASS"]
    # unconditionally calling this "switch play" was true for whichever
    # opponent it was first written against and silently wrong for a side
    # whose diagonal output is actually low or nonexistent
    if diag_o >= diag_u and diag_o > 0:
        diag_txt = (f"They switch play: diagonals gave them {diag_o:.1f} "
                    f"bypassed defenders, against our {diag_u:.1f}.")
    elif diag_o > 0:
        diag_txt = (f"They rarely switch play: diagonals gave them just "
                    f"{diag_o:.1f} bypassed defenders (ours {diag_u:.1f}).")
    else:
        diag_txt = (f"They don't switch play through diagonals in this "
                    f"sample at all (ours {diag_u:.1f}).")
    fig.text(0.075, 0.335, diag_txt, ha="left", va="center", color=MUTED,
             fontsize=6.8)

    types = ["LOW_PASS", "CHIPPED_PASS", "DIAGONAL_PASS", "HIGH_CROSS"]
    tlab = ["Low pass", "Chipped", "Diagonal", "High cross"]
    ok = [f["passtype"][t][0] for t in types]
    bad = [f["passtype"][t][1] for t in types]
    fig.text(0.545, 0.590, "…and how often it works", ha="left", va="center",
             color=INK, fontsize=11, weight="bold")
    ax = fig.add_axes([0.545, 0.385, 0.385, 0.170])
    style_bar_axis(ax, ylabel="Passes", label_size=8.5, tick_size=7)
    x = np.arange(len(types))
    ax.bar(x, ok, color=GOOD, width=.7, zorder=3, label="completed")
    ax.bar(x, bad, bottom=ok, color=BAD, width=.7, zorder=3, label="not")
    for i, (o, t) in enumerate(zip(ok, np.array(ok) + np.array(bad))):
        if t:
            ax.text(i, t + max(np.array(ok) + np.array(bad)) * .04,
                    f"{int(o)}/{int(t)}", ha="center", va="bottom",
                    fontsize=6.4, color=INK, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(tlab, rotation=35, ha="right", fontsize=6.8)
    ax.set_ylim(0, max(np.array(ok) + np.array(bad)) * 1.3)
    ax.legend(frameon=False, fontsize=6.4, labelcolor=INK, loc="upper right", ncol=2)
    cp = f["passtype"]["CHIPPED_PASS"]
    cp_rate = cp[0] / (cp[0] + cp[1]) * 100 if (cp[0] + cp[1]) else 0
    cp_word = "Only" if cp_rate < 55 else "A healthy"
    fig.text(0.545, 0.335, f"{cp_word} {cp_rate:.0f}% of their chipped "
                           f"passes found a teammate.",
             ha="left", va="center", color=MUTED, fontsize=6.8)

    zl = ["FIRST_THIRD", "MIDDLE_THIRD", "FINAL_THIRD", "OPPONENT_BOX"]
    zn = ["Own third", "Middle third", "Final third", "Opp box"]
    zv = [f["wins_zone"][z] for z in zl]
    high = (zv[2] + zv[3]) / sum(zv) * 100
    fig.text(0.075, 0.310, "Where they win the ball back", ha="left",
             va="center", color=INK, fontsize=11, weight="bold")
    ax = fig.add_axes([0.075, 0.105, 0.385, 0.170])
    style_bar_axis(ax, ylabel="Ball wins", label_size=8.5, tick_size=7)
    x = np.arange(len(zv))
    ax.bar(x, zv, color=OPP_C, width=.7, zorder=3)
    for i, v in enumerate(zv):
        ax.text(i, v + max(zv) * .04, f"{v:.0f}", ha="center", va="bottom",
                fontsize=6.6, color=INK, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(zn, rotation=25, ha="right", fontsize=6.8)
    ax.set_ylim(0, max(zv) * 1.25)
    fig.text(0.075, 0.055, f"{high:.0f}% of their ball wins came in the final "
                           f"third or the box.",
             ha="left", va="center", color=MUTED, fontsize=6.8)

    # every category with at least one shot - a fixed 4- or 5-bar list quietly
    # dropped free kicks/open goals/corners/penalties whenever an opponent had
    # one, which understated their true shot total on this chart alone.
    st = ["LONG_RANGE_SHOT", "MID_RANGE_SHOT", "CLOSE_RANGE_SHOT",
          "ONE_VS_ONE_AGAINST_GK", "HEADER", "OPEN_GOAL_SHOT", "FREE_KICK",
          "CORNER", "PENALTY_KICK"]
    sn = ["Long range", "Mid range", "Close range", "1v1 vs GK", "Header",
          "Open goal", "Free kick", "Corner", "Penalty"]
    all_sv = {t: f["shot_type"][t][0] for t in st}
    mid_n = all_sv["MID_RANGE_SHOT"]
    pairs = [(lab, all_sv[t]) for t, lab in zip(st, sn) if all_sv[t] > 0] or [("None", 0)]
    sn = [p[0] for p in pairs]; sv = [p[1] for p in pairs]
    fig.text(0.545, 0.310, "Where they shoot from", ha="left", va="center",
             color=INK, fontsize=11, weight="bold")
    ax = fig.add_axes([0.545, 0.105, 0.385, 0.170])
    style_bar_axis(ax, ylabel="Shots", label_size=8.5, tick_size=7)
    x = np.arange(len(sv))
    ax.bar(x, sv, color=OPP_C, width=.7, zorder=3)
    for i, v in enumerate(sv):
        ax.text(i, v + max(sv) * .05, f"{v:.0f}", ha="center", va="bottom",
                fontsize=6.6, color=INK, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(sn, rotation=25, ha="right", fontsize=6.8)
    ax.set_ylim(0, max(sv) * 1.3)
    fig.text(0.545, 0.055, f"{mid_n:.0f} of their {f['shots']:.0f} shots were "
                           f"from mid range — low-value efforts.",
             ha="left", va="center", color=MUTED, fontsize=6.8)


def page_personnel(fig, f):
    title(fig, "KEY PERSONNEL",
         f"{OPPONENT} — combined across their last {len(f['opp_matches'])} "
         f"matches ({f['opp_comp_mix']})")
    p = f["players"]
    if len(p) == 0:
        fig.text(0.5, 0.5, "No player data available for this match.",
                 ha="center", va="center", color=MUTED, fontsize=11)
        return
    fig.text(0.075, 0.878, "Who progresses the ball", ha="left", va="center",
             color=INK, fontsize=11, weight="bold")
    player_bar(fig, [0.075, 0.660, 0.385, 0.180], p, "BPD",
               "Bypassed defenders", OPP_C, labels=f["labels"])
    top = p.nlargest(1, "BPD").iloc[0]
    fig.text(0.075, 0.610, f"{surname(top.playerName)} ({top.position.replace('_',' ').lower()}) "
                           f"is the pivot — {top.BPD:.1f} bypassed defenders.",
             ha="left", va="center", color=MUTED, fontsize=6.8)

    fig.text(0.545, 0.878, "Who receives it", ha="left", va="center",
             color=INK, fontsize=11, weight="bold")
    player_bar(fig, [0.545, 0.660, 0.385, 0.180], p, "BPDR",
               "Bypassed defenders received", GOOD, labels=f["labels"])
    topr = p.nlargest(1, "BPDR").iloc[0]
    fig.text(0.545, 0.610, f"{surname(topr.playerName)} is the out-ball — "
                           f"{topr.BPDR:.1f} received.", ha="left", va="center",
             color=MUTED, fontsize=6.8)

    fig.text(0.075, 0.585, "Who gives it away (costliest losses)", ha="left",
             va="center", color=INK, fontsize=11, weight="bold")
    player_bar(fig, [0.075, 0.368, 0.385, 0.180], p, "loss",
               "Ball loss removed teammates", BAD, labels=f["labels"])
    # named by their actual positions rather than assumed to always be "the
    # back line and wing-backs" - that was true for one opponent and not
    # guaranteed for the next (a defensive midfielder topped this list here).
    top_losers = p[p.loss != 0].sort_values("loss", ascending=False).head(3)
    loss_desc = ", ".join(f"{surname(x.playerName)} ({x.position.replace('_', ' ').lower()})"
                          for _, x in top_losers.iterrows())
    fig.text(0.075, 0.318, f"Costliest here: {loss_desc}.", ha="left",
             va="center", color=MUTED, fontsize=6.8)

    fig.text(0.545, 0.585, "Aerial duels", ha="left", va="center", color=INK,
             fontsize=11, weight="bold")
    # sorted by total duels, not wins - sorting by wins alone buries a player
    # who contests plenty of headers but loses most of them (Wigan's own
    # centre-forward, 0 of 4, dropped off a wins-sorted top 9 while the text
    # below kept citing him).
    cf = p[p.position == "CENTER_FORWARD"].nlargest(1, "mins")
    d = (p[(p.aw + p.al) > 0].assign(_tot=lambda x: x.aw + x.al)
        .sort_values("_tot", ascending=False, kind="stable").head(9))
    if len(cf) and cf.iloc[0].playerName not in d.playerName.values:
        d = pd.concat([d.iloc[:-1], cf.assign(_tot=cf.aw + cf.al)])
    d = d.sort_values("aw", ascending=False, kind="stable")
    ax = fig.add_axes([0.545, 0.368, 0.385, 0.180])
    style_bar_axis(ax, ylabel="Aerial duels", label_size=8.5, tick_size=7)
    x = np.arange(len(d))
    ax.bar(x, d.aw.values, color=GOOD, width=.7, zorder=3, label="won")
    ax.bar(x, d.al.values, bottom=d.aw.values, color=BAD, width=.7, zorder=3,
           label="lost")
    for i, (w, l) in enumerate(zip(d.aw.values, d.al.values)):
        ax.text(i, w + l + 0.15, f"{int(w)}/{int(w+l)}", ha="center",
                va="bottom", fontsize=6.4, color=INK, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f["labels"].get(n, surname(n)) for n in d.playerName],
                       rotation=40, ha="right", fontsize=6.4)
    ax.set_ylim(0, (d.aw + d.al).max() * 1.35)
    ax.legend(frameon=False, fontsize=6.4, labelcolor=INK, loc="upper right", ncol=2)
    cf = p[p.position == "CENTER_FORWARD"].nlargest(1, "mins")
    if len(cf):
        c = cf.iloc[0]
        fig.text(0.545, 0.318, f"Their centre-forward {surname(c.playerName)} "
                               f"won {int(c.aw)} of {int(c.aw + c.al)} in the air.",
                 ha="left", va="center", color=MUTED, fontsize=6.8)

    # the XI, for reference - average minutes per game, not a raw total that
    # only grows with the sample size. Fringe players (under 30 minutes total
    # across the whole sample) are dropped, and the panel is capped at the 16
    # players with the most total minutes so a big squad rotation doesn't
    # turn the x-axis into an unreadable wall of names.
    n_opp_g = len(f["opp_matches"])
    MAX_PERSONNEL = 16
    d = p[p.mins >= 30].sort_values("mins", ascending=False).head(MAX_PERSONNEL).copy()
    d["avg_mins"] = d.mins / n_opp_g
    d = d.sort_values("avg_mins", ascending=False)
    capped = len(p[p.mins >= 30]) > MAX_PERSONNEL
    subtitle = (f"Minutes played (average per game, top {MAX_PERSONNEL} by total)"
               if capped else "Minutes played (average per game)")
    fig.text(0.075, 0.292, subtitle, ha="left", va="center", color=INK,
             fontsize=11, weight="bold")
    ax = fig.add_axes([0.075, 0.085, 0.855, 0.175])
    style_bar_axis(ax, ylabel="Avg minutes", label_size=8.5, tick_size=7)
    x = np.arange(len(d))
    cols = [OPP_C if m >= 81 else _fade(OPP_C) for m in d.avg_mins]
    ax.bar(x, d.avg_mins.values, color=cols, width=.7, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f["labels"].get(n, surname(n)) for n in d.playerName],
                       rotation=40, ha="right", fontsize=6.2)
    ax.set_ylim(0, max(d.avg_mins.max() * 1.15, 1))
    varied = " (varied)" if f["formation_varied"] else ""
    fig.text(0.075, 0.035, f"Bright = averaged close to a full 90 across "
                           f"their last {n_opp_g} games. Players under 30 "
                           f"minutes total left off{', capped at the top 16' if capped else ''}. "
                           f"Usual shape: {f.get('formation', 'unknown')}{varied}.",
             ha="left", va="center", color=MUTED, fontsize=6.8)


def stat_tiles(fig, rect, tiles):
    ax = fig.add_axes(rect); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    n = len(tiles); w = 1.0 / n
    for i, (label, ours, theirs, note) in enumerate(tiles):
        x0 = i * w
        ax.add_patch(FancyBboxPatch((x0 + 0.006, 0.04), w - 0.012, 0.92,
                                    boxstyle="round,pad=0,rounding_size=0.02",
                                    fc=PANEL, ec=GRID, lw=0.8,
                                    mutation_aspect=0.35))
        cx = x0 + w / 2
        ax.text(cx, 0.82, label, ha="center", va="center", color=INK,
                fontsize=7.4, weight="bold")
        ax.text(cx - w * 0.19, 0.48, ours, ha="center", va="center",
                color=OPP_C, fontsize=13.5, weight="bold")
        ax.text(cx, 0.48, "|", ha="center", va="center", color=GRID, fontsize=11)
        ax.text(cx + w * 0.19, 0.48, theirs, ha="center", va="center",
                color=REF_C, fontsize=13.5, weight="bold")
        ax.text(cx, 0.16, note, ha="center", va="center", color=MUTED,
                fontsize=6.2)


def page_press(fig, f):
    varied = " (varied)" if f["formation_varied"] else ""
    n_opp, n_our = len(f["opp_matches"]), len(f["our_matches"])
    title(fig, "PRESS & SHAPE", f"usual formation {f['formation']}{varied}  ·  "
                               f"{OPPONENT} (last {n_opp}) vs {f['ref']} (last {n_our})")
    sc = f["scores"]

    def s2(k, m=1, fmt="{:.1f}"):
        if k not in sc:
            return "-", "-"
        return fmt.format(sc[k][0] * m), fmt.format(sc[k][1] * m)

    a, b = s2("PPDA"); c, d = s2("MEAN_PRESSURE_HEIGHT", 1, "{:.0f}m")
    e, g_ = s2("PRESSURE_DURING_OPP_BUILD_UP_PERCENT", 100, "{:.0f}%")
    h, i_ = s2("PRESSURE_OPP_GK_PERCENT", 100, "{:.0f}%")
    j, k_ = s2("RATIO_BALL_POSSESSION", 100, "{:.0f}%")
    l, m_ = s2("FIELD_TILT", 100, "{:.0f}%")
    fig.text(0.075, 0.878, "How hard they press", ha="left", va="center",
             color=INK, fontsize=11, weight="bold")
    stat_tiles(fig, [0.055, 0.780, 0.890, 0.078], [
        ("PPDA", a, b, "lower = more aggressive"),
        ("Press height", c, d, "metres upfield"),
        ("Opp build-up pressed", e, g_, "higher = more"),
        ("Opp GK pressed", h, i_, "they do press the keeper"),
    ])
    # Every clause here used to be a fixed assertion ("they had less of it",
    # "A pressing counter-attacking side") true for Wigan's low-possession
    # style and never re-checked since - Barnsley actually had MORE
    # possession and territory than Orient across this sample, so the old
    # text was flatly backwards for them. Each comparison is now read off
    # the real numbers.
    poss_opp, poss_us = sc.get("RATIO_BALL_POSSESSION", (0.0, 0.0))
    tilt_opp, tilt_us = sc.get("FIELD_TILT", (0.0, 0.0))
    ep_opp, ep_us = sc.get("EXPECTED_POINTS_SHOT_XG", (0.0, 0.0))
    poss_more = poss_opp >= poss_us
    tilt_more = tilt_opp >= tilt_us
    bpd_more = f["bpd"] >= f["bpd_r"]
    ep_more = ep_opp >= ep_us

    fig.text(0.075, 0.745,
             "…and they dominate the ball too" if poss_more else
             "…but they do not dominate the ball",
             ha="left", va="center", color=INK, fontsize=11, weight="bold")
    stat_tiles(fig, [0.055, 0.647, 0.890, 0.078], [
        ("Possession", j, k_, "they had more of it" if poss_more else "they had less of it"),
        ("Field tilt", l, m_, "and more territory" if tilt_more else "and less territory"),
        ("Bypassed defenders", f"{f['bpd']:.0f}", f"{f['bpd_r']:.0f}",
         "more progression than us" if bpd_more else "less progression than us"),
        ("Expected points", *s2("EXPECTED_POINTS_SHOT_XG"), "on xG"),
    ])
    style_word = "A possession-dominant side" if poss_more else "A pressing counter-attacking side"
    bits = [
        "more of the ball" if poss_more else "less of the ball",
        "more territory" if tilt_more else "less territory",
        "more progression" if bpd_more else "less progression",
        "more chances" if ep_more else "fewer chances",
    ]
    fig.text(0.5, 0.622, f"{style_word}: {', '.join(bits)}.",
             ha="center", va="center", color=MUTED, fontsize=8)

    # Shot location map dropped from pre-match: with only 5-6 games in the
    # sample it read as noise, and the post-match report already carries a
    # proper shot map with a full match's data - pre-match keeps the ball
    # recoveries panel, widened to use the space this freed up.
    fig.text(0.075, 0.585, "Where they win the ball back", ha="left",
             va="center", color=INK, fontsize=11, weight="bold")
    ax = fig.add_axes([0.055, 0.392, 0.890, 0.180])
    draw_pitch(ax, lw=0.8)
    r = f.get("recov")
    if r is not None and len(r):
        ax.scatter(r.startAdjCoordinatesX, r.startAdjCoordinatesY, s=11,
                   color=OPP_C, alpha=.75, zorder=4, edgecolors="none")
    fig.text(0.5, 0.376, f"attacking right · {len(r) if r is not None else 0} "
                         f"ball recoveries across their last "
                         f"{f['opp_match_count']} games", ha="center",
             va="center", color=MUTED, fontsize=6.6)

    # ---- starting XI passing network
    cut = f.get("first_change")
    fig.text(0.075, 0.345, "Starting XI — passing network", ha="left",
             va="center", color=INK, fontsize=11, weight="bold")
    ax = fig.add_axes([0.055, 0.108, 0.520, 0.222])
    draw_pitch(ax, lw=0.9)
    nodes, edges = f.get("net", (None, None))
    if nodes is not None and len(edges):
        mx = edges.n.max()
        pos = nodes.set_index("playerName")[["x", "y"]].to_dict("index")
        for _, e in edges.iterrows():
            a, b = e.playerName, e.passReceiverPlayerName
            if a in pos and b in pos and e.n >= 2:
                ax.plot([pos[a]["x"], pos[b]["x"]], [pos[a]["y"], pos[b]["y"]],
                        color="#D8E2EE", lw=0.4 + 3.2 * (e.n / mx),
                        alpha=0.25 + 0.55 * (e.n / mx), zorder=2,
                        solid_capstyle="round")
    if nodes is not None and len(nodes):
        smax = nodes.n.max()
        offs = RM.label_offsets(nodes, ax, s_base=95, s_extra=520)
        for k, (_, r2) in enumerate(nodes.iterrows()):
            ax.scatter([r2.x], [r2.y], s=95 + 520 * (r2.n / smax), color=OPP_C,
                       edgecolors="#BFD9F5", linewidths=1.2, zorder=3)
            lab = str(f["shirts"].get(r2.playerName, ""))
            if not lab:
                lab = surname(r2.playerName)[:3].upper()
            lx, ly, leader = offs[k]
            if leader:
                ax.plot([r2.x, lx], [r2.y, ly], color="#BFD9F5", lw=0.7,
                        alpha=.8, zorder=4)
                ax.text(lx, ly, lab, ha="center", va="center", color="#0B1B2B",
                        fontsize=7, weight="bold", zorder=6,
                        bbox=dict(boxstyle="circle,pad=0.28", fc="#BFD9F5",
                                  ec="none"))
            else:
                ax.text(r2.x, r2.y, lab, ha="center", va="center",
                        color="#0B1B2B", fontsize=7, weight="bold", zorder=5)
    fig.text(0.315, 0.092, f"0–{cut}' before the first change · "
                           f"{f.get('net_passes', 0)} successful passes · "
                           f"vs {f['net_opp']}, {f['net_date']} · attacking right",
             ha="center", va="center", color=MUTED, fontsize=6.6)

    # ---- set pieces, narrower column beside the network
    tk = f.get("takers", {})
    corner_bits = []
    if "CORNER_LEFT" in tk:
        corner_bits.append(f"{surname(tk['CORNER_LEFT'])} (left)")
    if "CORNER_RIGHT" in tk:
        corner_bits.append(f"{surname(tk['CORNER_RIGHT'])} (right)")
    if f["corners"] == 0 or f["corner_xg"] <= 0:
        corner_line = f"{f['corners']} corners, no threat created from them at all."
    elif f["corner_concentrated"]:
        corner_line = (f"{f['corners']} corners produced {f['corner_xg']:.2f} xG, "
                       f"but most of it is one costly phase, not a repeatable pattern.")
    else:
        corner_line = (f"{f['corners']} corners produced {f['corner_xg']:.2f} xG, "
                       f"spread across several deliveries - a real if modest threat.")
    items = [corner_line]
    if corner_bits:
        line = "Corners: " + ", ".join(corner_bits) + "."
        if "FREE_KICK" in tk:
            line += f"  Free kicks: {surname(tk['FREE_KICK'])}."
        items.append(line)
    if f["throwin_taker"]:
        name, cnt = f["throwin_taker"]
        items.append(f"{f['throwins']} throw-ins, {surname(name)} takes most "
                     f"of them ({cnt}).")

    fig.text(0.615, 0.345, f"Set pieces (last {f['opp_match_count']} games)",
             ha="left", va="center", color=INK, fontsize=11, weight="bold")
    ax = fig.add_axes([0.615, 0.108, 0.330, 0.222]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1,
                                boxstyle="round,pad=0,rounding_size=0.02",
                                fc=PANEL, ec=GRID, lw=0.8, mutation_aspect=0.32))
    # one text block with blank separators: the previous per-item cursor
    # over-advanced and pushed the last line outside the panel
    block = "\n\n".join("\n".join(textwrap.wrap(t, 40)) for t in items)
    ax.text(0.06, 0.90, block, ha="left", va="top", color="#B9C4D2",
            fontsize=7.6, linespacing=1.55)



def page_buildup(fig, f):
    title(fig, "BUILD-UP MAP", "which channel to use against them, and why")
    fig.text(0.5, 0.898,
             "All three charts use the BUILDING team's own lanes, so they read "
             "directly against each other.",
             ha="center", va="center", color=MUTED, fontsize=7.8)

    # --- 1. how hard they press each channel
    tp = f["their_press_lane"]
    press = [tp[l][0] for l in LANES]
    fail = [tp[l][1] for l in LANES]
    worst = int(np.argmax(press))
    fig.text(0.055, 0.862, "1 · How hard they press each channel", ha="left",
             va="center", color=INK, fontsize=10.5, weight="bold")
    ax = fig.add_axes([0.055, 0.655, 0.415, 0.185])
    style_bar_axis(ax, ylabel="Mean pressure on the ball", label_size=8.5,
                   tick_size=7)
    x = np.arange(len(LANES))
    ax.bar(x, press, color=[BAD if i == worst else _fade(BAD) for i in x],
           width=.62, zorder=3)
    for i, (pv, fv) in enumerate(zip(press, fail)):
        ax.text(i, pv + max(press) * .04, f"{pv:.0f}", ha="center", va="bottom",
                fontsize=7.4, color=INK, weight="bold")
        ax.text(i, max(press) * .045, f"{fv:.0f}% lost", ha="center",
                va="bottom", fontsize=6.2, color="#10161F", weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(LANE_LABEL, fontsize=7)
    ax.set_ylim(0, max(press) * 1.25)
    # how much heavier the worst channel is than the rest, described in
    # proportion to the real gap - this used to just assert "about twice any
    # other channel" regardless of the actual numbers, which was true for the
    # opponent it was first written for and silently wrong for every one since.
    others = [v for i, v in enumerate(press) if i != worst]
    others_mean = sum(others) / len(others) if others else 0
    ratio = press[worst] / others_mean if others_mean else 1.0
    if ratio >= 1.6:
        press_detail = f"about {ratio:.1f}x any other channel"
    elif ratio >= 1.12:
        press_detail = f"{(ratio - 1) * 100:.0f}% heavier than the rest average"
    else:
        press_detail = "though not by much - their press is fairly even across the pitch"
    fig.text(0.055, 0.628, f"Their heaviest press is on the "
                           f"{LANE_LABEL[worst].replace(chr(10),' ').lower()}, "
                           f"{press_detail}.",
             ha="left", va="center", color=MUTED, fontsize=7)

    # --- 2. which part of our shape their turnovers hurt
    zd = dict(list(f["zone_damage"].items())[:7])
    fig.text(0.525, 0.862, "2 · Which part of our shape their turnovers hurt",
             ha="left", va="center", color=INK, fontsize=10.5, weight="bold")
    ax = fig.add_axes([0.525, 0.655, 0.415, 0.185])
    style_bar_axis(ax, ylabel="Opponents removed", label_size=8.5, tick_size=7)
    ks, vs = list(zd.keys()), list(zd.values())
    x = np.arange(len(ks))
    ax.bar(x, vs, color=[BAD if i == 0 else _fade(BAD) for i in x], width=.62,
           zorder=3)
    for i, v in enumerate(vs):
        ax.text(i, v + max(vs) * .04, f"{v:.0f}", ha="center", va="bottom",
                fontsize=7, color=INK, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([ZONE_LABEL.get(k, k) for k in ks], fontsize=6.6,
                       rotation=20, ha="right")
    ax.set_ylim(0, max(vs) * 1.25)
    fbl = f["zone_damage"].get("FBL", 0.0)
    fbr = f["zone_damage"].get("FBR", 0.0)
    side = ("left back" if fbl > fbr else "right back")
    fig.text(0.525, 0.628, f"Impect's own zone labels. Worst hit: "
                           f"{ZONE_LABEL.get(ks[0], ks[0]).lower()}. Of the two "
                           f"full-backs the {side} took more "
                           f"({max(fbl,fbr):.0f} to {min(fbl,fbr):.0f}).",
             ha="left", va="center", color=MUTED, fontsize=7)

    # --- 3. where we actually build well
    ob = f["our_build"]
    ofail = [ob[l][0] for l in LANES]
    opress = [ob[l][1] for l in LANES]
    on = [ob[l][2] for l in LANES]
    best = int(np.argmin([v if n >= 8 else 999 for v, n in zip(ofail, on)]))
    fig.text(0.055, 0.588, f"3 · Where {US} actually build well", ha="left",
             va="center", color=INK, fontsize=10.5, weight="bold")
    ax = fig.add_axes([0.055, 0.372, 0.885, 0.195])
    style_bar_axis(ax, ylabel="% of build-up passes lost", label_size=8.5,
                   tick_size=7)
    x = np.arange(len(LANES))
    ax.bar(x, ofail, color=[GOOD if i == best else _fade(GOOD) for i in x],
           width=.55, zorder=3)
    for i, (fv, pv, nv) in enumerate(zip(ofail, opress, on)):
        ax.text(i, fv + max(ofail) * .04, f"{fv:.0f}%", ha="center",
                va="bottom", fontsize=8, color=INK, weight="bold")
        ax.text(i, max(ofail) * .04, f"n={nv} · pressure {pv:.0f}", ha="center",
                va="bottom", fontsize=6.2, color="#10161F", weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(chr(10), " ") for l in LANE_LABEL], fontsize=8)
    ax.set_ylim(0, max(ofail) * 1.28)
    n = f.get("our_match_count", 1)
    span = f"our last {n} matches" if n >= 2 else "our most recent match"
    fig.text(0.055, 0.345, f"From {span}. Lower is better — this "
                           f"half of the answer is about us, not them.",
             ha="left", va="center", color=MUTED, fontsize=7)

    # --- 4. the verdict, derived from the three views above
    rank = lambda v: np.argsort(np.argsort(v))
    combined = rank(press) + rank(fail) + rank(ofail)
    route = int(np.argmin(combined)); avoid = int(np.argmax(combined))
    top_zone = ZONE_LABEL.get(list(zd)[0], list(zd)[0])
    # "both halves agree" is only true when the combined-rank channel is
    # ALSO the individual worst on both charts separately - the combined
    # score can pick a channel that's merely bad on both without being the
    # single worst on either, which used to get the same "strongest signal"
    # claim regardless.
    both_agree = (avoid == int(np.argmax(press))) and (avoid == int(np.argmax(ofail)))
    avoid_tail = ("Both halves of the analysis agree, which is the strongest "
                 "signal on this page." if both_agree else
                 "The combined read across both charts, weighing their "
                 "pressure and our build-up success together.")
    cards = [
        ("THE ROUTE", LANE_LABEL[route].replace("\n", " "), GOOD,
         f"Their pressure there is {press[route]:.0f} and they won only "
         f"{fail[route]:.0f}% of build-up back from it — and it is our most "
         f"reliable channel at {ofail[route]:.0f}% lost."),
        ("AVOID", LANE_LABEL[avoid].replace("\n", " "), BAD,
         f"Their heaviest press ({press[avoid]:.0f}) meets our weakest channel "
         f"({ofail[avoid]:.0f}% lost). {avoid_tail}"),
        ("ALSO WATCH", top_zone, "#F5C518",
         f"Chart 2 measures roles, not flanks, so it sees something chart 1 "
         f"cannot: their turnovers hurt the {top_zone.lower()} more than any "
         f"other zone. With {f['press_kind']['BETWEEN_THE_LINES'][0]:.0f} "
         f"presses between the lines, playing into the 6 under pressure is the "
         f"risk whichever side we favour."),
    ]
    for i, (kicker, head, colour, body) in enumerate(cards):
        x0 = 0.055 + i * 0.303
        ax = fig.add_axes([x0, 0.088, 0.283, 0.225]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1,
                                    boxstyle="round,pad=0,rounding_size=0.025",
                                    fc=PANEL, ec=GRID, lw=0.9,
                                    mutation_aspect=0.30))
        ax.add_patch(Rectangle((0, 0.05), 0.016, 0.90, fc=colour, ec="none"))
        ax.text(0.075, 0.90, kicker, ha="left", va="top", color=colour,
                fontsize=8, weight="bold")
        ax.text(0.075, 0.745, head, ha="left", va="top", color=INK,
                fontsize=13.5, weight="bold")
        ax.text(0.075, 0.545, "\n".join(textwrap.wrap(body, 34)), ha="left",
                va="top", color="#B9C4D2", fontsize=7.6, linespacing=1.55)

    division_note = (f"{OPPONENT} play in a different competition to Orient"
                     if f["ref_diff_division"] else
                     f"both sides play in {OUR_COMPETITION} this season")
    fig.text(0.5, 0.048,
             f"{n} matches on our side, {f['opp_match_count']} on theirs, and "
             f"{division_note}. The agreement in the first two cards is more "
             f"reliable than "
             f"either chart alone.",
             ha="center", va="center", color=MUTED, fontsize=7.2)


def page_channels(fig, f):
    title(fig, "THE CHANNELS", "which side they use, and which side gave way")
    our_cover_side = "RIGHT" if f["prog_side"] == "LEFT" else "LEFT"
    prog_tail = (f"— close to an even split, not a clear lean" if f["prog_close"]
                else f"— our {our_cover_side.lower()} side must be ready")
    conceded_tail = (f"— close to an even split, not yet a clear pattern to attack"
                     if f["conceded_close"] else "— that is the side to attack")
    specs = [
        ("Where they build from", f["lane_build"], OPP_C, 0.700,
         "Their first-third passes. They start centrally."),
        ("Where they progress", f["lane_prog"], GOOD, 0.435,
         f"Their bypassed defenders. "
         f"{f['prog_share']:.0f}% of it comes down THEIR "
         f"{f['prog_side']} {prog_tail}."),
        ("Where they were opened up", f["lane_conceded"], BAD, 0.170,
         f"The opposition's progression, mirrored onto their shape. "
         f"{f['conceded_share']:.0f}% came down "
         f"THEIR {f['conceded_side']} {conceded_tail}."),
    ]
    for label, vals, colour, y, note in specs:
        fig.text(0.075, y + 0.205, label, ha="left", va="center", color=INK,
                 fontsize=11.5, weight="bold")
        ax = fig.add_axes([0.075, y, 0.855, 0.190])
        style_bar_axis(ax, ylabel="% of total", label_size=9, tick_size=8)
        x = np.arange(len(LANES))
        cols = [colour if v == max(vals) else _fade(colour) for v in vals]
        ax.bar(x, vals, color=cols, width=.62, zorder=3)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * .04, f"{v:.0f}%", ha="center",
                    va="bottom", fontsize=8, color=INK, weight="bold")
        ax.set_xticks(x); ax.set_xticklabels(LANE_LABEL, fontsize=8)
        ax.set_ylim(0, max(vals) * 1.25)
        fig.text(0.075, y - 0.030, note, ha="left", va="center", color=MUTED,
                 fontsize=8)
    fig.text(0.5, 0.055,
             "Lanes are always from the attacking team's point of view. The "
             "bottom chart is mirrored so it reads as their defensive shape.",
             ha="center", va="center", color=MUTED, fontsize=7.4)


def _force_long_card(cp):
    """The 'Force them long' exploit card - only makes sense as advice when
    their chipped passes actually go astray often. It used to recommend
    cutting the short option regardless of the real completion rate, which
    would be backwards advice against a side that's genuinely good in the air."""
    total = cp[0] + cp[1]
    rate = cp[0] / total if total else 0
    if rate < 0.55:
        body = ("When they skip midfield they lose it. Cutting the short "
                "option is worth more than chasing the ball.")
    else:
        body = ("They actually retain the ball well when they go long - "
                "don't expect an easy turnover just by inviting it.")
    return ("Force them long", f"{rate*100:.0f}%",
           "of chipped passes found a teammate", body)


def _turnover_card(bwrd, bwrd_r):
    """The 'their turnovers hurt' beware card, with wording that actually
    matches the ratio rather than a fixed 'nearly three times the damage'
    claim regardless of the real numbers - true for whichever opponent it
    was first written against, silently wrong (sometimes backwards, as for
    a side whose BWRD is actually LOWER than ours) for every other one."""
    ratio = bwrd / bwrd_r if bwrd_r else None
    if ratio is None:
        body = "No comparable figure for our own turnovers in this sample."
    elif ratio >= 1.6:
        body = (f"About {ratio:.1f}x the damage per loss we do. When they "
                f"win it, they win it somewhere that matters.")
    elif ratio >= 1.12:
        body = (f"{(ratio - 1) * 100:.0f}% more damage per loss than our "
                f"own turnovers average.")
    elif ratio >= 0.88:
        body = "About the same damage per loss as our own turnovers - not a distinguishing feature of their game."
    else:
        body = (f"Actually less damaging per loss than our own turnovers "
                f"({bwrd:.1f} to {bwrd_r:.1f}) - this isn't where the game gets lost.")
    return ("Their turnovers hurt", f"{bwrd:.1f}",
           f"defenders removed by their ball wins (ours {bwrd_r:.1f})", body)


def _fade(colour, amount=0.55, toward="#5A6676"):
    from matplotlib.colors import to_rgb
    c, t = np.array(to_rgb(colour)), np.array(to_rgb(toward))
    return tuple(c * (1 - amount) + t * amount)


def _top(p, mask, sort_col, ascending=False):
    """Top row of p[mask] by sort_col, or None if no player fits.

    Formations vary: a back-3 side has no WINGER-tagged position at all (wing-
    backs instead), some sides play with no out-and-out CENTER_FORWARD. Every
    position-filtered lookup on this page has to tolerate that rather than
    assume Sheffield Wednesday's shape generalises to every opponent.
    """
    d = p[mask].sort_values(sort_col, ascending=ascending)
    return d.iloc[0] if len(d) else None


def _press_target(position):
    """Which line of the pitch a position belongs to, for the 'Press their
    X' card headline. The body text already lists the real positions of the
    top losers rather than assuming a back-line/wing-back archetype - the
    headline has to agree with it instead of asserting 'back line'
    regardless of whether the actual costliest loser plays there."""
    if position in ("GOALKEEPER", "CENTRAL_DEFENDER",
                    "LEFT_WINGBACK_DEFENDER", "RIGHT_WINGBACK_DEFENDER",
                    "LEFT_FULLBACK_DEFENDER", "RIGHT_FULLBACK_DEFENDER"):
        return "back line"
    if position in ("DEFENSE_MIDFIELD", "CENTRAL_MIDFIELD", "ATTACKING_MIDFIELD"):
        return "midfield"
    return "front line"


def page_plan(fig, f):
    title(fig, "THE PLAN")
    p = f["players"]
    if len(p) == 0:
        fig.text(0.5, 0.5, "No player data available for this match.",
                 ha="center", va="center", color=MUTED, fontsize=11)
        return
    losers = p.nlargest(3, "loss")
    cp = f["passtype"]["CHIPPED_PASS"]
    cf = _top(p, p.position == "CENTER_FORWARD", "mins")
    # len(p)>0 is already guaranteed by the early return above, so this always
    # finds someone - unlike cf, which depends on a position that is not
    # guaranteed to exist (a side playing false-9 has no CENTER_FORWARD tag).
    bannan = _top(p, pd.Series(True, index=p.index), "BPD")
    tr = f["phase"]["ATTACKING_TRANSITION"][0] / sum(v[0] for v in f["phase"].values()) * 100

    sc = f["scores"]
    attack_side = f["conceded_side"].lower()      # their weaker defensive side
    prog_side = f["prog_side"].lower()             # their stronger attacking side
    cover_side = "left" if f["prog_side"] == "RIGHT" else "right"
    right_open = f["conceded_share"]
    left_prog = f["prog_share"]
    ppda = sc.get("PPDA", (0, 0))
    ph = sc.get("MEAN_PRESSURE_HEIGHT", (0, 0))
    gk = sc.get("PRESSURE_OPP_GK_PERCENT", (0, 0))
    poss = sc.get("RATIO_BALL_POSSESSION", (0, 0))

    weak_flank = p[p.position.str.startswith(f"{attack_side.upper()}_WING")]
    weak_flank_name = (surname(weak_flank.nlargest(1, "loss").playerName.iloc[0])
                       if len(weak_flank) else f"their {attack_side}-back")
    attack_body = (
        f"Close to an even split with their other side, so this isn't a "
        f"clear-cut weakness yet - but {weak_flank_name} is still among "
        f"their costliest losers of the ball, worth testing early."
        if f["conceded_close"] else
        f"Their clearest defensive weakness, and {weak_flank_name} "
        f"is among their costliest losers of the ball. Overload it.")
    exploit = [
        (f"Attack their {attack_side}", f"{right_open:.0f}%",
         "of the opposition's progression came down that side", attack_body),
        (f"Press their {_press_target(losers.iloc[0].position)}", f"{losers.iloc[0].loss:.1f}",
         f"teammates taken out by {surname(losers.iloc[0].playerName)}'s losses",
         f"The three most damaging losers were "
         + ", ".join(f"{surname(x.playerName)} "
                     f"({x.position.replace('_', ' ').lower()})"
                     for _, x in losers.iterrows()) + "."),
        _force_long_card(cp),
    ]
    if f["corners"] == 0 or f["corner_xg"] <= 0:
        dead_ball_head, dead_ball_body = "Nothing from dead balls", (
            f"No direct set-piece threat at all, and their {f['fouls']:.1f} "
            f"fouls a game should give us dead balls of our own.")
    elif f["corner_concentrated"]:
        dead_ball_head, dead_ball_body = "Their corners aren't a pattern", (
            f"Most of that xG is one costly phase, not something repeatable. "
            f"Their {f['fouls']:.1f} fouls a game should still give us dead "
            f"balls of our own.")
    else:
        dead_ball_head, dead_ball_body = "A real if modest corner threat", (
            f"Spread across deliveries rather than one moment - worth "
            f"defending properly, not dismissing. Their {f['fouls']:.1f} "
            f"fouls a game should still give us dead balls of our own.")
    exploit.append((dead_ball_head, f"{f['corner_xg']:.2f}",
                    f"xG from their {f['corners']} corners", dead_ball_body))
    if cf is not None:
        # Inserted rather than appended so it sits with the other individual-
        # matchup points. Skipped entirely for a side with no CENTER_FORWARD-
        # tagged player (false-9, or a front two both tagged differently).
        # Wording used to unconditionally call him "a runner, not a target
        # man" regardless of his actual aerial record - true for whichever
        # centre-forward that was first written about, backwards for any CF
        # who actually wins most of what comes his way in the air.
        cf_total = cf.aw + cf.al
        cf_rate = cf.aw / cf_total if cf_total else 0.5
        if cf_rate <= 0.45:
            cf_body = ("Their centre-forward is a runner, not a target man. "
                      "Our centre-backs should back themselves and squeeze up.")
        elif cf_rate >= 0.55:
            cf_body = ("Their centre-forward wins most of what comes his way "
                      "in the air - a genuine threat, not a soft touch.")
        else:
            cf_body = ("Roughly an even contest with their centre-forward in "
                      "the air - no clear edge either way.")
        exploit.insert(3, (
            "Win the first ball", f"{int(cf.aw)}/{int(cf.aw+cf.al)}",
            f"aerials won by {surname(cf.playerName)}", cf_body))
    beware = [
        ("Transitions, above all", f"{tr:.0f}%",
         "of their progression came in transition",
         f"Plus {f['xg_transition']:.2f} xG from transition alone. Careless "
         f"turnovers in our own half are how this game is lost."),
        ("The press is real", f"{ppda[0]:.1f}",
         f"PPDA, against our own {ppda[1]:.1f}",
         f"They pressed at {ph[0]:.0f}m and hunted between the lines. Our 8s "
         f"need one-touch options."),
        (f"They come down their {prog_side}", f"{left_prog:.0f}%",
         "of their progression is down that side",
         (f"Close to an even split with their other side, so don't over-"
          f"shift for it - keep both flanks covered rather than loading up "
          f"on one."
          if f["prog_close"] else
          f"Their {prog_side} wing alone accounts for the biggest single "
          f"share. Our {cover_side}-sided pair will need cover.")),
        _turnover_card(f["bwrd"], f["bwrd_r"]),
        (f"{surname(bannan.playerName)} is the pivot", f"{bannan.BPD:.1f}",
         "bypassed defenders — most on the pitch",
         "Cutting the line into him is the clearest individual job in the game."),
    ]
    # PRESSURE_OPP_GK_PERCENT is how much THEY press an opposing keeper - only
    # worth a beware card when that's actually true. At 0% the old
    # unconditional card said "they press the keeper" from a stat that says
    # the opposite.
    GK_PRESS_THRESHOLD = 0.08
    if gk[0] >= GK_PRESS_THRESHOLD:
        # "only X% and didn't want it" only makes sense when that share is
        # genuinely low - it used to fire unconditionally, which read as
        # nonsense for a side that actually had the majority of the ball.
        if poss[0] < 0.48:
            gk_body = (f"They had only {poss[0]*100:.0f}% of the ball and "
                      f"did not want it — expect our goalkeeper and "
                      f"centre-backs to be hurried.")
        else:
            gk_body = (f"They press the keeper even with {poss[0]*100:.0f}% "
                      f"of the ball themselves — expect our goalkeeper and "
                      f"centre-backs to be hurried regardless.")
        beware.append(("They press the keeper", f"{gk[0]*100:.0f}%",
                       f"of GK actions pressed (ours {gk[1]*100:.0f}%)", gk_body))

    for x0, head, items, accent in ((0.048, "WHERE TO ATTACK THEM", exploit, GOOD),
                                    (0.512, "WHAT TO BE READY FOR", beware, BAD)):
        ax = fig.add_axes([x0, 0.868, 0.440, 0.036]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.add_patch(FancyBboxPatch((0, 0.05), 1, 0.9,
                                    boxstyle="round,pad=0,rounding_size=0.03",
                                    fc=accent, ec="none", mutation_aspect=0.12))
        ax.text(0.5, 0.5, head, ha="center", va="center", color="#10161F",
                fontsize=11.5, weight="bold")
        n = len(items)
        top, gap = 0.850, 0.009
        h = min(0.128, (top - 0.085 - gap * (n - 1)) / n)
        for i, (hl, stat, slab, body) in enumerate(items):
            y = top - (i + 1) * h - i * gap
            point_card(fig, [x0, y, 0.440, h], i + 1, hl, stat, slab, body, accent)

    n_opp, n_our = len(f["opp_matches"]), len(f["our_matches"])
    match_note = (f"{n_opp} matches for them, {n_our} for us" if n_opp != n_our
                 else f"{n_opp} matches each")
    division_qualifier = (", who play in a different competition to Orient"
                          if f["ref_diff_division"] else "")
    fig.text(0.5, 0.062,
             f"{match_note}{division_qualifier}. Treat the patterns as "
             f"reasonably reliable and the individual figures as indicative.",
             ha="center", va="center", color=MUTED, fontsize=7.6)
    footer = (f"Built from {OPPONENT} ({f['opp_comp_mix']}) and Orient "
             f"({f['our_comp_mix']}). The channel read is the most "
             f"actionable part and the most sample-sensitive — worth "
             f"re-checking as more matches come in.")
    fig.text(0.5, 0.030, "\n".join(textwrap.wrap(footer, 115)), ha="center",
             va="center", color=MUTED, fontsize=7.0, linespacing=1.6)


def main():
    apply_style()
    opp, ours = load()
    f = facts(opp, ours)
    with PdfPages(OUT) as pdf:
        for builder in (page_cover, page_profile, page_style, page_press,
                        page_channels, page_buildup, page_personnel,
                        page_plan):
            fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor=BG)
            builder(fig, f)
            pdf.savefig(fig, facecolor=BG)
            plt.close(fig)
    print(f"wrote {OUT}  (8 pages)")


if __name__ == "__main__":
    main()
