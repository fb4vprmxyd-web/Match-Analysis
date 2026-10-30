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
                           draw_pitch, contrast_ink, PITCH_X, PITCH_Y,
                           stat_table, player_table, line_pitch)
from envconfig import CACHE_ROOT, REPORTS_ROOT


def draw_pitch_portrait(ax, lw=0.9, colour=None):
    """Pitch drawn tall (pitch WIDTH along the page's horizontal axis,
    pitch LENGTH running vertically) rather than landscape - every
    coordinate below is report_config.draw_pitch()'s own, transposed: a +90
    degree rotation, not a mirror, so a penalty-arc bulge still points the
    right way at each end. Needed for the lane/channel views: a lane is a
    strip of pitch WIDTH, and "left wing reads as the left column on the
    page" only holds if the pitch itself is portrait. Same dark-theme
    colours as report_config.draw_pitch() - this is the one shape it
    doesn't already provide a portrait version of."""
    c = colour or INK
    ax.set_facecolor(BG)
    X, Y = PITCH_X, PITCH_Y
    r = dict(fill=False, color=c, lw=lw, zorder=1)

    from matplotlib.patches import Rectangle, Circle, Arc
    ax.add_patch(Rectangle((-Y, -X), 2 * Y, 2 * X, **r))
    ax.plot([-Y, Y], [0, 0], color=c, lw=lw, zorder=1)
    ax.add_patch(Circle((0, 0), 9.15, **r))
    ax.plot([0], [0], marker=".", ms=2, color=c, zorder=1)

    for s in (-1, 1):
        ax.add_patch(Rectangle((-20.16, s * X - s * 16.5), 40.32, s * 16.5, **r))
        ax.add_patch(Rectangle((-9.16, s * X - s * 5.5), 18.32, s * 5.5, **r))
        ax.plot([0], [s * (X - 11)], marker=".", ms=2, color=c, zorder=1)
        ax.add_patch(Arc((0, s * (X - 11)), 18.3, 18.3, angle=0,
                         theta1=218 if s > 0 else 38,
                         theta2=322 if s > 0 else 142, **r))

    ax.set_xlim(-Y - 1.5, Y + 1.5)
    ax.set_ylim(-X - 1.5, X + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")


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
LANE_SHORT = ["Left wing", "Left half-space", "Centre", "Right half-space",
             "Right wing"]
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
                 ("goals", "GOALS"), ("on_target", "SHOT_AT_GOAL_NUMBER_ON_TARGET"),
                 ("shots_in", "SHOT_AT_GOAL_NUMBER_IN_PITCH_POSITION_OPPONENT_BOX"),
                 ("box_passes", "SUCCESSFUL_PASSES_TO_PITCH_POSITION_OPPONENT_BOX"),
                 # No raw "interception count" field in the matchsums (unlike
                 # the post-match report's event-level data) - the closest is
                 # ball wins specifically via an interception, which should be
                 # very close in practice since an interception by definition
                 # wins the ball.
                 ("ints", "BALL_WIN_NUMBER_BY_ACTION_INTERCEPTION"),
                 ("cross_ok", None), ("cross_bad", None)]:
        if c is None:
            continue
        f[k], f[k + "_r"] = g(c), g_us(c)
    f["shots_out"], f["shots_out_r"] = f["shots"] - f["shots_in"], f["shots_r"] - f["shots_in_r"]

    # crosses completed % - both cross action types combined, both sides.
    # No single Impect column for "crosses"; built from the same
    # successful/unsuccessful-by-action columns passtype already reads.
    def _cross_pct(get_fn):
        ok = get_fn("SUCCESSFUL_PASSES_BY_ACTION_HIGH_CROSS") + get_fn("SUCCESSFUL_PASSES_BY_ACTION_LOW_CROSS")
        bad = get_fn("UNSUCCESSFUL_PASSES_BY_ACTION_HIGH_CROSS") + get_fn("UNSUCCESSFUL_PASSES_BY_ACTION_LOW_CROSS")
        return (ok / (ok + bad) * 100) if (ok + bad) else 0.0, ok, bad
    f["cross_pct"], f["cross_ok"], f["cross_bad"] = _cross_pct(g)
    f["cross_pct_r"], f["cross_ok_r"], f["cross_bad_r"] = _cross_pct(g_us)

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

    # ---- goalkeeper pass length: no direct Impect column for this, built
    # from event coordinates for whichever player(s) are tagged GOALKEEPER.
    def _gk_pass_length(evdf, psdf, squad):
        if evdf is None or len(evdf) == 0 or psdf is None or len(psdf) == 0:
            return None
        pmap = (psdf.groupby("playerName")["position"]
               .agg(lambda x: x.value_counts().index[0]))
        gks = set(pmap[pmap == "GOALKEEPER"].index)
        d = evdf[(evdf.squadName == squad) & (evdf.actionType == "PASS")
                & (evdf.result == "SUCCESS") & evdf.playerName.isin(gks)]
        if len(d) == 0:
            return None
        dx = d.endAdjCoordinatesX - d.startAdjCoordinatesX
        dy = d.endAdjCoordinatesY - d.startAdjCoordinatesY
        return float(np.sqrt(dx ** 2 + dy ** 2).mean())

    f["gk_pass_len"] = _gk_pass_length(ev, opp["ps"], OPPONENT)
    f["gk_pass_len_r"] = _gk_pass_length(ours["ev"], ours["ps"], US)

    # ---- passes before reaching the box: for each possession sequence
    # (Impect's sequenceIndex, unique within a match) that reaches the box
    # via a pass, how many of this squad's passes came before and including
    # that one - averaged across every sequence that got there at all.
    # sequenceIndex resets per match, so it's grouped with matchId to avoid
    # colliding sequences from different cached matches into one.
    def _passes_before_box(evdf, squad):
        if evdf is None or len(evdf) == 0:
            return None
        d = evdf[evdf.squadName == squad]
        counts = []
        for _, g_ in d.groupby(["matchId", "sequenceIndex"]):
            passes = g_[g_.actionType == "PASS"].reset_index(drop=True)
            box = passes.index[passes.endPitchPosition == "OPPONENT_BOX"]
            if len(box):
                counts.append(int(box[0]) + 1)
        return (sum(counts) / len(counts)) if counts else None

    f["passes_to_box"] = _passes_before_box(ev, OPPONENT)
    f["passes_to_box_r"] = _passes_before_box(ours["ev"], US)

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


def draw_lane_pitch(ax, values, colour, lane_labels, fmt="{:.0f}%"):
    """The lane/channel share charts, drawn on a portrait pitch (explicit
    request: "make them look as a pitch", "vertical not horizontal, left
    wing on the left and right wing on the right"). `values` are in LANES
    order (LEFT_WING..RIGHT_WING); this function is the one place that
    decides which page position each named lane gets, and it does so
    directly from that order - band 0 (LEFT_WING) always renders as the
    leftmost column, band 4 (RIGHT_WING) always the rightmost - so there's
    no coordinate-sign guess involved, just the label order already in
    LANES itself."""
    draw_pitch_portrait(ax, lw=1.0)
    X, Y = PITCH_X, PITCH_Y
    edges = np.linspace(-Y, Y, len(lane_labels) + 1)
    mx = max(values) if max(values) > 0 else 1
    for i, v in enumerate(values):
        x0, x1 = edges[i], edges[i + 1]
        alpha = 0.12 + 0.68 * (v / mx)
        ax.add_patch(Rectangle((x0, -X), x1 - x0, 2 * X, fc=colour, ec="none",
                               alpha=alpha, zorder=0.5))
        ax.plot([x0, x0], [-X, X], color=GRID, lw=0.6, zorder=1)
        ax.text((x0 + x1) / 2, 0, f"{lane_labels[i]}\n{fmt.format(v)}",
                ha="center", va="center", fontsize=8.2, weight="bold",
                color=INK, zorder=5, linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.3", fc=PANEL, ec=GRID, lw=0.7))


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
    # Coach-specified profile list and order, split into Attacking/
    # Defending panels - same list build_report.py's page_match_stats()
    # now uses on the post-match side, see report_metrics.team_stats().
    # Two definitional gaps versus the post-match report, both accepted:
    # Interceptions reads ball-wins-via-interception (no raw interception
    # count in the matchsums this averages across several matches), and
    # every row here is a per-game mean across the sample rather than one
    # match's raw total.
    n_opp, n_our = len(f["opp_matches"]), len(f["our_matches"])
    sample = (f"last {n_opp} for them, {n_our} for us" if n_opp != n_our
             else f"last {n_opp} matches each")
    title(fig, "THE PROFILE", f"{OPPONENT} vs {f['ref']} — {sample}, per game")

    attacking = [
        ("Goals", f"{f['goals']:.1f}", f"{f['goals_r']:.1f}"),
        ("xG", f"{f['xg']:.2f}", f"{f['xg_r']:.2f}"),
        ("Post-Shot xG", f"{f['psxg']:.2f}", f"{f['psxg_r']:.2f}"),
        ("Shots / on target", f"{f['shots']:.0f}/{f['on_target']:.0f}",
         f"{f['shots_r']:.0f}/{f['on_target_r']:.0f}"),
        ("Inside Box", f"{f['shots_in']:.0f}", f"{f['shots_in_r']:.0f}"),
        ("Outside Box", f"{f['shots_out']:.0f}", f"{f['shots_out_r']:.0f}"),
        ("Packing xT", f"{f['pxt']:.2f}", f"{f['pxt_r']:.2f}"),
        ("Passes Into Box", f"{f['box_passes']:.0f}", f"{f['box_passes_r']:.0f}"),
        ("Crosses (Accurate)", f"({f['cross_pct']:.0f}%) {f['cross_ok']+f['cross_bad']:.0f}",
         f"{f['cross_ok_r']+f['cross_bad_r']:.0f} ({f['cross_pct_r']:.0f}%)"),
    ]
    defending = [
        ("Interceptions", f"{f['ints']:.0f}", f"{f['ints_r']:.0f}"),
        ("Pressures", f"{f['press']:.0f}", f"{f['press_r']:.0f}"),
        ("Ball Wins", f"{f['wins']:.0f}", f"{f['wins_r']:.0f}"),
        ("Ball Win Removed Defenders", f"{f['bwrd']:.1f}", f"{f['bwrd_r']:.1f}"),
    ]
    y_top, y_bottom, gap = 0.875, 0.075, 0.022
    n_att, n_def = len(attacking), len(defending)
    total = y_top - y_bottom - gap
    h_def = total * n_def / (n_att + n_def)
    h_att = total * n_att / (n_att + n_def)
    stat_table(fig, [0.075, y_bottom + h_def + gap, 0.85, h_att], attacking,
              OPPONENT, f["ref"], header="ATTACKING",
              left_colour=OPP_C, right_colour=TEAM_C)
    stat_table(fig, [0.075, y_bottom, 0.85, h_def], defending,
              OPPONENT, f["ref"], header="DEFENDING",
              left_colour=OPP_C, right_colour=TEAM_C)

    poss = f["scores"].get("RATIO_BALL_POSSESSION", (0.0, 0.0))
    bits = []
    more_less = "Less" if poss[0] < poss[1] else "More"
    bits.append(f"{more_less} of the ball ({poss[0]*100:.0f}% to {poss[1]*100:.0f}%)")
    bpd_cmp = "more" if f["bpd"] > f["bpd_r"] else "less"
    bits.append(f"{bpd_cmp} progression ({f['bpd']:.1f} to {f['bpd_r']:.1f} "
               f"bypassed defenders)")
    if f["bwrd_r"] > 0:
        bits.append(f"turnovers that hurt {f['bwrd']/f['bwrd_r']:.1f}x as much")
    fig.text(0.5, 0.042, ", ".join(bits) + ".",
             ha="center", va="center", color=MUTED, fontsize=8.5)


def page_style(fig, f):
    # Page 3 feedback: everything here becomes a table; "how they press"
    # moved out to page 4 and the passing network moved in from there;
    # "where they win the ball back" removed (it duplicated page 4's pitch
    # panel, which is the better version of the same thing); goalkeeper
    # pass length, passes before the box, and crosses % added. "How they
    # move the ball forward" was flagged as maybe-remove, not a definite
    # cut - kept, as a table, since the ask was tentative.
    title(fig, "HOW THEY PLAY")

    ph = ["IN_POSSESSION", "ATTACKING_TRANSITION", "SET_PIECE", "SECOND_BALL"]
    lab = ["Settled possession", "Attacking transition", "Set piece", "Second ball"]
    a = [f["phase"][p][0] for p in ph]; b = [f["phase"][p][1] for p in ph]
    share = a[1] / sum(a) * 100 if sum(a) else 0
    rows = [(lab[i], f"{a[i]:.1f}", f"{b[i]:.1f}") for i in range(len(ph))]
    stat_table(fig, [0.055, 0.660, 0.415, 0.230], rows, OPPONENT, f["ref"],
              header="Where their progression comes from",
              left_colour=OPP_C, right_colour=TEAM_C)
    fig.text(0.055, 0.638, f"{share:.0f}% of their progression came in "
                           f"transition (bypassed defenders).",
             ha="left", va="center", color=MUTED, fontsize=7.4)

    acts = ["LOW_PASS", "CHIPPED_PASS", "DIAGONAL_PASS", "DRIBBLE",
            "HIGH_CROSS", "LOW_CROSS"]
    alab = ["Low pass", "Chipped pass", "Diagonal", "Dribble", "High cross", "Low cross"]
    aa = [f["action"][x][0] for x in acts]; ab = [f["action"][x][1] for x in acts]
    rows = [(alab[i], f"{aa[i]:.1f}", f"{ab[i]:.1f}") for i in range(len(acts))]
    stat_table(fig, [0.525, 0.660, 0.415, 0.230], rows, OPPONENT, f["ref"],
              header="How they move the ball forward",
              left_colour=OPP_C, right_colour=TEAM_C)
    diag_o, diag_u = f["action"]["DIAGONAL_PASS"]
    if diag_o >= diag_u and diag_o > 0:
        diag_txt = (f"They switch play: diagonals gave them {diag_o:.1f} "
                    f"bypassed defenders, against our {diag_u:.1f}.")
    elif diag_o > 0:
        diag_txt = (f"They rarely switch play: diagonals gave them just "
                    f"{diag_o:.1f} bypassed defenders (ours {diag_u:.1f}).")
    else:
        diag_txt = (f"They don't switch play through diagonals in this "
                    f"sample at all (ours {diag_u:.1f}).")
    fig.text(0.525, 0.638, diag_txt, ha="left", va="center", color=MUTED, fontsize=7.4)

    types = ["LOW_PASS", "CHIPPED_PASS", "DIAGONAL_PASS", "HIGH_CROSS"]
    tlab = ["Low pass", "Chipped pass", "Diagonal", "High cross"]
    trows = []
    for t, lb in zip(types, tlab):
        ok, bad = f["passtype"][t]
        rate = ok / (ok + bad) * 100 if (ok + bad) else 0
        trows.append((lb, f"{int(ok)}/{int(ok+bad)}", f"{rate:.0f}%"))
    player_table(fig, [0.055, 0.365, 0.415, 0.250], trows,
                ["Pass type", "Completed", "Rate"],
                header="…and how often it works")
    cp = f["passtype"]["CHIPPED_PASS"]
    cp_rate = cp[0] / (cp[0] + cp[1]) * 100 if (cp[0] + cp[1]) else 0
    cp_word = "Only" if cp_rate < 55 else "A healthy"
    fig.text(0.055, 0.345, f"{cp_word} {cp_rate:.0f}% of their chipped "
                           f"passes found a teammate.",
             ha="left", va="center", color=MUTED, fontsize=7.4)

    st = ["LONG_RANGE_SHOT", "MID_RANGE_SHOT", "CLOSE_RANGE_SHOT",
          "ONE_VS_ONE_AGAINST_GK", "HEADER", "OPEN_GOAL_SHOT", "FREE_KICK",
          "CORNER", "PENALTY_KICK"]
    sn = ["Long range", "Mid range", "Close range", "1v1 vs GK", "Header",
          "Open goal", "Free kick", "Corner", "Penalty"]
    srows = [(lb, f"{f['shot_type'][t][0]:.0f}", f"{f['shot_type'][t][1]:.2f}")
            for t, lb in zip(st, sn) if f["shot_type"][t][0] > 0] or [("None", "0", "0.00")]
    player_table(fig, [0.525, 0.365, 0.415, 0.250], srows,
                ["Shot type", "Shots", "xG"], header="Where they shoot from")
    mid_n = f["shot_type"]["MID_RANGE_SHOT"][0]
    fig.text(0.525, 0.345, f"{mid_n:.0f} of their {f['shots']:.0f} shots were "
                           f"from mid range — low-value efforts.",
             ha="left", va="center", color=MUTED, fontsize=7.4)

    extra = [
        ("Goalkeeper pass length",
         f"{f['gk_pass_len']:.0f}m" if f["gk_pass_len"] is not None else "-",
         f"{f['gk_pass_len_r']:.0f}m" if f["gk_pass_len_r"] is not None else "-"),
        ("Passes before reaching the box",
         f"{f['passes_to_box']:.1f}" if f["passes_to_box"] is not None else "-",
         f"{f['passes_to_box_r']:.1f}" if f["passes_to_box_r"] is not None else "-"),
        ("Crosses completed",
         f"{f['cross_ok']:.0f}/{f['cross_ok']+f['cross_bad']:.0f}  {f['cross_pct']:.0f}%",
         f"{f['cross_ok_r']:.0f}/{f['cross_ok_r']+f['cross_bad_r']:.0f}  {f['cross_pct_r']:.0f}%"),
    ]
    stat_table(fig, [0.055, 0.078, 0.415, 0.220], extra, OPPONENT, f["ref"],
              header="More on their build-up", left_colour=OPP_C, right_colour=TEAM_C)

    # ---- starting XI passing network (moved in from the press page)
    cut = f.get("first_change")
    fig.text(0.525, 0.298, "Starting XI — passing network", ha="left",
             va="center", color=INK, fontsize=13, weight="bold")
    ax = fig.add_axes([0.525, 0.078, 0.415, 0.205])
    draw_pitch(ax, lw=0.9)
    nodes, edges = f.get("net", (None, None))
    if nodes is not None and len(edges):
        mx = edges.n.max()
        pos = nodes.set_index("playerName")[["x", "y"]].to_dict("index")
        for _, e in edges.iterrows():
            a2, b2 = e.playerName, e.passReceiverPlayerName
            if a2 in pos and b2 in pos and e.n >= 2:
                ax.plot([pos[a2]["x"], pos[b2]["x"]], [pos[a2]["y"], pos[b2]["y"]],
                        color="#D8E2EE", lw=0.4 + 3.2 * (e.n / mx),
                        alpha=0.25 + 0.55 * (e.n / mx), zorder=2,
                        solid_capstyle="round")
    if nodes is not None and len(nodes):
        smax = nodes.n.max()
        offs = RM.label_offsets(nodes, ax, s_base=95, s_extra=520)
        for k, (_, r2) in enumerate(nodes.iterrows()):
            ax.scatter([r2.x], [r2.y], s=95 + 520 * (r2.n / smax), color=OPP_C,
                       edgecolors="#BFD9F5", linewidths=1.2, zorder=3)
            lab_ = str(f["shirts"].get(r2.playerName, ""))
            if not lab_:
                lab_ = surname(r2.playerName)[:3].upper()
            lx, ly, leader = offs[k]
            if leader:
                ax.plot([r2.x, lx], [r2.y, ly], color="#BFD9F5", lw=0.7,
                        alpha=.8, zorder=4)
                ax.text(lx, ly, lab_, ha="center", va="center", color="#0B1B2B",
                        fontsize=7, weight="bold", zorder=6,
                        bbox=dict(boxstyle="circle,pad=0.28", fc="#BFD9F5",
                                  ec="none"))
            else:
                ax.text(r2.x, r2.y, lab_, ha="center", va="center",
                        color="#0B1B2B", fontsize=7, weight="bold", zorder=5)
    fig.text(0.7325, 0.062, f"0–{cut}' before the first change · "
                            f"{f.get('net_passes', 0)} successful passes · "
                            f"vs {f['net_opp']}, {f['net_date']} · attacking right",
             ha="center", va="center", color=MUTED, fontsize=6.8)

def page_personnel(fig, f):
    # Page 7 feedback: tables throughout, minutes played dropped entirely -
    # the space that freed up goes to making the four tables below taller
    # (more rows each) rather than sitting empty.
    title(fig, "KEY PERSONNEL",
         f"{OPPONENT} — combined across their last {len(f['opp_matches'])} "
         f"matches ({f['opp_comp_mix']})")
    p = f["players"]
    if len(p) == 0:
        fig.text(0.5, 0.5, "No player data available for this match.",
                 ha="center", va="center", color=MUTED, fontsize=11)
        return
    labels = f["labels"]

    def _name(n):
        return labels.get(n, surname(n))

    top = p.nlargest(1, "BPD").iloc[0]
    d = p[p.BPD != 0].sort_values("BPD", ascending=False).head(8)
    rows = [(_name(r.playerName), r.position.replace("_", " ").title(), f"{r.BPD:.1f}")
           for _, r in d.iterrows()]
    player_table(fig, [0.075, 0.560, 0.415, 0.335], rows,
                ["Player", "Position", "Bypassed def."],
                header="Who progresses the ball")
    fig.text(0.075, 0.538, f"{surname(top.playerName)} "
                           f"({top.position.replace('_',' ').lower()}) is the "
                           f"pivot — {top.BPD:.1f} bypassed defenders.",
             ha="left", va="center", color=MUTED, fontsize=7.4)

    topr = p.nlargest(1, "BPDR").iloc[0]
    d = p[p.BPDR != 0].sort_values("BPDR", ascending=False).head(8)
    rows = [(_name(r.playerName), r.position.replace("_", " ").title(), f"{r.BPDR:.1f}")
           for _, r in d.iterrows()]
    player_table(fig, [0.525, 0.560, 0.415, 0.335], rows,
                ["Player", "Position", "Received"],
                header="Who receives it")
    fig.text(0.525, 0.538, f"{surname(topr.playerName)} is the out-ball — "
                           f"{topr.BPDR:.1f} received.", ha="left", va="center",
             color=MUTED, fontsize=7.4)

    # named by their actual positions rather than assumed to always be "the
    # back line and wing-backs" - that was true for one opponent and not
    # guaranteed for the next (a defensive midfielder topped this list here).
    d = p[p.loss != 0].sort_values("loss", ascending=False).head(8)
    rows = [(_name(r.playerName), r.position.replace("_", " ").title(), f"{r.loss:.1f}")
           for _, r in d.iterrows()]
    player_table(fig, [0.075, 0.150, 0.415, 0.335], rows,
                ["Player", "Position", "Ball loss removed"],
                header="Who gives it away (costliest losses)")
    top_losers = p[p.loss != 0].sort_values("loss", ascending=False).head(3)
    loss_desc = ", ".join(f"{surname(x.playerName)} ({x.position.replace('_', ' ').lower()})"
                          for _, x in top_losers.iterrows())
    fig.text(0.075, 0.128, "\n".join(textwrap.wrap(f"Costliest here: {loss_desc}.", 62)),
             ha="left", va="top", color=MUTED, fontsize=7.4, linespacing=1.5)

    # sorted by total duels, not wins - sorting by wins alone buries a player
    # who contests plenty of headers but loses most of them (Wigan's own
    # centre-forward, 0 of 4, dropped off a wins-sorted top 9 while the text
    # below kept citing him).
    cf = p[p.position == "CENTER_FORWARD"].nlargest(1, "mins")
    d = (p[(p.aw + p.al) > 0].assign(_tot=lambda x: x.aw + x.al)
        .sort_values("_tot", ascending=False, kind="stable").head(8))
    if len(cf) and cf.iloc[0].playerName not in d.playerName.values:
        d = pd.concat([d.iloc[:-1], cf.assign(_tot=cf.aw + cf.al)])
    d = d.sort_values("aw", ascending=False, kind="stable")
    rows = [(_name(r.playerName), f"{int(r.aw)}/{int(r.aw+r.al)}",
             f"{r.aw/(r.aw+r.al)*100:.0f}%") for _, r in d.iterrows()]
    player_table(fig, [0.525, 0.150, 0.415, 0.335], rows,
                ["Player", "Won/contested", "Win rate"], header="Aerial duels")
    cf = p[p.position == "CENTER_FORWARD"].nlargest(1, "mins")
    if len(cf):
        c = cf.iloc[0]
        fig.text(0.525, 0.128, f"Their centre-forward {surname(c.playerName)} "
                               f"won {int(c.aw)} of {int(c.aw + c.al)} in the air.",
                 ha="left", va="center", color=MUTED, fontsize=7.4)

    varied = " (varied)" if f["formation_varied"] else ""
    fig.text(0.5, 0.045, f"Usual shape: {f.get('formation', 'unknown')}{varied}.",
             ha="center", va="center", color=MUTED, fontsize=7.6)


def page_press(fig, f):
    # Page 4 feedback: tables throughout, expected points and the set-piece
    # panel dropped, "how they press" moved in from page 3 (its passing
    # network moved the other way, to page 3), opening-balls-pressed added
    # as the extra pressing metric - already computed in facts() but never
    # shown anywhere before this. The recoveries pitch panel is unaffected
    # and widened into the space the set-piece panel and network freed up.
    varied = " (varied)" if f["formation_varied"] else ""
    n_opp, n_our = len(f["opp_matches"]), len(f["our_matches"])
    title(fig, "PRESS & SHAPE", f"usual formation {f['formation']}{varied}  ·  "
                               f"{OPPONENT} (last {n_opp}) vs {f['ref']} (last {n_our})")
    sc = f["scores"]

    def s2(k, m=1, fmt="{:.1f}"):
        if k not in sc:
            return "-", "-"
        return fmt.format(sc[k][0] * m), fmt.format(sc[k][1] * m)

    press_rows = [
        ("PPDA (lower = more aggressive)", *s2("PPDA")),
        ("Press height (m upfield)", *s2("MEAN_PRESSURE_HEIGHT", 1, "{:.0f}")),
        ("Opp build-up pressed", *s2("PRESSURE_DURING_OPP_BUILD_UP_PERCENT", 100, "{:.0f}%")),
        ("Opp GK pressed", *s2("PRESSURE_OPP_GK_PERCENT", 100, "{:.0f}%")),
        ("Opening balls pressed", *s2("OPP_OPENING_BALLS_PRESSED_PERCENT", 100, "{:.0f}%")),
    ]
    stat_table(fig, [0.055, 0.660, 0.415, 0.230], press_rows, OPPONENT, f["ref"],
              header="How hard they press", left_colour=OPP_C, right_colour=TEAM_C)

    kinds = ["BUILD_UP", "BETWEEN_THE_LINES", "COUNTER_PRESS"]
    klab = ["In their build-up", "Between the lines", "Counter-press"]
    krows = [(klab[i], f"{f['press_kind'][kinds[i]][0]:.0f}",
             f"{f['press_kind'][kinds[i]][1]:.0f}") for i in range(3)]
    stat_table(fig, [0.525, 0.660, 0.415, 0.230], krows, OPPONENT, f["ref"],
              header="How they press", left_colour=OPP_C, right_colour=TEAM_C)
    fig.text(0.525, 0.638, f"Between the lines: {f['press_kind']['BETWEEN_THE_LINES'][0]:.0f} "
                           f"to {f['press_kind']['BETWEEN_THE_LINES'][1]:.0f}.",
             ha="left", va="center", color=MUTED, fontsize=7.4)

    # Every clause here used to be a fixed assertion ("they had less of it",
    # "A pressing counter-attacking side") true for Wigan's low-possession
    # style and never re-checked since - Barnsley actually had MORE
    # possession and territory than Orient across this sample, so the old
    # text was flatly backwards for them. Each comparison is now read off
    # the real numbers.
    poss_opp, poss_us = sc.get("RATIO_BALL_POSSESSION", (0.0, 0.0))
    tilt_opp, tilt_us = sc.get("FIELD_TILT", (0.0, 0.0))
    poss_more = poss_opp >= poss_us
    tilt_more = tilt_opp >= tilt_us
    bpd_more = f["bpd"] >= f["bpd_r"]

    style_rows = [
        ("Possession", *s2("RATIO_BALL_POSSESSION", 100, "{:.0f}%")),
        ("Field tilt", *s2("FIELD_TILT", 100, "{:.0f}%")),
        ("Bypassed defenders", f"{f['bpd']:.0f}", f"{f['bpd_r']:.0f}"),
    ]
    stat_table(fig, [0.055, 0.330, 0.415, 0.280], style_rows, OPPONENT, f["ref"],
              header="Do they dominate the ball?", left_colour=OPP_C, right_colour=TEAM_C)
    style_word = "A possession-dominant side" if poss_more else "A pressing counter-attacking side"
    bits = [
        "more of the ball" if poss_more else "less of the ball",
        "more territory" if tilt_more else "less territory",
        "more progression" if bpd_more else "less progression",
    ]
    fig.text(0.055, 0.308, f"{style_word}: {', '.join(bits)}.",
             ha="left", va="center", color=MUTED, fontsize=7.6)

    fig.text(0.525, 0.612, "Where they win the ball back", ha="left",
             va="center", color=INK, fontsize=14, weight="bold")
    ax = fig.add_axes([0.525, 0.150, 0.415, 0.440])
    draw_pitch(ax, lw=0.8)
    r = f.get("recov")
    if r is not None and len(r):
        ax.scatter(r.startAdjCoordinatesX, r.startAdjCoordinatesY, s=13,
                   color=OPP_C, alpha=.8, zorder=4, edgecolors="none")
    fig.text(0.7325, 0.128, f"attacking right · {len(r) if r is not None else 0} "
                            f"ball recoveries across their last "
                            f"{f['opp_match_count']} games", ha="center",
             va="center", color=MUTED, fontsize=7.2)

    fig.text(0.5, 0.045,
             f"Formation and pressing figures are means across {OPPONENT}'s "
             f"last {n_opp} matches ({f['opp_comp_mix']}).",
             ha="center", va="center", color=MUTED, fontsize=7.4)


def page_buildup(fig, f):
    # Page 6 feedback: chart 2 ("which part of our shape their turnovers
    # hurt") dropped as its own panel - its data (zd/top_zone) still feeds
    # the ALSO WATCH card below, just isn't charted on its own any more.
    # "Where they were opened up" moved in from page 5 to take that slot,
    # and everything channel/lane-based here is now a pitch view instead of
    # a bar chart, same as page 5.
    title(fig, "BUILD-UP MAP", "which channel to use against them, and why")
    fig.text(0.5, 0.900,
             "Charts use the BUILDING team's own lanes, so the first two "
             "read directly against each other.",
             ha="center", va="center", color=MUTED, fontsize=7.6)

    COLW, COLH = 0.265, 0.302
    COLX = [0.055, 0.368, 0.681]
    label_y, chart_top = 0.860, 0.815
    chart_bottom = chart_top - COLH
    caption_top = chart_bottom - 0.025

    # --- 1. how hard they press each channel
    tp = f["their_press_lane"]
    press = [tp[l][0] for l in LANES]
    fail = [tp[l][1] for l in LANES]
    worst = int(np.argmax(press))
    fig.text(COLX[0], label_y, "1 · How hard they press", ha="left",
             va="center", color=INK, fontsize=10.5, weight="bold")
    ax = fig.add_axes([COLX[0], chart_bottom, COLW, COLH])
    draw_lane_pitch(ax, press, BAD, LANE_LABEL, fmt="{:.0f}")
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
        press_detail = f"{(ratio - 1) * 100:.0f}% heavier than the rest"
    else:
        press_detail = "fairly even across the pitch"
    fig.text(COLX[0], caption_top,
             "\n".join(textwrap.wrap(
                 f"Heaviest press: {LANE_SHORT[worst].lower()}, "
                 f"{press_detail} ({fail[worst]:.0f}% won back there).", 34)),
             ha="left", va="top", color=MUTED, fontsize=7.2, linespacing=1.45)

    # --- 2. where they were opened up (moved in from THE CHANNELS)
    zd = dict(list(f["zone_damage"].items())[:7])   # still feeds ALSO WATCH below
    conceded_tail = ("close to an even split, not yet a clear pattern"
                     if f["conceded_close"] else "that is the side to attack")
    fig.text(COLX[1], label_y, "2 · Where they opened up", ha="left",
             va="center", color=INK, fontsize=10.5, weight="bold")
    ax = fig.add_axes([COLX[1], chart_bottom, COLW, COLH])
    draw_lane_pitch(ax, f["lane_conceded"], BAD, LANE_LABEL)
    fig.text(COLX[1], caption_top,
             "\n".join(textwrap.wrap(
                 f"Mirrored onto their shape. {f['conceded_share']:.0f}% came "
                 f"down THEIR {f['conceded_side']} - {conceded_tail}.", 34)),
             ha="left", va="top", color=MUTED, fontsize=7.2, linespacing=1.45)

    # --- 3. where we actually build well
    ob = f["our_build"]
    ofail = [ob[l][0] for l in LANES]
    on = [ob[l][2] for l in LANES]
    best = int(np.argmin([v if n >= 8 else 999 for v, n in zip(ofail, on)]))
    fig.text(COLX[2], label_y, f"3 · Where we build well", ha="left",
             va="center", color=INK, fontsize=10.5, weight="bold")
    ax = fig.add_axes([COLX[2], chart_bottom, COLW, COLH])
    draw_lane_pitch(ax, ofail, GOOD, LANE_LABEL)
    n = f.get("our_match_count", 1)
    span = f"our last {n} matches" if n >= 2 else "our most recent match"
    fig.text(COLX[2], caption_top,
             "\n".join(textwrap.wrap(
                 f"% of build-up passes lost by lane, {span} - lower is "
                 f"better. Best: {LANE_SHORT[best].lower()} "
                 f"({ofail[best]:.0f}% lost).", 34)),
             ha="left", va="top", color=MUTED, fontsize=7.2, linespacing=1.45)

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
        ax = fig.add_axes([x0, 0.032, 0.283, 0.375]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1,
                                    boxstyle="round,pad=0,rounding_size=0.02",
                                    fc=PANEL, ec=GRID, lw=0.9,
                                    mutation_aspect=0.18))
        ax.add_patch(Rectangle((0, 0.03), 0.014, 0.94, fc=colour, ec="none"))
        ax.text(0.075, 0.94, kicker, ha="left", va="top", color=colour,
                fontsize=8.6, weight="bold")
        ax.text(0.075, 0.83, head, ha="left", va="top", color=INK,
                fontsize=14.5, weight="bold")
        ax.text(0.075, 0.62, "\n".join(textwrap.wrap(body, 36)), ha="left",
                va="top", color="#B9C4D2", fontsize=8.2, linespacing=1.55)

    division_note = (f"{OPPONENT} play in a different competition to Orient"
                     if f["ref_diff_division"] else
                     f"both sides play in {OUR_COMPETITION} this season")
    fig.text(0.5, 0.012,
             f"{n} matches on our side, {f['opp_match_count']} on theirs, and "
             f"{division_note}. The agreement in the first two cards is more "
             f"reliable than either chart alone.",
             ha="center", va="center", color=MUTED, fontsize=6.8)


def page_channels(fig, f):
    # Page 5 feedback: "where they were opened up" moved to page 6 (it sits
    # better there next to the pressure-per-channel view it directly
    # answers); the two that stay here are drawn on a PORTRAIT pitch, side
    # by side, so "left wing" is genuinely the left column on the page -
    # a landscape pitch with lanes stacked top-to-bottom couldn't promise
    # that without a coordinate-sign guess; a portrait one with lanes as
    # vertical bands just reads directly off the label order.
    title(fig, "THE CHANNELS", "where they build, and where they progress")
    our_cover_side = "RIGHT" if f["prog_side"] == "LEFT" else "LEFT"
    prog_tail = (f"— close to an even split, not a clear lean" if f["prog_close"]
                else f"— our {our_cover_side.lower()} side must be ready")
    specs = [
        ("Where they build from", f["lane_build"], OPP_C, 0.055,
         "Their first-third passes. They start centrally."),
        ("Where they progress", f["lane_prog"], GOOD, 0.525,
         f"Their bypassed defenders. {f['prog_share']:.0f}% of it comes "
         f"down THEIR {f['prog_side']} {prog_tail}."),
    ]
    label_y, chart_w, chart_h = 0.870, 0.420, 0.480
    chart_top = label_y - 0.030
    chart_bottom = chart_top - chart_h
    for label, vals, colour, x0, note in specs:
        fig.text(x0, label_y, label, ha="left", va="center", color=INK,
                 fontsize=14.5, weight="bold")
        ax = fig.add_axes([x0, chart_bottom, chart_w, chart_h])
        draw_lane_pitch(ax, vals, colour, LANE_LABEL)
        fig.text(x0, chart_bottom - 0.030,
                 "\n".join(textwrap.wrap(note, 44)), ha="left", va="top",
                 color=MUTED, fontsize=8, linespacing=1.5)
    fig.text(0.5, 0.085,
             "Lanes are always from the attacking team's point of view.",
             ha="center", va="center", color=MUTED, fontsize=7.6)


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
                f"({bwrd:.1f} to {bwrd_r:.1f} a game) - this isn't where "
                f"the game gets lost.")
    return ("Their turnovers hurt", f"{bwrd:.1f}",
           f"defenders removed by their ball wins a game (ours {bwrd_r:.1f})", body)


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
    # Replaces the old card-grid layout with a written summary - three
    # paragraphs (style, strengths, weaknesses) instead of point cards. Most
    # of the underlying logic (close-split qualifiers, the turnover ratio
    # bands, the GK-press threshold, corner concentration) is unchanged from
    # the old card version; only how it's assembled into sentences is new.
    # The centre-forward read is the one branch that's genuinely new logic,
    # not just reworded - see the BPDR-per-90 comment below.
    #
    # Numbers on this page come from two genuinely different kinds of stat,
    # and any new sentence has to say which: most team-level figures (xg,
    # ppda, bwrd, fouls, xg_transition, ...) are a MEAN across the sample's
    # matches (see facts()'s own docstring) and need "a game"/"on average"
    # in the sentence, or a reader has no way to tell a per-match figure
    # from a season total (confirmed confusing in practice - Blackpool's
    # "0.45 xG" transition line read as a season total until this was
    # added). Player-level figures from `p` (BPD, loss, BPDR, ...) are the
    # opposite: a SUM across however many of the sample's matches that
    # player featured in, so those need "across this sample" instead - also
    # not safe to leave bare, just wrong in the other direction. Percentage
    # shares (xG%, progression splits, press%) don't need either - a rate
    # is already self-describing.
    title(fig, "THE PLAN", "how they set up, their threats, and where to hurt them")
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
    tilt = sc.get("FIELD_TILT", (0, 0))
    varied = " (though it varied)" if f["formation_varied"] else ""

    # ------------------------------------------------------------- style
    style_word = ("a possession-dominant side" if poss[0] >= poss[1]
                  else "a pressing counter-attacking side")
    territory = ("more territory to go with it" if tilt[0] >= tilt[1]
                else "without controlling territory to match")
    prog_lean = ("fairly evenly split between both flanks" if f["prog_close"]
                else f"{left_prog:.0f}% of it down their {prog_side} alone")
    style_para = (
        f"{OPPONENT} usually set up in a {f['formation']}{varied} and play as "
        f"{style_word}: {poss[0]*100:.0f}% of the ball across this sample to "
        f"our own {poss[1]*100:.0f}%, {territory}. Out of possession they "
        f"press at {ppda[0]:.1f} PPDA (ours {ppda[1]:.1f}) and {ph[0]:.0f}m "
        f"up the pitch - a genuinely active press, not a passive mid-block. "
        f"Their own progression is {prog_lean}, and {tr:.0f}% of it comes in "
        f"transition rather than settled build-up, which says as much about "
        f"how they hurt teams as anything they do in a settled shape."
    )

    # --------------------------------------------------------- strengths
    _, _, _, turnover_body = _turnover_card(f["bwrd"], f["bwrd_r"])
    flank_threat = (
        "it's close enough to an even split that we shouldn't over-shift for "
        "it - keep both flanks covered rather than loading up on one"
        if f["prog_close"] else
        f"their {prog_side} wing alone accounts for the biggest single "
        f"share, so our {cover_side}-sided pair will need real cover")
    strength_bits = [
        f"Transitions are their biggest weapon: {tr:.0f}% of their "
        f"progression and {f['xg_transition']:.2f} xG a game comes from "
        f"them alone, so careless turnovers in our own half are how this "
        f"game gets lost.",
        f"The press is real - {ppda[0]:.1f} PPDA against our own "
        f"{ppda[1]:.1f}, hunting between the lines - so our 8s will need "
        f"one-touch options rather than time on the ball.",
        f"When they progress through a settled shape, {flank_threat}.",
        f"Their ball-winning matters just as much as their press: "
        f"{turnover_body[0].lower()}{turnover_body[1:]}",
        f"{surname(bannan.playerName)} is their pivot, with {bannan.BPD:.1f} "
        f"bypassed defenders across this sample - the most on the pitch. "
        f"Cutting the passing line into him is the clearest individual job "
        f"in the game.",
    ]
    # PRESSURE_OPP_GK_PERCENT is how much THEY press an opposing keeper - only
    # worth mentioning when that's actually true. At 0% an unconditional line
    # here would say "they press the keeper" from a stat that says the
    # opposite.
    # gk[0]/poss[0] are THEIR season-wide averages against whoever they've
    # actually played so far (we're not in that sample yet - this fixture
    # hasn't happened) - phrasing it as "pressed our goalkeeper" read as
    # something already observed against us specifically, when it's really
    # their general tendency against opposing keepers, projected onto us.
    GK_PRESS_THRESHOLD = 0.08
    if gk[0] >= GK_PRESS_THRESHOLD:
        if poss[0] < 0.48:
            strength_bits.append(
                f"They had only {poss[0]*100:.0f}% of the ball across this "
                f"sample but still pressed the opposition's goalkeeper - "
                f"expect the same against us, with our goalkeeper and "
                f"centre-backs hurried regardless of who has it.")
        else:
            strength_bits.append(
                f"They press the opposition's goalkeeper even with "
                f"{poss[0]*100:.0f}% of the ball themselves - expect our "
                f"goalkeeper and centre-backs to be hurried regardless.")
    # Centre-forward profile: bypassed defenders received per 90, not aerial
    # duel win-rate - most centre-forwards in this league lose more aerials
    # than they win regardless of actual playing style (two centre-backs
    # against one striker), so a win-rate threshold fired on nearly every
    # opponent and said nothing distinctive. BPDR-as-receiver spreads far
    # more meaningfully across real players (measured across every cached
    # centre-forward this season: roughly 5 to 28 per 90, median ~15) and
    # actually separates "gets found in behind the back line" from "comes
    # short to feet" - the distinction this sentence is trying to make.
    if cf is not None and cf.mins > 0:
        cf_bpdr90 = cf.BPDR / cf.mins * 90
        if cf_bpdr90 >= 18:
            strength_bits.append(
                f"Their centre-forward {surname(cf.playerName)} averages "
                f"{cf_bpdr90:.1f} bypassed defenders per 90 as a receiver - "
                f"he prefers to run in behind the defence rather than drop "
                f"deep for it, so our centre-backs need to watch the line "
                f"rather than stepping up on him.")
    strength_para = " ".join(strength_bits)

    # -------------------------------------------------------- weaknesses
    weak_flank = p[p.position.str.startswith(f"{attack_side.upper()}_WING")]
    weak_flank_name = (surname(weak_flank.nlargest(1, "loss").playerName.iloc[0])
                       if len(weak_flank) else f"their {attack_side}-back")
    attack_qualifier = (
        " (close enough to an even split that it isn't a clear-cut weakness "
        "yet, but still worth testing early)" if f["conceded_close"] else "")
    _, _, _, force_long_body = _force_long_card(cp)
    top_loser = losers.iloc[0]
    loser_names = ", ".join(
        f"{surname(x.playerName)} ({x.position.replace('_', ' ').lower()})"
        for _, x in losers.iterrows())
    weak_bits = [
        f"Their clearest defensive gap is down their {attack_side}: "
        f"{right_open:.0f}% of the opposition's progression against them "
        f"came down that side{attack_qualifier}, and {weak_flank_name} is "
        f"among their costliest losers of the ball - "
        f"{'worth testing early' if f['conceded_close'] else 'overload it'}.",
        f"When they go long they're vulnerable too: "
        f"{force_long_body[0].lower()}{force_long_body[1:]}",
        f"Press their {_press_target(top_loser.position)} directly - "
        f"{surname(top_loser.playerName)} alone was responsible for "
        f"{top_loser.loss:.1f} teammates taken out of the game by his "
        f"losses across this sample. The three costliest overall were "
        f"{loser_names}.",
    ]
    if cf is not None and cf.mins > 0:
        cf_bpdr90 = cf.BPDR / cf.mins * 90
        if cf_bpdr90 <= 10:
            weak_bits.append(
                f"Their centre-forward {surname(cf.playerName)} averages "
                f"just {cf_bpdr90:.1f} bypassed defenders per 90 as a "
                f"receiver - he likes dropping deep to come to the ball "
                f"rather than running in behind, so our centre-backs can "
                f"squeeze up without much fear of being played through.")
    if f["corners"] == 0 or f["corner_xg"] <= 0:
        weak_bits.append(
            f"They offer nothing from dead balls ({f['corners']} corners, "
            f"no threat created), and their {f['fouls']:.1f} fouls a game "
            f"should still give us dead balls of our own.")
    elif f["corner_concentrated"]:
        weak_bits.append(
            f"Their corners aren't a real pattern - most of the "
            f"{f['corner_xg']:.2f} xG from {f['corners']} of them is one "
            f"costly phase, not something repeatable - and their "
            f"{f['fouls']:.1f} fouls a game should still give us dead balls "
            f"of our own.")
    else:
        weak_bits.append(
            f"Their {f['corner_xg']:.2f} xG from {f['corners']} corners is "
            f"a real if modest threat, spread across deliveries rather than "
            f"one moment - worth defending properly. Their {f['fouls']:.1f} "
            f"fouls a game should still give us dead balls of our own.")
    weak_para = " ".join(weak_bits)

    # ------------------------------------------------------------ layout
    sections = [
        ("STYLE OF PLAY", TEAM_C, style_para),
        ("STRENGTHS — WHAT TO BE READY FOR", BAD, strength_para),
        ("WEAKNESSES — WHERE TO ATTACK THEM", GOOD, weak_para),
    ]
    HEADER_STEP, SECTION_GAP, LINESPACING = 0.040, 0.045, 1.75
    wrapped = [(head, accent, textwrap.wrap(para, 100)) for head, accent, para in sections]
    total_lines = sum(len(lines) for _, _, lines in wrapped)

    # A wider sample (more games each side -> longer paragraphs) can need
    # more vertical room than three sections at a fixed 10pt leave for the
    # footer below - confirmed directly on an 11-a-side sample, where the
    # last paragraph's final line ran under the footer notes. The font
    # shrinks to fit the measured content instead of the footer silently
    # overlapping whatever paragraph happened to run long.
    TOP, FOOTER_RESERVE = 0.885, 0.095
    fixed_overhead = len(sections) * (HEADER_STEP + SECTION_GAP)
    budget = TOP - FOOTER_RESERVE - fixed_overhead
    FONTSIZE = 10.0
    per_line = line_pitch(fig, FONTSIZE, LINESPACING)
    needed = total_lines * per_line
    if needed > budget and total_lines:
        FONTSIZE = max(7.5, FONTSIZE * budget / needed)
        per_line = line_pitch(fig, FONTSIZE, LINESPACING)

    y = TOP
    for head, accent, lines in wrapped:
        ax = fig.add_axes([0.075, y - 0.006, 0.02, 0.022]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.add_patch(Rectangle((0, 0), 1, 1, fc=accent, ec="none"))
        fig.text(0.108, y, head, ha="left", va="center", color=INK,
                 fontsize=13, weight="bold")
        y -= HEADER_STEP
        fig.text(0.075, y, "\n".join(lines), ha="left", va="top",
                 color="#B9C4D2", fontsize=FONTSIZE, linespacing=LINESPACING)
        y -= len(lines) * per_line + SECTION_GAP

    # Anchored to wherever the three sections actually ended, not a fixed
    # position - the budget above keeps this from running off the page in
    # the normal case, but a dynamic anchor is what actually guarantees no
    # overlap regardless.
    n_opp, n_our = len(f["opp_matches"]), len(f["our_matches"])
    match_note = (f"{n_opp} matches for them, {n_our} for us" if n_opp != n_our
                 else f"{n_opp} matches each")
    division_qualifier = (", who play in a different competition to Orient"
                          if f["ref_diff_division"] else "")
    y -= 0.005
    fig.text(0.5, y,
             f"{match_note}{division_qualifier}. Treat the patterns as "
             f"reasonably reliable and the individual figures as indicative.",
             ha="center", va="top", color=MUTED, fontsize=7.6)
    y -= line_pitch(fig, 7.6, 1.0) + 0.012
    footer = (f"Built from {OPPONENT} ({f['opp_comp_mix']}) and Orient "
             f"({f['our_comp_mix']}). The channel read is the most "
             f"actionable part and the most sample-sensitive — worth "
             f"re-checking as more matches come in.")
    fig.text(0.5, y, "\n".join(textwrap.wrap(footer, 115)), ha="center",
             va="top", color=MUTED, fontsize=7.0, linespacing=1.6)


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
