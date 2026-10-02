"""Set-piece pre-match template, for the set-piece coach.

    python3 build_corners.py

Third version - broadened from corners-only to every attacking set piece
with a real delivery pattern: corners, wide free kicks (crosses/high balls
into the box - direct shots and pure restarts excluded, they're not a
delivery pattern), and throw-ins taken close enough to the box to be a
goal-threat rather than a possession restart. Free kicks and throw-ins get
a text breakdown (side split, delivery type, who takes it, who wins first
contact, what happens to the second ball) rather than the corner pages'
plotted box diagrams - Impect's end-zone taxonomy for those two is a
whole-pitch vocabulary (wings/half-spaces/thirds), not the tight box grid
corners have, so forcing it onto the same diagram would mean inventing a
zone mapping nobody has verified the way the corner grid's left/right
convention was verified against real corners (see _first_touch_points()'s
docstring on that bug). Worth building properly later if it turns out to
matter enough.

The old page 2 ("CORNERS — OURS": our own delivery/takers/first-contact)
was dropped per the set-piece coach's own call - this report exists to
brief them on the OPPONENT, and repeating our own routine back to us
wasn't earning its page. What stayed: "what we know about them defending
corners," now with a goals/xG headline and a second-ball note alongside
it.

Corner-page conventions carried over unchanged:
  - box diagrams are drawn goal-at-top
  - left and right are TRUE pitch sides throughout, not a "near post always
    on the left" schematic - a right corner's near post renders on the
    right, matching how the coach draws these on a whiteboard
  - "who delivers" lists every individual taker's share on each side, not
    just the single most frequent one

Built from each side's cached matches via build_prematch.py's own
OPP_CACHES/OUR_CACHES config - not fixed at two, not fixed to league matches.
Whatever sample that report was built with (mixed competitions, uneven counts
either side) is what this one inherits automatically.
"""

import os
import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

from report_config import (apply_style, BG, PANEL, INK, MUTED, GRID,
                           GOOD, BAD, BARC, PAGE_W, PAGE_H, surname, player_table)
from build_prematch import OPPONENT, OPP_CACHES, OPP_C, title
from envconfig import REPORTS_ROOT

os.makedirs(REPORTS_ROOT, exist_ok=True)
OUT = os.path.join(str(REPORTS_ROOT), f"Set Piece Report - {OPPONENT}.pdf")

# ------------------------------------------------------------- zone taxonomy
# Every value seen in Impect's setPieceSubPhaseCornerEndZone. NEAR/CENTRAL/
# FAR_POST x CLOSE/WIDE/BACK_OF_BOX makes a 3x3 grid; FLICK_ON and
# OPEN_PLAY(_ATTACK) sit outside it (a flick, or a short corner recycled into
# normal play, isn't a box location).
BOX_COLS = ["NEAR_POST", "CENTRAL", "FAR_POST"]
BOX_ROWS = ["CLOSE", "WIDE", "BACK_OF_BOX"]
ROW_LABEL = {"CLOSE": "close (6-yd)", "WIDE": "wide (penalty spot)",
            "BACK_OF_BOX": "edge of box"}


def _bucket(zone):
    if not isinstance(zone, str):
        return None, None
    if zone == "FLICK_ON":
        return "FLICK_ON", None
    if zone in ("OPEN_PLAY", "OPEN_PLAY_ATTACK"):
        return "OPEN_PLAY", None
    for col in BOX_COLS:
        if zone.startswith(col):
            row = zone[len(col) + 1:]
            if row in BOX_ROWS:
                return col, row
    return None, None


def corner_phases(setp, attacking=None, defending=None):
    """One row per physical corner, not per Impect subphase.

    A corner can span two subphases (the delivery, then a separate touch
    event) - collapsing to setPieceId keeps 'how many corners' meaningful and
    lets a first touch recorded on a later subphase inherit the delivery
    zone when it has none of its own, which is the common case.
    """
    cols = ["setPieceId", "side", "taker", "zone", "attacking", "defending",
            "first_touch_player", "first_touch_won", "first_touch_zone",
            "second_touch_player", "second_touch_won", "xg", "goals"]
    if setp is None or len(setp) == 0:
        return pd.DataFrame(columns=cols)
    d = setp[setp.setPieceCategory.isin(["CORNER_LEFT", "CORNER_RIGHT"])]
    if attacking:
        d = d[d.attackingSquadName == attacking]
    if defending:
        d = d[d.defendingSquadName == defending]
    rows = []
    for pid, g in d.groupby("setPieceId"):
        deliv = g[g.setPieceSubPhaseCornerType.notna()]
        dr = deliv.iloc[0] if len(deliv) else g.iloc[0]
        ft = g[g.setPieceSubPhaseFirstTouchPlayerName.notna()]
        if len(ft):
            fr = ft.iloc[0]
            ft_player = fr.setPieceSubPhaseFirstTouchPlayerName
            ft_won = fr.setPieceSubPhaseFirstTouchWon
            ft_zone = (fr.setPieceSubPhaseCornerEndZone
                      if pd.notna(fr.setPieceSubPhaseCornerEndZone)
                      else dr.setPieceSubPhaseCornerEndZone)
        else:
            ft_player, ft_won, ft_zone = None, None, None
        st = g[g.setPieceSubPhaseSecondTouchPlayerName.notna()]
        sr = st.iloc[0] if len(st) else None
        rows.append(dict(
            setPieceId=pid,
            side="LEFT" if dr.setPieceCategory == "CORNER_LEFT" else "RIGHT",
            taker=dr.setPieceSubPhaseMainEventPlayerName,
            zone=dr.setPieceSubPhaseCornerEndZone,
            attacking=dr.attackingSquadName, defending=dr.defendingSquadName,
            first_touch_player=ft_player, first_touch_won=ft_won,
            first_touch_zone=ft_zone,
            second_touch_player=(sr.setPieceSubPhaseSecondTouchPlayerName if sr is not None else None),
            second_touch_won=(sr.setPieceSubPhaseSecondTouchWon if sr is not None else None),
            xg=float(g.setPieceSubPhase_SHOT_XG.sum()),
            goals=int(g.setPieceSubPhase_GOALS.sum())))
    return pd.DataFrame(rows, columns=cols)


# Genuine box deliveries only - direct shots and pure restarts
# (INTO_POSSESSION/INTO_ATTACK/OTHER/VARIANT) aren't a delivery pattern the
# way a cross is, so they're excluded rather than counted as "a free kick
# with no delivery type."
FREEKICK_DELIVERY_TYPES = {
    "FREE_KICK_CROSS_NEAR_POST", "FREE_KICK_CROSS_FAR_POST",
    "FREE_KICK_CROSS_CENTRAL", "FREE_KICK_CROSS_BACK_OF_BOX",
    "FREE_KICK_HIGH_BALL", "FREE_KICK_HIGH_BALL_BOX",
}
# A throw taken from open midfield is a possession tool, not a goal threat -
# only ones started from beside or in front of the opponent's box are a
# set-piece delivery pattern worth scouting here.
THROWIN_NEAR_BOX_STARTS = ("WING_IN_FRONT_OF_BOX", "WING_BESIDES_BOX")


def _side_from_zone(zone: str):
    if not isinstance(zone, str):
        return None
    if "LEFT" in zone:
        return "LEFT"
    if "RIGHT" in zone:
        return "RIGHT"
    return None


def freekick_phases(setp, attacking=None, defending=None):
    """One row per wide free kick delivered into the box - side is read
    from where the kick was taken (there's no FREE_KICK_LEFT/RIGHT category
    the way corners have)."""
    cols = ["setPieceId", "side", "taker", "delivery_type", "start_zone",
            "attacking", "defending", "first_touch_player", "first_touch_won",
            "second_touch_player", "second_touch_won", "xg", "goals"]
    if setp is None or len(setp) == 0:
        return pd.DataFrame(columns=cols)
    d = setp[setp.setPieceCategory == "FREE_KICK"]
    if attacking:
        d = d[d.attackingSquadName == attacking]
    if defending:
        d = d[d.defendingSquadName == defending]
    rows = []
    for pid, g in d.groupby("setPieceId"):
        deliv = g[g.setPieceSubPhaseFreeKickType.isin(FREEKICK_DELIVERY_TYPES)]
        if len(deliv) == 0:
            continue
        dr = deliv.iloc[0]
        ft = g[g.setPieceSubPhaseFirstTouchPlayerName.notna()]
        fr = ft.iloc[0] if len(ft) else None
        st = g[g.setPieceSubPhaseSecondTouchPlayerName.notna()]
        sr = st.iloc[0] if len(st) else None
        rows.append(dict(
            setPieceId=pid, side=_side_from_zone(dr.setPieceSubPhaseStartZone),
            taker=dr.setPieceSubPhaseMainEventPlayerName,
            delivery_type=dr.setPieceSubPhaseFreeKickType,
            start_zone=dr.setPieceSubPhaseStartZone,
            attacking=dr.attackingSquadName, defending=dr.defendingSquadName,
            first_touch_player=(fr.setPieceSubPhaseFirstTouchPlayerName if fr is not None else None),
            first_touch_won=(fr.setPieceSubPhaseFirstTouchWon if fr is not None else None),
            second_touch_player=(sr.setPieceSubPhaseSecondTouchPlayerName if sr is not None else None),
            second_touch_won=(sr.setPieceSubPhaseSecondTouchWon if sr is not None else None),
            xg=float(g.setPieceSubPhase_SHOT_XG.sum()),
            goals=int(g.setPieceSubPhase_GOALS.sum())))
    return pd.DataFrame(rows, columns=cols)


def throwin_phases(setp, attacking=None, defending=None):
    """One row per throw-in taken close enough to the box to be a goal
    threat - side is read from the start zone, same as free kicks."""
    cols = ["setPieceId", "side", "taker", "delivery_type", "start_zone",
            "attacking", "defending", "first_touch_player", "first_touch_won",
            "second_touch_player", "second_touch_won", "xg", "goals"]
    if setp is None or len(setp) == 0:
        return pd.DataFrame(columns=cols)
    d = setp[setp.setPieceCategory == "THROW_IN"]
    if attacking:
        d = d[d.attackingSquadName == attacking]
    if defending:
        d = d[d.defendingSquadName == defending]
    rows = []
    for pid, g in d.groupby("setPieceId"):
        dr = g.iloc[0]
        start = dr.setPieceSubPhaseStartZone
        if not isinstance(start, str) or not any(k in start for k in THROWIN_NEAR_BOX_STARTS):
            continue
        ft = g[g.setPieceSubPhaseFirstTouchPlayerName.notna()]
        fr = ft.iloc[0] if len(ft) else None
        st = g[g.setPieceSubPhaseSecondTouchPlayerName.notna()]
        sr = st.iloc[0] if len(st) else None
        rows.append(dict(
            setPieceId=pid, side=_side_from_zone(start),
            taker=dr.setPieceSubPhaseMainEventPlayerName,
            delivery_type=dr.setPieceSubPhaseThrowInType, start_zone=start,
            attacking=dr.attackingSquadName, defending=dr.defendingSquadName,
            first_touch_player=(fr.setPieceSubPhaseFirstTouchPlayerName if fr is not None else None),
            first_touch_won=(fr.setPieceSubPhaseFirstTouchWon if fr is not None else None),
            second_touch_player=(sr.setPieceSubPhaseSecondTouchPlayerName if sr is not None else None),
            second_touch_won=(sr.setPieceSubPhaseSecondTouchWon if sr is not None else None),
            xg=float(g.setPieceSubPhase_SHOT_XG.sum()),
            goals=int(g.setPieceSubPhase_GOALS.sum())))
    return pd.DataFrame(rows, columns=cols)


# ------------------------------------------------------------------- loading
def _load_side(caches):
    """Combine setpieces, events and player matchsums across N cached
    matches - the same two-match aggregation build_prematch.py uses."""
    names = ("setpieces", "ev", "ps")
    out = {n: [] for n in names}
    for c in caches:
        c = c.strip()
        for n in names:
            p = os.path.join(c, f"{n}.parquet")
            if os.path.exists(p):
                out[n].append(pd.read_parquet(p))
    return {n: (pd.concat(fs, ignore_index=True) if fs else None)
            for n, fs in out.items()}


def _first_touch_points(ev, attacking_squad):
    """Real (x, y) location of each corner's first touch, converted to box
    coordinates (x=box width, y=depth with 0 at the goal line, negative
    going into the pitch - matching the goal-at-top diagrams below).

    Impect's coordinates are relative to whichever team the touching PLAYER
    belongs to, so a defender's touch has to be mirrored (negate both axes)
    to sit in the same attacking-team frame as the delivery itself.

    Separately - and this was the actual bug the set-piece coach caught -
    Impect's Y axis and this diagram's left/right are opposite conventions.
    Checked directly against setPieceSubPhaseCornerEndZone (Impect's own
    NEAR_POST/FAR_POST tag for the same touch): a raw Y of the same sign as
    the corner taker's own Y plots as FAR post here unless negated, for both
    CORNER_LEFT and CORNER_RIGHT alike - confirmed on four real corners
    (two near-post, two far-post) before applying the extra negation below.

    A short/recycled corner (zone OPEN_PLAY / OPEN_PLAY_ATTACK) also gets a
    'first touch player' from Impect - whoever received the ball out near
    the flag - which is a real touch but not a delivery into the box. Left
    in, it plots miles outside the box on this diagram (caught on a
    Springett short corner: he's tagged as the first touch simply for
    receiving the ball wide near the corner flag). zone_grid already
    excludes these from the delivery-zone breakdown; this needs the same
    exclusion so a short corner can't masquerade as an aerial contest.
    """
    cols = ["setPieceId", "side", "playerName", "x_box", "y_box", "won"]
    if ev is None or len(ev) == 0 or "setPieceId" not in ev.columns:
        return pd.DataFrame(columns=cols)
    d = ev[ev.setPieceCategory.isin(["CORNER_LEFT", "CORNER_RIGHT"])
           & (ev.attackingSquadName == attacking_squad)
           & ev.setPieceSubPhaseFirstTouchPlayerName.notna()
           & (ev.playerName == ev.setPieceSubPhaseFirstTouchPlayerName)
           & ~ev.setPieceSubPhaseCornerEndZone.isin(["OPEN_PLAY", "OPEN_PLAY_ATTACK"])]
    if len(d) == 0:
        return pd.DataFrame(columns=cols)
    d = d.sort_index().groupby("setPieceId", as_index=False).first()
    own = (d.squadName == attacking_squad).values
    ix = np.where(own, d.startAdjCoordinatesX.values, -d.startAdjCoordinatesX.values)
    iy = np.where(own, d.startAdjCoordinatesY.values, -d.startAdjCoordinatesY.values)
    side = np.where(d.setPieceCategory.values == "CORNER_LEFT", "LEFT", "RIGHT")
    return pd.DataFrame({"setPieceId": d.setPieceId.values, "side": side,
                          "playerName": d.playerName.values,
                          "x_box": -iy, "y_box": ix - 52.5,
                          "won": d.setPieceSubPhaseFirstTouchWon.values})


def _touch_points(ev, attacking_squad, categories):
    """Real (x, y) first-touch location for any set piece category - the
    same mirroring/sign transform as _first_touch_points() above (see its
    docstring; that's the one actually verified against real corners),
    generalized rather than duplicated with different math. Unlike the
    corner version this doesn't compute 'side' itself - free kicks and
    throw-ins don't carry it on the touch event the way CORNER_LEFT/
    CORNER_RIGHT does, so callers join the result onto their own phases
    frame (freekick_phases()/throwin_phases()) by setPieceId, which
    already derived each delivery's side correctly from its start zone.
    Also doesn't need the corner version's OPEN_PLAY exclusion - callers
    only ever pass this a `d` already filtered to genuine box deliveries,
    so a setPieceId that isn't a real delivery is never in the join."""
    cols = ["setPieceId", "playerName", "x_box", "y_box", "won"]
    if ev is None or len(ev) == 0 or "setPieceId" not in ev.columns:
        return pd.DataFrame(columns=cols)
    d = ev[ev.setPieceCategory.isin(categories)
           & (ev.attackingSquadName == attacking_squad)
           & ev.setPieceSubPhaseFirstTouchPlayerName.notna()
           & (ev.playerName == ev.setPieceSubPhaseFirstTouchPlayerName)]
    if len(d) == 0:
        return pd.DataFrame(columns=cols)
    d = d.sort_index().groupby("setPieceId", as_index=False).first()
    own = (d.squadName == attacking_squad).values
    ix = np.where(own, d.startAdjCoordinatesX.values, -d.startAdjCoordinatesX.values)
    iy = np.where(own, d.startAdjCoordinatesY.values, -d.startAdjCoordinatesY.values)
    return pd.DataFrame({"setPieceId": d.setPieceId.values,
                          "playerName": d.playerName.values,
                          "x_box": -iy, "y_box": ix - 52.5,
                          "won": d.setPieceSubPhaseFirstTouchWon.values})


def _within_diagram_frame(pts):
    """Drops touch points too far from the box to read as a meaningful dot
    on it. Corners never needed this - a corner's first touch is always
    right at the box - but a free kick or throw-in can legitimately be won
    (headed clear, contested) well out past the box. The diagram's own
    axis limits are sized generously for internal label/caption placement,
    not as a "this is near the box" cutoff, so filtering on those let a
    header won 10m past the box edge still render as a stray dot with a
    half-clipped label floating in open space. This uses a tighter, purely
    visual threshold instead - a few metres of margin around the box - and
    drops anything past it rather than clamping it to the edge, since a
    clamped dot would misreport a clearance well outside the box as a
    near-post contest."""
    if pts is None or len(pts) == 0:
        return pts
    xmax = BOX_HW + 3
    ymin = -BOX_DEPTH - 3
    return pts[(pts.x_box.abs() <= xmax) & (pts.y_box >= ymin)]


# ------------------------------------------------------------------ drawing
# Real box proportions (metres). Goal sits at y=0 at the TOP of every
# diagram; depth into the box runs negative (downward on the page) from
# there, to y=-BOX_DEPTH at the edge of the box.
BOX_DEPTH, BOX_HW = 16.5, 20.16
SIX_YD_DEPTH, SIX_YD_HW = 5.5, 9.16
GOAL_HW = 3.66
PEN_SPOT_Y = -11.0
COL_SPLIT = BOX_HW / 3


def _col_range(col, side):
    """x-range for a zone column, mirrored so a delivery lands on its true
    physical side - a right corner's near post is the right goalpost, not a
    fixed 'near post = left' convention."""
    left_to_right = {"NEAR_POST": (-BOX_HW, -COL_SPLIT),
                     "CENTRAL": (-COL_SPLIT, COL_SPLIT),
                     "FAR_POST": (COL_SPLIT, BOX_HW)}
    x0, x1 = left_to_right[col]
    return (x0, x1) if side == "LEFT" else (-x1, -x0)


def _row_range(row):
    return {"CLOSE": (0, -SIX_YD_DEPTH),
            "WIDE": (-SIX_YD_DEPTH, PEN_SPOT_Y),
            "BACK_OF_BOX": (PEN_SPOT_Y, -BOX_DEPTH)}[row]


def _true_side(col, side):
    """Which physical third of the box a (zone-column, corner-side) pair
    actually falls in - Left/Central/Right, not near/far post."""
    x0, x1 = _col_range(col, side)
    if x1 <= 1e-6:
        return "Left"
    if x0 >= -1e-6:
        return "Right"
    return "Central"


def draw_box_outline(ax, line_c):
    lw = 1.1
    ax.plot([-BOX_HW - 3, BOX_HW + 3], [0, 0], color=line_c, lw=lw, zorder=1)
    ax.add_patch(Rectangle((-BOX_HW, -BOX_DEPTH), 2 * BOX_HW, BOX_DEPTH,
                           fill=False, ec=line_c, lw=lw, zorder=1))
    ax.add_patch(Rectangle((-SIX_YD_HW, -SIX_YD_DEPTH), 2 * SIX_YD_HW, SIX_YD_DEPTH,
                           fill=False, ec=line_c, lw=lw, zorder=1))
    ax.add_patch(Rectangle((-GOAL_HW, 0), 2 * GOAL_HW, 1.4, fill=False,
                           ec=INK, lw=1.3, zorder=1))
    ax.plot([0], [PEN_SPOT_Y], marker=".", ms=3, color=line_c, zorder=1)
    ax.add_patch(Arc((0, PEN_SPOT_Y), 18.3, 18.3, angle=0, theta1=0,
                     theta2=180, ec=line_c, lw=lw, zorder=1))
    for cx in (-BOX_HW - 3, BOX_HW + 3):
        ax.add_patch(Arc((cx, 0), 4, 4, angle=0,
                         theta1=270 if cx > 0 else 180,
                         theta2=360 if cx > 0 else 270, ec=line_c, lw=lw, zorder=1))


_YLIM = (-BOX_DEPTH - 8.0, 6.0)
_LABEL_Y = -BOX_DEPTH - 2.0
_EXTRA_Y = -BOX_DEPTH - 4.0
_EXTRA_Y2 = -BOX_DEPTH - 5.8


def zone_grid(fig, rect, zones, n, colour, subtitle, side, flick_pts=None):
    ax = fig.add_axes(rect); ax.axis("off")
    ax.set_xlim(-BOX_HW - 5, BOX_HW + 5)
    ax.set_ylim(*_YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.text(0, 4.0, subtitle, ha="center", va="center", color=INK,
            fontsize=9.5, weight="bold")
    draw_box_outline(ax, "#4A5568")
    flag_x = -BOX_HW - 3 if side == "LEFT" else BOX_HW + 3
    ax.plot([flag_x], [0], marker="o", ms=4, color=colour, zorder=4)

    if n == 0:
        ax.text(0, -BOX_DEPTH / 2, "No corners recorded", ha="center",
                va="center", color=MUTED, fontsize=8.5)
        return
    # flicked-on deliveries have no static Impect zone (the ball moves on
    # arrival, so there's nothing to bucket into one of the 9 cells below) -
    # but the real first-touch coordinate is still in the event data, so
    # it's plotted directly rather than left as just a count in the footnote.
    if flick_pts is not None and len(flick_pts):
        ax.scatter(flick_pts.x_box, flick_pts.y_box, s=40, marker="D",
                   color=GOOD, edgecolors="#10161F", linewidths=0.7, zorder=5)
        for _, r3 in flick_pts.iterrows():
            ax.text(r3.x_box, r3.y_box - 1.5, surname(r3.playerName),
                    ha="center", va="top", fontsize=5.4, color=INK,
                    weight="bold", zorder=6)
    counts = zones.value_counts() if len(zones) else pd.Series(dtype=int)
    vmax = max([counts.get(f"{c}_{r}", 0) for c in BOX_COLS for r in BOX_ROWS] + [0])
    for c in BOX_COLS:
        x0, x1 = _col_range(c, side)
        for r in BOX_ROWS:
            y0, y1 = _row_range(r)
            ylo, yhi = min(y0, y1), max(y0, y1)
            v = counts.get(f"{c}_{r}", 0)
            alpha = 0.15 + 0.75 * (v / vmax) if vmax else 0.0
            ax.add_patch(Rectangle((x0, ylo), x1 - x0, yhi - ylo, fc=colour,
                                   alpha=alpha, ec=GRID, lw=0.7, zorder=2))
            if v:
                ax.text((x0 + x1) / 2, (ylo + yhi) / 2, f"{v/n*100:.0f}%\nn={v}",
                        ha="center", va="center", fontsize=7.4, color=INK,
                        weight="bold", linespacing=1.2, zorder=3)
    for x, lab in ((-1.5 * COL_SPLIT, "Left"), (0, "Central"),
                  (1.5 * COL_SPLIT, "Right")):
        ax.text(x, _LABEL_Y, lab, ha="center", va="top", color=MUTED,
                fontsize=6.8, weight="bold")

    fo = counts.get("FLICK_ON", 0)
    op = counts.get("OPEN_PLAY", 0) + counts.get("OPEN_PLAY_ATTACK", 0)
    # a residual, not an enumerated list of "every other zone name" - Impect
    # has zone values beyond the 3x3 grid + flick-on + short/recycled (e.g.
    # OPPOSITE_SIDE_1, first seen in Barnsley's data) that would otherwise be
    # silently uncounted here even though they're still in n, making the
    # displayed breakdown add up to less than the total the diagram claims.
    grid_total = sum(counts.get(f"{c}_{r}", 0) for c in BOX_COLS for r in BOX_ROWS)
    other = n - grid_total - fo - op

    # FLICK_ON means the ball WAS delivered into the box and won first-time -
    # Impect just can't give it a static landing zone since it's redirected
    # on arrival, not a "no delivery happened" outcome. Grouping it with
    # short/recycled corners (which are a genuinely different corner TYPE -
    # CORNER_OPEN_PLAY, passed short to a teammate outside the box, no
    # delivery at all) implied they were the same kind of exception. They
    # aren't, so they're shown on separate lines with different framing.
    if fo:
        # not every flick-on corner has a first-touch player recorded in
        # Impect's own feed (a genuine gap in their data, not something this
        # code can recover) - only claim a location is plotted when one
        # actually is, rather than overstate what's on the diagram.
        n_shown = len(flick_pts) if flick_pts is not None else 0
        located = (" — real location shown" if n_shown >= fo else
                  f" ({n_shown} of {fo} located)" if n_shown else
                  " — no location recorded")
        ax.text(0, _EXTRA_Y, f"♦ {fo/n*100:.0f}% flicked on — delivered, won "
                             f"first time{located} (n={fo})",
                ha="center", va="top", color=colour, fontsize=6.6, weight="bold")
    not_delivered = []
    if op:
        not_delivered.append(f"{op/n*100:.0f}% short/recycled (n={op})")
    if other > 0:
        not_delivered.append(f"{other/n*100:.0f}% other (n={other})")
    if not_delivered:
        ax.text(0, _EXTRA_Y2, "No delivery into the box: "
                             + "  ·  ".join(not_delivered), ha="center",
                va="top", color=MUTED, fontsize=6.6)


_MINI_YLIM = (-BOX_DEPTH - 2.0, 5.0)


def _mini_first_contact(fig, rect, points, player_label, colour):
    """One player's own first-contact points on their own small box diagram -
    both corner sides combined, since x_box/y_box are already in a physical
    left/right frame that doesn't depend on which side the corner came from.
    A flag dot marks each side this player actually has a point from, same
    marker/placement as the other box diagrams on this page.
    """
    ax = fig.add_axes(rect); ax.axis("off")
    ax.set_xlim(-BOX_HW - 5, BOX_HW + 5)
    ax.set_ylim(*_MINI_YLIM)
    ax.set_aspect("equal", adjustable="box")
    draw_box_outline(ax, "#4A5568")
    sides = points.side.unique()
    # a player with contacts from only one side is unambiguous as-is (single
    # flag, section colour). Only once a diagram mixes both sides - Johnston
    # is the one case in this sample - is there anything for a second colour
    # to disambiguate, so the left/right split only kicks in there.
    split = len(sides) > 1
    side_colour = {"LEFT": BAD, "RIGHT": BARC}
    for s in sides:
        flag_x = -BOX_HW - 3 if s == "LEFT" else BOX_HW + 3
        fc = side_colour[s] if split else colour
        ax.plot([flag_x], [0], marker="o", ms=4, color=fc, zorder=4)
    won = points[points.won == True]
    lost = points[points.won != True]
    if len(lost):
        ax.scatter(lost.x_box, lost.y_box, s=32, color=BAD,
                   edgecolors="#10161F", linewidths=0.6, zorder=5)
    if len(won):
        if split:
            wc = won.side.map(side_colour)
            ax.scatter(won.x_box, won.y_box, s=32, color=wc,
                       edgecolors="#10161F", linewidths=0.6, zorder=5)
        else:
            ax.scatter(won.x_box, won.y_box, s=32, color=GOOD,
                       edgecolors="#10161F", linewidths=0.6, zorder=5)
    record = f"{len(won)}W {len(lost)}L" if len(won) or len(lost) else ""
    ax.text(0, 4.3, f"{surname(player_label)}  {record}", ha="center",
            va="center", color=INK, fontsize=7.6, weight="bold")


def player_first_contact_grid(fig, rect, cp, pts, colour, max_cols=4, label="corners"):
    """A separate first-contact diagram per individual player, rather than
    every player's points overlaid on one shared per-side diagram - request
    from the set-piece coach, so an aerial threat's own tendency reads on
    its own picture instead of being lost among everyone else's dots.
    Generic over any phases frame with the same .side/.playerName/.won/
    .x_box/.y_box shape - corners, wide free kicks, near-box throw-ins."""
    x0, y0, w, h = rect
    n_total = len(cp)
    if pts is None or len(pts) == 0:
        ax = fig.add_axes(rect); ax.axis("off")
        ax.text(0.5, 0.5, "No first-contact data recorded", ha="center",
                va="center", color=MUTED, fontsize=8.5)
        return
    order = pts.playerName.value_counts().index.tolist()
    n = len(order)
    ncols = min(n, max_cols)
    nrows = -(-n // ncols)
    cell_w, cell_h = w / ncols, h / nrows
    pad = 0.06
    for i, player in enumerate(order):
        r, c = divmod(i, ncols)
        sub = [x0 + c * cell_w + cell_w * pad / 2,
               y0 + h - (r + 1) * cell_h + cell_h * pad / 2,
               cell_w * (1 - pad), cell_h * (1 - pad)]
        _mini_first_contact(fig, sub, pts[pts.playerName == player], player, colour)
    fig.text(x0 + w / 2, y0 - 0.022,
             f"{len(pts)} of {n_total} {label} have a recorded "
             f"first-contact location, across {n} player{'s' if n != 1 else ''}",
             ha="center", va="top", color=MUTED, fontsize=6.8)


def touch_diagram(fig, rect, points, subtitle, n_total, side, colour):
    """Real touch-location box diagram for one side - the free-kick/
    throw-in equivalent of the corner pages' first-contact plotting (the
    function this was originally built as, before it briefly went unused
    when those pages were text-only). Dot position is where the ball was
    first touched - effectively where the delivery arrived, since there's
    no separately verified 'delivery end zone' grid for these two
    categories the way corners have (see module docstring) - colour is
    won/lost, so this single diagram covers both delivery location and
    first contact at once."""
    ax = fig.add_axes(rect); ax.axis("off")
    ax.set_xlim(-BOX_HW - 5, BOX_HW + 5)
    ax.set_ylim(*_YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.text(0, 4.0, subtitle, ha="center", va="center", color=INK,
            fontsize=9.5, weight="bold")
    draw_box_outline(ax, "#4A5568")
    flag_x = -BOX_HW - 3 if side == "LEFT" else BOX_HW + 3
    ax.plot([flag_x], [0], marker="o", ms=4, color=colour, zorder=4)

    if points is None or len(points) == 0:
        ax.text(0, -BOX_DEPTH / 2, "No location data recorded",
                ha="center", va="center", color=MUTED, fontsize=8.5)
        return
    won = points[points.won == True]
    lost = points[points.won != True]
    if len(lost):
        ax.scatter(lost.x_box, lost.y_box, s=36, color=BAD,
                   edgecolors="#10161F", linewidths=0.7, zorder=5, label="lost")
    if len(won):
        ax.scatter(won.x_box, won.y_box, s=36, color=GOOD,
                   edgecolors="#10161F", linewidths=0.7, zorder=5, label="won")
    for _, r2 in points.iterrows():
        # clip_on=True: matplotlib text isn't clipped to its axes by
        # default, so a point right at the edge of the (already-filtered)
        # plotted range can still have its LABEL spill past the box into
        # neighbouring page real estate otherwise.
        ax.text(r2.x_box, r2.y_box - 1.6, surname(r2.playerName), ha="center",
                va="top", fontsize=5.6, color=INK, weight="bold", zorder=6,
                clip_on=True)
    ax.legend(loc="upper right", frameon=False, fontsize=6.6, labelcolor=INK,
              bbox_to_anchor=(0.99, 0.97), handletextpad=0.3)
    ax.text(0, _LABEL_Y, f"{len(points)} of {n_total} have a recorded "
                        f"location", ha="center", va="top",
            color=MUTED, fontsize=6.8)


def flick_on_points(cp, pts, side):
    """Real first-touch coordinates for this side's flicked-on corners only -
    the one delivery outcome with no static Impect zone to plot in the grid."""
    ids = cp[(cp.side == side) & (cp.zone == "FLICK_ON")].setPieceId
    if len(pts) == 0 or len(ids) == 0:
        return pts.iloc[0:0]
    return pts[pts.setPieceId.isin(ids)]


def caveat(fig, text, y=0.045):
    fig.text(0.5, y, text, ha="center", va="center", color=MUTED, fontsize=7.4)


def taker_rows(d):
    """Player rows for a 'who delivers' player_table() - every individual
    taker's share on each side, as table rows instead of wrapped text.
    Works on any phases frame with .side/.taker columns (corner_phases(),
    freekick_phases(), throwin_phases() all have them)."""
    rows = []
    for side in ("LEFT", "RIGHT"):
        dd = d[d.side == side]
        n = len(dd)
        if n == 0:
            continue
        for name, cnt in dd.taker.value_counts().items():
            rows.append((surname(name), side.title(), int(cnt), f"{cnt/n*100:.0f}%"))
    return rows




def headline_stat(fig, x, y, d, label, colour):
    """Goals/xG this sample - a single top-line number so the coach sees
    how big a threat this phase actually is before reading the detail."""
    goals = int(d.goals.sum()) if len(d) else 0
    xg = float(d.xg.sum()) if len(d) else 0.0
    fig.text(x, y, f"{goals} goal{'s' if goals != 1 else ''} · {xg:.2f} xG "
                   f"from {len(d)} {label} this sample",
             ha="left", va="center", color=colour, fontsize=9.5, weight="bold")


# --------------------------------------------------------------------- pages
def page_opposition(fig, cp, pts, n_matches):
    title(fig, "CORNERS — OPPOSITION",
         f"{OPPONENT} attacking, last {n_matches} matches")
    n = len(cp)

    fig.text(0.055, 0.895, "Where they deliver", ha="left", va="center",
             color=INK, fontsize=11.5, weight="bold")
    zone_grid(fig, [0.06, 0.630, 0.40, 0.24], cp[cp.side == "LEFT"].zone,
             len(cp[cp.side == "LEFT"]), OPP_C, "Corners from the left", "LEFT",
             flick_on_points(cp, pts, "LEFT"))
    zone_grid(fig, [0.54, 0.630, 0.40, 0.24], cp[cp.side == "RIGHT"].zone,
             len(cp[cp.side == "RIGHT"]), OPP_C, "Corners from the right", "RIGHT",
             flick_on_points(cp, pts, "RIGHT"))

    fig.text(0.055, 0.575, "Who delivers", ha="left", va="center", color=INK,
             fontsize=11.5, weight="bold")
    player_table(fig, [0.075, 0.400, 0.85, 0.150], taker_rows(cp),
                ["Player", "Side", "N", "Share"])

    fig.text(0.055, 0.370, "First contact — by player", ha="left", va="center",
             color=INK, fontsize=11.5, weight="bold")
    player_first_contact_grid(fig, [0.055, 0.130, 0.885, 0.22], cp, pts, OPP_C)

    caveat(fig, f"n={n} corners from {n_matches} matches. Enough to spot "
                f"a preference, not yet a settled pattern.", y=0.070)


def _defensive_header_winners(ev, corner_ids, squad):
    """Genuine defensive headers at a defended corner, by player.

    setPieceSubPhaseFirstTouchPlayerName only ever names the ATTACKING side
    - checked across every corner cached anywhere in this project (109 of
    109 recorded first touches belonged to the attacker, never the
    defence), so filtering that field for the defending team can never
    return a result regardless of sample size. A defender's touch is
    tagged as a plain action == 'HEADER' event instead, just never linked
    into that subphase field, so this counts those directly.
    """
    if ev is None or len(ev) == 0 or "setPieceId" not in ev.columns or len(corner_ids) == 0:
        return pd.Series(dtype=int)
    d = ev[ev.setPieceId.isin(corner_ids) & (ev.squadName == squad)
           & (ev.action == "HEADER")]
    return d.playerName.value_counts()


def page_corner_defence(fig, opp_def_cp, opp_ev, n_opp):
    """Everything the old 'CORNERS — OURS' page had beyond our own
    delivery/takers/first-contact, which the set-piece coach dropped - this
    report scouts the opponent, and our own routine wasn't earning its page
    back. What's here now: how big a threat their conceded corners already
    are (goals/xG), then the same defensive scouting cards as before, plus
    a second-ball note using the freed space."""
    title(fig, "CORNERS — HOW THEY DEFEND",
         f"{OPPONENT} defending, last {n_opp} matches")
    nd = len(opp_def_cp)

    headline_stat(fig, 0.055, 0.885, opp_def_cp, "corners conceded", "#FF2D2D")

    fig.text(0.055, 0.790, f"What we know about {OPPONENT} defending corners",
             ha="left", va="center", color=INK, fontsize=12.5, weight="bold")
    header_wins = _defensive_header_winners(
        opp_ev, opp_def_cp.setPieceId.unique() if nd else [], OPPONENT)
    if len(header_wins):
        n_top = int(header_wins.iloc[0])
        # value_counts() only orders ties by internal sort stability, not
        # merit - naming just the first one implied a lone standout when
        # Wycombe's Taylor and Grimmer were genuinely tied at 4 each.
        tied = [surname(p) for p in header_wins[header_wins == n_top].index]
        header_txt = (f"{', '.join(tied)} — {n_top} defensive header"
                     f"{'s' if n_top != 1 else ''} won each across their "
                     f"{nd} corners defended") if len(tied) > 1 else (
                     f"{tied[0]} — {n_top} defensive header"
                     f"{'s' if n_top != 1 else ''} won across their "
                     f"{nd} corners defended")
    else:
        header_txt = f"No defensive headers recorded in {nd} corners defended"

    if nd:
        # true physical side/row, computed per-row from that corner's own
        # side - combining LEFT and RIGHT corners under a raw zone name like
        # NEAR_POST would silently mix opposite sides of the box.
        labels = []
        for _, r in opp_def_cp.iterrows():
            col, row = _bucket(r.zone)
            if col in BOX_COLS and row in BOX_ROWS:
                labels.append(f"{_true_side(col, r.side)} / {ROW_LABEL[row]}")
        if labels:
            vc = pd.Series(labels).value_counts()
            n_top = int(vc.iloc[0])
            # ties broken only by value_counts()'s internal sort order would
            # silently name one zone "the" vulnerable area when another was
            # conceded from equally often - list every zone tied at the top.
            tied = [z.capitalize() for z in vc[vc == n_top].index]
            vuln_txt = (f"{' and '.join(tied)} — {n_top}/{nd} of the "
                       f"corners they conceded targeted this area"
                       f"{' each' if len(tied) > 1 else ''}")
        else:
            vuln_txt = f"No clear zone pattern in the {nd} corners they conceded"
    else:
        vuln_txt = f"No corners conceded in their last {n_opp} matches"

    st = opp_def_cp[opp_def_cp.second_touch_player.notna()] if nd else opp_def_cp
    if len(st):
        won_pct = float((st.second_touch_won == True).mean()) * 100
        top = st.second_touch_player.value_counts()
        second_txt = (f"Recorded on {len(st)} of their conceded corners — "
                      f"won back {won_pct:.0f}% of the time, most often by "
                      f"{surname(top.index[0])} ({int(top.iloc[0])})")
    else:
        second_txt = "No second-ball contests recorded"

    ax = fig.add_axes([0.055, 0.480, 0.885, 0.310]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    w = 0.315
    # matplotlib's own wrap=True wraps to the whole figure width, not the
    # axes/panel it's drawn in - with three narrower panels (this used to
    # be two) that bled text across the panel boundary, so this wraps by
    # hand at a character width sized to the panel instead, the same way
    # every other multi-line block on these pages already does.
    for i, (kicker, body, colour) in enumerate([
            ("STRONGEST HEADER", header_txt, "#FF2D2D"),
            ("MOST VULNERABLE AREA", vuln_txt, "#8FE388"),
            ("SECOND BALL", second_txt, BARC)]):
        x0 = i * (w + 0.02)
        ax.add_patch(Rectangle((x0, 0.02), w, 0.96, fc=PANEL, ec=GRID, lw=0.9))
        ax.text(x0 + 0.035, 0.86, kicker, ha="left", va="top", color=colour,
                fontsize=9, weight="bold")
        ax.text(x0 + 0.035, 0.68, "\n".join(textwrap.wrap(body, 28)),
                ha="left", va="top", color=INK, fontsize=9, linespacing=1.4)

    # The "MOST VULNERABLE AREA" card above names a zone in words - this is
    # the same box-grid diagram CORNERS — OPPOSITION uses for "where they
    # deliver", pointed at the corners THEY conceded instead, so the named
    # zone reads as a picture too, not just a sentence.
    fig.text(0.055, 0.440, "Where it's targeted", ha="left", va="center",
             color=INK, fontsize=12.5, weight="bold")
    zone_grid(fig, [0.06, 0.190, 0.40, 0.22], opp_def_cp[opp_def_cp.side == "LEFT"].zone,
             len(opp_def_cp[opp_def_cp.side == "LEFT"]), "#8FE388",
             "Conceded from the left", "LEFT")
    zone_grid(fig, [0.54, 0.190, 0.40, 0.22], opp_def_cp[opp_def_cp.side == "RIGHT"].zone,
             len(opp_def_cp[opp_def_cp.side == "RIGHT"]), "#8FE388",
             "Conceded from the right", "RIGHT")

    caveat(fig, f"Their defensive corners: {nd} from {n_opp} matches.", y=0.055)


def _setpiece_opposition_body(fig, d, pts, colour, label, n_matches):
    """Shared layout for the wide-free-kick and near-box-throw-in pages -
    same visual language as CORNERS — OPPOSITION: real touch-location box
    diagrams split by side for 'where they deliver' (dots, not zone-grid
    percentages - see module docstring on why these two categories don't
    get the corner grid), a player_table() for who takes it, and the same
    per-player first-contact grid corners uses. `pts` is _touch_points()
    already joined onto `d` for side."""
    headline_stat(fig, 0.055, 0.900, d, label, colour)

    fig.text(0.055, 0.855, "Where they deliver", ha="left", va="center",
             color=INK, fontsize=12, weight="bold")
    touch_diagram(fig, [0.06, 0.600, 0.40, 0.24], pts[pts.side == "LEFT"],
                 "From the left", len(d[d.side == "LEFT"]), "LEFT", colour)
    touch_diagram(fig, [0.54, 0.600, 0.40, 0.24], pts[pts.side == "RIGHT"],
                 "From the right", len(d[d.side == "RIGHT"]), "RIGHT", colour)

    fig.text(0.055, 0.545, "Who delivers", ha="left", va="center", color=INK,
             fontsize=12, weight="bold")
    player_table(fig, [0.075, 0.370, 0.85, 0.150], taker_rows(d),
                ["Player", "Side", "N", "Share"])

    fig.text(0.055, 0.340, "First contact — by player", ha="left",
             va="center", color=INK, fontsize=12, weight="bold")
    player_first_contact_grid(fig, [0.055, 0.130, 0.885, 0.185], d, pts,
                              colour, label=label)

    second = pts_second_ball_line(d)
    fig.text(0.5, 0.095, second, ha="center", va="center", color=MUTED,
             fontsize=7.2)

    caveat(fig, f"n={len(d)} {label} from {n_matches} matches. Enough to "
                f"spot a preference, not yet a settled pattern.", y=0.055)


def pts_second_ball_line(d):
    """One-line second-ball summary, for when there's no room left on the
    page for a full section (see _setpiece_opposition_body)."""
    st = d[d.second_touch_player.notna()]
    if len(st) == 0:
        return "No second-ball contests recorded."
    won_pct = float((st.second_touch_won == True).mean()) * 100
    top = st.second_touch_player.value_counts()
    return (f"Second ball: recorded on {len(st)}, won back {won_pct:.0f}% "
           f"of the time — most involved: {surname(top.index[0])} "
           f"({int(top.iloc[0])}).")


def page_freekicks(fig, d, pts, n_matches):
    title(fig, "WIDE FREE KICKS — OPPOSITION",
         f"{OPPONENT} attacking, last {n_matches} matches")
    _setpiece_opposition_body(fig, d, pts, OPP_C, "wide free kicks", n_matches)


def page_throwins(fig, d, pts, n_matches):
    title(fig, "THROW-INS (NEAR THE BOX) — OPPOSITION",
         f"{OPPONENT} attacking, last {n_matches} matches")
    _setpiece_opposition_body(fig, d, pts, OPP_C, "near-box throw-ins", n_matches)


def main():
    apply_style()
    opp = _load_side(OPP_CACHES)

    n_opp_matches = opp["ev"].matchId.nunique() if opp["ev"] is not None else 0

    opp_cp = corner_phases(opp["setpieces"], attacking=OPPONENT)
    opp_def_cp = corner_phases(opp["setpieces"], defending=OPPONENT)
    opp_pts = _first_touch_points(opp["ev"], OPPONENT)

    opp_fk = freekick_phases(opp["setpieces"], attacking=OPPONENT)
    opp_ti = throwin_phases(opp["setpieces"], attacking=OPPONENT)
    opp_fk_pts = _within_diagram_frame(_touch_points(opp["ev"], OPPONENT, ["FREE_KICK"]).merge(
        opp_fk[["setPieceId", "side"]], on="setPieceId", how="inner"))
    opp_ti_pts = _within_diagram_frame(_touch_points(opp["ev"], OPPONENT, ["THROW_IN"]).merge(
        opp_ti[["setPieceId", "side"]], on="setPieceId", how="inner"))

    with PdfPages(OUT) as pdf:
        fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor=BG)
        page_opposition(fig, opp_cp, opp_pts, n_opp_matches)
        pdf.savefig(fig, facecolor=BG); plt.close(fig)

        fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor=BG)
        page_corner_defence(fig, opp_def_cp, opp["ev"], n_opp_matches)
        pdf.savefig(fig, facecolor=BG); plt.close(fig)

        fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor=BG)
        page_freekicks(fig, opp_fk, opp_fk_pts, n_opp_matches)
        pdf.savefig(fig, facecolor=BG); plt.close(fig)

        fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor=BG)
        page_throwins(fig, opp_ti, opp_ti_pts, n_opp_matches)
        pdf.savefig(fig, facecolor=BG); plt.close(fig)

    print(f"wrote {OUT}  (4 pages)")


if __name__ == "__main__":
    main()
