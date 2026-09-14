"""Corner-routine pre-match template, for the set-piece coach.

    python3 build_corners.py

Second version, revised per the set-piece coach's own review of the first
draft:
  - box diagrams are drawn goal-at-top (the first draft had the goal at the
    bottom, which nobody actually reads a corner diagram as)
  - left and right are TRUE pitch sides throughout, not a "near post always
    on the left" schematic - a right corner's near post renders on the
    right, matching how the coach draws these on a whiteboard
  - "who delivers" lists every individual taker's share on each side, not
    just the single most frequent one
  - "first contact" is a box diagram with real individual contact points
    plotted from event coordinates (labelled, coloured by won/lost),
    not a bar chart of zone percentages

Also now built from each side's cached matches via build_prematch.py's own
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

from report_config import (apply_style, BG, PANEL, INK, MUTED, GRID, TEAM_C,
                           GOOD, BAD, BARC, PAGE_W, PAGE_H, surname)
from build_prematch import OPPONENT, OPP_CACHES, OUR_CACHES, US, OPP_C, title
from envconfig import REPORTS_ROOT

os.makedirs(REPORTS_ROOT, exist_ok=True)
OUT = os.path.join(str(REPORTS_ROOT), f"Corner Report - {OPPONENT}.pdf")

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
            "first_touch_player", "first_touch_won", "first_touch_zone", "xg"]
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
        rows.append(dict(
            setPieceId=pid,
            side="LEFT" if dr.setPieceCategory == "CORNER_LEFT" else "RIGHT",
            taker=dr.setPieceSubPhaseMainEventPlayerName,
            zone=dr.setPieceSubPhaseCornerEndZone,
            attacking=dr.attackingSquadName, defending=dr.defendingSquadName,
            first_touch_player=ft_player, first_touch_won=ft_won,
            first_touch_zone=ft_zone, xg=float(g.setPieceSubPhase_SHOT_XG.sum())))
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


def first_contact_diagram(fig, rect, points, subtitle, n_total, side, colour):
    """Individual first-contact locations for corners from one side, plotted
    for real rather than binned into a zone percentage - keeps left and
    right apart the same way the delivery diagrams do, and avoids the
    left/right conflation a combined-sides zone grid would have (a 'near
    post' cell means opposite physical sides depending which corner it
    came from)."""
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
        ax.text(0, -BOX_DEPTH / 2, "No first-contact data recorded",
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
        ax.text(r2.x_box, r2.y_box - 1.6, surname(r2.playerName), ha="center",
                va="top", fontsize=5.6, color=INK, weight="bold", zorder=6)
    ax.legend(loc="upper right", frameon=False, fontsize=6.6, labelcolor=INK,
              bbox_to_anchor=(0.99, 0.97), handletextpad=0.3)
    ax.text(0, _LABEL_Y, f"{len(points)} of {n_total} corners have a recorded "
                        f"first-contact location", ha="center", va="top",
            color=MUTED, fontsize=6.8)


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


def player_first_contact_grid(fig, rect, cp, pts, colour, max_cols=4):
    """A separate first-contact diagram per individual player, rather than
    every player's points overlaid on one shared per-side diagram - request
    from the set-piece coach, so an aerial threat's own tendency reads on
    its own picture instead of being lost among everyone else's dots."""
    x0, y0, w, h = rect
    n_corners = len(cp)
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
             f"{len(pts)} of {n_corners} corners have a recorded "
             f"first-contact location, across {n} player{'s' if n != 1 else ''}",
             ha="center", va="top", color=MUTED, fontsize=6.8)


def taker_breakdown(fig, x, y, cp, side, width=100):
    """Every individual taker's share on this side, not just the top one."""
    d = cp[cp.side == side]
    n = len(d)
    if n == 0:
        fig.text(x, y, f"{side.title()}: no corners recorded", ha="left",
                 va="top", color=INK, fontsize=9)
        return
    vc = d.taker.value_counts()
    parts = ", ".join(f"{surname(name)} {cnt/n*100:.0f}% ({cnt})"
                      for name, cnt in vc.items())
    text = f"{side.title()} (n={n}): {parts}"
    fig.text(x, y, "\n".join(textwrap.wrap(text, width)), ha="left", va="top",
             color=INK, fontsize=8.6, linespacing=1.5)


def flick_on_points(cp, pts, side):
    """Real first-touch coordinates for this side's flicked-on corners only -
    the one delivery outcome with no static Impect zone to plot in the grid."""
    ids = cp[(cp.side == side) & (cp.zone == "FLICK_ON")].setPieceId
    if len(pts) == 0 or len(ids) == 0:
        return pts.iloc[0:0]
    return pts[pts.setPieceId.isin(ids)]


def caveat(fig, text, y=0.045):
    fig.text(0.5, y, text, ha="center", va="center", color=MUTED, fontsize=7.4)


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
    taker_breakdown(fig, 0.075, 0.548, cp, "LEFT")
    taker_breakdown(fig, 0.075, 0.508, cp, "RIGHT")

    fig.text(0.055, 0.450, "First contact — by player", ha="left", va="center",
             color=INK, fontsize=11.5, weight="bold")
    player_first_contact_grid(fig, [0.055, 0.190, 0.885, 0.24], cp, pts, OPP_C)

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


def page_ours(fig, our_cp, our_pts, opp_def_cp, opp_ev, n_matches, n_opp):
    title(fig, "CORNERS — OURS", f"{US} attacking, last {n_matches} matches")
    n = len(our_cp)

    fig.text(0.055, 0.910, "Where we deliver", ha="left", va="center",
             color=INK, fontsize=11.5, weight="bold")
    zone_grid(fig, [0.06, 0.650, 0.40, 0.24], our_cp[our_cp.side == "LEFT"].zone,
             len(our_cp[our_cp.side == "LEFT"]), TEAM_C, "Corners from the left", "LEFT",
             flick_on_points(our_cp, our_pts, "LEFT"))
    zone_grid(fig, [0.54, 0.650, 0.40, 0.24], our_cp[our_cp.side == "RIGHT"].zone,
             len(our_cp[our_cp.side == "RIGHT"]), TEAM_C, "Corners from the right", "RIGHT",
             flick_on_points(our_cp, our_pts, "RIGHT"))

    fig.text(0.055, 0.598, "Who delivers", ha="left", va="center", color=INK,
             fontsize=11.5, weight="bold")
    taker_breakdown(fig, 0.075, 0.571, our_cp, "LEFT")
    taker_breakdown(fig, 0.075, 0.531, our_cp, "RIGHT")

    fig.text(0.055, 0.470, "First contact", ha="left", va="center",
             color=INK, fontsize=11.5, weight="bold")
    first_contact_diagram(fig, [0.06, 0.225, 0.40, 0.240],
                          our_pts[our_pts.side == "LEFT"], "Corners from the left",
                          len(our_cp[our_cp.side == "LEFT"]), "LEFT", TEAM_C)
    first_contact_diagram(fig, [0.54, 0.225, 0.40, 0.240],
                          our_pts[our_pts.side == "RIGHT"], "Corners from the right",
                          len(our_cp[our_cp.side == "RIGHT"]), "RIGHT", TEAM_C)

    # opponent defensive intel, for when we attack
    fig.text(0.055, 0.185, f"What we know about {OPPONENT} defending corners",
             ha="left", va="center", color=INK, fontsize=11.5, weight="bold")

    nd = len(opp_def_cp)
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

    ax = fig.add_axes([0.055, 0.100, 0.885, 0.075]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for i, (kicker, body, colour) in enumerate([
            ("OPP STRONGEST HEADER", header_txt, "#FF2D2D"),
            ("THEIR MOST VULNERABLE AREA", vuln_txt, "#8FE388")]):
        x0 = i * 0.505
        ax.add_patch(Rectangle((x0, 0.02), 0.48, 0.96, fc=PANEL, ec=GRID, lw=0.9))
        ax.text(x0 + 0.03, 0.78, kicker, ha="left", va="top", color=colour,
                fontsize=8, weight="bold")
        ax.text(x0 + 0.03, 0.52, body, ha="left", va="top", color=INK,
                fontsize=8.2, wrap=True)

    caveat(fig, f"Our corners: {n} from {n_matches} matches. Their "
                f"defensive corners: {nd} from {n_opp} matches.", y=0.028)


def main():
    apply_style()
    opp = _load_side(OPP_CACHES)
    ours = _load_side(OUR_CACHES)

    n_opp_matches = opp["ev"].matchId.nunique() if opp["ev"] is not None else 0
    n_our_matches = ours["ev"].matchId.nunique() if ours["ev"] is not None else 0

    opp_cp = corner_phases(opp["setpieces"], attacking=OPPONENT)
    our_cp = corner_phases(ours["setpieces"], attacking=US)
    opp_def_cp = corner_phases(opp["setpieces"], defending=OPPONENT)

    opp_pts = _first_touch_points(opp["ev"], OPPONENT)
    our_pts = _first_touch_points(ours["ev"], US)

    with PdfPages(OUT) as pdf:
        fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor=BG)
        page_opposition(fig, opp_cp, opp_pts, n_opp_matches)
        pdf.savefig(fig, facecolor=BG); plt.close(fig)

        fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor=BG)
        page_ours(fig, our_cp, our_pts, opp_def_cp, opp["ev"],
                 n_our_matches, n_opp_matches)
        pdf.savefig(fig, facecolor=BG); plt.close(fig)

    print(f"wrote {OUT}  (2 pages)")


if __name__ == "__main__":
    main()
