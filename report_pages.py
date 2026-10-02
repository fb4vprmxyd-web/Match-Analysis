"""Page renderers - one function per page of the debrief."""

import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Arc
from matplotlib.lines import Line2D

from report_config import (BG, PANEL, INK, MUTED, GRID, TEAM_C, OPP_C, GOOD, BAD,
                           BARC, PAGE_W, PAGE_H, FONT, draw_pitch, style_bar_axis,
                           surname, contrast_ink, PITCH_X, PITCH_Y,
                           player_table, stat_table)
import report_metrics as M


def _draw_pitch_portrait(ax, lw=1.0):
    """Pitch drawn tall (width along the page's horizontal axis, length
    running vertically) - a same-shape copy of build_prematch.py's
    draw_pitch_portrait(), kept local rather than imported (see this
    module's import comment on why). Used for the lane/channel diagram
    below: a lane is a strip of pitch width, and "left wing reads as the
    left column" only holds if the pitch itself is portrait."""
    ax.set_facecolor(BG)
    X, Y = PITCH_X, PITCH_Y
    r = dict(fill=False, color=INK, lw=lw, zorder=1)
    ax.add_patch(Rectangle((-Y, -X), 2 * Y, 2 * X, **r))
    ax.plot([-Y, Y], [0, 0], color=INK, lw=lw, zorder=1)
    ax.add_patch(Circle((0, 0), 9.15, **r))
    ax.plot([0], [0], marker=".", ms=2, color=INK, zorder=1)
    for s in (-1, 1):
        ax.add_patch(Rectangle((-20.16, s * X - s * 16.5), 40.32, s * 16.5, **r))
        ax.add_patch(Rectangle((-9.16, s * X - s * 5.5), 18.32, s * 5.5, **r))
        ax.plot([0], [s * (X - 11)], marker=".", ms=2, color=INK, zorder=1)
        ax.add_patch(Arc((0, s * (X - 11)), 18.3, 18.3, angle=0,
                         theta1=218 if s > 0 else 38,
                         theta2=322 if s > 0 else 142, **r))
    ax.set_xlim(-Y - 1.5, Y + 1.5)
    ax.set_ylim(-X - 1.5, X + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")


def _lane_pitch(ax, values, colour, lane_labels, fmt="{:.1f}", fontsize=10.5):
    """Lane/channel share on a portrait pitch - same layout as
    build_prematch.py's draw_lane_pitch(), with a larger default label size
    (explicit request: keep the writing on these bigger and clearer than
    the pre-match version)."""
    _draw_pitch_portrait(ax, lw=1.0)
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
                ha="center", va="center", fontsize=fontsize, weight="bold",
                color=INK, zorder=5, linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.35", fc=PANEL, ec=GRID, lw=0.8))


def _fig():
    return plt.figure(figsize=(PAGE_W, PAGE_H), facecolor=BG)


def _title(fig, text, y=0.965, size=30):
    fig.text(0.5, y, text, ha="center", va="center", color=INK,
             fontsize=size, weight="bold")


# ============================================================== page 1
def page_cover(fig, team, opp, comp, matchday, venue_date, score):
    """Quadrant colour-block cover mirroring the template's layout.

    The source uses club crests; those are club IP and not in the data feed,
    so this uses a typographic treatment in the clubs' own colours instead.
    """
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Horizontal bands keep every string on a single, high-contrast ground.
    ax.add_patch(Rectangle((0, .60), 1, .40, color=TEAM_C, zorder=0))
    ax.add_patch(Rectangle((0, .36), 1, .24, color=BG, zorder=0))
    ax.add_patch(Rectangle((0, 0), 1, .36, color=OPP_C, zorder=0))

    ax.text(.5, .775, team.upper(), ha="center", va="center", color="#FFFFFF",
            fontsize=40, weight="bold", zorder=2)
    ax.text(.5, .705, "—" * 9, ha="center", va="center", color="#FFFFFF",
            fontsize=13, alpha=.55, zorder=2)

    ax.text(.5, .545, "POST-MATCH DEBRIEF", ha="center", va="center",
            color="#FFFFFF", fontsize=27, weight="bold", zorder=2)
    ax.text(.5, .445, score, ha="center", va="center", color="#FFFFFF",
            fontsize=64, weight="bold", zorder=2)

    ink = contrast_ink(OPP_C)
    muted = "#3A3320" if ink == "#141414" else "#D8D3C4"
    ax.text(.5, .265, opp.upper(), ha="center", va="center", color=ink,
            fontsize=36, weight="bold", zorder=2)
    ax.text(.5, .135, f"{comp}   ·   {matchday}", ha="center", va="center",
            color=ink, fontsize=15, weight="bold", zorder=2)
    ax.text(.5, .088, venue_date, ha="center", va="center",
            color=muted, fontsize=12.5, zorder=2)


# ============================================================== page 2
def page_match_stats(fig, rows, team, opp):
    """Team-vs-team comparison table - same stat_table() picture-style
    layout the pre-match report uses, replacing the old two-colour bar per
    row."""
    _title(fig, "MATCH STATS", y=0.955, size=32)
    table_rows = [(label, lt, rt) for label, lt, rt, lv, rv in rows]
    stat_table(fig, [0.075, 0.045, 0.85, 0.865], table_rows, team, opp,
              left_colour=TEAM_C, right_colour=OPP_C)


# ============================================================== page 3
def page_team_charts_a(fig, ev, team, opp, score_line, ncols=6, nrows=5,
                       xt_totals=None):
    _title(fig, "TEAM CHARTS")
    fig.text(0.5, 0.925, score_line, ha="center", va="center", color=INK,
             fontsize=19, weight="bold")

    e = ev.copy()
    e["min"] = M.minute(e)
    fig.text(0.10, 0.895, "Timelines - xG, xT & BPD", ha="left", va="center",
             color=INK, fontsize=12.5, weight="bold")
    if xt_totals:
        fig.text(0.91, 0.895,
                 "xT curve shaped from events, scaled to Impect's match total",
                 ha="right", va="center", color=MUTED, fontsize=6.8)

    specs = [("SHOT_XG", "xG", 0.700), ("PXT_ATTACK", "xT", 0.505),
             ("BYPASSED_DEFENDERS", "BPD", 0.310)]
    for col, lab, y0 in specs:
        ax = fig.add_axes([0.11, y0, 0.80, 0.165])
        style_bar_axis(ax, ylabel=lab, label_size=10, tick_size=8)
        ax.xaxis.grid(True, color=GRID, ls=(0, (4, 4)), lw=0.7)
        for sq, c in ((team, TEAM_C), (opp, OPP_C)):
            d = e[e.squadName == sq].sort_values("min")
            cum = d[col].fillna(0).cumsum()
            # The event feed's pxT doesn't aggregate exactly to Impect's
            # published match total, so the curve is rescaled to finish on it.
            # Shape is event-derived; the endpoint matches page 2.
            if col == "PXT_ATTACK" and xt_totals and len(cum):
                fin = cum.iloc[-1]
                tgt = xt_totals.get(sq)
                if tgt is not None and abs(fin) > 1e-9:
                    cum = cum * (tgt / fin)
            xs = np.concatenate([[0], d["min"].values])
            ys = np.concatenate([[0], cum.values])
            ax.step(xs, ys, where="post", color=c, lw=1.6, zorder=3)
            if len(ys):
                ax.text(xs[-1] + 1.2, ys[-1], f"{ys[-1]:.2f}" if col != "BYPASSED_DEFENDERS"
                        else f"{ys[-1]:.2f}", color=c, fontsize=9,
                        weight="bold", va="center")
        # goal markers
        for _, g in e[(e.GOALS > 0)].iterrows():
            c = TEAM_C if g.squadName == team else OPP_C
            d = e[(e.squadName == g.squadName) & (e["min"] <= g["min"])]
            yv = d[col].fillna(0).sum()
            if col == "PXT_ATTACK" and xt_totals:
                full = e[e.squadName == g.squadName][col].fillna(0).sum()
                tgt = xt_totals.get(g.squadName)
                if tgt is not None and abs(full) > 1e-9:
                    yv = yv * (tgt / full)
            ax.plot([g["min"]], [yv], marker="o", ms=5, color=c, zorder=4)
        ax.set_xlim(0, max(e["min"].max() + 6, 96))
        ax.set_xticks([15, 30, 45, 60, 75, 90])
        if col != "BYPASSED_DEFENDERS":
            ax.set_xticklabels([])

    # ---- shot map
    ax = fig.add_axes([0.045, 0.055, 0.42, 0.225])
    fig.text(0.045, 0.288, "Shot Map", ha="left", va="center", color=INK,
             fontsize=12.5, weight="bold")
    draw_pitch(ax)
    s = M.shots(ev)
    for _, r in s.iterrows():
        home_side = r.squadName == team
        x = r.startAdjCoordinatesX * (1 if home_side else -1)
        y = r.startAdjCoordinatesY * (1 if home_side else -1)
        c = TEAM_C if home_side else OPP_C
        size = 12 + (r.xg or 0) * 620
        if r.is_goal:
            ax.scatter([x], [y], s=size + 45, marker="*", color=c,
                       edgecolors="white", linewidths=0.5, zorder=5)
        else:
            ax.scatter([x], [y], s=size, color=c, alpha=.85, zorder=4)
    tot = ev.groupby("squadName")[["SHOT_XG", "POSTSHOT_XG"]].sum()
    # clip_on=False: these sit on the pitch border and would otherwise be cut
    for xa, ha_, sq, col in ((0.985, "right", team, TEAM_C),
                             (0.015, "left", opp, OPP_C)):
        ax.text(xa, 0.975, f"xG: {tot.loc[sq,'SHOT_XG']:.2f}",
                transform=ax.transAxes, ha=ha_, va="top", color=col,
                fontsize=7.5, weight="bold", clip_on=False, zorder=6)
        ax.text(xa, 0.895, f"PSxG: {tot.loc[sq,'POSTSHOT_XG']:.2f}",
                transform=ax.transAxes, ha=ha_, va="top", color=col,
                fontsize=7.5, weight="bold", clip_on=False, zorder=6)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=TEAM_C, label=team),
                       Line2D([], [], marker="o", ls="", color=OPP_C, label=opp)],
              loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=2,
              frameon=False, fontsize=7, labelcolor=INK, handletextpad=0.25,
              columnspacing=0.9)
    fig.text(0.045, 0.036, "Size of points represents the xG | Stars represent goals",
             ha="left", va="center", color=MUTED, fontsize=7)

    # ---- recovery grid
    grid, thirds = M.recovery_grid(ev, team, ncols, nrows)
    fig.text(0.545, 0.288, "Ball Recoveries", ha="left", va="center", color=INK,
             fontsize=12.5, weight="bold")
    fig.text(0.545, 0.262,
             " | ".join(f"{k}: {v}" for k, v in thirds.items()),
             ha="left", va="center", color=MUTED, fontsize=8)
    gax = fig.add_axes([0.545, 0.075, 0.415, 0.175]); gax.axis("off")
    gax.set_xlim(0, ncols); gax.set_ylim(0, nrows)
    vmax = grid.max() if grid.max() > 0 else 1
    for i in range(nrows):
        for j in range(ncols):
            v = grid[i, j]
            shade = plt.get_cmap("Blues")(0.08 + 0.92 * (v / vmax))
            gax.add_patch(Rectangle((j, nrows - 1 - i), 1, 1, fc=shade,
                                    ec="#C9D4E2", lw=0.6))
            gax.text(j + .5, nrows - 1 - i + .5, f"{v:.0f}%", ha="center",
                     va="center", fontsize=8.5,
                     color="#0B1B2B" if v / vmax < .55 else "white")
    aax = fig.add_axes([0.545, 0.048, 0.415, 0.022]); aax.axis("off")
    aax.set_xlim(0, 1); aax.set_ylim(0, 1)
    aax.annotate("", xy=(0.78, 0.78), xytext=(0.22, 0.78),
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1))
    aax.text(0.5, 0.12, "Direction of Attack", ha="center", va="center",
             color=INK, fontsize=8, weight="bold")


# ============================================================== page 4
def _draw_network(fig, rect, ev, team, shirts, title, subtitle,
                  start_min=None, end_min=None):
    """One passing network in a given time window."""
    fig.text(rect[0], rect[1] + rect[3] + 0.030, title, ha="left", va="center",
             color=INK, fontsize=11, weight="bold")
    fig.text(rect[0], rect[1] + rect[3] + 0.011, subtitle, ha="left",
             va="center", color=MUTED, fontsize=7.2)
    ax = fig.add_axes(rect)
    draw_pitch(ax, lw=0.8)
    nodes, edges = M.passing_network(ev, team, start_min, end_min)
    if len(edges):
        mx = edges.n.max()
        pos = nodes.set_index("playerName")[["x", "y"]].to_dict("index")
        for _, r in edges.iterrows():
            a, b = r.playerName, r.passReceiverPlayerName
            if a in pos and b in pos and r.n >= 2:
                ax.plot([pos[a]["x"], pos[b]["x"]], [pos[a]["y"], pos[b]["y"]],
                        color="#D8E2EE", lw=0.4 + 3.2 * (r.n / mx),
                        alpha=0.28 + 0.55 * (r.n / mx), zorder=2,
                        solid_capstyle="round")
    if len(nodes):
        smax = nodes.n.max()
        offs = M.label_offsets(nodes, ax, s_base=90, s_extra=560)
        for k, (_, r) in enumerate(nodes.iterrows()):
            ax.scatter([r.x], [r.y], s=90 + 560 * (r.n / smax), color="#0E6BA8",
                       edgecolors="#4EB6E3", linewidths=1.3, zorder=3)
            lab = str(shirts.get(r.playerName, "")) if shirts else ""
            if not lab:
                lab = surname(r.playerName)[:3].upper()
            lx, ly, leader = offs[k]
            if leader:
                ax.plot([r.x, lx], [r.y, ly], color="#4EB6E3", lw=0.7,
                        alpha=.85, zorder=4)
                ax.text(lx, ly, lab, ha="center", va="center", color="#0B1B2B",
                        fontsize=6.8, weight="bold", zorder=6,
                        bbox=dict(boxstyle="circle,pad=0.28", fc="#4EB6E3",
                                  ec="none"))
            else:
                ax.text(r.x, r.y, lab, ha="center", va="center", color="white",
                        fontsize=6.8, weight="bold", zorder=5)
    return len(nodes)


def page_team_charts_b(fig, ev, team, shirts=None, first_change=None,
                       changes=None):
    _title(fig, "TEAM CHARTS")
    fig.text(0.5, 0.925, "Passing Network & Grid", ha="center", va="center",
             color=INK, fontsize=13.5, weight="bold")

    # Two windows rather than one all-match network: the starting XI and the
    # side after the changes are different teams and should not share a plot.
    cut = first_change
    W, H, Y = 0.415, 0.255, 0.585
    if cut:
        p = ev[(ev.squadName == team) & (ev.actionType == "PASS")
               & (ev.result == "SUCCESS") & (ev.passReceiverType == "TEAMMATE")].copy()
        p["_min"] = M.minute(p)
        n_before, n_after = int((p["_min"] < cut).sum()), int((p["_min"] >= cut).sum())
        _draw_network(fig, [0.055, Y, W, H], ev, team, shirts,
                      "Starting XI", f"0-{cut}' · {n_before} successful passes",
                      end_min=cut)
        _draw_network(fig, [0.530, Y, W, H], ev, team, shirts,
                      "After the changes",
                      f"{cut}'-FT · {n_after} successful passes", start_min=cut)
    else:
        _draw_network(fig, [0.16, Y, 0.68, H], ev, team, shirts,
                      "Passing Network", "full match")

    fig.text(0.5, Y - 0.016,
             "Locations are average positions when making a pass | point size = "
             "successful passes | links shown from 2 passes",
             ha="center", va="center", color=MUTED, fontsize=6.8)

    # ---- substitutions, so the shape change is legible
    if changes:
        # ASCII arrow: the font has no glyph for U+2192 and it rendered blank.
        txt = "     ".join(
            f"{m}'  {', '.join(surname(x) for x in off)} -> "
            f"{', '.join(surname(x) for x in on)}"
            for m, off, on in changes)
        fig.text(0.5, Y - 0.038, txt, ha="center", va="center", color=INK,
                 fontsize=7.2)

    # ---- passer x receiver grid (full match)
    p = ev[(ev.squadName == team) & (ev.actionType == "PASS")
           & (ev.result == "SUCCESS") & ev.playerName.notna()
           & ev.passReceiverPlayerName.notna()
           & (ev.passReceiverType == "TEAMMATE")]
    te = ev[ev.squadName == team]
    succ, unsucc = int(te.SUCCESSFUL_PASSES.sum()), int(te.UNSUCCESSFUL_PASSES.sum())
    fig.text(0.5, 0.512, f"Passer × Receiver — full match   ·   "
                         f"Total Passes {succ + unsucc}   ·   Accurate {succ}",
             ha="center", va="center", color=INK, fontsize=10, weight="bold")

    order = (p.groupby("playerName").size()
             .sort_values(ascending=False).index.tolist())
    mat = (p.groupby(["playerName", "passReceiverPlayerName"]).size()
           .unstack(fill_value=0).reindex(index=order, columns=order, fill_value=0))
    k = len(order)
    gax = fig.add_axes([0.245, 0.048, 0.72, 0.40]); gax.axis("off")
    gax.set_xlim(0, k); gax.set_ylim(0, k)
    vmax = mat.values.max() if mat.values.max() > 0 else 1
    for a in range(k):
        for b in range(k):
            v = mat.values[a, b]
            if a == b:
                gax.add_patch(Rectangle((b, k - 1 - a), 1, 1, fc=BG,
                                        ec="#C9D4E2", lw=.5))
                continue
            shade = plt.get_cmap("Blues")(0.06 + 0.94 * (v / vmax))
            gax.add_patch(Rectangle((b, k - 1 - a), 1, 1, fc=shade,
                                    ec="#C9D4E2", lw=.5))
            gax.text(b + .5, k - 1 - a + .5, str(int(v)), ha="center",
                     va="center", fontsize=6.2,
                     color="#0B1B2B" if v / vmax < .55 else "white")
    for a, nm in enumerate(order):
        gax.text(-0.25, k - 1 - a + .5, surname(nm), ha="right", va="center",
                 fontsize=6.4, color=INK)
        gax.text(a + .5, k + 0.22, surname(nm), ha="left", va="bottom",
                 fontsize=6.4, color=INK, rotation=45, rotation_mode="anchor")
    fig.text(0.150, 0.248, "Passer", ha="center", va="center", color=INK,
             fontsize=10.5, weight="bold", rotation=90)


# ============================================================== page 5
def _flanks(df):
    """Crosses originating from each flank (adjusted Y: +ve one side, -ve other)."""
    if df is None or len(df) == 0:
        return 0, 0
    return int((df.startAdjCoordinatesY > 0).sum()), int((df.startAdjCoordinatesY < 0).sum())


def _cross_note(ev, squad):
    """Single-team cross summary with the flank split."""
    d = M.crosses_and_cutbacks(ev, squad)
    if not len(d):
        return "no open-play crosses"
    l, r = _flanks(d)
    ok = int((d.result == "SUCCESS").sum())
    return f"{len(d)} crosses | {ok} completed | L {l} / R {r}"


def _pitch_panel(fig, rect, title, df, note, arrows=True, touches=False,
                 xmin=None, overlay=None, title_colour=None):
    fig.text(rect[0] + rect[2] / 2, rect[1] + rect[3] + 0.013, title,
             ha="center", va="center", color=title_colour or INK,
             fontsize=9.5 if title_colour else 10, weight="bold")
    ax = fig.add_axes(rect)
    draw_pitch(ax, lw=0.7)
    if xmin is not None:                     # crop to the relevant zone
        ax.set_xlim(xmin, 54.0)
    if touches:
        ax.scatter(df.startAdjCoordinatesX, df.startAdjCoordinatesY, s=16,
                   color=BARC, edgecolors="#0E6BA8", linewidths=.4, zorder=4)
    elif arrows:
        for _, r in df.iterrows():
            c = GOOD if r.result == "SUCCESS" else BAD
            if pd.isna(r.endAdjCoordinatesX):
                continue
            ax.annotate("", xy=(r.endAdjCoordinatesX, r.endAdjCoordinatesY),
                        xytext=(r.startAdjCoordinatesX, r.startAdjCoordinatesY),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=0.8,
                                        alpha=.9, mutation_scale=7), zorder=4)
    if overlay is not None and len(overlay):
        for _, r in overlay.iterrows():
            if pd.isna(r.endAdjCoordinatesX):
                continue
            ax.annotate("", xy=(r.endAdjCoordinatesX, r.endAdjCoordinatesY),
                        xytext=(r.startAdjCoordinatesX, r.startAdjCoordinatesY),
                        arrowprops=dict(arrowstyle="-|>", color=OPP_C, lw=0.8,
                                        alpha=.85, mutation_scale=7), zorder=3)
    # anchored to the top so a two-line note grows downward predictably and
    # cannot creep into the next panel's title
    cx = rect[0] + rect[2] / 2
    if isinstance(note, (list, tuple)):
        for k, (txt, col) in enumerate(note):
            fig.text(cx, rect[1] - 0.004 - k * 0.0105, txt, ha="center",
                     va="top", color=col, fontsize=6.4, weight="bold")
    else:
        fig.text(cx, rect[1] - 0.004, note, ha="center", va="top",
                 color=MUTED, fontsize=6.6)


def _plabels(names, mins=None, full=None):
    """Surname, with minutes appended for anyone short of a full match."""
    out = []
    for n in names:
        s = surname(n)
        if mins is not None and full and n in mins and mins[n] < full * 0.95:
            s = f"{s} {mins[n]:.0f}'"
        out.append(s)
    return out


def page_player_stats_a(fig, ev, team, mins=None, full=None, opp=None):
    """Pitch maps stay - they show WHERE, which a table can't - but every
    per-player breakdown next to them is now a player_table() instead of a
    bar chart, matching the pre-match report's personnel page."""
    _title(fig, "PLAYER STATS")
    if mins:
        fig.text(0.5, 0.928,
                 "pitch maps plot every recorded event this match · minutes "
                 "shown in the table for anyone short of a full match",
                 ha="center", va="center", color=MUTED, fontsize=7.2)
    # `xmin` crops each pitch to the zone the metric lives in, as the template does
    panels = [
        ("Passes Into Final Third", M.passes_into_final_third(ev, team), False, None),
        ("Passes in Final Third", M.passes_in_final_third(ev, team), False, 10.0),
        ("Open-Play Crosses", M.crosses_and_cutbacks(ev, team), False, 0.0),
        ("Passes Into the Box", M.passes_into_box(ev, team), False, -10.0),
        ("Box Touches", M.box_touches(ev, team), True, 10.0),
    ]
    top, h, gap = 0.893, 0.135, 0.042
    for idx, (title, df, is_touch, xmin) in enumerate(panels):
        y = top - (idx + 1) * h - idx * gap
        if is_touch:
            note = f"Number of Box Touches: {len(df)}"
        else:
            ok, bad = M.split_success(df)
            note = f"Successful: {len(ok)} | Unsuccessful: {len(bad)}"
        if title.startswith("Open-Play Crosses") and opp:
            # One map per team, side by side, so the two delivery patterns can
            # be read separately rather than untangled from a single pitch.
            # Arrows keep the page convention (green completed / red not);
            # the team is carried by the coloured panel title.
            _pitch_panel(fig, [0.035, y, 0.170, h], team,
                         M.crosses_and_cutbacks(ev, team),
                         _cross_note(ev, team), xmin=xmin, title_colour=TEAM_C)
            _pitch_panel(fig, [0.215, y, 0.170, h], opp,
                         M.crosses_and_cutbacks(ev, opp),
                         _cross_note(ev, opp), xmin=xmin, title_colour=OPP_C)
        else:
            _pitch_panel(fig, [0.035, y, 0.235, h], title, df, note,
                         touches=is_touch, xmin=xmin)

        if is_touch:
            cnt = df.groupby("playerName").size().sort_values(ascending=False)
            names = list(cnt.index)
            rows = list(zip(_plabels(names, mins, full), cnt.values.tolist()))
            player_table(fig, [0.345, y, 0.615, h], rows,
                        ["Player", "Box touches"])
        elif title.startswith("Open-Play Crosses") and opp:
            for x0, sq in ((0.435, team), (0.710, opp)):
                d = M.crosses_and_cutbacks(ev, sq)
                g = d.groupby("playerName").agg(
                    ok=("result", lambda s: (s == "SUCCESS").sum()),
                    tot=("result", "size")).sort_values("tot", ascending=False)
                labels = _plabels(list(g.index), mins, full)
                rows = [(lab, int(r.ok), int(r.tot - r.ok))
                       for lab, (_, r) in zip(labels, g.iterrows())]
                player_table(fig, [x0, y, 0.255, h], rows,
                            ["Player", "OK", "Missed"])
        else:
            g = df.groupby("playerName").agg(
                ok=("result", lambda s: (s == "SUCCESS").sum()),
                tot=("result", "size")).sort_values("tot", ascending=False)
            labels = _plabels(list(g.index), mins, full)
            rows = [(lab, int(r.ok), int(r.tot - r.ok))
                   for lab, (_, r) in zip(labels, g.iterrows())]
            player_table(fig, [0.345, y, 0.615, h], rows,
                        ["Player", "Successful", "Unsuccessful"])


# ============================================================== page 6
def page_player_stats_b(fig, tbl, bench=None, mins=None, full=None):
    """Same ten per-player metrics as before, now as player_table()s
    instead of bar charts, matching the pre-match report's personnel page.
    The season-median benchmark (previously a white tick behind the bar)
    becomes its own column instead."""
    _title(fig, "PLAYER STATS")
    has_bench = bench is not None and len(bench)
    if has_bench:
        fig.text(0.5, 0.928,
                 "\"Season median\" = the player's own median in matches he "
                 "played 60+ minutes this season",
                 ha="center", va="center", color=MUTED, fontsize=7.5)
    else:
        # Silence would read as "no benchmark exists"; say why it is absent.
        fig.text(0.5, 0.928,
                 "minutes shown for anyone short of a full match — season "
                 "benchmarks omitted: fewer than 5 matches of this season's "
                 "data, and last season's squad is not a fair comparison",
                 ha="center", va="center", color=MUTED, fontsize=7.0,
                 linespacing=1.6)

    def bval(name, key):
        if not has_bench or key not in bench.columns or name not in bench.index:
            return None
        v = bench.loc[name, key]
        return None if pd.isna(v) else v

    def label_above(rect, text):
        fig.text(rect[0], rect[1] + rect[3] + 0.014, text, ha="left",
                 va="center", color=INK, fontsize=10, weight="bold")

    def plain_table(rect, col, label, fmt="{:.1f}", ascending=False, bkey=None):
        label_above(rect, label)
        d = tbl[["playerName", col]].copy()
        d = d[d[col].notna()]
        d = d[d[col] != 0] if col in ("BPDR", "xG") else d
        d = d.sort_values(col, ascending=ascending)
        labels = _plabels(list(d.playerName), mins, full)
        cols = ["Player", label]
        rows = []
        for lab, (_, r) in zip(labels, d.iterrows()):
            row = [lab, fmt.format(r[col])]
            if has_bench:
                bv = bval(r.playerName, bkey or col)
                row.append(fmt.format(bv) if bv is not None else "-")
            rows.append(tuple(row))
        if has_bench:
            cols = cols + ["Season median"]
        player_table(fig, rect, rows, cols)

    def duel_table(rect, won, lost, label, bkey):
        label_above(rect, label)
        d = tbl[["playerName", won, lost]].copy()
        d["tot"] = d[won] + d[lost]
        d = d[d.tot > 0].sort_values("tot", ascending=False)
        labels = _plabels(list(d.playerName), mins, full)
        cols = ["Player", "Won", "Lost"]
        rows = []
        for lab, (_, r) in zip(labels, d.iterrows()):
            row = [lab, int(r[won]), int(r[lost])]
            if has_bench:
                bv = bval(r.playerName, bkey)
                row.append(f"{bv:.1f}" if bv is not None else "-")
            rows.append(tuple(row))
        if has_bench:
            cols = cols + ["Median"]
        player_table(fig, rect, rows, cols)

    def passes_table(rect):
        label_above(rect, "Passes")
        d = tbl[["playerName", "succ", "unsucc"]].copy()
        d["tot"] = d.succ + d.unsucc
        d = d[d.tot > 0].sort_values("tot", ascending=False)
        labels = _plabels(list(d.playerName), mins, full)
        cols = ["Player", "Successful", "Unsuccessful"]
        rows = []
        for lab, (_, r) in zip(labels, d.iterrows()):
            row = [lab, int(r.succ), int(r.unsucc)]
            if has_bench:
                bv = bval(r.playerName, "passes")
                row.append(f"{bv:.0f}" if bv is not None else "-")
            rows.append(tuple(row))
        if has_bench:
            cols = cols + ["Median"]
        player_table(fig, rect, rows, cols)

    L, R, W, H = 0.075, 0.560, 0.385, 0.122
    ys = [0.780, 0.598, 0.416, 0.234, 0.052]

    plain_table([L, ys[0], W, H], "BPD", "BPD")
    plain_table([R, ys[0], W, H], "BPDR", "BPDR")
    plain_table([L, ys[1], W, H], "xT", "xT", fmt="{:.2f}")
    plain_table([R, ys[1], W, H], "xG", "xG", fmt="{:.2f}")
    passes_table([L, ys[2], W, H])
    duel_table([R, ys[2], W, H], "aw", "al", "Aerial Duels", "aerials")
    duel_table([L, ys[3], W, H], "gw", "gl", "Ground Duels", "grounds")
    plain_table([R, ys[3], W, H], "BLRT", "BLRT", fmt="{:.1f}", ascending=True)
    plain_table([L, ys[4], W, H], "pressures", "Pressures", fmt="{:.0f}")
    plain_table([R, ys[4], W, H], "recoveries", "Ball Recoveries", fmt="{:.0f}")


# ========================================================== set-piece page
def _compare_rows(fig, rect, rows, team, opp, value_fmt=True):
    """Shared two-sided bar block used by the set-piece and defensive pages."""
    ax = fig.add_axes(rect); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    n = len(rows)
    slot = 1.0 / n
    bar_h = slot * 0.36
    for i, (label, fmt, lv, rv) in enumerate(rows):
        top = 1 - i * slot
        ax.text(.5, top - slot * 0.28, label, ha="center", va="center",
                color=INK, fontsize=9.5, weight="bold")
        y = top - slot * 0.72
        total = (lv or 0) + (rv or 0)
        frac = 0.5 if total == 0 else (lv or 0) / total
        frac = min(max(frac, 0.05), 0.95)
        ax.add_patch(FancyBboxPatch((0, y - bar_h / 2), frac, bar_h,
                                    boxstyle="round,pad=0,rounding_size=0.012",
                                    fc=TEAM_C, ec="none", mutation_aspect=0.05))
        ax.add_patch(FancyBboxPatch((frac, y - bar_h / 2), 1 - frac, bar_h,
                                    boxstyle="round,pad=0,rounding_size=0.012",
                                    fc=OPP_C, ec="none", mutation_aspect=0.05))
        ax.text(frac - 0.012, y, fmt.format(lv), ha="right", va="center",
                color="#FFFFFF", fontsize=9, weight="bold")
        ax.text(frac + 0.012, y, fmt.format(rv), ha="left", va="center",
                color="#111111", fontsize=9, weight="bold")


def page_set_pieces(fig, sp, ev, team, opp):
    _title(fig, "SET PIECES")
    fig.text(0.085, 0.925, team, ha="left", va="center", color=TEAM_C,
             fontsize=12, weight="bold")
    fig.text(0.915, 0.925, opp, ha="right", va="center", color=OPP_C,
             fontsize=12, weight="bold")

    tbl = M.setpiece_team(sp)

    def val(sq, grp, col):
        r = tbl[(tbl.attackingSquadName == sq) & (tbl.grp == grp)]
        return float(r[col].iloc[0]) if len(r) else 0.0

    # ---- volume + value by category
    fig.text(0.075, 0.885, "Phases & Value by Type", ha="left", va="center",
             color=INK, fontsize=12, weight="bold")
    groups = [g for g in M.SP_ORDER
              if (tbl[tbl.grp == g].phases.sum() if len(tbl) else 0) > 0]
    rows = []
    for g in groups:
        nice = g.replace("_", " ").title()
        rows.append((f"{nice} — Phases", "{:.0f}",
                     val(team, g, "phases"), val(opp, g, "phases")))
    _compare_rows(fig, [0.075, 0.635, 0.85, 0.235], rows, team, opp)

    fig.text(0.075, 0.610, "Attacking Value from Set Pieces", ha="left",
             va="center", color=INK, fontsize=12, weight="bold")
    rows2 = [("Set-Piece xG", "{:.2f}",
              sum(val(team, g, "xg") for g in groups),
              sum(val(opp, g, "xg") for g in groups)),
             ("Set-Piece Shots", "{:.0f}",
              sum(val(team, g, "shots") for g in groups),
              sum(val(opp, g, "shots") for g in groups)),
             ("Set-Piece Goals", "{:.0f}",
              sum(val(team, g, "goals") for g in groups),
              sum(val(opp, g, "goals") for g in groups)),
             ("Bypassed Defenders", "{:.1f}",
              sum(val(team, g, "bpd") for g in groups),
              sum(val(opp, g, "bpd") for g in groups))]
    _compare_rows(fig, [0.075, 0.425, 0.85, 0.175], rows2, team, opp)

    # ---- xG by category, grouped bars
    ax = fig.add_axes([0.075, 0.245, 0.38, 0.135])
    style_bar_axis(ax, ylabel="Set-Piece xG", label_size=8, tick_size=7)
    x = np.arange(len(groups))
    ax.bar(x - .19, [val(team, g, "xg") for g in groups], width=.36,
           color=TEAM_C, zorder=3)
    ax.bar(x + .19, [val(opp, g, "xg") for g in groups], width=.36,
           color=OPP_C, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("_", " ").title() for g in groups],
                       rotation=30, ha="right", fontsize=6.8)
    fig.text(0.075, 0.393, "xG by Set-Piece Type", ha="left", va="center",
             color=INK, fontsize=10, weight="bold")

    # ---- set-piece shot map
    fig.text(0.545, 0.393, "Set-Piece Shots", ha="left", va="center",
             color=INK, fontsize=10, weight="bold")
    sax = fig.add_axes([0.545, 0.235, 0.40, 0.150])
    draw_pitch(sax, lw=0.7)
    sax.set_xlim(-6, 54)
    s = ev[(ev.actionType == "SHOT") & ev.phase.eq("SET_PIECE")]
    for _, r in s.iterrows():
        is_team = r.squadName == team
        c = TEAM_C if is_team else OPP_C
        size = 14 + (r.SHOT_XG or 0) * 600
        goal = (r.SHOT_AT_GOAL_NUMBER_SUCCESS or 0) > 0
        sax.scatter([r.startAdjCoordinatesX], [r.startAdjCoordinatesY],
                    s=size + (45 if goal else 0),
                    marker="*" if goal else "o", color=c,
                    edgecolors="white" if goal else "none",
                    linewidths=0.5, alpha=.9, zorder=5)
    fig.text(0.745, 0.222, "Both teams shown attacking right | stars = goals",
             ha="center", va="center", color=MUTED, fontsize=6.8)

    # ---- takers
    tk = M.setpiece_players(sp, team).head(9)
    fig.text(0.075, 0.185, f"{team} — Set Pieces Taken", ha="left",
             va="center", color=INK, fontsize=10, weight="bold")
    ax2 = fig.add_axes([0.075, 0.055, 0.38, 0.115])
    style_bar_axis(ax2, ylabel="Taken", label_size=8, tick_size=7)
    if len(tk):
        xs = np.arange(len(tk))
        ax2.bar(xs, tk.ok, color=GOOD, width=.7, zorder=3)
        ax2.bar(xs, tk.taken - tk.ok, bottom=tk.ok, color=BAD, width=.7, zorder=3)
        for i, (o, t) in enumerate(zip(tk.ok, tk.taken)):
            ax2.text(i, t + tk.taken.max() * .05, f"{int(o)}/{int(t)}",
                     ha="center", va="bottom", fontsize=6.4, color=INK,
                     weight="bold")
        ax2.set_xticks(xs)
        ax2.set_xticklabels([surname(n) for n in tk.playerName], rotation=45,
                            ha="right", fontsize=6.4)
        ax2.set_ylim(0, tk.taken.max() * 1.3)

    # ---- first contact
    fc = M.setpiece_first_contact(sp)
    fig.text(0.545, 0.185, "First Contact on Deliveries", ha="left",
             va="center", color=INK, fontsize=10, weight="bold")
    ax3 = fig.add_axes([0.545, 0.055, 0.40, 0.115])
    style_bar_axis(ax3, ylabel="Deliveries", label_size=8, tick_size=7)
    sides = [team, opp]
    won = [float(fc.loc[s, True]) if (s in fc.index and True in fc.columns) else 0.0
           for s in sides]
    lost = [float(fc.loc[s, False]) if (s in fc.index and False in fc.columns) else 0.0
            for s in sides]
    xs = np.arange(2)
    ax3.bar(xs, won, color=GOOD, width=.5, zorder=3, label="Won")
    ax3.bar(xs, lost, bottom=won, color=BAD, width=.5, zorder=3, label="Lost")
    ax3.set_xticks(xs); ax3.set_xticklabels(sides, fontsize=7)
    ax3.legend(frameon=False, fontsize=6.5, labelcolor=INK, loc="upper right",
               ncol=2, handlelength=1.1)
    ax3.set_ylim(0, max(1, max(w + l for w, l in zip(won, lost)) * 1.45))
    fig.text(0.745, 0.026,
             "Only deliveries where a first contact was recorded",
             ha="center", va="center", color=MUTED, fontsize=6.6)


def _stat_tiles(fig, rect, tiles, team, opp):
    """Row of small two-value tiles. Used where a bar would mislead."""
    ax = fig.add_axes(rect); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    n = len(tiles)
    if n == 0:
        return
    w = 1.0 / n
    for i, (label, fmt, lv, rv, note) in enumerate(tiles):
        x0 = i * w
        ax.add_patch(FancyBboxPatch((x0 + 0.006, 0.04), w - 0.012, 0.92,
                                    boxstyle="round,pad=0,rounding_size=0.02",
                                    fc=PANEL, ec=GRID, lw=0.8,
                                    mutation_aspect=0.35))
        cx = x0 + w / 2
        ax.text(cx, 0.80, label, ha="center", va="center", color=INK,
                fontsize=7.4, weight="bold")
        ax.text(cx - w * 0.20, 0.47, fmt.format(lv), ha="center", va="center",
                color=TEAM_C, fontsize=13, weight="bold")
        ax.text(cx, 0.47, "|", ha="center", va="center", color=GRID, fontsize=11)
        ax.text(cx + w * 0.20, 0.47, fmt.format(rv), ha="center", va="center",
                color=OPP_C, fontsize=13, weight="bold")
        ax.text(cx, 0.17, note, ha="center", va="center", color=MUTED,
                fontsize=6.2)


# ===================================================== out-of-possession page
def page_out_of_possession(fig, ev, psum, squad_sums, dtbl, team, opp,
                           scores=None):
    _title(fig, "OUT OF POSSESSION")
    fig.text(0.085, 0.925, team, ha="left", va="center", color=TEAM_C,
             fontsize=12, weight="bold")
    fig.text(0.915, 0.925, opp, ha="right", va="center", color=OPP_C,
             fontsize=12, weight="bold")

    rows = M.defensive_team(squad_sums, team, opp)
    _compare_rows(fig, [0.075, 0.645, 0.85, 0.258], rows, team, opp)
    # These are defensive quantities: on the first three rows a longer bar is
    # worse, unlike the match-stats page. Say so rather than let the eye assume.
    fig.text(0.5, 0.632,
             "Bars show volume, not merit — for Suffered BPD and the two "
             "Conceded rows, a longer bar is worse",
             ha="center", va="center", color=MUTED, fontsize=7.2)

    # ---- press profile (scores endpoint; PPDA can't go in the bars above)
    tiles = M.press_profile(scores, team, opp)
    if tiles:
        fig.text(0.075, 0.607, "Press Profile", ha="left", va="center",
                 color=INK, fontsize=10.5, weight="bold")
        fig.text(0.925, 0.607, f"{team} | {opp}", ha="right", va="center",
                 color=MUTED, fontsize=7)
        _stat_tiles(fig, [0.075, 0.520, 0.85, 0.072], tiles, team, opp)

    # ---- where the opposition progressed - a portrait pitch, not a bar
    # chart (explicit request, matching the pre-match report's channel
    # diagrams). Sized larger than the first pass at this (still bigger/
    # clearer than the pre-match version's own labels) after direct
    # feedback that it was too small to read comfortably - the rest of
    # this page's lower section is compressed further to make room.
    lane = M.opponent_progression(ev, opp)
    lane_labels = ["L\nWing", "L\nHS", "Centre", "R\nHS", "R\nWing"]
    fig.text(0.075, 0.503, "Bypassed Defenders by Channel", ha="left",
             va="center", color=INK, fontsize=10.5, weight="bold")
    lax = fig.add_axes([0.075, 0.208, 0.238, 0.275])
    _lane_pitch(lax, lane.values, OPP_C, lane_labels, fmt="{:.1f}", fontsize=10.8)

    # ---- conceded shot map (shares the row with the lane pitch, centred
    # on it vertically since it needs less height)
    fig.text(0.360, 0.503, f"Shots Conceded to {opp}", ha="left", va="center",
             color=INK, fontsize=10.5, weight="bold")
    sax = fig.add_axes([0.360, 0.286, 0.590, 0.120])
    draw_pitch(sax, lw=0.7)
    sax.set_xlim(-6, 54)
    s = ev[(ev.actionType == "SHOT") & (ev.squadName == opp)]
    for _, r in s.iterrows():
        goal = (r.SHOT_AT_GOAL_NUMBER_SUCCESS or 0) > 0
        sax.scatter([r.startAdjCoordinatesX], [r.startAdjCoordinatesY],
                    s=14 + (r.SHOT_XG or 0) * 600 + (45 if goal else 0),
                    marker="*" if goal else "o", color=OPP_C,
                    edgecolors="white" if goal else "none", linewidths=0.5,
                    alpha=.9, zorder=5)
    tot = ev[ev.squadName == opp][["SHOT_XG", "POSTSHOT_XG"]].sum()
    fig.text(0.655, 0.274,
             f"Conceded xG {tot.SHOT_XG:.2f} | PSxG {tot.POSTSHOT_XG:.2f} "
             f"| stars = goals",
             ha="center", va="center", color=MUTED, fontsize=6.8)

    # ---- player defensive actions
    d = dtbl.sort_values("def_actions", ascending=False)
    d = d[d.def_actions > 0]
    fig.text(0.075, 0.190, f"{team} — Defensive Actions", ha="left",
             va="center", color=INK, fontsize=10.5, weight="bold")
    ax2 = fig.add_axes([0.075, 0.118, 0.38, 0.052])
    style_bar_axis(ax2, ylabel="Actions", label_size=8, tick_size=7)
    xs = np.arange(len(d))
    b = np.zeros(len(d))
    for col, c, lab in (("INTERCEPTION", "#4EB6E3", "Int"),
                        ("CLEARANCE", "#2E86AB", "Clr"),
                        ("BLOCK", "#8FE388", "Blk")):
        ax2.bar(xs, d[col].values, bottom=b, color=c, width=.7, zorder=3,
                label=lab)
        b = b + d[col].values
    ax2.set_xticks(xs)
    ax2.set_xticklabels([surname(n) for n in d.playerName], rotation=45,
                        ha="right", fontsize=6.3)
    ax2.legend(frameon=False, fontsize=6.2, labelcolor=INK, ncol=3,
               loc="upper right", handlelength=1.0, columnspacing=.8)
    ax2.set_ylim(0, max(b.max() * 1.35, 1))

    # ---- pressures by type
    p = dtbl.sort_values("NUMBER_OF_PRESSES", ascending=False)
    p = p[p.NUMBER_OF_PRESSES > 0]
    fig.text(0.545, 0.190, f"{team} — Pressures by Type", ha="left",
             va="center", color=INK, fontsize=10.5, weight="bold")
    ax3 = fig.add_axes([0.545, 0.118, 0.40, 0.052])
    style_bar_axis(ax3, ylabel="Pressures", label_size=8, tick_size=7)
    xs = np.arange(len(p))
    b = np.zeros(len(p))
    for col, c, lab in (
            ("NUMBER_OF_PRESSES_COUNTER_PRESS", "#E4002B", "Counter"),
            ("NUMBER_OF_PRESSES_BETWEEN_THE_LINES", "#F5C518", "BTL"),
            ("NUMBER_OF_PRESSES_BUILD_UP", "#8FE388", "Build-up"),
            ("press_other", "#4EB6E3", "Other")):
        ax3.bar(xs, p[col].values, bottom=b, color=c, width=.7, zorder=3,
                label=lab)
        b = b + p[col].values
    ax3.set_xticks(xs)
    ax3.set_xticklabels([surname(n) for n in p.playerName], rotation=45,
                        ha="right", fontsize=6.3)
    ax3.legend(frameon=False, fontsize=6.0, labelcolor=INK, ncol=4,
               loc="upper right", handlelength=1.0, columnspacing=.6)
    ax3.set_ylim(0, max(b.max() * 1.4, 1))

    # ---- turnover quality
    q = dtbl[dtbl.bwrd > 0].sort_values("bwrd", ascending=False).head(12)
    fig.text(0.075, 0.080, f"{team} — Ball Win Removed Defenders "
                           f"(turnover quality)", ha="left", va="center",
             color=INK, fontsize=9.0, weight="bold")
    ax4 = fig.add_axes([0.075, 0.032, 0.85, 0.030])
    style_bar_axis(ax4, ylabel="BWRD", label_size=7.0, tick_size=6.0)
    xs = np.arange(len(q))
    ax4.bar(xs, q.bwrd.values, color=BARC, width=.66, hatch="///",
            edgecolor="#0E6BA8", zorder=3)
    for i, v in enumerate(q.bwrd.values):
        ax4.text(i, v + max(q.bwrd.max(), .01) * .06, f"{v:.1f}", ha="center",
                 va="bottom", fontsize=6.0, color=INK, weight="bold")
    ax4.set_xticks(xs)
    ax4.set_xticklabels([surname(n) for n in q.playerName], rotation=30,
                        ha="right", fontsize=6.0)
    ax4.set_ylim(0, max(q.bwrd.max() * 1.35, 1))


# ========================================================== coach's brief
VERDICT_C = {"GREEN": "#8FE388", "AMBER": "#F5C518", "RED": "#FF2D2D",
             "INFO": "#4EB6E3", "N/A": "#5A6676"}
VERDICT_LABEL = {"GREEN": "MET", "AMBER": "MARGINAL", "RED": "MISSED",
                 "INFO": "NOTED", "N/A": "NO DATA"}


def page_coach_brief(fig, rows, team, opp, score_line, provisional=True):
    """Grades the match against the coaching staff's pre-match instructions."""
    _title(fig, "COACH'S BRIEF", y=0.960, size=30)
    fig.text(0.5, 0.928, score_line, ha="center", va="center", color=MUTED,
             fontsize=10.5)

    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    tally = "   ".join(
        f"{VERDICT_LABEL[k]} {counts.get(k,0)}"
        for k in ("GREEN", "AMBER", "RED") if counts.get(k))
    fig.text(0.5, 0.905, tally, ha="center", va="center", color=INK,
             fontsize=9.5, weight="bold")

    # geometry
    x0, x1 = 0.050, 0.950
    y = 0.878
    sec_h, row_h, gap = 0.026, 0.0295, 0.004
    last_section = None

    for r in rows:
        if r["section"] != last_section:
            y -= sec_h
            ax = fig.add_axes([x0, y, x1 - x0, sec_h * 0.78]); ax.axis("off")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.add_patch(FancyBboxPatch((0, 0.05), 1, 0.9,
                                        boxstyle="round,pad=0,rounding_size=0.02",
                                        fc=GRID, ec="none", mutation_aspect=0.10))
            ax.text(0.012, 0.5, r["section"], ha="left", va="center", color=INK,
                    fontsize=9, weight="bold")
            last_section = r["section"]
            y -= gap

        y -= row_h
        ax = fig.add_axes([x0, y, x1 - x0, row_h]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.add_patch(FancyBboxPatch((0, 0.06), 1, 0.88,
                                    boxstyle="round,pad=0,rounding_size=0.018",
                                    fc=PANEL, ec=GRID, lw=0.6,
                                    mutation_aspect=0.10))
        col = VERDICT_C[r["verdict"]]
        ax.add_patch(Rectangle((0, 0.10), 0.006, 0.80, fc=col, ec="none"))

        ax.text(0.016, 0.66, r["instruction"], ha="left", va="center", color=INK,
                fontsize=8.2, weight="bold")
        ax.text(0.016, 0.28, r["metric"], ha="left", va="center", color=MUTED,
                fontsize=6.6)
        # Wrap defensively: readout strings vary a lot in length brief to
        # brief, and an unwrapped long one used to run straight into the
        # verdict chip at x=0.882.
        import textwrap as _tw
        _lines = _tw.wrap(str(r["readout"]), width=44)[:2]
        if len(_lines) > 1:
            ax.text(0.62, 0.62, _lines[0], ha="left", va="center",
                    color="#C6D2E0", fontsize=6.9)
            ax.text(0.62, 0.36, _lines[1], ha="left", va="center",
                    color="#C6D2E0", fontsize=6.9)
        else:
            ax.text(0.62, 0.5, r["readout"], ha="left", va="center",
                    color="#C6D2E0", fontsize=7.6)

        # verdict chip
        ax.add_patch(FancyBboxPatch((0.882, 0.24), 0.106, 0.52,
                                    boxstyle="round,pad=0,rounding_size=0.03",
                                    fc=col, ec="none", mutation_aspect=0.18))
        ax.text(0.935, 0.5, VERDICT_LABEL[r["verdict"]], ha="center", va="center",
                color="#10161F", fontsize=6.4, weight="bold")
        y -= gap

    note = ("Thresholds are PROVISIONAL — replace them with the staff's own "
            "numbers before kick-off. Grading against targets chosen after the "
            "result is marking your own homework.") if provisional else ""
    if note:
        fig.text(0.5, 0.030, note, ha="center", va="center", color=MUTED,
                 fontsize=6.9)


# ======================================================= performance review
def page_performance_review(fig, pros, cons, how_it_went, team, opp, score_line):
    """One flowing narrative instead of a two-column card grid - explicit
    request: prose, not figures, describing how the game went, then what
    was achieved, then what could improve. pros/cons and how_it_went all
    still come from report_metrics.performance_review()'s own gated
    conditions (the verdict still changes with the data, nothing here is
    written in advance) - this only changes how they're assembled into
    sentences."""
    _title(fig, "PERFORMANCE REVIEW", y=0.955, size=30)
    fig.text(0.5, 0.917, score_line, ha="center", va="center", color=MUTED,
             fontsize=11)

    opening = how_it_went

    achieved = (" ".join(d for _, d in pros) if pros else
               "Nothing in this match stood out clearly enough to call a "
               "genuine strength - a quiet performance rather than a bad one.")
    improve = (" ".join(d for _, d in cons) if cons else
              "No clear weakness stood out either - there wasn't an obvious "
              "problem for the next session to fix.")

    sections = [
        ("HOW IT WENT", BARC, opening),
        ("WHAT WE ACHIEVED", GOOD, achieved),
        ("WHERE WE CAN IMPROVE", BAD, improve),
    ]
    # matplotlib's text line height is fontsize * linespacing, in points -
    # converting that to a figure-fraction-per-line step (rather than a
    # guessed constant) keeps the gap between sections consistent
    # regardless of how many lines a paragraph actually wraps to.
    FONTSIZE, LINESPACING = 11, 1.75
    per_line = FONTSIZE * LINESPACING / 72 / PAGE_H
    y = 0.865
    for head, accent, para in sections:
        ax = fig.add_axes([0.075, y - 0.006, 0.02, 0.022]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.add_patch(Rectangle((0, 0), 1, 1, fc=accent, ec="none"))
        fig.text(0.108, y, head, ha="left", va="center", color=INK,
                 fontsize=14, weight="bold")
        y -= 0.045
        lines = textwrap.wrap(para, 95)
        fig.text(0.075, y, "\n".join(lines), ha="left", va="top",
                 color="#B9C4D2", fontsize=FONTSIZE, linespacing=LINESPACING)
        y -= len(lines) * per_line + 0.055

    fig.text(0.5, 0.032,
             "Interpretation of the preceding pages. Each point is triggered by "
             "a condition in the match data, not written in advance.",
             ha="center", va="center", color=MUTED, fontsize=7.2)


# ============================================================== summary page
def page_summary(fig, bullets, heading="KEY STATS & SUMMARY", footnote=None):
    """Wraps explicitly: matplotlib's `wrap=True` ignores the figure margin and
    lets long bullets run off the right edge."""
    import textwrap
    _title(fig, heading, y=0.945, size=30)
    y = 0.882
    for b in bullets:
        lines = textwrap.wrap(b, width=86)
        fig.text(0.075, y, "•", ha="left", va="top", color=TEAM_C,
                 fontsize=13, weight="bold")
        fig.text(0.105, y, "\n".join(lines), ha="left", va="top", color=INK,
                 fontsize=11.2, linespacing=1.5)
        y -= 0.023 + len(lines) * 0.0228
    if footnote:
        fig.text(0.5, 0.038, "\n".join(textwrap.wrap(footnote, width=104)),
                 ha="center", va="bottom", color=MUTED, fontsize=7.6,
                 linespacing=1.5)
