"""Page renderers - one function per page of the debrief."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

from report_config import (BG, PANEL, INK, MUTED, GRID, TEAM_C, OPP_C, GOOD, BAD,
                           BARC, PAGE_W, PAGE_H, FONT, draw_pitch, style_bar_axis,
                           surname, contrast_ink)
import report_metrics as M


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
    _title(fig, "MATCH STATS", y=0.955, size=32)
    fig.text(0.085, 0.915, team, ha="left", va="center", color=TEAM_C,
             fontsize=14, weight="bold")
    fig.text(0.915, 0.915, opp, ha="right", va="center", color=OPP_C,
             fontsize=14, weight="bold")

    ax = fig.add_axes([0.07, 0.035, 0.86, 0.855]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    n = len(rows)
    slot = 1.0 / n
    bar_h = slot * 0.34

    for i, (label, lt, rt, lv, rv) in enumerate(rows):
        top = 1 - i * slot
        ax.text(.5, top - slot * 0.30, label, ha="center", va="center",
                color=INK, fontsize=11.5, weight="bold")
        y = top - slot * 0.74
        total = (lv or 0) + (rv or 0)
        frac = 0.5 if total == 0 else (lv or 0) / total
        frac = min(max(frac, 0.035), 0.965)

        ax.add_patch(FancyBboxPatch((0, y - bar_h / 2), frac, bar_h,
                                    boxstyle="round,pad=0,rounding_size=0.012",
                                    fc=TEAM_C, ec="none", mutation_aspect=0.05))
        ax.add_patch(FancyBboxPatch((frac, y - bar_h / 2), 1 - frac, bar_h,
                                    boxstyle="round,pad=0,rounding_size=0.012",
                                    fc=OPP_C, ec="none", mutation_aspect=0.05))

        ax.text(frac - 0.012, y, lt, ha="right", va="center",
                color="#FFFFFF", fontsize=10.5, weight="bold")
        ax.text(frac + 0.012, y, rt, ha="left", va="center",
                color="#111111", fontsize=10.5, weight="bold")


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


BENCH_C = "#FFFFFF"

# Bars are match totals, so a player who was not on for the whole game cannot be
# compared like-for-like. Minutes are shown for anyone who did not play the full
# match, and short appearances are drawn dimmed so the eye does not rank them
# naively against 95-minute starters. Per-90 is deliberately NOT used: one match
# extrapolated x8 turns an 11-minute cameo into a chart-topper.
PART_MATCH_MINS = 60


def _plabels(names, mins=None, full=None):
    """Surname, with minutes appended for anyone short of a full match."""
    out = []
    for n in names:
        s = surname(n)
        if mins is not None and full and n in mins and mins[n] < full * 0.95:
            s = f"{s} {mins[n]:.0f}'"
        out.append(s)
    return out


def _mute(colour, amount=0.58, toward="#5A6676"):
    """Desaturate towards neutral slate.

    Alpha was tried first and looked wrong: fading towards a dark navy ground
    darkens and muddies a saturated colour, so the bar reads as a *different*
    category rather than a quieter version of the same one. Blending towards
    grey keeps the luminance and kills only the saturation.
    """
    from matplotlib.colors import to_rgb
    c, t = np.array(to_rgb(colour)), np.array(to_rgb(toward))
    return tuple(c * (1 - amount) + t * amount)


def _dimmed(names, mins, base):
    """Per-bar colours: full colour for full-match players, muted for cameos."""
    if not mins:
        return base
    muted = _mute(base)
    return [base if mins.get(n, 999) >= PART_MATCH_MINS else muted
            for n in names]


def _draw_bench(ax, values, width=0.72):
    """Season-median tick behind each bar. NaN = not enough appearances."""
    drawn = False
    for i, v in enumerate(values):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        ax.plot([i - width / 2, i + width / 2], [v, v], color=BENCH_C, lw=1.3,
                solid_capstyle="butt", zorder=6, alpha=.95)
        drawn = True
    return drawn


def _stacked_bar(fig, rect, ylabel, labels, ok, bad, bench=None,
                 names=None, mins=None):
    ax = fig.add_axes(rect)
    style_bar_axis(ax, ylabel=ylabel, label_size=8, tick_size=7)
    x = np.arange(len(labels))
    ax.bar(x, ok, color=_dimmed(names, mins, GOOD) if names else GOOD,
           width=.72, zorder=3)
    ax.bar(x, bad, bottom=ok, color=_dimmed(names, mins, BAD) if names else BAD,
           width=.72, zorder=3)
    tot = np.array(ok) + np.array(bad)
    ceiling = max(tot) if len(tot) else 1
    if bench is not None and len(bench) and not np.all(np.isnan(bench)):
        ceiling = max(ceiling, np.nanmax(bench))
    for i, (o, t) in enumerate(zip(ok, tot)):
        if t > 0:
            # clear the season tick as well as the bar, or they collide
            base = t
            if bench is not None and i < len(bench) and not (
                    bench[i] is None or np.isnan(bench[i])):
                base = max(base, bench[i])
            # "3/4" not "75%": with 1-2 attempts a percentage is meaningless,
            # and a 1-of-1 "100%" outranked a 3-of-4 by eye.
            ax.text(i, base + ceiling * .04, f"{int(o)}/{int(t)}", ha="center",
                    va="bottom", fontsize=6.4, color=INK, weight="bold")
    if bench is not None:
        _draw_bench(ax, bench)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.6)
    top = max(tot.max() if len(tot) else 1,
              np.nanmax(bench) if bench is not None and len(bench) else 0)
    ax.set_ylim(0, max(top * 1.24, 1))
    ax.margins(x=0.02)


def page_player_stats_a(fig, ev, team, mins=None, full=None, opp=None):
    _title(fig, "PLAYER STATS")
    if mins:
        fig.text(0.5, 0.928,
                 "bars are match totals · minutes shown for anyone short of a "
                 "full match · faded bars played under 60 minutes",
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
            ax = fig.add_axes([0.345, y, 0.615, h])
            style_bar_axis(ax, ylabel="Box Touches", label_size=8, tick_size=7)
            xs = np.arange(len(cnt))
            names = list(cnt.index)
            ax.bar(xs, cnt.values, color=_dimmed(names, mins, BARC), width=.72,
                   hatch="///", edgecolor="#0E6BA8", zorder=3)
            for i, v in enumerate(cnt.values):
                ax.text(i, v + max(cnt.values) * .05, str(int(v)), ha="center",
                        va="bottom", fontsize=6.8, color=INK, weight="bold")
            ax.set_xticks(xs)
            ax.set_xticklabels(_plabels(names, mins, full), rotation=45,
                               ha="right", fontsize=6.4)
            ax.set_ylim(0, max(cnt.values) * 1.28)
        else:
            g = df.groupby("playerName").agg(
                ok=("result", lambda s: (s == "SUCCESS").sum()),
                tot=("result", "size")).sort_values("tot", ascending=False)
            g["bad"] = g.tot - g.ok
            bar_rect = ([0.435, y, 0.525, h]
                        if title.startswith("Open-Play Crosses") and opp
                        else [0.345, y, 0.615, h])
            _stacked_bar(fig, bar_rect, title,
                         _plabels(list(g.index), mins, full),
                         g.ok.tolist(), g.bad.tolist(),
                         names=list(g.index), mins=mins)


# ============================================================== page 6
def page_player_stats_b(fig, tbl, bench=None, mins=None, full=None):
    _title(fig, "PLAYER STATS")
    if bench is not None and len(bench):
        fig.text(0.5, 0.928,
                 "— white tick = player's season median in matches he played 60+ minutes",
                 ha="center", va="center", color=MUTED, fontsize=7.5)
    else:
        # Silence would read as "no benchmark exists"; say why it is absent.
        fig.text(0.5, 0.928,
                 "bars are match totals · minutes shown for anyone short of a full "
                 "match · faded bars played under 60 minutes\n"
                 "season benchmarks omitted — fewer than 5 matches of this "
                 "season's data, and last season's squad is not a fair comparison",
                 ha="center", va="center", color=MUTED, fontsize=7.0,
                 linespacing=1.6)

    def bvals(names, key):
        if bench is None or len(bench) == 0 or key not in bench.columns:
            return None
        return [bench[key].get(n, np.nan) for n in names]

    def plain(ax_rect, col, label, fmt="{:.1f}", ascending=False, bkey=None):
        d = tbl[["playerName", col]].copy()
        d = d[d[col].notna()]
        d = d[d[col] != 0] if col in ("BPDR", "xG") else d
        d = d.sort_values(col, ascending=ascending)
        ax = fig.add_axes(ax_rect)
        style_bar_axis(ax, ylabel=label, label_size=8, tick_size=7)
        xs = np.arange(len(d))
        names = list(d.playerName)
        ax.bar(xs, d[col].values, color=_dimmed(names, mins, BARC), width=.72,
               hatch="///", edgecolor="#0E6BA8", zorder=3)
        bv = bvals(d.playerName.tolist(), bkey or col)
        span = max(abs(d[col].max()), abs(d[col].min()), 1e-9)
        if bv is not None and len(bv) and not np.all(np.isnan(bv)):
            span = max(span, abs(np.nanmax(bv)))
        for i, v in enumerate(d[col].values):
            base = v
            if bv is not None and i < len(bv) and not (
                    bv[i] is None or np.isnan(bv[i])):
                base = max(v, bv[i]) if v >= 0 else min(v, bv[i])
            ax.text(i, base + (span * .05 if v >= 0 else -span * .05),
                    fmt.format(v), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=6.2,
                    color=INK, weight="bold")
        if bv is not None:
            _draw_bench(ax, bv)
        ax.set_xticks(xs)
        ax.set_xticklabels(_plabels(names, mins, full), rotation=45,
                           ha="right", fontsize=6.2)
        hi = d[col].max()
        if bv is not None and len(bv) and not np.all(np.isnan(bv)):
            hi = max(hi, np.nanmax(bv))
        lo = min(0, d[col].min() * 1.35)
        ax.set_ylim(lo, max(hi * 1.3, 1e-9))

    def duel(ax_rect, won, lost, label, bkey):
        d = tbl[["playerName", won, lost]].copy()
        d["tot"] = d[won] + d[lost]
        d = d[d.tot > 0].sort_values("tot", ascending=False)
        _stacked_bar(fig, ax_rect, label,
                     _plabels(list(d.playerName), mins, full),
                     d[won].tolist(), d[lost].tolist(),
                     bench=bvals(d.playerName.tolist(), bkey),
                     names=list(d.playerName), mins=mins)

    def passes(ax_rect):
        d = tbl[["playerName", "succ", "unsucc"]].copy()
        d["tot"] = d.succ + d.unsucc
        d = d[d.tot > 0].sort_values("tot", ascending=False)
        _stacked_bar(fig, ax_rect, "Passes",
                     _plabels(list(d.playerName), mins, full),
                     d.succ.tolist(), d.unsucc.tolist(),
                     bench=bvals(d.playerName.tolist(), "passes"),
                     names=list(d.playerName), mins=mins)

    L, R, W, H = 0.075, 0.560, 0.385, 0.132
    ys = [0.790, 0.612, 0.434, 0.256, 0.078]

    plain([L, ys[0], W, H], "BPD", "BPD")
    plain([R, ys[0], W, H], "BPDR", "BPDR")
    plain([L, ys[1], W, H], "xT", "xT", fmt="{:.2f}")
    plain([R, ys[1], W, H], "xG", "xG", fmt="{:.2f}")
    passes([L, ys[2], W, H])
    duel([R, ys[2], W, H], "aw", "al", "Aerial Duels", "aerials")
    duel([L, ys[3], W, H], "gw", "gl", "Ground Duels", "grounds")
    plain([R, ys[3], W, H], "BLRT", "BLRT", fmt="{:.1f}", ascending=True)
    plain([L, ys[4], W, H], "pressures", "Pressures", fmt="{:.0f}")
    plain([R, ys[4], W, H], "recoveries", "Ball Recoveries", fmt="{:.0f}")


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

    # ---- where the opposition progressed
    lane = M.opponent_progression(ev, opp)
    fig.text(0.075, 0.492, f"Where {opp} Bypassed Defenders", ha="left",
             va="center", color=INK, fontsize=10.5, weight="bold")
    ax = fig.add_axes([0.075, 0.355, 0.38, 0.120])
    style_bar_axis(ax, ylabel="Opponent BPD", label_size=8, tick_size=7)
    xs = np.arange(len(lane))
    ax.bar(xs, lane.values, color=OPP_C, width=.68, zorder=3)
    for i, v in enumerate(lane.values):
        ax.text(i, v + lane.values.max() * .05, f"{v:.1f}", ha="center",
                va="bottom", fontsize=6.6, color=INK, weight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([l.replace("_", " ").title() for l in lane.index],
                       rotation=35, ha="right", fontsize=6.5)
    ax.set_ylim(0, lane.values.max() * 1.3)

    # ---- conceded shot map
    fig.text(0.545, 0.492, f"Shots Conceded to {opp}", ha="left", va="center",
             color=INK, fontsize=10.5, weight="bold")
    sax = fig.add_axes([0.545, 0.350, 0.40, 0.130])
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
    fig.text(0.745, 0.338,
             f"Conceded xG {tot.SHOT_XG:.2f} | PSxG {tot.POSTSHOT_XG:.2f} "
             f"| stars = goals",
             ha="center", va="center", color=MUTED, fontsize=6.8)

    # ---- player defensive actions
    d = dtbl.sort_values("def_actions", ascending=False)
    d = d[d.def_actions > 0]
    fig.text(0.075, 0.305, f"{team} — Defensive Actions", ha="left",
             va="center", color=INK, fontsize=10.5, weight="bold")
    ax2 = fig.add_axes([0.075, 0.175, 0.38, 0.115])
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
    fig.text(0.545, 0.305, f"{team} — Pressures by Type", ha="left",
             va="center", color=INK, fontsize=10.5, weight="bold")
    ax3 = fig.add_axes([0.545, 0.175, 0.40, 0.115])
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
    fig.text(0.075, 0.130, f"{team} — Ball Win Removed Defenders "
                           f"(turnover quality)", ha="left", va="center",
             color=INK, fontsize=10.5, weight="bold")
    ax4 = fig.add_axes([0.075, 0.055, 0.85, 0.068])
    style_bar_axis(ax4, ylabel="BWRD", label_size=8, tick_size=7)
    xs = np.arange(len(q))
    ax4.bar(xs, q.bwrd.values, color=BARC, width=.66, hatch="///",
            edgecolor="#0E6BA8", zorder=3)
    for i, v in enumerate(q.bwrd.values):
        ax4.text(i, v + max(q.bwrd.max(), .01) * .06, f"{v:.1f}", ha="center",
                 va="bottom", fontsize=6.4, color=INK, weight="bold")
    ax4.set_xticks(xs)
    ax4.set_xticklabels([surname(n) for n in q.playerName], rotation=30,
                        ha="right", fontsize=6.5)
    ax4.set_ylim(0, max(q.bwrd.max() * 1.35, 1))
    fig.text(0.5, 0.022,
             "How many opponent defenders were taken out of the game by each "
             "player's ball wins — where a turnover happens matters more than "
             "how many you make. Players with none are omitted.",
             ha="center", va="center", color=MUTED, fontsize=6.8)


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
def page_performance_review(fig, pros, cons, team, opp, score_line):
    """Two-column verdict. Deliberately words, not figures - the eight pages
    before this carry the numbers."""
    import textwrap
    _title(fig, "PERFORMANCE REVIEW", y=0.955, size=30)
    fig.text(0.5, 0.917, score_line, ha="center", va="center", color=MUTED,
             fontsize=11)

    cols = [(0.055, "WHAT WORKED", GOOD, pros),
            (0.515, "WHAT NEEDS WORK", BAD, cons)]
    CW = 0.430

    for x0, title, accent, items in cols:
        # column header
        ax = fig.add_axes([x0, 0.855, CW, 0.038]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.add_patch(FancyBboxPatch((0, 0.05), 1, 0.9,
                                    boxstyle="round,pad=0,rounding_size=0.03",
                                    fc=accent, ec="none", mutation_aspect=0.12))
        ax.text(0.5, 0.5, title, ha="center", va="center", color="#10161F",
                fontsize=12.5, weight="bold")

        n = max(len(items), 1)
        top, gap = 0.836, 0.010
        h = min(0.118, (top - 0.055 - gap * (n - 1)) / n)
        for i, (headline, detail) in enumerate(items):
            y = top - (i + 1) * h - i * gap
            cax = fig.add_axes([x0, y, CW, h]); cax.axis("off")
            cax.set_xlim(0, 1); cax.set_ylim(0, 1)
            cax.add_patch(FancyBboxPatch((0, 0), 1, 1,
                                         boxstyle="round,pad=0,rounding_size=0.02",
                                         fc=PANEL, ec=GRID, lw=0.8,
                                         mutation_aspect=0.30))
            # accent spine so the two columns read apart at a glance
            cax.add_patch(Rectangle((0, 0.06), 0.012, 0.88, fc=accent, ec="none"))
            cax.text(0.045, 0.87, headline, ha="left", va="top", color=INK,
                     fontsize=9.4, weight="bold")
            body = "\n".join(textwrap.wrap(detail, width=50))
            cax.text(0.045, 0.63, body, ha="left", va="top", color="#B9C4D2",
                     fontsize=8.0, linespacing=1.45)

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
