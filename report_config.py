"""Shared style + config for the post-match debrief generator.

Visual language originated from an early Wycombe Wanderers template PDF:
dark navy ground, geometric sans, two team colours used consistently
across every chart.
"""

import matplotlib as mpl
from matplotlib import font_manager

# ---------------------------------------------------------------- match
# Resolved from match.toml, overridable per-run with the matching PMR_* env var.
import os as _os
from envconfig import setting as _s

MATCH_ID = _s("PMR_MATCH_ID", "id", 285457, int)
TEAM = _s("PMR_TEAM", "team", "Leyton Orient")   # the club it's written for
OPP = _s("PMR_OPP", "opponent", "Oxford United")
COMPETITION = _s("PMR_COMP", "competition", "EFL Cup 26/27")
MATCHDAY = _s("PMR_MATCHDAY", "matchday", "First Round")
VENUE_DATE = _s("PMR_VENUE", "venue_date",
                "Brisbane Road · Saturday 8 August 2026")

# ------------------------------------------------------- shared thresholds
# Defined once so the Performance Review and the Coach's Brief cannot reach
# opposite verdicts on the same metric. They did: a ratio-only rule called set
# pieces "a soft spot" on 0.32 xG conceded while the Brief passed the same
# figure against an absolute target.
SETPIECE_XG_CONCEDED_CONCERN = 0.40   # below this, conceding from dead balls
                                      # is not a weakness worth flagging
SETPIECE_XG_CREATED_GOOD = 0.30       # above this, dead balls were a weapon
SETPIECE_XG_CREATED_POOR = 0.20       # below this (over enough phases), the
                                      # attacking return is the real finding

# ---------------------------------------------------------------- colour
BG = "#171E29"          # sampled from the source PDF
PANEL = "#1E2735"
INK = "#FFFFFF"
MUTED = "#8B97A8"
GRID = "#2E3949"

TEAM_C = "#E4002B"      # Leyton Orient red
OPP_C = _s("PMR_OPP_C", "opponent_colour", "#FFDD00")   # Oxford yellow

GOOD = "#8FE388"        # successful (template green)
BAD = "#FF2D2D"         # unsuccessful (template red)
BARC = "#4EB6E3"        # neutral bar blue (template light blue)


def contrast_ink(hex_colour, dark="#141414", light="#FFFFFF"):
    """Pick readable text ink for a coloured band, by relative luminance.

    The cover page's opponent band used fixed dark text, chosen against
    Oxford's bright yellow. It went unreadable the first time an opponent
    colour was dark (Wigan's navy) instead. OPP_C is set per opponent, so the
    text colour on top of it has to adapt rather than assume "light bg".
    """
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return dark if luminance > 140 else light

# page geometry matches the source: 1174x1565px @110dpi
PAGE_W, PAGE_H = 10.67, 14.23
DPI = 150

# ---------------------------------------------------------------- pitch
PITCH_X, PITCH_Y = 52.5, 34.0   # Impect half-extents (105 x 68)


def _pick_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Avenir Next", "Futura", "Helvetica Neue", "Arial", "DejaVu Sans"):
        if name in have:
            return name
    return "sans-serif"


FONT = _pick_font()


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.facecolor": BG,
        "font.family": FONT,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRID,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": False,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "pdf.fonttype": 42,
    })


def surname(full_name: str) -> str:
    """Template labels players by surname only ('Morley', 'Harvie')."""
    if not isinstance(full_name, str) or not full_name.strip():
        return "?"
    parts = full_name.split()
    return parts[-1] if len(parts) > 1 else parts[0]


def draw_pitch(ax, lw=0.9, colour=None, vertical_pad=1.0):
    """Minimal white-line pitch in Impect coordinates, matching the template."""
    c = colour or INK
    ax.set_facecolor(BG)
    X, Y = PITCH_X, PITCH_Y
    r = dict(fill=False, color=c, lw=lw, zorder=1)

    from matplotlib.patches import Rectangle, Circle, Arc
    ax.add_patch(Rectangle((-X, -Y), 2 * X, 2 * Y, **r))
    ax.plot([0, 0], [-Y, Y], color=c, lw=lw, zorder=1)
    ax.add_patch(Circle((0, 0), 9.15, **r))
    ax.plot([0], [0], marker=".", ms=2, color=c, zorder=1)

    for s in (-1, 1):
        # penalty area (16.5m) and six-yard box (5.5m)
        ax.add_patch(Rectangle((s * X - s * 16.5, -20.16), s * 16.5, 40.32, **r))
        ax.add_patch(Rectangle((s * X - s * 5.5, -9.16), s * 5.5, 18.32, **r))
        ax.plot([s * (X - 11)], [0], marker=".", ms=2, color=c, zorder=1)
        ax.add_patch(Arc((s * (X - 11), 0), 18.3, 18.3, angle=0,
                         theta1=128 if s > 0 else -52,
                         theta2=232 if s > 0 else 52, **r))

    ax.set_xlim(-X - 1.5, X + 1.5)
    ax.set_ylim(-Y - vertical_pad, Y + vertical_pad)
    ax.set_aspect("equal")
    ax.axis("off")


def style_bar_axis(ax, ylabel=None, label_size=8.5, tick_size=7.5):
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=tick_size, length=0)
    ax.yaxis.grid(True, color=GRID, ls=(0, (4, 4)), lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_size, color=INK, weight="bold")


def stat_table(fig, rect, rows, left_label, right_label, header=None,
              left_colour=None, right_colour=None):
    """Picture-style stat table: a bold section header, a team-name sub-
    header row, then label / left-value / right-value rows with thin grey
    dividers - replaces a bar-chart comparison. Shared across every report
    in this project (pre-match, post-match) since it has no dependency on
    any one report's own config.

    `rows` is a list of (label, left_str, right_str) or (label, left_str,
    right_str, left_pct, right_pct) - the last two, if given, render as a
    small grey annotation next to the value (e.g. "29%").
    """
    from matplotlib.patches import Rectangle
    ax = fig.add_axes(rect); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    top = 1.0
    if header:
        ax.text(0, top, header, ha="left", va="top", color=INK,
                fontsize=18, weight="bold")
        top -= 0.145
    ax.text(0.64, top, left_label, ha="center", va="top",
            color=left_colour or INK, fontsize=10.5, weight="bold")
    ax.text(0.90, top, right_label, ha="center", va="top",
            color=right_colour or INK, fontsize=10.5, weight="bold")
    top -= 0.075
    ax.plot([0, 1], [top, top], color=GRID, lw=1.1)
    n = len(rows)
    row_h = top / n
    for i, row in enumerate(rows):
        label, lv, rv = row[0], row[1], row[2]
        lpct = row[3] if len(row) > 3 else None
        rpct = row[4] if len(row) > 4 else None
        y = top - (i + 0.5) * row_h
        if i % 2 == 1:
            ax.add_patch(Rectangle((-0.02, top - (i + 1) * row_h), 1.04, row_h,
                                   fc=PANEL, ec="none", zorder=0))
        ax.text(0.0, y, label, ha="left", va="center", color=INK, fontsize=9.6)
        ax.text(0.64, y, str(lv), ha="center", va="center", color=INK,
                fontsize=9.6, weight="bold")
        ax.text(0.90, y, str(rv), ha="center", va="center", color=INK,
                fontsize=9.6, weight="bold")
        if lpct:
            ax.text(0.735, y, lpct, ha="left", va="center", color=MUTED, fontsize=7.4)
        if rpct:
            ax.text(0.995, y, rpct, ha="left", va="center", color=MUTED, fontsize=7.4)
        if i < n - 1:
            liney = top - (i + 1) * row_h
            ax.plot([0, 1], [liney, liney], color=GRID, lw=0.7)


def line_pitch(fig, fontsize, linespacing):
    """Real per-line vertical step for a multi-line fig.text() call, in
    figure-fraction units - for stacking a wrapped paragraph's line count
    into a reserved block of page space (see page_plan() and
    page_performance_review()).

    matplotlib's actual rendered line spacing is the active font's own
    line-height metric times `linespacing`, not fontsize times linespacing
    directly - measured ~37% taller than that naive product for every font
    this project has rendered with (Avenir Next locally, DejaVu Sans in
    Docker). A layout formula using the naive product silently undershoots
    a paragraph's real height, and a long enough paragraph ends up with
    its last line under the next section's header instead of above it -
    confirmed directly on a Bradford City pre-match report (a 10-line
    paragraph) and, less visibly, on a post-match Performance Review page.
    Rendering two real lines and measuring the actual gap between them
    sidesteps ever needing to know that factor in advance, and stays
    correct if the active font changes.
    """
    t = fig.text(0.5, 0.5, "Ag\nAg", fontsize=fontsize, linespacing=linespacing)
    fig.canvas.draw()
    bbox = t.get_window_extent(fig.canvas.get_renderer())
    t.remove()
    return (bbox.height / fig.bbox.height) / 2


# player_table()'s own internal geometry, pulled out to module level so
# required_table_height() can invert the exact same formula rather than a
# second hand-tuned copy that could silently drift out of sync with it.
_PT_HEADER_OFFSET = 0.115   # local-axes fraction used only when header= is given
_PT_COLHEAD_OFFSET = 0.058  # column-label row + divider, always subtracted
_PT_MAX_FONTSIZE = 8.8
_PT_MIN_FONTSIZE = 5.8
_PT_PITCH_RATIO = 1.35      # matches line_pitch()'s own measured ~1.37x factor -
                            # real glyph line-height versus the naive fontsize
                            # product; going tighter than this is what used to
                            # let adjacent rows' text actually touch.


def required_table_height(n, target_fontsize=7.2, header=False):
    """Inverse of player_table()'s own row-height formula: the figure-
    fraction rect height an n-row table needs for every row to render at
    `target_fontsize` instead of being squeezed toward the readability
    floor. Used by page_player_stats_b() to size each row of its grid to
    what its actual roster count needs, rather than forcing every table -
    a 7-player list and a 16-player one alike - into the same fixed box
    (confirmed cramping real matches: FC Walsall/Plymouth/Barnsley/Wigan/
    Luton/Oxford all had a 16-row Passes or Pressures table this season,
    squeezed to the 5.8pt floor under the old fixed-height grid)."""
    n = max(n, 1)
    top = 1.0 - (_PT_HEADER_OFFSET if header else 0.0) - _PT_COLHEAD_OFFSET
    row_h_local = top / n
    row_h_pts_target = target_fontsize * _PT_PITCH_RATIO
    return row_h_pts_target / (row_h_local * PAGE_H * 72)


def player_table(fig, rect, rows, col_labels, header=None):
    """Same picture-style row/divider table as stat_table(), for a list of
    players rather than a two-team comparison. `rows` is a list of tuples
    matching col_labels (first column is always the player name,
    left-aligned; the rest are right-aligned under their header)."""
    from matplotlib.patches import Rectangle
    ax = fig.add_axes(rect); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    top = 1.0
    if header:
        ax.text(0, top, header, ha="left", va="top", color=INK,
                fontsize=13, weight="bold")
        top -= _PT_HEADER_OFFSET
    ncols = len(col_labels)
    col_x = [0.0] + [1.0 - (ncols - 1 - i) * (0.62 / max(ncols - 1, 1))
                     for i in range(1, ncols)]
    for j, lab in enumerate(col_labels):
        ax.text(col_x[j], top, lab, ha=("left" if j == 0 else "right"),
                va="top", color=MUTED, fontsize=8.2, weight="bold")
    top -= _PT_COLHEAD_OFFSET
    ax.plot([0, 1], [top, top], color=GRID, lw=1.0)
    n = len(rows) or 1
    row_h = top / n
    # A longer roster - every player who touched the ball in a given role -
    # shrinks the font to fit instead of stacking rows closer than the text
    # itself is tall, which is what was overlapping adjacent rows. Callers
    # with widely varying roster sizes (page_player_stats_b) should size
    # `rect` itself via required_table_height() first so this floor is
    # rarely the thing actually doing the work.
    row_h_pts = row_h * rect[3] * PAGE_H * 72
    fontsize = min(_PT_MAX_FONTSIZE, max(_PT_MIN_FONTSIZE, row_h_pts / _PT_PITCH_RATIO))
    for i, row in enumerate(rows):
        y = top - (i + 0.5) * row_h
        if i % 2 == 1:
            ax.add_patch(Rectangle((-0.02, top - (i + 1) * row_h), 1.04, row_h,
                                   fc=PANEL, ec="none", zorder=0))
        for j, val in enumerate(row):
            ax.text(col_x[j], y, str(val), ha=("left" if j == 0 else "right"),
                    va="center", color=INK, fontsize=fontsize,
                    weight=("bold" if j == 0 else "normal"))
        if i < n - 1:
            liney = top - (i + 1) * row_h
            ax.plot([0, 1], [liney, liney], color=GRID, lw=0.6)
