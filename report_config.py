"""Shared style + config for the post-match debrief generator.

Visual language is copied from the Wycombe Wanderers template
(`Post Match Report v Leyton Orient.pdf`): dark navy ground, geometric
sans, two team colours used consistently across every chart.
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
