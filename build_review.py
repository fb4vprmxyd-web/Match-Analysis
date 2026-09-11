"""Build a short review PDF containing only the pages the coach asked about.

    python3 build_review.py

Produces a 4-page document: a summary of what changed, then only the pages that
changed. Nothing is recomputed - pages are lifted from the full report, so the
two can never disagree.
"""

import os
import textwrap
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.backends.backend_pdf import PdfPages
import fitz

warnings.filterwarnings("ignore")

from report_config import (apply_style, BG, PANEL, INK, MUTED, GRID, TEAM_C,
                           GOOD, BARC, PAGE_W, PAGE_H, TEAM, OPP)

SOURCE = os.environ.get("PMR_OUT", f"Post Match Report - {TEAM} v {OPP}.pdf")
OUT = f"Post Match Report - {TEAM} v {OPP} - CHANGES.pdf"

# 0-indexed pages of the full report to carry across, in order
PAGES = [3, 6, 7]          # passing networks · player stats A · player stats B

CHANGES = [
    {
        "note": "“I think we can change some of the player stats and set "
                "piece data from percentages to totals.”",
        "done": "Success labels now read 6/10 rather than 60%.",
        "why": "On the crossing and passes-into-the-box charts, 10 of 11 "
               "players had two attempts or fewer, so the labels were almost "
               "all 0% or 100% — and a 1-of-1 “100%” outranked a "
               "3-of-4 by eye. Showing the count keeps the denominator "
               "visible. The bar height already gives the attempts.",
        "where": "Pages 3 and 4 — every bar label. Also the set-piece "
                 "takers chart in the full report.",
    },
    {
        "note": "“Did you look into pass networks, starting 11, after "
                "subs etc?”",
        "done": "Split into two networks: the starting XI, and the side after "
                "the changes.",
        "why": "There was one network built from all 16 players over the whole "
               "match — a starting XI that played 69 minutes plotted "
               "alongside substitutes who played 20, as though their average "
               "positions were comparable. They are now separated, with the "
               "substitutions listed underneath so the shape change is legible.",
        "where": "Page 2 — Starting XI 0–69' (216 passes) and After "
                 "the changes 69'–FT (56 passes).",
    },
    {
        "note": "“Is it possible to show crosses from both, whether it’s "
                "a visual or just a number?”",
        "done": "A separate map for each team, with the left/right split in numbers.",
        "why": "Leyton Orient 17 crosses, 2 completed, L 9 / R 8. Oxford United "
               "7 crosses, 1 completed, L 2 / R 5. Orient crossed evenly from "
               "both flanks; Oxford leaned right. Neither side could cross — "
               "3 completions from 24 between them.",
        "where": "Page 3, third row — one pitch per team. Arrows are green "
                 "for completed and red for not on both, so the two are "
                 "directly comparable.",
    },
]


def summary_page(fig):
    fig.text(0.5, 0.955, "WHAT'S CHANGED", ha="center", va="center", color=INK,
             fontsize=30, weight="bold")
    fig.text(0.5, 0.919, f"{TEAM} v {OPP}  ·  post-match report",
             ha="center", va="center", color=MUTED, fontsize=11)
    fig.text(0.5, 0.893, "Three changes, from your notes. Only the pages that "
                         "changed are included.",
             ha="center", va="center", color=MUTED, fontsize=8.5)

    top, h, gap = 0.866, 0.248, 0.024
    for i, c in enumerate(CHANGES):
        y = top - (i + 1) * h - i * gap
        ax = fig.add_axes([0.055, y, 0.89, h]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1,
                                    boxstyle="round,pad=0,rounding_size=0.015",
                                    fc=PANEL, ec=GRID, lw=0.9,
                                    mutation_aspect=0.28))
        ax.add_patch(Rectangle((0, 0.05), 0.005, 0.90, fc=GOOD, ec="none"))
        ax.text(0.024, 0.93, f"{i + 1}", ha="left", va="top", color=GOOD,
                fontsize=14, weight="bold")

        ax.text(0.058, 0.93, "\n".join(textwrap.wrap(c["note"], 86)),
                ha="left", va="top", color=MUTED, fontsize=9.5,
                style="italic", linespacing=1.4)
        ax.text(0.058, 0.70, c["done"], ha="left", va="top", color=INK,
                fontsize=11.5, weight="bold")
        ax.text(0.058, 0.555, "\n".join(textwrap.wrap(c["why"], 96)),
                ha="left", va="top", color="#B9C4D2", fontsize=8.8,
                linespacing=1.5)
        # a pointer, not a warning - red on the dark panel read as an error
        ax.text(0.058, 0.185, "See:  " + " ".join(c["where"].split()),
                ha="left", va="top", color=BARC, fontsize=8.6,
                weight="bold")

    fig.text(0.5, 0.028,
             "Pages are lifted directly from the full report, so the two cannot "
             "disagree.", ha="center", va="center", color=MUTED, fontsize=7.4)


def main():
    if not os.path.exists(SOURCE):
        raise SystemExit(f"Full report not found: {SOURCE}\n"
                         f"Run build_report.py first.")
    apply_style()

    tmp = ".changes_summary.pdf"
    with PdfPages(tmp) as pdf:
        fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor=BG)
        summary_page(fig)
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

    out = fitz.open()
    cover = fitz.open(tmp)
    out.insert_pdf(cover)
    cover.close()

    src = fitz.open(SOURCE)
    for i in PAGES:
        out.insert_pdf(src, from_page=i, to_page=i)
    src.close()

    out.save(OUT)
    out.close()
    os.remove(tmp)
    print(f"wrote {OUT}  ({len(PAGES) + 1} pages)")


if __name__ == "__main__":
    main()
