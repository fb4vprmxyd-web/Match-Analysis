"""One-off surgical patch: regenerates ONLY the "KEY STATS & SUMMARY" page
(page index 8) of an already-shipped post-match report and splices it back
into the existing PDF in place, leaving every other page - including any
real Coach's Brief - byte-for-byte untouched.

Why not just rebuild the whole report? Four already-shipped reports
(Barnsley, Wigan, Burton Albion, Sheffield Wednesday) predate the fix for
the self-inclusion ranking bug in build_report.py's season_context() (a
match's own row in the season pool could beat itself on a floating-point
technicality and inflate its own rank past the pool size - see that
function's docstring). That bug lives ENTIRELY in season_context(), which
only feeds the summary page - performance_review() and coach_brief.py's
CB.evaluate() don't touch the season pool at all, so they were never
affected. Wigan, Burton, and Sheffield Wednesday's Coach's Brief pages are
real, specific, dictated notes (verified by reading each one before writing
this script) - coach_brief.py is a single rotating config, so a full
rebuild would silently grade whichever match is being rebuilt against
today's resting config (currently Bradford's) instead of that match's own
real notes, destroying them. Patching just the one broken page sidesteps
that risk entirely instead of re-running the Oxford/Sheffield-Wednesday-
style reconstruction three more times for a bug that never touched the
Brief page to begin with.

Run in an isolated subprocess per match, same reasoning as
_golden_runner.py: build_report.py resolves TEAM/OPP/CACHE from the
environment only on first import.

    python3 _patch_summary.py <match_id> <team> <opp> <cache> <comp> \\
        <matchday> <venue> <opp_c> <target_pdf>
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")


def main():
    (match_id, team, opp, cache, comp, matchday, venue, opp_c,
     target_pdf) = sys.argv[1:10]
    os.environ["PMR_MATCH_ID"] = match_id
    os.environ["PMR_TEAM"] = team
    os.environ["PMR_OPP"] = opp
    os.environ["PMR_CACHE"] = cache
    os.environ["PMR_COMP"] = comp
    os.environ["PMR_MATCHDAY"] = matchday
    os.environ["PMR_VENUE"] = venue
    os.environ["PMR_OPP_C"] = opp_c

    import fitz
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import build_report as BR
    import report_metrics as M
    import report_pages as P
    from report_config import apply_style

    (ev, ps, season_sq, season_pl, shirts, setp, squad_sums, scores,
     formations) = BR.load()
    tbl = M.player_table(ev, ps, team)
    bullets_raw = BR.season_context(ev, ps, season_sq, season_pl, tbl, setp,
                                    squad_sums, scores)
    bullets, heading, footnote = BR._split_footnote(bullets_raw)

    apply_style()
    fig = plt.figure(figsize=(10.67, 14.23), facecolor="#171E29")
    P.page_summary(fig, bullets, heading, footnote)

    tmp_path = target_pdf + ".patch_page.pdf"
    fig.savefig(tmp_path, facecolor="#171E29")
    plt.close(fig)

    doc = fitz.open(target_pdf)
    assert "KEY STATS" in doc[8].get_text(), (
        f"page 8 of {target_pdf!r} is not the summary page - refusing to "
        f"touch it (starts: {doc[8].get_text()[:60]!r})")
    doc.delete_page(8)
    patch = fitz.open(tmp_path)
    doc.insert_pdf(patch, from_page=0, to_page=0, start_at=8)
    doc.save(target_pdf + ".tmp", garbage=4, deflate=True)
    doc.close()
    patch.close()
    os.replace(target_pdf + ".tmp", target_pdf)
    os.remove(tmp_path)
    print(f"patched: {target_pdf}")


if __name__ == "__main__":
    main()
