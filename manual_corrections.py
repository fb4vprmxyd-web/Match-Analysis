"""Manual, video-verified corrections to Impect's automated event tagging.

Impect groups a "set-piece phase" from the restart until its own definition of
the phase ending, which can span well past the point where the passage of play
has become normal open football. This has already produced one bad call: a
penalty that Impect attributes to a THROW_IN phase.

Every entry here is a deliberate, reviewed override of the data - not a
tuning parameter. Add one only after checking the footage, and say what you
saw. These apply to ALL set-piece-derived numbers for the match (Performance
Review, Coach's Brief, the Set Pieces page, the summary bullet), not just one
page, because leaving the same misattributed event correct on one page and
wrong on another would be worse than the original problem.

This does NOT change how set pieces are judged for any other match - only
these named phases, on these named matches, are excluded.
"""

# matchId -> [(setPieceId, reason), ...]
SETPIECE_EXCLUSIONS = {
    270229: [  # Leyton Orient v Sheffield Wednesday, 15 Aug 2026
        (90185897,
         "Sheffield Wednesday's 65' penalty (Jamal Lowe). Impect files it "
         "under a THROW_IN phase (restart 63:43), but the throw-in itself "
         "was 13.9s and several actions before the foul that won the "
         "penalty (a save, a dribble, multiple passes intervened) - "
         "confirmed against the footage. Genuine open play, not a set-piece "
         "delivery being punished. setPieceId was 90099794 until this "
         "match's data was revised partway through this project - the "
         "underlying event is the same penalty, Impect just re-numbered it "
         "on re-pull, and the old id silently stopped matching anything."),
    ],
}


def apply(setp, match_id: int):
    """Return `setp` with any reviewed phases removed for this match.

    Silently returns `setp` unchanged if there is nothing to correct or the
    dataframe is empty, so this is always safe to call.
    """
    if setp is None or len(setp) == 0:
        return setp
    exclusions = SETPIECE_EXCLUSIONS.get(int(match_id), [])
    if not exclusions:
        return setp
    out = setp.copy()
    for phase_id, reason in exclusions:
        hit = out.setPieceId == phase_id
        if hit.any():
            print(f"  manual correction: excluding set-piece phase "
                 f"{phase_id} from match {match_id} - {reason}")
            out = out[~hit]
        else:
            print(f"  ! manual correction for match {match_id} did not "
                 f"match any row (phase {phase_id}) - check it is still "
                 f"needed")
    return out
