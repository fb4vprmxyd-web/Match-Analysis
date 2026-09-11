"""Coach's Brief - grade a match against the staff's pre-match instructions.

The brief separates two questions that post-match reviews usually blur:

    COMPLIANCE     did the side do what was asked?
    EFFECTIVENESS  did doing it actually produce anything?

A side can be fully compliant and still lose; judging only on outcome bins a
good plan after one bad afternoon.

TARGETS below are PROVISIONAL. They should be replaced with the numbers the
coaching staff give you BEFORE kick-off - thresholds chosen after seeing the
result are marking your own homework.

This file's checks are written against the specific notes given for the
Leyton Orient v Bradford City brief (out of possession: pressure the ball to
limit forward play, force it wide when they do go forward, track their
forward runs, close distances when pressing to block lines and win 2nd
balls, do our work early off the ball with a coordinated shift; in
possession: overload their midfield two through the centre, target Hayden
Matthews directly, get the 10s receiving around their box, limit backward/
sideways play across our own back line since that's what triggers their
press).

"Play onto Matthew's" refers to Bradford's Hayden Matthews (their own
centre-back that match) - confirmed with the user rather than assumed, since
there is no Matthews on our own roster and guessing wrong here would have
made the whole check meaningless.
"""

import numpy as np
import pandas as pd

from report_config import SETPIECE_XG_CONCEDED_CONCERN

# --------------------------------------------------------------- targets
TARGETS = {
    "press_limit_forward": {"basis": "beat_opponent", "lower_better": True, "margin": 1.0},
    "force_wide":          {"basis": "absolute", "target": 0.30, "lower_better": True},
    "second_balls":        {"basis": "beat_opponent", "margin": 2},
    "early_engagement":    {"basis": "beat_opponent", "margin": 0.05},
    "overload_middle":     {"basis": "absolute", "target": 0.30},
    "tens_in_box":         {"basis": "beat_opponent", "margin": 1},
    "limit_backward":      {"basis": "absolute", "target": 20, "lower_better": True},
    "turnover_quality":    {"basis": "beat_opponent", "margin": 1.0},
    "xg":                  {"basis": "beat_opponent", "margin": 0.20},
    "setpiece_conceded": {"basis": "absolute",
                          "target": SETPIECE_XG_CONCEDED_CONCERN,
                          "lower_better": True},
}

GREEN, AMBER, RED, INFO, NA = "GREEN", "AMBER", "RED", "INFO", "N/A"

BACK_POS = ("CENTRAL_DEFENDER", "LEFT_WINGBACK_DEFENDER", "RIGHT_WINGBACK_DEFENDER",
           "LEFT_FULLBACK_DEFENDER", "RIGHT_FULLBACK_DEFENDER", "GOALKEEPER")
TEN_POS = ("ATTACKING_MIDFIELD",)


# --------------------------------------------------------------- grading
def _grade(key, value, other=None):
    cfg = TARGETS.get(key, {"basis": "info"})
    basis = cfg["basis"]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return NA
    if basis == "info":
        return INFO
    lower = cfg.get("lower_better", False)

    if basis == "absolute":
        t = cfg["target"]
        good = value <= t if lower else value >= t
        near = abs(value - t) <= abs(t) * 0.15
        return GREEN if good else (AMBER if near else RED)

    if basis == "beat_opponent":
        if other is None:
            return NA
        m = cfg.get("margin", 0)
        if lower:
            return GREEN if value <= other - m else (
                AMBER if value <= other + m else RED)
        return GREEN if value >= other + m else (
            AMBER if value >= other - m else RED)
    return INFO


def _row(section, instruction, metric, readout, verdict):
    return {"section": section, "instruction": instruction, "metric": metric,
            "readout": readout, "verdict": verdict}


# --------------------------------------------------------------- helpers
def _position_map(psum, squad):
    return (psum[psum.squadName == squad].groupby("playerName")["position"]
            .agg(lambda s: s.value_counts().index[0]).to_dict())


def _lane_share(ev, squad, lane):
    """Share of a squad's own progression that goes through a given lane -
    lane is in that squad's own attacking frame, no mirroring needed since
    both checks here (force them wide, overload the centre) are about a
    squad's own output, not what its opponents did against it."""
    d = ev[(ev.squadName == squad) & (ev.BYPASSED_DEFENDERS > 0)]
    tot = d.BYPASSED_DEFENDERS.sum()
    if tot <= 0:
        return None, 0
    return float(d[d.startLane == lane].BYPASSED_DEFENDERS.sum() / tot), float(tot)


def _balls_in_behind(ev, squad):
    """Receptions that got beyond the last line, near goal - restricted to
    the final third since AVAILABILITY_IN_THE_BACK otherwise fires anywhere
    on the pitch - plus AVAILABILITY_IN_THE_BOX."""
    d = ev[ev.squadName == squad]
    behind = d[(d.action == "AVAILABILITY_IN_THE_BACK")
              & (d.startAdjCoordinatesX > 17.5)]
    in_box = d[d.action == "AVAILABILITY_IN_THE_BOX"]
    return len(behind) + len(in_box), len(behind), len(in_box)


def _second_balls_won(ev, squad):
    d = ev[(ev.squadName == squad) & (ev.phase == "SECOND_BALL")
           & (ev.actionType == "LOOSE_BALL_REGAIN")]
    return len(d)


def _box_touches_by_position(ev, psum, squad, positions):
    pos = _position_map(psum, squad)
    d = ev[(ev.squadName == squad)
           & ev.playerName.map(lambda n: pos.get(n, "") in positions)
           & (ev.startPitchPosition == "OPPONENT_BOX") & (ev.OFFENSIVE_TOUCHES > 0)]
    return len(d)


def _player_contests(ev, squad, opp_player):
    """Direct 1v1s against a named opponent - ground/aerial duels plus
    dribbles faced, combined, since 'play onto him' means isolating him in
    individual contests regardless of which specific type it takes."""
    d = ev[(ev.squadName == squad) & ev.actionType.isin(("GROUND_DUEL", "AERIAL_DUEL"))
           & (ev.duelPlayerName == opp_player)]
    dr = ev[(ev.squadName == squad) & (ev.actionType == "DRIBBLE")
           & (ev.dribbleOpponentPlayerName == opp_player)]
    won = int((d.result == "SUCCESS").sum()) + int((dr.result == "SUCCESS").sum())
    return len(d) + len(dr), won


def _backline_recycling(ev, psum, squad):
    """Completed passes between two back-line players (incl. the keeper)
    that go backward or sideways - the literal 'playing backwards and
    sideways across the defence' the instruction names, not just any
    backward pass by anyone."""
    pos = _position_map(psum, squad)
    d = ev[(ev.squadName == squad) & (ev.actionType == "PASS") & (ev.result == "SUCCESS")]
    d = d[d.playerName.map(lambda n: pos.get(n, "") in BACK_POS)
         & d.passReceiverPlayerName.map(lambda n: pos.get(n, "") in BACK_POS)]
    back = d[d.endAdjCoordinatesX <= d.startAdjCoordinatesX]
    return len(back), len(d)


# --------------------------------------------------------------- evaluate
def evaluate(ev, psum, squad_sums, scores, setp, formations, team, opp):
    """Grade the match against the brief. Returns a list of check rows."""
    rows = []
    S = squad_sums.set_index("squadName") if squad_sums is not None else None
    C = scores.set_index("squadName") if scores is not None else None

    def sc(col, sq):
        if C is None or col not in C.columns:
            return None
        return float(C.loc[sq, col])

    def ss(col, sq):
        if S is None or col not in S.columns:
            return None
        return float(S.loc[sq, col])

    # ============================ OUT OF POSSESSION ============================
    sec = "OUT OF POSSESSION"

    v, o = sc("PPDA", team), sc("PPDA", opp)
    rows.append(_row(sec, "Pressure the ball to limit forward play",
                     "PPDA (lower = more press)",
                     f"{v:.1f}  (opp {o:.1f})" if v is not None else "-",
                     _grade("press_limit_forward", v, o)))

    share, tot = _lane_share(ev, opp, "CENTER")
    rows.append(_row(sec, "If they go forward, force it wide",
                     "Opp progression through the centre",
                     f"{share*100:.0f}% of {tot:.1f} bypassed defenders"
                     if share is not None else "-",
                     _grade("force_wide", share)))

    v, behind, in_box = _balls_in_behind(ev, opp)
    rows.append(_row(sec, "Track their forward runs",
                     "Opp receptions beyond our last line, final third + box",
                     f"{v} ({behind} in behind, {in_box} in the box)", INFO))

    sb, osb = _second_balls_won(ev, team), _second_balls_won(ev, opp)
    rows.append(_row(sec, "Close distances, block lines, win 2nd balls",
                     "2nd balls won", f"{sb}  (opp {osb})",
                     _grade("second_balls", sb, osb)))

    v, o = sc("PRESSURE_DURING_OPP_BUILD_UP_PERCENT", team), sc("PRESSURE_DURING_OPP_BUILD_UP_PERCENT", opp)
    rows.append(_row(sec, "Early work off the ball, shift and move",
                     "% of opp build-up pressed",
                     f"{v*100:.0f}%  (opp {o*100:.0f}%)" if v is not None else "-",
                     _grade("early_engagement", v, o)))

    # ============================== IN POSSESSION ==============================
    sec = "IN POSSESSION"

    share, tot = _lane_share(ev, team, "CENTER")
    rows.append(_row(sec, "Overload their midfield two, through the middle",
                     "Our progression through the centre",
                     f"{share*100:.0f}% of {tot:.1f} bypassed defenders"
                     if share is not None else "-",
                     _grade("overload_middle", share)))

    n, won = _player_contests(ev, team, "Hayden Matthews")
    rows.append(_row(sec, "Play onto Matthews",
                     "Direct 1v1s vs Matthews (duels + dribbles)",
                     f"{won} of {n} won" if n else "0 direct contests recorded", INFO))

    v = _box_touches_by_position(ev, psum, team, TEN_POS)
    o = _box_touches_by_position(ev, psum, opp, TEN_POS)
    rows.append(_row(sec, "10s receiving around their box",
                     "Box touches by our attacking mids", f"{v}  (opp {o})",
                     _grade("tens_in_box", v, o)))

    back, tot_bl = _backline_recycling(ev, psum, team)
    rows.append(_row(sec, "Limit backwards/sideways play across the defence",
                     "Back-line backward/sideways completions",
                     f"{back} of {tot_bl} back-line passes",
                     _grade("limit_backward", back)))

    # =============================== DID IT WORK ===============================
    sec = "DID IT WORK"

    v, o = (ss("BALL_WIN_REMOVED_OPPONENTS_DEFENDERS", team),
            ss("BALL_WIN_REMOVED_OPPONENTS_DEFENDERS", opp))
    rows.append(_row(sec, "…and those turnovers hurt them", "Ball win removed defenders",
                     f"{v:.1f}  (opp {o:.1f})" if v is not None else "-",
                     _grade("turnover_quality", v, o)))

    v, o = ss("SHOT_XG", team), ss("SHOT_XG", opp)
    rows.append(_row(sec, "The plan created chances", "xG",
                     f"{v:.2f}  (opp {o:.2f})" if v is not None else "-",
                     _grade("xg", v, o)))

    if setp is not None and len(setp):
        a = setp[setp.attackingSquadName == opp]["setPieceSubPhase_SHOT_XG"].sum()
        rows.append(_row(sec, "Set pieces didn't cost us", "Set-piece xG conceded",
                         f"{a:.2f} xG", _grade("setpiece_conceded", float(a))))

    return rows
