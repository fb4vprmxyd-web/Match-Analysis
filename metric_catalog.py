"""The fixed catalog of Coach's Brief metrics.

Every check ever built by hand this project - Bradford's, Bromley's,
Sheffield Wednesday's real briefs - turns out to be one of a small number of
underlying computations, just pointed at different lanes, players, or
zones. This module is that underlying set, pulled out of one-off per-match
code and made reusable, so a check gets built by SELECTING and
PARAMETERISING an entry here, not by writing new pandas each time.

This is the safety rail behind letting a coach describe a check in free
text: whatever turns his sentence into a structured check (today, a human;
later, an LLM - see coach_brief_engine.py) can only pick from what's
defined here. It can get the choice of metric or parameters wrong, and the
confirm-readback step exists to catch that - but it cannot invent a new,
unverified computation on the fly. Every entry below has already been
checked against real match data at least once (see the git history of
coach_brief.py for Bradford/Bromley/Sheffield Wednesday's reconstructions).

Each MetricSpec's `compute(ev, psum, squad_sums, scores, setp, squad,
**params)` returns a dict of named raw values for ONE squad in ONE match -
deliberately not "for this match's opponent" or "vs a baseline", so the same
function works whether it's being evaluated for the current match, a
previous match, or averaged across a season sample; comparison (beat the
opponent / hit a fixed number / beat a baseline) is a separate, later step
(see coach_brief_engine.py's _grade(), already proven across three real
briefs). `grading_key` names which entry in that dict is the number a
comparison actually grades on; `readout_template` is a str.format() string
over the same dict, used to render the actual page text.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

FINAL_THIRD = ("FINAL_THIRD", "OPPONENT_BOX")
CROSS_ACTIONS = ("LOW_CROSS", "HIGH_CROSS")
LEFT_LANES = ("LEFT_WING", "LEFT_HALF_SPACE")
RIGHT_LANES = ("RIGHT_WING", "RIGHT_HALF_SPACE")
MIRROR = {"LEFT_WING": "RIGHT_WING", "RIGHT_WING": "LEFT_WING",
         "LEFT_HALF_SPACE": "RIGHT_HALF_SPACE",
         "RIGHT_HALF_SPACE": "LEFT_HALF_SPACE", "CENTER": "CENTER"}
FB_POS = ("LEFT_WINGBACK_DEFENDER", "RIGHT_WINGBACK_DEFENDER",
         "LEFT_FULLBACK_DEFENDER", "RIGHT_FULLBACK_DEFENDER")
DEF_MID_POS = ("CENTRAL_DEFENDER", "DEFENSE_MIDFIELD", "CENTRAL_MIDFIELD")
BACK_POS = ("CENTRAL_DEFENDER", "LEFT_WINGBACK_DEFENDER", "RIGHT_WINGBACK_DEFENDER",
           "LEFT_FULLBACK_DEFENDER", "RIGHT_FULLBACK_DEFENDER", "GOALKEEPER")
TEN_POS = ("ATTACKING_MIDFIELD",)
LANE_CHOICES = ("LEFT_WING", "LEFT_HALF_SPACE", "CENTER",
                "RIGHT_HALF_SPACE", "RIGHT_WING")


def _position_map(psum, squad):
    return (psum[psum.squadName == squad].groupby("playerName")["position"]
            .agg(lambda s: s.value_counts().index[0]).to_dict())


@dataclass
class Param:
    name: str
    label: str
    kind: str            # "lane", "lanes", "player_name", "position_group", "zone"
    choices: tuple = None  # fixed choices, e.g. LANE_CHOICES; None = free (player name)


@dataclass
class MetricSpec:
    key: str
    label: str
    description: str
    unit: str             # "count", "percent", "xg", "metres"
    grading_key: str
    readout_template: str
    compute: Callable
    params: list = field(default_factory=list)


CATALOG: dict[str, MetricSpec] = {}


def register(spec: MetricSpec):
    CATALOG[spec.key] = spec
    return spec


# ------------------------------------------------------------ squad_scores
# Straight lookups from getSquadMatchScores - already matchsum-published,
# no derivation needed. `col` is fixed per entry (not a param) since each
# one is a genuinely distinct instruction, not a parameterised family.
def _score_lookup(col):
    def _compute(ev, psum, squad_sums, scores, setp, squad, **_):
        C = scores.set_index("squadName") if scores is not None else None
        if C is None or col not in C.columns or squad not in C.index:
            return {"value": None}
        return {"value": float(C.loc[squad, col])}
    return _compute


register(MetricSpec("ppda", "PPDA (press intensity)",
    "Passes allowed per defensive action - lower means more aggressive pressing.",
    "count", "value", "{value:.1f}", _score_lookup("PPDA")))

register(MetricSpec("press_height", "Mean press height",
    "Average distance upfield (metres) where defensive actions happen.",
    "metres", "value", "{value:.0f}m", _score_lookup("MEAN_PRESSURE_HEIGHT")))

register(MetricSpec("opp_buildup_pressed", "% of opponent build-up pressed",
    "Share of the opponent's build-up phases that were pressed.",
    "percent", "value", "{value:.0%}", _score_lookup("PRESSURE_DURING_OPP_BUILD_UP_PERCENT")))

register(MetricSpec("forced_high_pass_pct", "Forced-high-pass rate",
    "Share of a side's passes forced long/high under pressure - higher means "
    "more disrupted build-up.", "percent", "value", "{value:.0%}",
    _score_lookup("FORCED_HIGH_PASSES_PERCENT")))

register(MetricSpec("possession", "Possession share",
    "Share of match time in possession.", "percent", "value", "{value:.0%}",
    _score_lookup("RATIO_BALL_POSSESSION")))

register(MetricSpec("pass_accuracy", "Pass accuracy",
    "Impect's own published pass-accuracy figure for the match.",
    "percent", "value", "{value:.0%}", _score_lookup("RATIO_PASSING_ACCURACY")))


# --------------------------------------------------------------- squad_sums
def _sum_lookup(col):
    def _compute(ev, psum, squad_sums, scores, setp, squad, **_):
        S = squad_sums.set_index("squadName") if squad_sums is not None else None
        if S is None or col not in S.columns or squad not in S.index:
            return {"value": None}
        return {"value": float(S.loc[squad, col])}
    return _compute


register(MetricSpec("xg", "xG", "Expected goals from open play and set pieces.",
    "xg", "value", "{value:.2f}", _sum_lookup("SHOT_XG")))

register(MetricSpec("bwrd", "Ball win removed defenders",
    "How many opposition defenders a side's turnovers took out of the game - "
    "quality of ball-winning, not just volume.", "count", "value", "{value:.1f}",
    _sum_lookup("BALL_WIN_REMOVED_OPPONENTS_DEFENDERS")))

register(MetricSpec("bpd", "Bypassed defenders", "Team progression total.",
    "count", "value", "{value:.1f}", _sum_lookup("BYPASSED_DEFENDERS")))

register(MetricSpec("shot_creating_actions", "Shot-creating actions",
    "Actions that directly led to a shot.", "count", "value", "{value:.0f}",
    _sum_lookup("SHOT_CREATING_ACTIONS")))

register(MetricSpec("box_touches_matchsum", "Box touches",
    "Touches inside the opponent's box (matchsum figure).", "count", "value",
    "{value:.0f}", _sum_lookup("OFFENSIVE_TOUCHES_IN_PITCH_POSITION_OPPONENT_BOX")))

register(MetricSpec("box_passes_matchsum", "Successful passes into the box",
    "Completed passes ending inside the opponent's box (matchsum figure).",
    "count", "value", "{value:.0f}",
    _sum_lookup("SUCCESSFUL_PASSES_TO_PITCH_POSITION_OPPONENT_BOX")))


# ---------------------------------------------------------------- lane-based
def _lane_share_compute(ev, psum, squad_sums, scores, setp, squad, lanes=(), **_):
    lanes = lanes if isinstance(lanes, (list, tuple)) else (lanes,)
    d = ev[(ev.squadName == squad) & (ev.BYPASSED_DEFENDERS > 0)]
    tot = d.BYPASSED_DEFENDERS.sum()
    if tot <= 0:
        return {"value": None, "total": 0.0}
    share = float(d[d.startLane.isin(lanes)].BYPASSED_DEFENDERS.sum() / tot)
    return {"value": share, "total": float(tot)}


register(MetricSpec("lane_share", "Progression through a lane/lanes",
    "Share of a side's own progression (bypassed defenders) that goes through "
    "the given lane(s) of THEIR OWN attacking frame.", "percent", "value",
    "{value:.0%} of {total:.1f} bypassed defenders", _lane_share_compute,
    params=[Param("lanes", "Lane(s)", "lanes", LANE_CHOICES)]))


def _mirrored_ball_wins_compute(ev, psum, squad_sums, scores, setp, squad,
                                opp_side_lanes=(), **_):
    raw = (opp_side_lanes if isinstance(opp_side_lanes, (list, tuple))
          else (opp_side_lanes,))
    opp_side_lanes = tuple(
        lane for item in raw
        for lane in (item.split(",") if isinstance(item, str) else (item,))
    )
    own_lanes = tuple(MIRROR[l] for l in opp_side_lanes)
    d_all = ev[(ev.squadName == squad) & (ev.BALL_WIN_NUMBER > 0)]
    d = d_all[d_all.startLane.isin(own_lanes)]
    n, tot = len(d), len(d_all)
    return {"value": (n / tot) if tot else None, "n": n, "total": tot}


register(MetricSpec("mirrored_ball_wins", "Ball wins on a named opponent flank",
    "Own ball wins in the lane(s) that mirror onto a named side of the "
    "OPPONENT's shape - their left is our right, since the two sides face "
    "opposite directions.", "percent", "value", "{n} of {total} ({value:.0%})",
    _mirrored_ball_wins_compute,
    params=[Param("opp_side_lanes", "Opponent side", "lanes",
                  ("LEFT_WING,LEFT_HALF_SPACE", "RIGHT_WING,RIGHT_HALF_SPACE"))]))


# -------------------------------------------------------------------- events
def _second_balls_compute(ev, psum, squad_sums, scores, setp, squad, zone=None, **_):
    d = ev[(ev.squadName == squad) & (ev.phase == "SECOND_BALL")
           & (ev.actionType == "LOOSE_BALL_REGAIN")]
    if zone:
        d = d[d.startPitchPosition == zone]
    return {"value": float(len(d))}


register(MetricSpec("second_balls_won", "2nd balls won",
    "Loose-ball regains after an initial contest, optionally restricted to a "
    "pitch third.", "count", "value", "{value:.0f}", _second_balls_compute,
    params=[Param("zone", "Pitch zone (optional)", "zone",
                  ("OWN_THIRD", "MIDDLE_THIRD", "FINAL_THIRD"))]))


def _balls_in_behind_compute(ev, psum, squad_sums, scores, setp, squad, **_):
    d = ev[ev.squadName == squad]
    behind = d[(d.action == "AVAILABILITY_IN_THE_BACK") & (d.startAdjCoordinatesX > 17.5)]
    in_box = d[d.action == "AVAILABILITY_IN_THE_BOX"]
    return {"value": float(len(behind) + len(in_box)), "behind": len(behind),
            "in_box": len(in_box)}


register(MetricSpec("balls_in_behind", "Receptions beyond the last line",
    "Opponent receptions past the defensive line in the final third, plus box "
    "receptions.", "count", "value", "{value:.0f} ({behind} in behind, "
    "{in_box} in the box)", _balls_in_behind_compute))


def _crosses_compute(ev, psum, squad_sums, scores, setp, squad, **_):
    c = ev[(ev.squadName == squad) & ev.action.isin(CROSS_ACTIONS)]
    ok = int((c.result == "SUCCESS").sum())
    return {"value": float(ok), "ok": ok, "total": len(c)}


register(MetricSpec("crosses_completed", "Completed crosses",
    "Successful open-play + set-piece crosses.", "count", "value",
    "{ok} completed of {total} attempted", _crosses_compute))


def _passes_into_box_compute(ev, psum, squad_sums, scores, setp, squad, **_):
    p = ev[(ev.squadName == squad) & (ev.actionType == "PASS") & (ev.result == "SUCCESS")
          & (ev.endPitchPosition == "OPPONENT_BOX")]
    return {"value": float(len(p))}


register(MetricSpec("passes_into_box_events", "Successful passes into the box",
    "Event-derived count (any phase, any start position).", "count", "value",
    "{value:.0f}", _passes_into_box_compute))


def _headed_shots_compute(ev, psum, squad_sums, scores, setp, squad, **_):
    h = ev[(ev.squadName == squad) & (ev.action == "HEADER") & (ev.actionType == "SHOT")]
    on_target = int(h.SHOT_AT_GOAL_NUMBER_ON_TARGET.fillna(0).astype(bool).sum())
    goals = int(h.SHOT_AT_GOAL_NUMBER_SUCCESS.fillna(0).astype(bool).sum())
    return {"value": float(len(h)), "on_target": on_target, "goals": goals,
            "xg": float(h.SHOT_XG.sum())}


register(MetricSpec("headed_shots", "Headed shots",
    "Attempts, on-target, goals and xG from headers.", "count", "value",
    "{value:.0f} attempts, {on_target} on target, {goals} goals ({xg:.2f} xG)",
    _headed_shots_compute))


# ----------------------------------------------------------- player-specific
def _player_forward_passes_compute(ev, psum, squad_sums, scores, setp, squad,
                                   player_name="", **_):
    p = ev[(ev.playerName == player_name) & (ev.actionType == "PASS")]
    fwd = p[p.endAdjCoordinatesX > p.startAdjCoordinatesX]
    ok = int((fwd.result == "SUCCESS").sum())
    rate = (ok / len(fwd)) if len(fwd) else None
    return {"value": rate, "ok": ok, "attempted": len(fwd), "total_passes": len(p)}


register(MetricSpec("player_forward_pass_rate", "A named player's forward-pass rate",
    "Completion rate of a named player's forward passes, out of their total "
    "passes.", "percent", "value",
    "{ok} completed of {attempted} attempted ({total_passes} passes total)",
    _player_forward_passes_compute,
    params=[Param("player_name", "Player", "player_name")]))


def _player_contests_compute(ev, psum, squad_sums, scores, setp, squad,
                             opp_player="", **_):
    d = ev[(ev.squadName == squad) & ev.actionType.isin(("GROUND_DUEL", "AERIAL_DUEL"))
           & (ev.duelPlayerName == opp_player)]
    dr = ev[(ev.squadName == squad) & (ev.actionType == "DRIBBLE")
           & (ev.dribbleOpponentPlayerName == opp_player)]
    won = int((d.result == "SUCCESS").sum()) + int((dr.result == "SUCCESS").sum())
    n = len(d) + len(dr)
    return {"value": (won / n) if n else None, "won": won, "n": n}


register(MetricSpec("player_contests", "Direct 1v1s vs a named opponent",
    "Ground/aerial duels plus dribbles contested against one named opponent "
    "player.", "percent", "value", "{won} of {n} won", _player_contests_compute,
    params=[Param("opp_player", "Opponent player", "player_name")]))


def _box_touches_by_position_compute(ev, psum, squad_sums, scores, setp, squad,
                                     position_group=(), **_):
    position_group = (position_group if isinstance(position_group, (list, tuple))
                      else (position_group,))
    pos = _position_map(psum, squad)
    d = ev[(ev.squadName == squad)
          & ev.playerName.map(lambda n: pos.get(n, "") in position_group)
          & (ev.startPitchPosition == "OPPONENT_BOX") & (ev.OFFENSIVE_TOUCHES > 0)]
    return {"value": float(len(d))}


register(MetricSpec("box_touches_by_position", "Box touches by position group",
    "Box touches restricted to players in the given position group (e.g. the "
    "10s).", "count", "value", "{value:.0f}", _box_touches_by_position_compute,
    params=[Param("position_group", "Position group", "position_group",
                  ("ATTACKING_MIDFIELD", "CENTER_FORWARD", "DEFENSE_MIDFIELD"))]))


def _backline_recycling_compute(ev, psum, squad_sums, scores, setp, squad, **_):
    pos = _position_map(psum, squad)
    d = ev[(ev.squadName == squad) & (ev.actionType == "PASS") & (ev.result == "SUCCESS")]
    d = d[d.playerName.map(lambda n: pos.get(n, "") in BACK_POS)
         & d.passReceiverPlayerName.map(lambda n: pos.get(n, "") in BACK_POS)]
    back = d[d.endAdjCoordinatesX <= d.startAdjCoordinatesX]
    return {"value": float(len(back)), "total": len(d)}


register(MetricSpec("backline_recycling", "Back-line backward/sideways passes",
    "Completed passes between two back-line players (incl. keeper) that go "
    "backward or sideways.", "count", "value", "{value:.0f} of {total} "
    "back-line passes", _backline_recycling_compute))


def _diagonal_to_fb_compute(ev, psum, squad_sums, scores, setp, squad, **_):
    pos = _position_map(psum, squad)
    by_defmid = ev[(ev.squadName == squad) & (ev.action == "DIAGONAL_PASS")
                  & ev.playerName.map(lambda n: pos.get(n, "") in DEF_MID_POS)]
    to_fb = by_defmid[by_defmid.passReceiverPlayerName.map(
        lambda n: pos.get(n, "") in FB_POS)]
    ok = int((to_fb.result == "SUCCESS").sum())
    n_fb = len(to_fb)
    return {"value": (ok / n_fb) if n_fb else None, "ok": ok, "n_fb": n_fb,
            "n_total": len(by_defmid)}


register(MetricSpec("diagonal_to_fullback", "Diagonals from def/mid onto a full-back",
    "Diagonal passes made by a defender/midfielder, and how many reached a "
    "full-back.", "percent", "value", "{ok} of {n_fb} to a FB ({n_total} "
    "diagonals completed)", _diagonal_to_fb_compute))


# ------------------------------------------------------------------- crosses
def _cross_zone_share_compute(ev, psum, squad_sums, scores, setp, squad, **_):
    c = ev[(ev.squadName == squad) & ev.action.isin(CROSS_ACTIONS)
          & (ev.endPitchPosition == "OPPONENT_BOX")].dropna(
        subset=["endAdjCoordinatesY", "startAdjCoordinatesY"])
    if len(c) == 0:
        return {"value": None, "n": 0, "central": 0, "near": 0, "back": 0}
    side = np.sign(c.startAdjCoordinatesY)
    ey = c.endAdjCoordinatesY
    central = ey.abs() <= 10
    back = (~central) & (np.sign(ey) == -side)
    near = (~central) & (~back)
    n = len(c)
    return {"value": float((central | back).sum()) / n, "n": n,
            "central": int(central.sum()), "near": int(near.sum()),
            "back": int(back.sum())}


register(MetricSpec("cross_zone_share", "Cross target zone (central/back post share)",
    "Share of crosses reaching the box that land central or at the back post, "
    "vs near post.", "percent", "value",
    "{value:.0%} of {n} into the box (central {central}, near post {near}, "
    "back post {back})", _cross_zone_share_compute))


def _final_third_by_lane_compute(ev, psum, squad_sums, scores, setp, squad,
                                 lanes=(), **_):
    lanes = lanes if isinstance(lanes, (list, tuple)) else (lanes,)
    cr = ev[(ev.squadName == squad) & ev.action.isin(CROSS_ACTIONS)
           & (ev.startPitchPosition.isin(FINAL_THIRD)) & (ev.result == "SUCCESS")]
    bx = ev[(ev.squadName == squad) & (ev.actionType == "PASS")
           & (ev.action != "HEADER") & (~ev.action.isin(CROSS_ACTIONS))
           & (ev.startPitchPosition.isin(FINAL_THIRD))
           & (ev.endPitchPosition == "OPPONENT_BOX") & (ev.result == "SUCCESS")]
    sh = ev[(ev.squadName == squad) & (ev.actionType == "SHOT")
           & (ev.startPitchPosition.isin(FINAL_THIRD))]
    c = int(cr.startLane.isin(lanes).sum())
    b = int(bx.startLane.isin(lanes).sum())
    s = int(sh.startLane.isin(lanes).sum())
    return {"value": float(c + b + s), "cr": c, "bx": b, "sh": s}


register(MetricSpec("final_third_by_lane", "Crosses + box passes + shots, by lane",
    "Attacking output started in the final third, restricted to the given "
    "lane(s).", "count", "value", "{value:.0f} ({cr}cr {bx}bx {sh}sh)",
    _final_third_by_lane_compute,
    params=[Param("lanes", "Lane(s)", "lanes", LANE_CHOICES)]))


# ---------------------------------------------------------------- set pieces
def _setpiece_xg_conceded_compute(ev, psum, squad_sums, scores, setp, squad,
                                  opp="", **_):
    if setp is None or len(setp) == 0:
        return {"value": None}
    a = setp[setp.attackingSquadName == opp]["setPieceSubPhase_SHOT_XG"].sum()
    return {"value": float(a)}


register(MetricSpec("setpiece_xg_conceded", "Set-piece xG conceded",
    "xG conceded directly from the opponent's set pieces (corners, free "
    "kicks, throw-ins, goal kicks).", "xg", "value", "{value:.2f} xG",
    _setpiece_xg_conceded_compute))
