"""Builds glossary.json (the data behind /glossary): only the terms the
platform actually uses, with Impect's own wording where Impect provides it.

Usage: python3 build_glossary.py path/to/impect_definitions.csv
"""
import html
import json
import re
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "glossary.json"
CATEGORIES = ["Attacking", "Defending", "Transition", "Squad Stats", "Player Stats"]


def _expand(template, values):
    return [template.format(v) for v in values]


_ZONES = ("FBR", "CB", "FBL", "DM", "WR", "WL", "CM", "AM", "IBWR", "IBWL", "IB")
_THIRDS = ("FIRST_THIRD", "MIDDLE_THIRD", "FINAL_THIRD", "OPPONENT_BOX")
_PASS_TYPES = ("LOW_PASS", "CHIPPED_PASS", "DIAGONAL_PASS", "HIGH_CROSS", "LOW_CROSS")
_SHOT_TYPES = ("LONG_RANGE_SHOT", "MID_RANGE_SHOT", "CLOSE_RANGE_SHOT",
               "ONE_VS_ONE_AGAINST_GK", "HEADER", "OPEN_GOAL_SHOT", "FREE_KICK",
               "CORNER", "PENALTY_KICK")

# Every Impect column the reports, season pages and Coach's Brief read.
USED = [
    "ASSISTS", "BALL_LOSS_REMOVED_TEAMMATES", "BALL_WIN_NUMBER",
    "BALL_WIN_NUMBER_BY_ACTION_INTERCEPTION", "BALL_WIN_REMOVED_OPPONENTS",
    "BALL_WIN_REMOVED_OPPONENTS_DEFENDERS", "BYPASSED_DEFENDERS",
    "BYPASSED_DEFENDERS_RECEIVING", "BYPASSED_OPPONENTS", "CONCEDED_GOALS",
    "CONCEDED_POSTSHOT_XG", "CONCEDED_SHOT_XG", "CRITICAL_BALL_LOSS_NUMBER",
    "DEFENSIVE_TOUCHES", "GOALS", "LOST_AERIAL_DUELS", "LOST_GROUND_DUELS",
    "NUMBER_OF_FOULS", "NUMBER_OF_PRESSES", "NUMBER_OF_PRESSES_BETWEEN_THE_LINES",
    "NUMBER_OF_PRESSES_BUILD_UP", "NUMBER_OF_PRESSES_COUNTER_PRESS",
    "OFFENSIVE_TOUCHES", "OFFENSIVE_TOUCHES_IN_PITCH_POSITION_OPPONENT_BOX",
    "POSTSHOT_XG", "PXT_ATTACK", "PXT_DEFEND", "SHOT_AT_GOAL_NUMBER",
    "SHOT_AT_GOAL_NUMBER_IN_PITCH_POSITION_OPPONENT_BOX",
    "SHOT_AT_GOAL_NUMBER_ON_TARGET", "SHOT_AT_GOAL_NUMBER_SUCCESS",
    "SHOT_CREATING_ACTIONS", "SHOT_XG", "SHOT_XG_AT_PHASE_ATTACKING_TRANSITION",
    "SUCCESSFUL_PASSES", "SUCCESSFUL_PASSES_TO_PITCH_POSITION_OPPONENT_BOX",
    "SUFFERED_BYPASSED_DEFENDERS", "UNSUCCESSFUL_PASSES", "WON_AERIAL_DUELS",
    "WON_GROUND_DUELS",
    *_expand("BYPASSED_DEFENDERS_AT_PHASE_{}",
             ("IN_POSSESSION", "ATTACKING_TRANSITION", "SET_PIECE", "SECOND_BALL")),
    *_expand("BYPASSED_DEFENDERS_BY_ACTION_{}",
             ("LOW_PASS", "CHIPPED_PASS", "DIAGONAL_PASS", "DRIBBLE", "HIGH_CROSS",
              "LOW_CROSS", "HEADER")),
    *_expand("BALL_WIN_NUMBER_IN_PITCH_POSITION_{}", _THIRDS),
    *_expand("BALL_WIN_REMOVED_OPPONENTS_FROM_OPPONENT_PACKING_ZONE_{}", _ZONES),
    *_expand("SUCCESSFUL_PASSES_BY_ACTION_{}", _PASS_TYPES),
    *_expand("UNSUCCESSFUL_PASSES_BY_ACTION_{}", _PASS_TYPES),
    *_expand("SHOT_AT_GOAL_NUMBER_BY_ACTION_{}", _SHOT_TYPES),
    *_expand("SHOT_XG_BY_ACTION_{}", _SHOT_TYPES),
]

CATEGORY = {
    "GOALS": "Attacking", "ASSISTS": "Player Stats",
    "SHOT_CREATING_ACTIONS": "Player Stats",
    "BYPASSED_DEFENDERS_RECEIVING": "Player Stats",
    "NUMBER_OF_FOULS": "Squad Stats",
    "SUCCESSFUL_PASSES": "Squad Stats", "UNSUCCESSFUL_PASSES": "Squad Stats",
    "OFFENSIVE_TOUCHES": "Squad Stats", "DEFENSIVE_TOUCHES": "Squad Stats",
    "WON_GROUND_DUELS": "Squad Stats", "LOST_GROUND_DUELS": "Squad Stats",
    "WON_AERIAL_DUELS": "Squad Stats", "LOST_AERIAL_DUELS": "Squad Stats",
    "BALL_LOSS_REMOVED_TEAMMATES": "Transition",
    "CRITICAL_BALL_LOSS_NUMBER": "Transition",
    "BALL_WIN_REMOVED_OPPONENTS": "Transition",
    "NUMBER_OF_PRESSES_COUNTER_PRESS": "Transition",
    "BALL_WIN_NUMBER": "Defending",
    "BALL_WIN_REMOVED_OPPONENTS_DEFENDERS": "Defending",
    "NUMBER_OF_PRESSES": "Defending", "NUMBER_OF_PRESSES_BUILD_UP": "Defending",
    "NUMBER_OF_PRESSES_BETWEEN_THE_LINES": "Defending",
    "PXT_DEFEND": "Defending", "SUFFERED_BYPASSED_DEFENDERS": "Defending",
    "CONCEDED_GOALS": "Defending", "CONCEDED_SHOT_XG": "Defending",
    "CONCEDED_POSTSHOT_XG": "Defending",
}


def category(col, parent):
    if "ATTACKING_TRANSITION" in col or "SECOND_BALL" in col:
        return "Transition"
    if col in CATEGORY:
        return CATEGORY[col]
    if col.endswith(("HIGH_CROSS", "LOW_CROSS", "_OPPONENT_BOX")) and parent in (
            "SUCCESSFUL_PASSES", "UNSUCCESSFUL_PASSES", "OFFENSIVE_TOUCHES"):
        return "Attacking"
    if parent:
        return CATEGORY.get(parent, "Attacking")
    return "Attacking"


TERMS = {
    "CONCEDED_GOALS": "Goals Conceded",
    "CONCEDED_SHOT_XG": "Shot-based xG Conceded",
    "CONCEDED_POSTSHOT_XG": "Post-Shot xG Conceded",
    "PXT_ATTACK": "Packing xT", "PXT_DEFEND": "Packing xT Defend",
    "NUMBER_OF_PRESSES": "Pressures",
    "NUMBER_OF_PRESSES_BUILD_UP": "Pressures: Opponent Build-Up",
    "NUMBER_OF_PRESSES_BETWEEN_THE_LINES": "Pressures: Between the Lines",
    "NUMBER_OF_PRESSES_COUNTER_PRESS": "Pressures: Counter-Press",
    "BALL_WIN_NUMBER": "Ball Wins",
    "BALL_WIN_REMOVED_OPPONENTS_DEFENDERS": "Ball Win Removed Defenders",
    "NUMBER_OF_FOULS": "Fouls Committed",
    "SHOT_AT_GOAL_NUMBER": "Shots",
    "SHOT_AT_GOAL_NUMBER_ON_TARGET": "Shots On Target",
    "SHOT_AT_GOAL_NUMBER_SUCCESS": "Shots Scored",
    "SHOT_AT_GOAL_NUMBER_IN_PITCH_POSITION_OPPONENT_BOX": "Shots: Inside Box",
    "BALL_WIN_NUMBER_BY_ACTION_INTERCEPTION": "Interceptions (Ball Wins)",
    "OFFENSIVE_TOUCHES_IN_PITCH_POSITION_OPPONENT_BOX": "Box Touches",
    "SUCCESSFUL_PASSES_TO_PITCH_POSITION_OPPONENT_BOX": "Passes Into Box",
}

FORMULAS = {
    "SHOT_AT_GOAL_NUMBER_ON_TARGET": "Shots Scored + Shots Saved + Shots Caught",
}

# Read from Impect's getSquadMatchScores but absent from the definitions
# export, so these descriptions are ours, not Impect's wording.
SCORES = [
    ("RATIO_BALL_POSSESSION", "Possession", "Squad Stats", "Share of match time in possession.", ""),
    ("RATIO_PASSING_ACCURACY", "Passing Accuracy", "Squad Stats", "Impect's published pass-accuracy figure for the match.", ""),
    ("RATIO_DUELS", "Overall Duels Won %", "Squad Stats", "Share of all duels won, ground and aerial combined.",
     "(Won Ground + Won Aerial) / (Won Ground + Lost Ground + Won Aerial + Lost Aerial)"),
    ("RATIO_GROUND_DUELS", "Ground Duels Won %", "Squad Stats", "Share of ground duels won.", "Won Ground Duels / (Won Ground Duels + Lost Ground Duels)"),
    ("RATIO_AERIAL_DUELS", "Aerial Duels Won %", "Squad Stats", "Share of aerial duels won.", "Won Aerial Duels / (Won Aerial Duels + Lost Aerial Duels)"),
    ("FIELD_TILT", "Field Tilt", "Squad Stats", "How far the territorial balance of the game tilted towards a side.", ""),
    ("EXPECTED_POINTS_SHOT_XG", "Expected Points (xG)", "Squad Stats", "Points a side would expect from the match, based on shot-based xG.", ""),
    ("EXPECTED_POINTS_POSTSHOT_XG", "Expected Points (Post-Shot xG)", "Squad Stats", "Points a side would expect from the match, based on post-shot xG.", ""),
    ("WIN_PROBABLITY_SHOT_XG", "Win Probability", "Squad Stats", "Likelihood of a side winning, based on shot-based xG.", ""),
    ("DRAW_PROBABILITY_SHOT_XG", "Draw Probability", "Squad Stats", "Likelihood of a draw, based on shot-based xG.", ""),
    ("PXT_POSITIVE", "Attacking Threat (xT)", "Attacking", "Goal threat a side generated in the match.", ""),
    ("RATIO_REVERSE_PLAY", "Reverse Play Ratio", "Attacking", "Impect's ratio for reverse play. We treat lower as better, which is our reading and not confirmed by Impect.", ""),
    ("RATIO_ADDED_OPPONENTS", "Ball Control Ratio", "Attacking", "Closest available field to Impect's dashboard KPI \"Ratio - add opponents (without shots)\". It is an approximation because shots are not excluded here.", ""),
    ("PPDA", "Pressing Intensity (PPDA)", "Defending", "Passes allowed per defensive action. Lower means more aggressive pressing.", ""),
    ("MEAN_PRESSURE_HEIGHT", "Press Height", "Defending", "Average distance upfield, in metres, where defensive actions happen.", ""),
    ("PRESSURE_DURING_OPP_BUILD_UP_PERCENT", "Press Opponent Build-Up", "Defending", "Share of the opponent's build-up phases that were pressed.", ""),
    ("OPP_OPENING_BALLS_PRESSED_PERCENT", "Press Opening Balls", "Defending", "Share of the opponent's opening balls that were pressed.", ""),
    ("PRESSURE_OPP_GK_PERCENT", "Press Opposition Goalkeeper", "Defending", "How much a side presses the opposing goalkeeper.", ""),
    ("FORCED_HIGH_PASSES_PERCENT", "Force Long/High Passes", "Defending", "Share of the opponent's passes forced long or high under pressure. Higher means more disrupted build-up.", ""),
    ("RATIO_SECOND_BALL_WINS", "Second Balls Won %", "Transition", "Share of second balls won.", ""),
]

# Calculated by this platform from Impect's data.
OURS = [
    ("Shots: Outside Box", "Attacking", "Shots taken from outside the penalty area.", "Shots − Shots: Inside Box"),
    ("Crosses (Accurate)", "Attacking", "All crosses, open play and set piece, and how many were completed.", "Successful high + low crosses, out of all crosses"),
    ("Cross Completion %", "Attacking", "Share of crosses completed.", "Successful crosses / (Successful + Unsuccessful crosses)"),
    ("Set-Piece xG", "Attacking", "xG created from corners and free kicks.", "Shot xG: Corner + Shot xG: Free Kick"),
    ("Goals vs xG", "Attacking", "Positive means scoring more than the chances deserved.", "Goals − Shot-based xG"),
    ("Headed Shots", "Attacking", "Attempts, on-target efforts, goals and xG from headers.", "Count of headed shot events"),
    ("Cross Target Zone", "Attacking", "Share of crosses reaching the box that land central or at the back post, versus near post.", "Central + back-post crosses / Crosses reaching the box"),
    ("Lane Share", "Attacking", "Share of a side's own progression that goes through the given lane or lanes.", "Bypassed Defenders in lane / Total Bypassed Defenders"),
    ("Interceptions", "Defending", "Interceptions in a single match, used on post-match reports. Pre-match reports use Interceptions (Ball Wins) instead.", "Count of INTERCEPTION events"),
    ("Conceded vs xG", "Defending", "Positive means the defence has held up better than the chances faced.", "Shot-based xG Conceded − Goals Conceded"),
    ("Set-Piece xG Conceded", "Defending", "xG conceded directly from the opponent's set pieces.", "Σ xG of opponent corners, free kicks, throw-ins and goal kicks"),
    ("Receptions Beyond the Last Line", "Defending", "Opponent receptions past the defensive line in the final third, plus box receptions.", "Count of qualifying reception events"),
    ("2nd Balls Won", "Transition", "Loose-ball regains after an initial contest, optionally restricted to a pitch third.", "Count of loose-ball regains"),
]


def _clean(s):
    if pd.isna(s):
        return ""
    s = re.sub(r"<br\s*/?>", "\n", str(s))
    s = re.sub(r"</?b>", "", s)
    return re.sub(r"[ \t]+\n", "\n", html.unescape(s).strip())


def build(csv_path):
    d = pd.read_csv(csv_path, encoding="utf-8").set_index("column")
    missing = [c for c in USED if c not in d.index]
    if missing:
        raise SystemExit(f"Not in Impect's export: {missing}")

    rows = []
    for col in dict.fromkeys(USED):
        r = d.loc[col]
        parent = "" if pd.isna(r.parent_kpi) else r.parent_kpi
        base = TERMS.get(parent) or _clean(d.loc[parent, "label"]) if parent else ""
        term = TERMS.get(col) or (f"{base}: {_clean(r.context_label)}" if parent else _clean(r.label))
        if parent:
            definition = f"{base}, counted only for this split. {_clean(r.context_label)}: {_clean(r.context_definition)}"
            meaning = ""
        else:
            definition, meaning = _clean(r.official_definition), _clean(r.meaning)
        rows.append({"term": term, "column": col, "category": category(col, parent),
                     "source": "impect", "definition": definition, "meaning": meaning,
                     "formula": FORMULAS.get(col, ""), "lower_is_better": bool(r.inverted)})
    for col, term, cat, definition, formula in SCORES:
        rows.append({"term": term, "column": col, "category": cat, "source": "score",
                     "definition": definition, "meaning": "", "formula": formula,
                     "lower_is_better": col == "PPDA"})
    for term, cat, definition, formula in OURS:
        rows.append({"term": term, "column": "", "category": cat, "source": "ours",
                     "definition": definition, "meaning": "", "formula": formula,
                     "lower_is_better": False})
    rows.sort(key=lambda m: (CATEGORIES.index(m["category"]), m["term"].lower()))
    return {"categories": CATEGORIES, "metrics": rows}


if __name__ == "__main__":
    data = build(sys.argv[1])
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter
    print(len(data["metrics"]), Counter(m["category"] for m in data["metrics"]),
          Counter(m["source"] for m in data["metrics"]))
