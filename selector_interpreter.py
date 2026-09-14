"""Free-text sample description ("last 10 games each team", "last 5 cup
games each team") -> two match_selector.py specs, via Claude.

Same shape as brief_interpreter.py: the model only ever fills in a narrow
{team, count, competition} spec per side - it never touches the fixture
list itself. match_selector.resolve_selection() does the actual, real
lookup deterministically. Uses Haiku rather than Sonnet - unlike Coach's
Brief interpretation (which maps open-ended tactical language onto one of
many metrics), this is close to pure extraction: there's essentially one
correct reading of "last 5 cup games."

    specs = interpret_selection(raw_text, "Leyton Orient", "Bradford City")
    for line in readback(specs):
        print(line)
    # confirmed, then:
    opp_caches = resolve_selection(cfg, specs["opponent"])
    our_caches = resolve_selection(cfg, specs["us"])
"""

import os

import anthropic

from envconfig import anthropic_key

MODEL = os.environ.get("SELECTOR_INTERPRETER_MODEL", "claude-haiku-4-5-20251001")

_TOOL_NAME = "record_selection"
_COMPETITIONS = ("all", "league", "cup")


def _system_prompt(us: str, opp: str) -> str:
    return f"""You turn a coach's free-text description of a sample size into two \
match-selection specs for a pre-match report - one for "{us}" (us), one for \
"{opp}" (the opponent). You never look up real fixtures yourself; you only \
produce {{count, competition}} per side, which is executed deterministically \
against the live fixture list afterwards.

- count: how many of that side's most recent PLAYED matches this season to \
include. If the coach only describes one side, use the same count and \
competition for both sides.
- competition: one of {list(_COMPETITIONS)} - "league" or "cup" only when \
the coach is explicit about it, otherwise "all".

Counts are capped to however many matches actually exist this season, so \
it's fine and expected for the two sides to end up with different totals -
never invent extra matches or reach back into a prior season to make a \
count exact.

Examples:
- "analyze the last 10 games from each team" -> us: count 10, competition \
all; opponent: count 10, competition all
- "last 5 cup games for each team" -> both sides: count 5, competition cup
- "last 8 league games for them, last 4 for us" -> opponent: count 8, \
competition league; us: count 4, competition league
"""


_SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "count": {"type": "integer", "minimum": 1},
        "competition": {"type": "string", "enum": list(_COMPETITIONS)},
    },
    "required": ["count", "competition"],
}

_TOOLS = [{
    "name": _TOOL_NAME,
    "description": "Record the match-selection spec for each side.",
    "input_schema": {
        "type": "object",
        "properties": {"us": _SPEC_SCHEMA, "opponent": _SPEC_SCHEMA},
        "required": ["us", "opponent"],
    },
}]


def interpret_selection(raw_text: str, us: str, opp: str, model: str = None) -> dict:
    """Returns {"us": {"team", "count", "competition"},
    "opponent": {"team", "count", "competition"}} - each ready to pass
    straight into match_selector.resolve_selection()."""
    client = anthropic.Anthropic(api_key=anthropic_key())
    resp = client.messages.create(
        model=model or MODEL,
        max_tokens=1024,
        system=_system_prompt(us, opp),
        tools=_TOOLS,
        tool_choice={"type": "tool", "name": _TOOL_NAME},
        messages=[{"role": "user", "content": raw_text}],
    )
    block = next(b for b in resp.content if b.type == "tool_use")
    data = block.input
    return {
        "us": {"team": us, **data["us"]},
        "opponent": {"team": opp, **data["opponent"]},
    }


def readback(specs: dict) -> list:
    lines = []
    for side, label in (("opponent", "Opponent"), ("us", "Us")):
        s = specs[side]
        comp = {"all": "all competitions", "league": "league only",
               "cup": "cup only"}[s["competition"]]
        lines.append(f"{label} ({s['team']}): last {s['count']} played "
                     f"matches this season, {comp}")
    return lines
