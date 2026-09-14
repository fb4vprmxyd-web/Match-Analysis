"""Free-text Coach's Brief notes -> structured checks, via Claude.

The model never computes anything and never invents a metric - it can only
SELECT and PARAMETERISE entries from metric_catalog.CATALOG, forced by a
tool-use JSON schema whose enums are built straight from the real catalog.
That's the same safety rail metric_catalog.py's own docstring describes:
the model can pick the wrong entry, and the confirm/readback step (see
readback() below, and interpret_brief.py's CLI) exists to catch that - but
it cannot fabricate a computation that was never verified against real
match data.

    checks = interpret_brief(raw_text, cfg, "Leyton Orient", "Bradford City")
    for line in readback(checks):
        print(line)
    # coach confirms/edits, then:
    db.save_coach_brief_notes(match_id, raw_text)
    db.save_coach_brief_checks(match_id, checks)

readback() is deliberately NOT a second LLM call - it's a pure function of
the structured check, formatted from the same catalog data the model chose
from. The interpretation step is the only place free text ever meets an
LLM; everything after that is as deterministic as the rest of this project.
"""

import ast
import json
import os

import anthropic

from envconfig import anthropic_key
import metric_catalog as MC
import match_selector as MSEL

MODEL = os.environ.get("BRIEF_INTERPRETER_MODEL", "claude-sonnet-5")

_TOOL_NAME = "record_checks"

_SECTIONS = ("OUT OF POSSESSION", "IN POSSESSION", "DID IT WORK")
_BASES = ("beat_opponent", "absolute", "vs_baseline", "info")
_COMPETITIONS = ("all", "league", "cup")


def _catalog_prompt() -> str:
    lines = []
    for spec in MC.CATALOG.values():
        params = ""
        if spec.params:
            bits = []
            for p in spec.params:
                choices = f", one of {list(p.choices)}" if p.choices else ""
                bits.append(f"{p.name} ({p.kind}{choices})")
            params = "  Params: " + "; ".join(bits)
        lines.append(f"- {spec.key}: {spec.label} - {spec.description}{params}")
    return "\n".join(lines)


def _system_prompt(team: str, opp: str) -> str:
    return f"""You turn a football coach's free-text pre-match notes into a list of \
structured checks for a post-match Coach's Brief. You do not compute \
anything yourself and you do not invent new metrics - you SELECT and \
PARAMETERISE from this fixed catalog only:

{_catalog_prompt()}

Our side is "{team}"; the opponent is "{opp}". Every check needs:
- section: exactly one of {list(_SECTIONS)}
- instruction: the coach's own instruction, kept close to their wording
- metric_key: one of the catalog keys above
- subject: "us" or "opp" - whose data the metric is computed on
- params: object matching the metric's params (empty object {{}} if none)
- basis: one of {list(_BASES)}
    "beat_opponent" - us vs the opponent on the same metric, same match
    "absolute"      - vs a fixed target number you choose
    "vs_baseline"   - vs this side's OWN earlier form (only when the coach
                      explicitly asks for that, e.g. "compared to their last
                      few games" or "beat our own season average") - also
                      set baseline_spec: {{"count": int, "competition": one
                      of {list(_COMPETITIONS)}}} naming how many of the
                      SAME side's own past matches to average
    "info"          - just report the number, no grading (use this when
                      there's no natural target or opponent comparison,
                      e.g. a first-time check on a named player)
- lower_better: true if a SMALLER number is the good outcome
- target: only for basis="absolute" - the fixed number to hit
- margin: only for basis="beat_opponent"/"vs_baseline" - how much of a gap
  counts as a real win vs a rough tie (graded AMBER)
- source_note: null normally; if an instruction doesn't map cleanly onto
  any catalog entry, still produce your best-attempt check using the
  closest entry, and explain the mismatch here rather than dropping it

Calibration examples, from real briefs already built by hand:
- "Pressure the ball to limit forward play" -> ppda, subject us, \
beat_opponent, lower_better, margin ~1.0
- "If they go forward, force it wide" -> lane_share(lanes=["CENTER"]), \
subject opp, absolute, lower_better, target ~0.30 (want THEIR central \
progression low)
- "Overload their midfield two through the centre" -> \
lane_share(lanes=["CENTER"]), subject us, absolute, target ~0.30
- "Target Hayden Matthews directly" -> \
player_contests(opp_player="Hayden Matthews"), subject us, info
- "The plan created chances" -> xg, subject us, beat_opponent, margin ~0.20
- "Compared to their last 3 league games, press them higher" -> \
press_height, subject opp, vs_baseline, baseline_spec {{"count": 3, \
"competition": "league"}}, lower_better=false (assuming "higher" is good \
for us, i.e. opponent forced deeper - read the coach's intent carefully)
"""


_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "section": {"type": "string", "enum": list(_SECTIONS)},
        "instruction": {"type": "string"},
        "metric_key": {"type": "string", "enum": list(MC.CATALOG.keys())},
        "subject": {"type": "string", "enum": ["us", "opp"]},
        "params": {"type": "object"},
        "basis": {"type": "string", "enum": list(_BASES)},
        "lower_better": {"type": "boolean"},
        "target": {"type": ["number", "null"]},
        "margin": {"type": ["number", "null"]},
        "baseline_spec": {
            "type": ["object", "null"],
            "properties": {
                "count": {"type": "integer"},
                "competition": {"type": "string", "enum": list(_COMPETITIONS)},
            },
        },
        "source_note": {"type": ["string", "null"]},
    },
    "required": ["section", "instruction", "metric_key", "subject", "params",
               "basis", "lower_better"],
}

_TOOLS = [{
    "name": _TOOL_NAME,
    "description": "Record the structured Coach's Brief checks interpreted "
                   "from the coach's free-text notes.",
    "input_schema": {
        "type": "object",
        "properties": {"checks": {"type": "array", "items": _CHECK_SCHEMA}},
        "required": ["checks"],
    },
}]


def _resolve_baseline(cfg, subject_team, spec, before_date=None):
    """A vs_baseline check's baseline_spec {count, competition} into concrete
    cache directories - same resolver match_selector.py uses for pre-match
    reports, reused here so 'their own last 3 league games' means exactly
    the same thing whether it's a Coach's Brief baseline or a pre-match
    sample."""
    caches = MSEL.resolve_selection(cfg, {
        "team": subject_team,
        "count": spec.get("count", 3),
        "competition": spec.get("competition", "all"),
        "before_date": before_date,
    })
    return {"caches": caches} if caches else None


def _unstringify(value, expect):
    """Claude's tool-use occasionally serialises a nested array/object as a
    STRING instead of the real structure - seen here at more than one
    nesting level (the whole "checks" array, individual check objects,
    "params"/"baseline_spec" within one), and in two different encodings:
    valid JSON (double-quoted) and a Python dict/list repr (single-quoted,
    e.g. "{'section': 'OUT OF POSSESSION', ...}"). Most likely provoked by
    the wide-open "params" object in the schema having no fixed
    sub-properties for the model to anchor on. Decoding it here doesn't
    loosen what's trusted - the section/basis/metric_key checks below still
    apply to whatever comes out - it just unwraps however the model chose
    to encode the same content. ast.literal_eval is safe here (literals
    only, no code execution) unlike a real eval()."""
    if isinstance(value, str):
        if not value:
            return expect()
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return expect()
    return value if isinstance(value, type(expect())) else expect()


def _extract_checks(value, depth=0):
    """Digs out the actual checks array, however many layers of wrapping
    Claude put around it this time. Seen in practice: the array itself as a
    plain list (the schema's intent); as a JSON string of that list; and -
    the deepest case, from a real Wycombe-brief call - the WHOLE tool input
    re-encoded as a JSON string and stuffed back under the same "checks"
    key, i.e. {"checks": "{\\"checks\\": [...]}"} instead of {"checks":
    [...]}. Recurses through string-decoding and dict-unwrapping until it
    finds a real list or gives up - depth-capped so a pathological response
    can't loop forever."""
    if depth > 5:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return _extract_checks(value.get("checks", []), depth + 1)
    if isinstance(value, str):
        if not value:
            return []
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            try:
                decoded = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return []
        return _extract_checks(decoded, depth + 1)
    return []


def interpret_brief(raw_text: str, cfg: dict, team: str, opp: str,
                    match_date=None, model: str = None) -> list:
    """Returns a list of check dicts ready for db.save_coach_brief_checks()."""
    client = anthropic.Anthropic(api_key=anthropic_key())
    resp = client.messages.create(
        model=model or MODEL,
        max_tokens=4096,
        system=_system_prompt(team, opp),
        tools=_TOOLS,
        tool_choice={"type": "tool", "name": _TOOL_NAME},
        messages=[{"role": "user", "content": raw_text}],
    )
    block = next(b for b in resp.content if b.type == "tool_use")
    raw_checks = _extract_checks(block.input.get("checks", []))

    section_counters = {s: 0 for s in _SECTIONS}
    checks = []
    for rc in raw_checks:
        rc = _unstringify(rc, dict)
        section = rc.get("section")
        if section not in _SECTIONS:
            section = "DID IT WORK"
        order = section_counters[section]
        section_counters[section] += 1

        baseline = None
        if rc.get("basis") == "vs_baseline":
            subject_team = team if rc.get("subject", "us") == "us" else opp
            spec = _unstringify(rc.get("baseline_spec") or {}, dict)
            baseline = _resolve_baseline(cfg, subject_team, spec, match_date)
            if baseline is None:
                # couldn't resolve any reference matches - fall back to a
                # plain readout rather than a check that can never grade
                rc["basis"] = "info"

        checks.append({
            "section": section,
            "sort_order": order,
            "instruction": rc["instruction"],
            "metric_key": rc["metric_key"],
            "subject": rc.get("subject", "us"),
            "params": _unstringify(rc.get("params") or {}, dict),
            "basis": rc["basis"],
            "lower_better": bool(rc.get("lower_better", False)),
            "target": rc.get("target"),
            "margin": rc.get("margin"),
            "baseline": baseline,
            "source_note": rc.get("source_note"),
        })
    return checks


def _fmt_param(v):
    return ", ".join(v) if isinstance(v, (list, tuple)) else str(v)


def readback(checks: list) -> list:
    """One plain-English line per check, for the coach to confirm or edit
    BEFORE anything is saved - not a wall of numbers after the fact."""
    lines = []
    for c in checks:
        spec = MC.CATALOG.get(c["metric_key"])
        metric_desc = spec.label if spec else c["metric_key"]
        if c.get("params"):
            metric_desc += " (" + ", ".join(
                f"{k}={_fmt_param(v)}" for k, v in c["params"].items()) + ")"
        subject = "us" if c.get("subject", "us") == "us" else "the opponent"
        direction = "lower is better" if c.get("lower_better") else "higher is better"

        basis = c["basis"]
        if basis == "beat_opponent":
            tail = f"beat the opponent on this ({direction}, margin {c.get('margin') or 0})"
        elif basis == "absolute":
            tail = f"hit a target of {c.get('target')} ({direction})"
        elif basis == "vs_baseline":
            n = len(c["baseline"]["caches"]) if c.get("baseline") else 0
            tail = (f"beat {subject}'s own form over their last {n} cached "
                    f"match(es) ({direction}, margin {c.get('margin') or 0})")
        else:
            tail = "reported only, not graded"

        line = (f"[{c['section']}] \"{c['instruction']}\" -> {metric_desc} "
               f"for {subject}: {tail}")
        if c.get("source_note"):
            line += f"  ⚠ {c['source_note']}"
        lines.append(line)
    return lines
