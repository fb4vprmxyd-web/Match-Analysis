"""Generic Coach's Brief grading engine - reads structured checks (from
Postgres, see db.py) instead of the old per-match hardcoded coach_brief.py.

Produces exactly the same row shape report_pages.page_coach_brief() has
always consumed ({section, instruction, metric, readout, verdict}), so the
rendering side needs no changes at all - only where the checks come from
changes.

Grading logic (_grade) is a direct port of coach_brief.py's own, already
proven across three real, hand-verified briefs this project has built
(Bradford, Bromley, Sheffield Wednesday) - not a rewrite, a relocation.
"""

import os

import pandas as pd

from envconfig import CACHE_ROOT
import metric_catalog as MC

GREEN, AMBER, RED, INFO, NA = "GREEN", "AMBER", "RED", "INFO", "N/A"

_UNIT_FORMAT = {
    "percent": "{:.0%}",
    "xg": "{:.2f}",
    "metres": "{:.0f}m",
    "count": "{:.1f}",
}

_ENDPOINTS = {
    "ev": "ev.parquet", "psum": "ps.parquet", "squad_sums": "ss.parquet",
    "scores": "squad_scores.parquet", "setp": "setpieces.parquet",
}


def _grade(basis, value, other, target, margin, lower_better):
    if value is None:
        return NA
    if basis == "info":
        return INFO

    if basis == "absolute":
        t = target
        good = value <= t if lower_better else value >= t
        near = abs(value - t) <= abs(t) * 0.15
        return GREEN if good else (AMBER if near else RED)

    if basis in ("beat_opponent", "vs_baseline"):
        if other is None:
            return NA
        m = margin or 0
        if lower_better:
            return GREEN if value <= other - m else (
                AMBER if value <= other + m else RED)
        return GREEN if value >= other + m else (
            AMBER if value >= other - m else RED)
    return INFO


def _load_cache(cache_dir):
    """Loads the subset of a cache's endpoints a metric might need. Missing
    optional files (setpieces, squad_scores on an older pull) come back as
    None, matching how build_report.py's own load() already handles it."""
    path = CACHE_ROOT / cache_dir if not os.path.isabs(cache_dir) else cache_dir
    out = {}
    for key, fname in _ENDPOINTS.items():
        fp = os.path.join(str(path), fname)
        out[key] = pd.read_parquet(fp) if os.path.exists(fp) else None
    return out


def _baseline_value(spec, squad, params, baseline):
    """Mean of the metric's grading value across whichever cache(s) the
    baseline names - one cache for 'their previous game', several for a
    season-average reference. Both are the same operation: compute the same
    catalog metric on different data and average it."""
    if not baseline or not baseline.get("caches"):
        return None
    vals = []
    for cache_dir in baseline["caches"]:
        d = _load_cache(cache_dir)
        if d["ev"] is None:
            continue
        result = spec.compute(d["ev"], d["psum"], d["squad_sums"], d["scores"],
                              d["setp"], squad, **params)
        v = result.get(spec.grading_key)
        if v is not None:
            vals.append(v)
    return (sum(vals) / len(vals)) if vals else None


def evaluate(ev, psum, squad_sums, scores, setp, team, opp, checks):
    """checks: list of dicts as returned by db.load_coach_brief_checks().
    Returns rows in report_pages.page_coach_brief()'s expected shape."""
    rows = []
    for c in checks:
        spec = MC.CATALOG.get(c["metric_key"])
        if spec is None:
            rows.append({"section": c["section"], "instruction": c["instruction"],
                        "metric": "?", "readout": f"unknown metric "
                        f"'{c['metric_key']}'", "verdict": NA})
            continue

        params = c.get("params") or {}
        squad = team if c.get("subject", "us") == "us" else opp
        other_squad = opp if squad == team else team

        result = spec.compute(ev, psum, squad_sums, scores, setp, squad,
                              opp=opp, **params)
        value = result.get(spec.grading_key)

        basis = c["basis"]
        other = None
        if basis == "beat_opponent":
            other_result = spec.compute(ev, psum, squad_sums, scores, setp,
                                        other_squad, opp=opp, **params)
            other = other_result.get(spec.grading_key)
        elif basis == "vs_baseline":
            other = _baseline_value(spec, squad, params, c.get("baseline"))

        verdict = _grade(basis, value, other, c.get("target"), c.get("margin"),
                         c.get("lower_better", False))

        try:
            readout = spec.readout_template.format(**result) if value is not None else "-"
        except (KeyError, ValueError):
            readout = str(value) if value is not None else "-"
        if basis in ("beat_opponent", "vs_baseline") and other is not None:
            label = "opp" if basis == "beat_opponent" else "baseline"
            if spec.unit == "count" and other == int(other):
                other_str = f"{other:.0f}"          # true integer counts (2nd
            else:                                    # balls, box touches) -
                other_str = _UNIT_FORMAT.get(spec.unit, "{:.1f}").format(other)
            readout = f"{readout}  ({label} {other_str})"

        rows.append({
            "section": c["section"],
            "instruction": c["instruction"],
            "metric": spec.label,
            "readout": readout,
            "verdict": verdict,
        })
    return rows


def evaluate_from_db(ev, psum, squad_sums, scores, setp, team, opp, match_id):
    import db as DB
    checks = DB.load_coach_brief_checks(match_id)
    return evaluate(ev, psum, squad_sums, scores, setp, team, opp, checks)
