"""Shared helper for the test_*_golden.py regression tests - runs
_golden_runner.py in an isolated subprocess per match and caches the
result, so each golden test file gets its own match's real numbers no
matter what order pytest collects the files in."""

import json
import os
import subprocess
import sys
from functools import lru_cache

ROOT = os.path.dirname(os.path.abspath(__file__))


@lru_cache(maxsize=None)
def load_golden(match_id, team, opp, cache, comp, matchday, venue):
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "_golden_runner.py"),
         str(match_id), team, opp, cache, comp, matchday, venue],
        cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"_golden_runner.py failed for match {match_id}:\n"
            f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
    return json.loads(r.stdout.strip().splitlines()[-1])
