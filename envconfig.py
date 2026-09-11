"""Configuration loading: secrets from .env, match settings from match.toml.

Deliberately dependency-free - no python-dotenv, no PyYAML. `tomllib` is in the
standard library from Python 3.11.

Precedence is always: real environment variable > file > default. That way a
one-off override (`PMR_MATCH_ID=123 python3 build_report.py`) still works
without editing anything.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
MATCH_PATH = ROOT / "match.toml"


def load_env(path: Path = ENV_PATH) -> None:
    """Read KEY=VALUE lines into os.environ without clobbering the real env."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()
        # strip matching surrounding quotes, so passwords with # or spaces survive
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        os.environ.setdefault(key, val)      # real environment wins


def load_match(path: Path = MATCH_PATH) -> dict:
    """Read match settings from TOML. Returns {} if the file is absent."""
    if not path.exists():
        return {}
    try:
        import tomllib
    except ModuleNotFoundError:                     # Python < 3.11
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh).get("match", {})


def setting(key: str, toml_key: str, default, cast=str):
    """Resolve one setting: env var, then match.toml, then default."""
    raw = os.environ.get(key)
    if raw is None:
        raw = _MATCH.get(toml_key)
    if raw is None:
        raw = default
    return cast(raw)


def credentials() -> tuple:
    """Impect username and password, with an actionable error if unset."""
    load_env()
    user = os.environ.get("IMPECT_USER")
    pw = os.environ.get("IMPECT_PASS")
    if not user or not pw:
        sys.exit(
            "Impect credentials not found.\n"
            f"  Create {ENV_PATH} with:\n"
            "      IMPECT_USER=you@example.com\n"
            "      IMPECT_PASS=your-password\n"
            "  then:  chmod 600 .env\n"
            "  (copy .env.example to get started)"
        )
    return user, pw


# load once at import so report_config can resolve settings immediately
load_env()
_MATCH = load_match()
