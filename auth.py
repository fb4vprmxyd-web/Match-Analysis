"""Session-based login for webapp.py - a handful of named accounts for the
coaching staff, not public self-signup. Credentials live in one env var as
username:encoded-hash pairs, never in code or in Postgres, so adding or
rotating an account is "edit .env, restart the webapp container" - no
migration, no admin UI to build for three people.

    AUTH_USERS=coach:c2NyeXB0...,assistant1:c2NyeXB0...

Generate a new account's entry with:

    python3 auth.py "the person's password"

then paste the printed `username:...` line into AUTH_USERS and restart.

Why "encoded-hash" and not the raw werkzeug hash: werkzeug's own hash
format (`scrypt:N:r:p$salt$hash`) uses `$` as a field separator, and
`docker compose` parses `.env` for its OWN `$VAR`/`${VAR}` substitution
before a value ever reaches the container - a real salt starting with
something that happens to look like a variable name gets silently
replaced with an empty string, corrupting the hash and locking every
account out (confirmed directly: a real generated hash came out the other
side of `docker compose up` with its salt stripped). Base64-encoding the
hash for storage/transport removes every `$` from what's actually written
to `.env`, so there's nothing for Compose's interpolation to misread.
"""

import base64
import os
import sys

from werkzeug.security import check_password_hash, generate_password_hash


def _load_users() -> dict:
    """{username: password_hash} from AUTH_USERS - re-read on every call
    (not cached at import time) so a restart is all a credential change
    needs, matching every other setting's env-var-first resolution."""
    raw = os.environ.get("AUTH_USERS", "")
    users = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        name, encoded = pair.split(":", 1)
        name = name.strip()
        if not name:
            continue
        try:
            users[name] = base64.b64decode(encoded.strip()).decode()
        except Exception:
            # A malformed entry should block that one login, not crash
            # every request that checks who's signed in.
            continue
    return users


def verify(username: str, password: str) -> bool:
    """True only for a real username with a matching password. Doesn't
    distinguish "unknown user" from "wrong password" in its return value,
    on purpose - the login page shouldn't either."""
    if not username or not password:
        return False
    users = _load_users()
    h = users.get(username)
    return bool(h) and check_password_hash(h, password)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit('usage: python3 auth.py "<password>"')
    pw_hash = generate_password_hash(sys.argv[1])
    encoded = base64.b64encode(pw_hash.encode()).decode()
    print(f"username:{encoded}")
