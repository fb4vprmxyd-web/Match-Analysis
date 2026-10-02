"""Runs run.py and build_upcoming.py automatically, twice a week - Monday
and Thursday, to catch the weekend's games and the midweek round
respectively. run.py rebuilds the match just played; build_upcoming.py
then does the same job in the other direction, filling in a pre-match +
set piece report for whatever fixture is next if nothing exists for it yet
(see that module's docstring for how it avoids overwriting a coach's own
/selector choice).

    python3 scheduler.py

Meant to run as its own long-lived container (see docker-compose.yml's
`scheduler` service), not on a laptop: the whole point is that data
collection stops depending on anyone being awake at a given time - once
this is deployed on the droplet, it just runs, the same way it does in a
local `docker compose up -d scheduler` test. An earlier version of this
tried to schedule it locally via macOS's launchd, which only fires while
the machine is awake and defeats the purpose entirely - this replaces
that, not adds to it.

Deliberately not real cron: a real cron daemon inside the container adds a
PID-1/syslog dance for the sake of two triggers a week. A minute-by-minute
polling loop is simple, has no extra dependencies, and is easy to read.
"""

import subprocess
import sys
import time
from datetime import datetime, timezone

# Python's weekday(): Monday=0 ... Sunday=6.
RUN_DAYS = {0, 3}       # Monday, Thursday
RUN_HOUR_UTC = 8        # adjust if the droplet's clock isn't UTC
CHECK_INTERVAL_SECONDS = 60


def _due(now, last_run_date):
    return (now.weekday() in RUN_DAYS and now.hour == RUN_HOUR_UTC
           and now.date() != last_run_date)


def main():
    print(f"scheduler started - watching for "
         f"Mon/Thu {RUN_HOUR_UTC:02d}:00 UTC", flush=True)
    last_run_date = None
    while True:
        now = datetime.now(timezone.utc)
        if _due(now, last_run_date):
            print(f"=== {now.isoformat()} triggering run.py ===", flush=True)
            result = subprocess.run([sys.executable, "run.py"])
            print(f"run.py exited {result.returncode}", flush=True)

            print(f"=== {now.isoformat()} triggering build_upcoming.py ===", flush=True)
            result = subprocess.run([sys.executable, "build_upcoming.py"])
            print(f"build_upcoming.py exited {result.returncode}", flush=True)

            last_run_date = now.date()
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
