# Leyton Orient match analysis

An analytics pipeline *and self-service web platform* for **Leyton Orient**,
built on **Impect** match-level API data. It is not a one-off recreation of
a single report — it is the club's running toolkit for analysing every
match across a season: scouting the next opponent before kick-off,
debriefing every match Orient plays afterward, breaking down set-piece
routines for both teams, and tracking form, minutes and player impact
across the whole campaign. `webapp.py` is the front door to all of it —
one dashboard, reachable from a browser, that a coach's assistant can run
end to end without ever opening a terminal.

The visual language (dark navy ground, geometric sans, two-team-colour
charts, picture-style tables) started as a recreation of a single, now-
retired Wycombe Wanderers template PDF — but the report types, the data
pipeline, the season tracker, and the dashboard were all built from there,
for every fixture Orient plays, not for that one match.

## What this produces

| Output | Built by | Covers |
|---|---|---|
| **Post-Match Report** | `build_report.py` | One played match: full team stats, set pieces, personnel, out-of-possession, a prose Performance Review, and (when real notes exist) a graded Coach's Brief |
| **Pre-Match Analysis** | `build_prematch.py` | An upcoming opponent's recent form vs Orient's own, aggregated across each side's last N cached matches (or every match played this season) |
| **Set Piece Report** | `build_corners.py` | Both sides' corners, plus the opponent's wide free kicks and near-box throw-ins — delivery by side, takers, first contact, defensive vulnerabilities, all as pitch diagrams and tables |
| **Season Tracker** | `build_season.py` | `season-tracker.html` — league table, form, set-piece efficiency, minutes/workload, most-impactful and best individual performances, upcoming fixtures |
| **Dashboard** | `webapp.py` + `templates/dashboard.html` | The hub everything else lives inside — embeds the season tracker, and the Pre-Match Analysis / Set Piece Analysis panes are generated live from whatever's actually been built, not hand-maintained |

All of it runs from data pulled once per match and cached locally as
parquet, so report builders work offline and can be re-run freely.

## Architecture

```
Impect API  →  fetch_data.pull_match()  →  DATA_ROOT/cache/<slug>_<matchId>/*.parquet
                                                     │
        ┌───────────────────────┬────────────────────┼──────────────────────┐
        ▼                       ▼                     ▼                      ▼
 build_report.py        build_prematch.py      build_season.py        webapp.py
 (one played match,     + build_corners.py     (whole season log)     (Flask app -
  via run.py)           (N cached matches,                             the dashboard,
        │                each side)                   │                /brief, /selector)
        ▼                       │                      ▼                      │
 Post Match Report*.pdf         ▼        DATA_ROOT/reports/season-tracker.html │
        │          Pre-Match Analysis*.pdf /                │                 │
        │          Set Piece Report*.pdf                    │                 │
        └──────────────────────┬───────────────────────────┴─────────────────┘
                                ▼
                    templates/dashboard.html
                    (served at "/" by webapp.py)
```

- **`run.py`** is the routine entry point for **post-match** reports: it
  checks every configured Impect iteration, works out which of the club's
  own matches are new or have been revised, pulls only those, and
  rebuilds.
- **`build_upcoming.py`** is the mirror image: it finds whatever fixture is
  next and, the first time nothing exists for it yet, auto-builds a
  Pre-Match Analysis + Set Piece Report from every game each side has
  played this season. Once anything exists for that fixture — its own
  first pass, or a coach's assistant picking a specific sample through
  `/selector` — it's left alone; a deliberate choice always sticks until
  the fixture is played and a new one becomes next.
- **`scheduler.py`** runs both of the above twice a week (Monday and
  Thursday) inside its own long-lived container — see *Deployment* below.
  Nobody needs to trigger a pull by hand.
- **`manual_corrections.py`** holds video-verified overrides to Impect's
  own event tagging (e.g. a penalty Impect miscategorised under a
  throw-in phase) — applied automatically wherever set-piece figures are
  computed, so a correction can't be right on one page and wrong on
  another.
- **`DATA_ROOT`** (see `envconfig.py`) is where all of that actually gets
  written — `cache/` for pulled parquet, `reports/` for every generated
  PDF and HTML page. It defaults to the project directory locally, and
  becomes the mounted Docker volume once deployed (see *Deployment*
  below).

## Setup (once)

```bash
cp .env.example .env
chmod 600 .env
# edit .env and put your Impect username and password in it
```

`.env` holds credentials and is gitignored — it must never be committed,
emailed, or pasted into a chat window. Match settings live separately in
`match.toml`, which contains no secrets and is safe to share.

Resolution order for every setting is **real environment variable → file →
default**, so a one-off override never requires editing anything.

### Optional: Postgres for shared state

`state/processed.json` (which matches have been built), the Coach's
Brief's saved checks, and which upcoming fixtures already have a Pre-Match/
Set Piece Report all live in Postgres once `DATABASE_URL` is set — the
single source of truth once this runs as more than one process (the
scheduler and the web dashboard, for a start). Unset it and everything
falls back to local JSON files, which is fine for solo local testing but
not for deployment.

```bash
docker compose up -d postgres            # starts a local Postgres
cp .env.example .env                     # if you haven't already
# edit .env: DATABASE_URL=postgresql://match_analysis:devpassword@localhost:5432/match_analysis
python3 migrate_state_to_db.py           # one-off: carries over existing history
python3 run.py --dry-run                 # now reading from Postgres instead
```

The cached parquet files and generated PDFs/HTML are **not** going into
Postgres — they're large, columnar, and a poor fit for relational storage.
They live on a Docker volume instead (matching how the recruiting platform
already stores its own files on the shared droplet).

## Running it

### The dashboard (how this is actually meant to be used)

```bash
python3 webapp.py        # http://localhost:5050
```

Everything lives at one URL. The sidebar switches between:

- **Season Progress** — the live season tracker.
- **Pre-Match Analysis** / **Set Piece Analysis** — every report that
  exists, captioned with its sample and whether it was built automatically
  or picked by hand; click straight through to the PDF.
- **Post-Match Reports** — still hand-maintained; add a finished report's
  card by editing `templates/dashboard.html` directly.
- **Coach's Brief** (`/brief`) — pick a match, paste the coach's notes in
  plain English, see the plain-English readback, confirm, and the match's
  Post-Match Report rebuilds itself with the graded Brief page included.
  No CLI, no editing `match.toml`.
- **Pre-Match Selector** (`/selector`) — pick an opponent from a dropdown
  (populated from the live fixture list, so there's no guessing Impect's
  exact spelling of a club name), describe a sample in free text ("last 10
  games each team", "last 5 cup games each team"), confirm the readback,
  and both the Pre-Match Analysis and Set Piece Report build immediately —
  the second report always shares the first one's sample, since
  `build_corners.py` reads its cache lists straight from
  `build_prematch.py`.

Both free-text steps (`/brief` and `/selector`) go through Claude via
`brief_interpreter.py` / `selector_interpreter.py`, but the model never
computes anything itself — it only *selects and parameterises* entries
from a fixed, pre-verified catalog (`metric_catalog.py` for the Brief,
`match_selector.py` for sample specs). The confirm/readback step exists so
a misread never reaches a report un-checked.

### Automatic data collection

Deployed, `scheduler.py` runs inside its own container (`docker compose up
-d scheduler`) and triggers `run.py` + `build_upcoming.py` every Monday
and Thursday — no laptop needs to be awake, nothing needs a human to
trigger it. Locally, run either script directly whenever you want to pull
by hand:

```bash
python3 run.py                 # post-match: new/revised matches only
python3 run.py --dry-run       # decide and print, touch nothing
python3 run.py --list          # every fixture the accounts can see
python3 run.py --match 285457  # one specific match
python3 run.py --force         # rebuild even if nothing changed

python3 build_upcoming.py      # pre-match + set piece, for whatever's next
```

Everything on a post-match cover — opponent, competition, round, venue,
date, colour — is **derived from the match record**, so `match.toml` needs
no editing between fixtures. Impect labels rounds in German (`1. Runde`,
`12. Spieltag`); those are translated automatically.

#### Why it re-checks matches it has already built

Impect **revises** data after first publication — there is a verification
pass that moves the packing numbers by roughly a fifth, and recalculations
have been observed up to a week after kick-off. The processed-matches
ledger records each match's `lastCalculationDate`, and a report is
rebuilt whenever that moves. A report is therefore never left carrying
pre-verification figures.

#### When to expect data

Measured on the current season, from kick-off to data landing: **nothing
before ~12 hours**, median **18 hours**, **81% within 24h**, **97% within
48h** — though a specific match has been seen taking a full week. A
Saturday afternoon kick-off is typically ready by Sunday morning; the
scheduler just keeps checking until it is.

### Lower-level tools

For a one-off build without going through the web UI:

```bash
python3 fetch_data.py --list-matches      # fixtures + ids
python3 fetch_data.py --match 285457      # pull one match + season context
python3 build_report.py                   # build from whatever is in the cache
python3 build_review.py                   # 4-page excerpt: only what changed vs the last build

PRE_OPP="Wycombe Wanderers" \
PRE_OPP_CACHES="data_wycombe_1,data_wycombe_2,data_wycombe_3" \
PRE_OUR_CACHES="data_swfc,data_wigan,data_barnsley" \
PRE_FIXTURE="League One · Brisbane Road · Saturday 12 September 2026" \
python3 build_prematch.py
python3 build_corners.py    # same env vars - reads OPP_CACHES/OUR_CACHES from build_prematch.py
```

Credentials are read from `.env` by the scripts themselves and are never
passed on the command line, so they cannot end up in shell history or a
terminal log.

For a one-off post-match build without touching `match.toml`:

```bash
PMR_CACHE=data_burton_albion PMR_MATCH_ID=216445 PMR_OPP="Burton Albion" \
  PMR_COMP="League One 25/26" PMR_OPP_C="#F5C518" python3 build_report.py
```

The output filename is derived from the team and opponent, so a stale
setting cannot silently overwrite a different fixture's report.

### Deployment: Docker

```bash
docker compose build app
docker compose up -d postgres
docker compose run --rm app python3 run.py --dry-run
docker compose run --rm app python3 run.py
docker compose up -d scheduler    # automated Mon/Thu pulls, persistent
```

`app`/`scheduler`/`webapp` all write to `APP_DATA_DIR` (`/data` in the
container, see `envconfig.py`), which is the named `appdata` volume in
`docker-compose.yml` locally, and will be whatever's mounted at that path
on the droplet — the code never knows the difference. `.env` is passed in
via `env_file`, never baked into the image (see `.dockerignore`).

Tested end-to-end locally: built the image, pulled a real match through
the containerised pipeline against the compose Postgres, and confirmed
the parquet cache, the PDF, and the state row all landed on the
volume/database rather than local disk.

### Licences

The league competitions and the EFL Cup are on **separate Impect
licences**, and an account may hold one but not the other. `match.toml`
lists the iterations to search (League One 25/26 & 26/27, EFL Cup 25/26 &
26/27); use whichever account covers the match you are building. If an
iteration isn't covered, `run.py` reports it and carries on with the rest
rather than failing.

## File map

| File | Role |
|---|---|
| `run.py` | Routine entry point — pulls new/revised matches, rebuilds post-match reports, regenerates the season tracker |
| `build_upcoming.py` | The pre-match mirror of `run.py` — builds Pre-Match Analysis + Set Piece Report for whatever's next, once, never overwriting a deliberate `/selector` choice |
| `scheduler.py` | Long-lived container process — triggers `run.py` + `build_upcoming.py` every Monday and Thursday |
| `fetch_data.py` | Impect API pull for one match (`pull_match`, shared by `run.py` and the CLI) |
| `build_report.py` | Assembles a post-match report: match stats, set pieces, personnel, out-of-possession, prose Performance Review, Coach's Brief |
| `build_prematch.py` | Opponent-form pre-match report, aggregated across N cached matches per side (or every match this season) |
| `build_corners.py` | Set-piece report for the same fixture as a pre-match build, sharing its cache config — corners, wide free kicks, near-box throw-ins |
| `build_season.py` | Rebuilds `season-tracker.html` from the live match list plus whatever post-match caches exist |
| `build_review.py` | Short excerpt PDF of only the pages that changed since the last build — nothing recomputed, pages are lifted from the full report |
| `webapp.py` | The Flask app — serves the dashboard, `/brief`, `/selector`, and every generated report |
| `match_selector.py` | Turns a `{team, count, competition}` spec into real cache directories against the live fixture list; also owns the `prematch_builds` auto/manual tracking table |
| `match_lookup.py` | Shared opponent/date lookup from the processed-matches ledger, used by both the CLI and the web UI |
| `brief_interpreter.py` | Free-text Coach's Brief notes → structured checks, via Claude — selects from `metric_catalog.py`, never computes anything itself |
| `selector_interpreter.py` | Free-text sample description → `match_selector.py` specs, via Claude |
| `metric_catalog.py` | The fixed, pre-verified catalog of Coach's Brief metrics a check can be built from |
| `coach_brief_engine.py` | Grades one match's Postgres-saved checks against that match's real data |
| `manual_corrections.py` | Video-verified overrides to Impect's own event tagging, applied everywhere set-piece figures are computed |
| `report_config.py` | Match identity, palette, page geometry, pitch drawing, shared table renderers (`stat_table`/`player_table`), shared thresholds |
| `report_metrics.py` | Every metric definition (the auditable layer) |
| `report_pages.py` | One renderer per post-match report page |
| `envconfig.py` | Settings resolution: env var → `match.toml` → default; also defines `DATA_ROOT`/`CACHE_ROOT`/`REPORTS_ROOT` |
| `db.py` | Postgres backend for the processed-matches ledger, Coach's Brief checks, and the pre-match build tracking table |
| `migrate_state_to_db.py` | One-off: imports `state/processed.json` into Postgres |
| `templates/` | Jinja templates for every web page, including `dashboard.html` |
| `Dockerfile` | Builds the pipeline + web app into one image, used three ways — see *Deployment* |
| `docker-compose.yml` | Postgres + the app/scheduler/webapp services |
| `cache/`, `reports/` | Generated — pulled parquet and every built PDF/HTML respectively, under `DATA_ROOT` |
| `test_*_golden.py` | Regression tests pinned to already-verified reports — see *Testing* below |

## Testing

`test_luton_golden.py` and `test_bradford_golden.py` pin known-correct
numbers from two already fully-verified post-match reports (`team_stats()`
rows, `player_table()` rows, `season_context()`'s summary bullets, the
scoreline) and assert the shared code still produces them. They exist so a
future change to shared logic (`report_metrics.py`, `build_report.py`)
can't silently re-break something that was already found and fixed —
Luton's report covers the season-ranking self-inclusion bug specifically;
Bradford's covers own-goal score crediting, the one path Luton's match
doesn't exercise.

```bash
pytest -v                         # both, isolated from each other
python3 test_luton_golden.py -v   # either one standalone
```

Each test's data loading happens in an isolated subprocess
(`_golden_runner.py` via `_golden_helpers.py`), not in the pytest process
itself — `build_report.py` resolves its match config from the environment
only on first import, so two golden tests importing it directly in the same
process would silently share whichever match got imported first.

## Coach's Brief

Grades a match against the coaching staff's real pre-match instructions.
Two questions kept deliberately apart:

- **Compliance** — did the side do what was asked?
- **Effectiveness** — did doing it produce anything?

A side can be fully compliant and still lose. Judging only on outcome bins
a good plan after one bad afternoon; judging only on compliance keeps a
plan that isn't working.

A check is built from a coach's free-text notes (`/brief` in the
dashboard, or `brief_interpreter.py` directly), which the model maps onto
one of a fixed set of pre-verified computations in `metric_catalog.py` — it
never invents a new one. Every match's checks are saved to Postgres
(`coach_brief_checks`) independently of every other match's, so rebuilding
one match's report never risks overwriting another's notes, and a brief
can always be edited and re-saved from its own original wording.

Each check grades on one of three bases:

| basis | meaning |
|---|---|
| `beat_opponent` | compare against the opposition in the same match — needs no historical data, so it survives squad turnover |
| `absolute` | compare against a fixed threshold |
| `vs_baseline` | compare against the team's own previous match(es) |
| `info` | report the value, no pass/fail — used when no defensible threshold exists (e.g. a named opponent player who didn't feature) |

Thresholds are **provisional by default and the page says so**. Replace
them with the staff's own figures *before* kick-off — thresholds picked
after seeing the result are marking your own homework.

**Shared thresholds.** Where the Performance Review and the Coach's Brief
judge the same metric, both read one constant from `report_config.py`
(`SETPIECE_XG_CONCEDED_CONCERN` and friends) so the two pages cannot reach
opposite verdicts on the same number.

**Known gap:** the free-text interpretation layer matches player names as
written in the coach's notes against Impect's own squad list by simple
string matching — a nickname, misspelling, or unusual transliteration can
fail to resolve. The confirm/readback step surfaces this before anything
is saved, but it's worth knowing about rather than assuming every name
will just work.

## Data traps worth knowing

Lessons learned auditing this pipeline against Impect's own matchsums,
which are internally consistent (player totals sum exactly to squad
totals) and therefore the ground truth to check event-summed figures
against.

1. **Own goals.** An own goal's event row is credited to the *scorer's own*
   squad (`GOALS` is null there) — it has to be added to the *other* side's
   tally by hand, or a match with one shows the wrong scoreline.
2. **Duels.** The event feed only ever flags the *winner* of a duel —
   `LOST_*_DUELS` is always null there, and the loser is named in
   `duelPlayerName`. Summing events gives every team a 100% duel rate.
   Duels (and pressures, for the same reason) are taken from the
   **matchsums**, which populate both sides.
3. **First touch on set pieces.** `setPieceSubPhaseFirstTouchPlayerName`
   structurally never names a defender — checked across every corner
   cached in this project. A defender's touch is a plain `action ==
   'HEADER'` event, never linked into that subphase field.
4. **xT (pxT).** Summing `PXT_ATTACK` over events under-reports the
   published total — the feed leaves it null on defensive/second-ball
   events that still carry value, and splits pass value between passer and
   receiver. Taken from the **matchsums** at both team and player level.
5. **Possession & Field Tilt.** Not in the matchsums. They're published by
   a separate endpoint, **`getSquadMatchScores`**, as
   `RATIO_BALL_POSSESSION` and `FIELD_TILT` — deriving them as time-on-ball
   or final-third-event-share instead can overstate a side by several
   points.
6. **Matchsum vs event-sum drift.** The same metric computed via the
   squad-matchsum pipeline and via summing that match's own events can
   differ in the third decimal place — different Impect aggregation paths.
   This matters most in season-pool comparisons: a match's own row in the
   pool can appear to "beat itself" on a rounding hair and claim an
   impossible rank ("6th highest of 5 matches"). Ranking code must exclude
   the current match's own id from its comparison pool, not just from the
   headline "of N matches" count.
7. **Set-piece phase grouping.** Impect groups a "set-piece phase" from the
   restart until its own definition of the phase ending, which can span
   well past the point where play has become normal open football — this
   has produced a real penalty being filed under a throw-in phase.
   `manual_corrections.py` exists for exactly this, and a correction's
   `setPieceId` can go stale if the match is later revised and Impect
   re-numbers the event — the exclusion then silently stops matching and
   needs re-pointing at the new id.
8. **Zone tags vs raw coordinates can disagree.** A set-piece delivery's
   categorical end-zone tag (e.g. `FAR_POST_WIDE`) and the real x/y
   coordinate of who touched it first are two separate fields in Impect's
   data, and they don't always agree — seen directly on a real corner
   where the zone tag said far post but the coordinate placed the touch
   centrally. Diagrams in this project always plot the real coordinate,
   not the tag, so a diagram can be correct even when the tag it's next to
   looks inconsistent with it.
9. **Per-90 extrapolation.** Deliberately not used on single-match player
   pages. Extrapolating one match ×8 produces confident nonsense (an
   11-minute cameo with 4 recoveries becomes 32.7-per-90 and tops a chart
   whose real leader made 22 in 95 minutes). Per-90 belongs in season
   comparisons across many matches, which is why the season tracker's
   benchmarks filter to players with 5+ appearances of 60+ minutes.
