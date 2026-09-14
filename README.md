# Leyton Orient match analysis

An analytics pipeline for **Leyton Orient**, built on **Impect** match-level
API data. It is not a one-off recreation of a single report — it is the
club's running toolkit for analysing every match across a season: scouting
the next opponent before kick-off, debriefing every match Orient plays
afterward, breaking down set-piece routines for both teams, and tracking
form, minutes and player impact across the whole campaign. `dashboard.html`
is the front door to all of it.

The visual language (dark navy ground, geometric sans, two-team-colour
charts) started as a recreation of a single, now-retired Wycombe Wanderers
template PDF — but the report types, the data pipeline, the season tracker,
and the dashboard were all built from there, for every fixture Orient
plays, not for that one match.

## What this produces

| Output | Built by | Covers |
|---|---|---|
| **Post-Match Report** | `build_report.py` | One played match: full team stats, set pieces, personnel, out-of-possession, Performance Review, and (when real notes exist) a graded Coach's Brief |
| **Pre-Match Analysis** | `build_prematch.py` | An upcoming opponent's recent form vs Orient's own, aggregated across each side's last N cached matches |
| **Corner Report** | `build_corners.py` | Both sides' corner routines for the same fixture — delivery zones, takers, first contact, defensive vulnerabilities |
| **Season Tracker** | `build_season.py` | `season-tracker.html` — league table, form, set-piece efficiency, minutes/workload, most-impactful and best/worst individual performances, upcoming fixtures, all for the current season |
| **Dashboard** | hand-maintained | `dashboard.html` — embeds the season tracker and links every report ever built, organised by type |

All of it runs from data pulled once per match and cached locally as
parquet, so report builders work offline and can be re-run freely.

## Architecture

```
Impect API  →  fetch_data.pull_match()  →  DATA_ROOT/cache/<slug>_<matchId>/*.parquet
                                                     │
                     ┌───────────────────────────────┼───────────────────────┐
                     ▼                                ▼                       ▼
              build_report.py                  build_prematch.py       build_season.py
              (one played match)          + build_corners.py           (whole season log)
                     │                          (N cached matches            │
                     ▼                           per side)                   ▼
   DATA_ROOT/reports/Post Match Report*.pdf         │        DATA_ROOT/reports/season-tracker.html
                     │                               ▼                       │
                     └──────────────► dashboard.html ◄─────────────────────┘
                       DATA_ROOT/reports/{Pre-Match Analysis,Corner Report}*.pdf
```

- `run.py` is the routine entry point for **post-match** reports: it checks
  every configured Impect iteration, works out which of the club's own
  matches are new or have been revised, pulls only those, and rebuilds.
- **Pre-match and corner reports** are opponent-form pieces, not tied to one
  of Orient's own matches — they're built by pointing `build_prematch.py` /
  `build_corners.py` at a comma-separated list of single-match caches per
  side (see *Running it* below), and re-added to `dashboard.html` by hand.
- `manual_corrections.py` holds video-verified overrides to Impect's own
  event tagging (e.g. a penalty Impect miscategorised under a throw-in
  phase) — applied automatically wherever set-piece figures are computed,
  so a correction can't be right on one page and wrong on another.
- **`DATA_ROOT`** (see `envconfig.py`) is where all of that actually gets
  written — `cache/` for pulled parquet, `reports/` for every generated PDF
  and HTML page. It defaults to the project directory (today's behaviour,
  unchanged), and becomes the mounted Docker volume once deployed (see
  *Deployment* below). `dashboard.html` is the one exception: it's
  hand-maintained, lives at the repo root regardless of `DATA_ROOT`, and
  just links into `reports/...` — which is why that folder has to sit next
  to it, in Docker as much as locally.

## Setup (once)

```bash
cp .env.example .env
chmod 600 .env
# edit .env and put your Impect username and password in it
```

`.env` holds **only** the credentials and is gitignored — it must never be
committed, emailed, or pasted into a chat window. Match settings live
separately in `match.toml`, which contains no secrets and is safe to share.

Resolution order for every setting is **real environment variable → file →
default**, so a one-off override never requires editing anything.

### Optional: Postgres for the processed-matches ledger

`state/processed.json` (which matches have been built, and Impect's
`lastCalculationDate` for each — how `run.py` knows what's new or revised)
is the one piece of local state genuinely worth a database instead of a
file, ahead of deploying this alongside the recruiting platform on the same
droplet. It's optional and off by default — nothing else here needs it.

```bash
docker compose up -d                    # starts a local Postgres
cp .env.example .env                    # if you haven't already
# edit .env: DATABASE_URL=postgresql://match_analysis:devpassword@localhost:5432/match_analysis
python3 migrate_state_to_db.py          # one-off: carries over existing history
python3 run.py --dry-run                # now reading from Postgres instead
```

Unset `DATABASE_URL` (or leave it out of `.env` entirely) and everything
goes straight back to `state/processed.json` on local disk — `db.py`'s
`available()` check is the only thing that branches, so this is a strict
opt-in, not a new hard dependency. `build_season.py` picks up the same
backend automatically, since it imports `load_state` from `run.py` rather
than reading the file itself.

The cached parquet files and generated PDFs/HTML are **not** going into
Postgres — they're large, columnar, and a poor fit for relational storage.
Containerised, they're the part that lives on a Docker volume (matching how
the recruiting platform already stores its own files on the droplet), not
the database — see *Deployment* below.

## Running it

### Post-match reports — one command

```bash
python3 run.py
```

Run it whenever you know data has landed. **Nothing runs on a timer.** It
checks every configured iteration across both accounts, works out what is
new or has been revised, pulls only that, rebuilds the report, and
regenerates `season-tracker.html`.

```bash
python3 run.py --dry-run       # decide and print, touch nothing
python3 run.py --list          # every fixture the accounts can see
python3 run.py --match 285457  # one specific match
python3 run.py --force         # rebuild even if nothing changed
```

Everything on the cover — opponent, competition, round, venue, date,
colour — is **derived from the match record**, so `match.toml` needs no
editing between fixtures. Impect labels rounds in German (`1. Runde`,
`12. Spieltag`); those are translated automatically.

#### Why it re-checks matches it has already built

Impect **revises** data after first publication — there is a verification
pass that moves the packing numbers by roughly a fifth, and recalculations
have been observed up to a week after kick-off. `state/processed.json`
records each match's `lastCalculationDate`, and a report is rebuilt whenever
that moves. A report is therefore never left carrying pre-verification
figures.

#### When to run it

Measured on the current season, from kick-off to data landing: **nothing
before ~12 hours**, median **18 hours**, **81% within 24h**, **97% within
48h**. A Saturday afternoon kick-off is typically ready by Sunday morning.
There is no point checking the same evening.

### Pre-match analysis and corner reports

These aren't tied to one of Orient's own matches, so they aren't part of
`run.py`'s loop. Pull each side's recent matches individually, then point
the builder at the cache list:

```bash
# pull however many recent matches you want per side (fetch_data.pull_match,
# or fetch_data.py --match <id> for a one-off)

PRE_OPP="Wycombe Wanderers" \
PRE_OPP_CACHES="data_wycombe_1,data_wycombe_2,data_wycombe_3,data_wycombe_4,data_wycombe_5" \
PRE_OUR_CACHES="data_swfc,data_wigan,data_barnsley,data_bromley,data_luton" \
PRE_OUR_COMP="League One" \
PRE_FIXTURE="League One - Brisbane Road - Saturday 12 September 2026" \
python3 build_prematch.py

# same env vars, same cache lists - build_corners.py reads them from
# build_prematch.py directly, so both reports stay in sync automatically
python3 build_corners.py
```

Sample size isn't fixed — Walsall's brief used their last 2 League Two
games; Wycombe's used 5 League One games each side. Every match-count
caption on the page is computed from the actual cache list, not assumed.

### Lower-level tools

```bash
python3 fetch_data.py --list-matches      # fixtures + ids
python3 fetch_data.py --match 285457      # pull one match + season context
python3 build_report.py                   # build from whatever is in the cache
python3 build_review.py                   # 4-page excerpt: only what changed vs the last build
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

The image is the pipeline, not a web server — there isn't one yet, so the
files this produces are opened directly (or served some other way) until
that's built. Deployed, it's invoked on demand rather than left running,
matching how `run.py` already describes its own usage.

```bash
docker compose build app
docker compose up -d postgres
docker compose run --rm app python3 run.py --dry-run
docker compose run --rm app python3 run.py
```

`app` writes to `APP_DATA_DIR` (`/data` in the container, see
`envconfig.py`), which is the named `appdata` volume in
`docker-compose.yml` locally, and will be whatever's mounted at that path
on the droplet — the code never knows the difference. `.env` is passed in
via `env_file`, never baked into the image (see `.dockerignore`).

Tested end-to-end locally: built the image, pulled a real match through the
containerised pipeline against the compose Postgres, and confirmed the
parquet cache, the PDF, and the state row all landed on the volume/database
rather than local disk.

Not yet built: the serving layer (something to actually host `reports/` and
gate it behind a login) and the sign-in page itself — see the project's own
notes on that; it's a deliberately separate decision from this piece.

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
| `fetch_data.py` | Impect API pull for one match (`pull_match`, shared by `run.py` and the CLI) |
| `build_report.py` | Assembles a post-match report: match stats, set pieces, personnel, out-of-possession, Performance Review, Coach's Brief |
| `build_prematch.py` | Opponent-form pre-match report, aggregated across N cached matches per side |
| `build_corners.py` | Corner-routine report for the same fixture as a pre-match build, sharing its cache config |
| `build_season.py` | Rebuilds `season-tracker.html` from the live match list plus whatever post-match caches exist |
| `build_review.py` | Short excerpt PDF of only the pages that changed since the last build — nothing recomputed, pages are lifted from the full report |
| `coach_brief.py` | Grades one match against the coaching staff's real pre-match instructions — see below |
| `manual_corrections.py` | Video-verified overrides to Impect's own event tagging, applied everywhere set-piece figures are computed |
| `report_config.py` | Match identity, palette, page geometry, pitch drawing, shared thresholds |
| `report_metrics.py` | Every metric definition (the auditable layer) |
| `report_pages.py` | One renderer per post-match report page |
| `envconfig.py` | Settings resolution: env var → `match.toml` → default; also defines `DATA_ROOT`/`CACHE_ROOT`/`REPORTS_ROOT` |
| `db.py` | Optional Postgres backend for the processed-matches ledger — see *Setup* above |
| `migrate_state_to_db.py` | One-off: imports `state/processed.json` into Postgres |
| `Dockerfile` | Builds the pipeline (not a server yet) into an image — see *Deployment* |
| `docker-compose.yml` | Local Postgres + the app image, for testing before deployment |
| `dashboard.html` | Hand-maintained hub — embeds the season tracker, links every report ever built |
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

**`coach_brief.py` holds one match's config at a time.** It's rewritten for
whichever match currently has real dictated notes, and `build_report.py`
grades whatever match it's pointed at against whatever `coach_brief.py`
currently holds — so a rebuild of a *different* match must either pass
`PMR_SKIP_COACH_BRIEF=1` (no real notes exist for it) or have that match's
own brief reconstructed first. Once a match's real Brief has been
overwritten by a later one, it can only be recovered by reading the
instruction wording back off the already-shipped PDF page and
re-verifying every figure against the match's own data — never by copying
the old numbers, since a data revision or bug fix may have changed them.

Each check grades on one of three bases, set in `TARGETS` at the top of the
file:

| basis | meaning |
|---|---|
| `beat_opponent` | compare against the opposition in the same match — needs no historical data, so it survives squad turnover |
| `absolute` | compare against a fixed threshold |
| `info` | report the value, no pass/fail — used when no defensible threshold exists (e.g. a named opponent player who didn't feature) |

Thresholds are **provisional by default and the page says so**. Replace
them with the staff's own figures *before* kick-off — thresholds picked
after seeing the result are marking your own homework.

**Shared thresholds.** Where the Performance Review and the Coach's Brief
judge the same metric, both read one constant from `report_config.py`
(`SETPIECE_XG_CONCEDED_CONCERN` and friends) so the two pages cannot reach
opposite verdicts on the same number.

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
8. **Per-90 extrapolation.** Deliberately not used on single-match player
   pages. Extrapolating one match ×8 produces confident nonsense (an
   11-minute cameo with 4 recoveries becomes 32.7-per-90 and tops a chart
   whose real leader made 22 in 95 minutes). Per-90 belongs in season
   comparisons across many matches, which is why the season tracker's
   benchmarks filter to players with 5+ appearances of 60+ minutes.
