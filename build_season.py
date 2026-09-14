"""Season tracker: form, minutes and player-impact views for the current season.

    python3 build_season.py

Rebuilds season-tracker.html from every playable match in the club's current
season. The results log comes straight from the live match list (so it is
never stale); the player views use whatever post-match caches run.py has
already pulled - a match without a local ps.parquet is skipped for those
views and named in the page's own footnote rather than silently dropped.

Run it after run.py brings in a new match. Nothing here pulls fresh data on
its own beyond the match list itself.
"""

import html
import warnings
from datetime import datetime, timezone

import pandas as pd

warnings.filterwarnings("ignore")

from envconfig import load_env, load_match, ROOT, CACHE_ROOT, REPORTS_ROOT
from run import discover, slug, pretty_matchday, load_state, OPP_COLOURS, DEFAULT_OPP_COLOUR
import manual_corrections as MC

REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
OUT = REPORTS_ROOT / "season-tracker.html"

MIN_MINUTES_ROLLING = 180   # floor for the rolling "most impactful" ranking
MIN_MINUTES_SINGLE = 45     # floor for a single game to count for best/worst


def resolve_cache(row, team, cfg, state):
    """Local cache dir for one match, or None if we haven't pulled it."""
    key = str(int(row.id))
    if key in state:
        d = CACHE_ROOT / state[key]["cache"]
        return d if d.exists() else None
    if int(row.id) == int(cfg.get("id", -1)):
        d = CACHE_ROOT / cfg.get("cache", "data")
        return d if d.exists() else None
    opp = row.awaySquadName if row.homeSquadName == team else row.homeSquadName
    guess = CACHE_ROOT / slug(opp)
    return guess if guess.exists() else None


def fmt_date(ts) -> str:
    return ts.strftime("%a %-d %b %Y")


def build_log(matches, team, cfg, state):
    fallback_id = int(cfg.get("id", 0))
    fb_row = matches[matches.id == fallback_id]
    season = fb_row.season.iloc[0] if len(fb_row) else matches.season.max()

    cur = (matches[(matches.season == season) & (matches.available == True)]
           .sort_values("scheduledDate"))

    log = []
    for _, row in cur.iterrows():
        home = row.homeSquadName == team
        opp = row.awaySquadName if home else row.homeSquadName
        gf = int(row.homeSquadGoals if home else row.awaySquadGoals)
        ga = int(row.awaySquadGoals if home else row.homeSquadGoals)
        is_cup = "cup" in str(row.competitionName).lower()
        result = "W" if gf > ga else ("L" if gf < ga else "D")
        log.append(dict(
            id=int(row.id),
            date=pd.to_datetime(row.scheduledDate, utc=True),
            comp=str(row.competitionName),
            matchday=pretty_matchday(row.matchDayName),
            opponent=str(opp), home=home, gf=gf, ga=ga, result=result,
            is_cup=is_cup, cache=resolve_cache(row, team, cfg, state),
            colour=OPP_COLOURS.get(str(opp), DEFAULT_OPP_COLOUR),
        ))

    pts_map = {"W": 3, "D": 1, "L": 0}
    running_pts = running_gf = running_ga = 0
    for m in log:
        if not m["is_cup"]:
            running_pts += pts_map[m["result"]]
            running_gf += m["gf"]
            running_ga += m["ga"]
        m["running_pts"], m["running_gf"], m["running_ga"] = (
            running_pts, running_gf, running_ga)

    return log, season


def build_upcoming(matches, team, season, n=5):
    """Next N fixtures still to be played - kept separate from matches that
    have already kicked off but that Impect hasn't published data for yet,
    since those aren't "upcoming" even though available is still False.
    """
    unplayed = matches[(matches.season == season) & (matches.available == False)].copy()
    unplayed["_dt"] = pd.to_datetime(unplayed.scheduledDate, utc=True)
    now = pd.Timestamp.now(tz="UTC")

    def as_row(row):
        home = row.homeSquadName == team
        opp = row.awaySquadName if home else row.homeSquadName
        return dict(
            date=row._dt, comp=str(row.competitionName),
            matchday=pretty_matchday(row.matchDayName),
            opponent=str(opp), home=home,
            is_cup="cup" in str(row.competitionName).lower(),
            colour=OPP_COLOURS.get(str(opp), DEFAULT_OPP_COLOUR),
        )

    fut = unplayed[unplayed._dt >= now].sort_values("_dt").head(n)
    pending = unplayed[unplayed._dt < now].sort_values("_dt")
    return [as_row(r) for _, r in fut.iterrows()], [as_row(r) for _, r in pending.iterrows()]


def division_matches(fb_row, apis):
    """Every played match in Orient's own league iteration this season - the
    shared basis for the league table and the percentile rankings, so both
    read the same live pull instead of hitting the API twice.
    """
    if not len(fb_row):
        return None, pd.DataFrame()
    iteration = int(fb_row._iteration.iloc[0])
    acct = fb_row._account.iloc[0]
    api = apis.get(acct)
    if api is None:
        return None, pd.DataFrame()
    try:
        m = api.getMatches(iteration=iteration)
    except Exception:
        return api, pd.DataFrame()
    return api, m[m.available == True]


def league_table(div_matches, team):
    """Full division standings, computed from raw results - won't reflect an
    external points deduction, since Impect doesn't encode that."""
    if not len(div_matches):
        return pd.DataFrame()

    rows = {}
    for _, r in div_matches.iterrows():
        for name, gf, ga in ((r.homeSquadName, r.homeSquadGoals, r.awaySquadGoals),
                              (r.awaySquadName, r.awaySquadGoals, r.homeSquadGoals)):
            gf, ga = int(gf), int(ga)
            d = rows.setdefault(name, dict(played=0, w=0, d=0, l=0, gf=0, ga=0))
            d["played"] += 1
            d["gf"] += gf
            d["ga"] += ga
            if gf > ga:
                d["w"] += 1
            elif gf == ga:
                d["d"] += 1
            else:
                d["l"] += 1

    tbl = pd.DataFrame([{"squad": k, **v} for k, v in rows.items()])
    if not len(tbl):
        return tbl
    tbl["gd"] = tbl.gf - tbl.ga
    tbl["pts"] = tbl.w * 3 + tbl.d
    tbl = (tbl.sort_values(["pts", "gd", "gf"], ascending=False)
              .reset_index(drop=True))
    tbl["pos"] = tbl.index + 1
    return tbl


PERCENTILE_METRICS = [
    # column, label, higher-is-better, display format
    ("RATIO_BALL_POSSESSION", "Possession", True, "{:.0%}"),
    ("FIELD_TILT", "Field Tilt", True, "{:.0%}"),
    ("MEAN_PRESSURE_HEIGHT", "Press Height", True, "{:.1f}m"),
    ("PPDA", "Pressing Intensity", False, "{:.1f} PPDA"),
    ("PXT_POSITIVE", "Attacking Threat (xT)", True, "{:.2f}"),
    ("RATIO_PASSING_ACCURACY", "Passing Accuracy", True, "{:.0%}"),
]


def league_percentiles(api, div_matches, team):
    """Orient's season-to-date average for each metric, ranked against every
    other squad's season-to-date average in the same league matches. Doesn't
    cover set-piece threat specifically - that needs its own pull from the
    set-piece feed rather than squad match scores.
    """
    if api is None or not len(div_matches):
        return []
    ids = [int(i) for i in div_matches.id.tolist()]
    try:
        sc = api.getSquadMatchScores(matches=ids)
    except Exception:
        return []

    cols = [c for c, *_ in PERCENTILE_METRICS]
    agg = sc.groupby("squadName")[cols].mean()

    rows = []
    for col, label, higher_better, fmt in PERCENTILE_METRICS:
        vals = agg[col].dropna()
        if team not in vals.index or len(vals) < 2:
            continue
        us = vals[team]
        n = len(vals)
        rank = int((vals > us).sum() + 1) if higher_better else int((vals < us).sum() + 1)
        pct = round((n - rank) / (n - 1) * 100)
        rows.append(dict(label=label, value=us, fmt=fmt, pct=pct, rank=rank, n=n))
    return rows


def build_league_extras(log, team):
    """League-only extras: clean sheets, points per game, and goals scored /
    conceded directly from a set piece - every setPieceCategory Impect
    tracks (corners, free kicks, throw-ins), matching how the rest of this
    project already sums setPieceSubPhase_SHOT_XG across all of them, not
    just corners. Needs each match's setpieces.parquet, so a match without
    one is skipped and named, same as the player views do.
    """
    league = [m for m in log if not m["is_cup"]]
    played = len(league)
    clean_sheets = sum(1 for m in league if m["ga"] == 0)
    pts_map = {"W": 3, "D": 1, "L": 0}
    pts = sum(pts_map[m["result"]] for m in league)
    ppg = pts / played if played else 0.0

    sp_for = sp_against = 0
    missing = []
    for m in league:
        f = m["cache"] / "setpieces.parquet" if m["cache"] else None
        if f is None or not f.exists():
            missing.append(m["opponent"])
            continue
        setp = MC.apply(pd.read_parquet(f), m["id"])
        sp_for += setp[setp.attackingSquadName == team].setPieceSubPhase_GOALS.sum()
        sp_against += setp[(setp.attackingSquadName == m["opponent"])
                           & (setp.defendingSquadName == team)].setPieceSubPhase_GOALS.sum()

    return dict(played=played, clean_sheets=clean_sheets, ppg=ppg,
                sp_for=int(sp_for), sp_against=int(sp_against),
                n_sp_matches=played - len(missing), missing=missing)


def build_player_frames(log, team):
    frames, missing = [], []
    for m in log:
        f = m["cache"] / "ps.parquet" if m["cache"] else None
        if f is None or not f.exists():
            missing.append(m["opponent"])
            continue
        ps = pd.read_parquet(f)
        ps = ps[ps.squadName == team]
        # Impect gives one row per position a player occupied in the match -
        # collapse those back into one row per player before aggregating.
        g = (ps.groupby(["playerId", "playerName"], as_index=False)
               .agg(position=("position", lambda s: s.mode().iat[0] if len(s) else ""),
                    playDuration=("playDuration", "sum"),
                    PXT_ATTACK=("PXT_ATTACK", "sum"),
                    PXT_DEFEND=("PXT_DEFEND", "sum"),
                    GOALS=("GOALS", "sum"),
                    ASSISTS=("ASSISTS", "sum")))
        g["matchId"] = m["id"]
        g["opponent"] = m["opponent"]
        g["date"] = m["date"]
        frames.append(g)
    players = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return players, missing


PRETTY_POS = {
    "GOALKEEPER": "GK", "CENTRAL_DEFENDER": "CB",
    "LEFT_WINGBACK_DEFENDER": "LWB", "RIGHT_WINGBACK_DEFENDER": "RWB",
    "DEFENSE_MIDFIELD": "DM", "CENTRAL_MIDFIELD": "CM",
    "LEFT_WINGER": "LW", "RIGHT_WINGER": "RW", "CENTER_FORWARD": "CF",
}


def pos(p):
    return PRETTY_POS.get(p, p.replace("_", " ").title() if p else "")


def f2(x: float) -> str:
    """.2f that never prints a confusing '-0.00' for a rounding-to-zero value."""
    v = round(x, 2)
    return f"{v + 0.0:.2f}"


def minutes_table(players):
    if not len(players):
        return pd.DataFrame()
    m = (players.groupby(["playerId", "playerName"], as_index=False)
                 .agg(position=("position", lambda s: s.mode().iat[0] if len(s) else ""),
                      apps=("matchId", "nunique"),
                      mins=("playDuration", "sum"),
                      goals=("GOALS", "sum"), assists=("ASSISTS", "sum")))
    m["mins"] = m.mins / 60
    m["avg_mins"] = m.mins / m.apps
    return m.sort_values("mins", ascending=False)


def impact_table(players):
    if not len(players):
        return pd.DataFrame()
    i = (players.groupby(["playerId", "playerName"], as_index=False)
                 .agg(position=("position", lambda s: s.mode().iat[0] if len(s) else ""),
                      apps=("matchId", "nunique"),
                      mins=("playDuration", "sum"),
                      xt=("PXT_ATTACK", "sum"), defxt=("PXT_DEFEND", "sum"),
                      goals=("GOALS", "sum"), assists=("ASSISTS", "sum")))
    i["mins"] = i.mins / 60
    i = i[i.mins >= MIN_MINUTES_ROLLING].copy()
    i["xt90"] = i.xt / (i.mins / 90)
    return i.sort_values("xt90", ascending=False)


def single_match_extremes(players):
    if not len(players):
        return pd.DataFrame(), pd.DataFrame()
    s = players.copy()
    s["mins"] = s.playDuration / 60
    s = s[s.mins >= MIN_MINUTES_SINGLE]
    best = s.sort_values("PXT_ATTACK", ascending=False).head(5)
    worst = s.sort_values("PXT_ATTACK", ascending=True).head(5)
    return best, worst


# ---------------------------------------------------------------- rendering

def result_pill(r):
    colour = {"W": "#8FE388", "D": "#8B97A8", "L": "#FF2D2D"}[r]
    return f'<span class="pill" style="background:{colour}22;color:{colour};border:1px solid {colour}55">{r}</span>'


def log_rows(log):
    rows = []
    for m in log:
        venue = "H" if m["home"] else "A"
        score = f'{m["gf"]}–{m["ga"]}' if m["home"] else f'{m["ga"]}–{m["gf"]}'
        comp_tag = "EFL Cup" if m["is_cup"] else "League"
        pts_cell = "" if m["is_cup"] else f'{m["running_pts"]} pts · GD {m["running_gf"]-m["running_ga"]:+d}'
        rows.append(f'''
      <tr>
        <td class="mut">{fmt_date(m["date"])}</td>
        <td><span class="comp-tag">{html.escape(comp_tag)}</span></td>
        <td><span class="bar" style="background:{m["colour"]}"></span>{html.escape(m["opponent"])} <span class="mut">({venue})</span></td>
        <td class="num">{score}</td>
        <td>{result_pill(m["result"])}</td>
        <td class="mut num">{pts_cell}</td>
      </tr>''')
    return "".join(rows)


def upcoming_rows(fixtures):
    rows = []
    for m in fixtures:
        venue = "H" if m["home"] else "A"
        comp_tag = "EFL Cup" if m["is_cup"] else "League"
        rows.append(f'''
      <tr>
        <td class="mut">{m["date"].strftime("%a %-d %b %Y")}</td>
        <td class="mut">{m["date"].strftime("%H:%M")} UTC</td>
        <td><span class="comp-tag">{html.escape(comp_tag)}</span></td>
        <td><span class="bar" style="background:{m["colour"]}"></span>{html.escape(m["opponent"])} <span class="mut">({venue})</span></td>
        <td class="mut">{html.escape(m["matchday"])}</td>
      </tr>''')
    return "".join(rows)


def pending_note(pending):
    if not pending:
        return ""
    bits = [f'{html.escape(m["opponent"])} ({m["date"].strftime("%-d %b")})' for m in pending]
    return (f'<div class="note">Already played but not yet published by Impect: '
            f'{", ".join(bits)}. Will move into the match log once the data lands.</div>')


def percentile_rows(pcts):
    rows = []
    for r in pcts:
        cls = "good" if r["pct"] >= 60 else ("bad" if r["pct"] <= 40 else "mut")
        rows.append(f'''
      <div class="pctl-row">
        <div class="pctl-label">{html.escape(r["label"])}</div>
        <div class="pctl-track"><div class="pctl-fill" style="width:{r['pct']}%"></div></div>
        <div class="pctl-val mut">{r['fmt'].format(r['value'])}</div>
        <div class="pctl-rank {cls}">{r['rank']} of {r['n']}</div>
      </div>''')
    return "".join(rows)


def league_table_rows(tbl, team):
    rows = []
    for _, r in tbl.iterrows():
        us = r.squad == team
        row_cls = ' class="us"' if us else ""
        name = f'<strong>{html.escape(r.squad)}</strong>' if us else html.escape(r.squad)
        rows.append(f'''
      <tr{row_cls}>
        <td class="mut">{int(r.pos)}</td>
        <td>{name}</td>
        <td class="num">{int(r.played)}</td>
        <td class="num mut">{int(r.w)}</td>
        <td class="num mut">{int(r.d)}</td>
        <td class="num mut">{int(r.l)}</td>
        <td class="num mut">{int(r.gf)}</td>
        <td class="num mut">{int(r.ga)}</td>
        <td class="num">{int(r.gd):+d}</td>
        <td class="num good">{int(r.pts)}</td>
      </tr>''')
    return "".join(rows)


def minutes_rows(m):
    rows = []
    for _, r in m.iterrows():
        rows.append(f'''
      <tr>
        <td>{html.escape(r.playerName)}</td>
        <td class="mut">{pos(r.position)}</td>
        <td class="num">{int(r.apps)}</td>
        <td class="num">{r.mins:.0f}</td>
        <td class="num mut">{r.avg_mins:.0f}</td>
        <td class="num">{int(r.goals)}</td>
        <td class="num">{int(r.assists)}</td>
      </tr>''')
    return "".join(rows)


def impact_rows(i):
    rows = []
    for rank, (_, r) in enumerate(i.iterrows(), start=1):
        rows.append(f'''
      <tr>
        <td class="mut">{rank}</td>
        <td>{html.escape(r.playerName)}</td>
        <td class="mut">{pos(r.position)}</td>
        <td class="num">{int(r.apps)}</td>
        <td class="num">{r.mins:.0f}</td>
        <td class="num good">{f2(r.xt90)}</td>
        <td class="num mut">{f2(r.xt)}</td>
        <td class="num mut">{f2(r.defxt)}</td>
        <td class="num">{int(r.goals)}</td>
        <td class="num">{int(r.assists)}</td>
      </tr>''')
    return "".join(rows)


def extreme_rows(df, good):
    rows = []
    cls = "good" if good else "bad"
    for _, r in df.iterrows():
        rows.append(f'''
      <tr>
        <td>{html.escape(r.playerName)}</td>
        <td class="mut">{pos(r.position)}</td>
        <td class="mut">v {html.escape(r.opponent)}</td>
        <td class="mut">{fmt_date(r.date)}</td>
        <td class="num {cls}">{f2(r.PXT_ATTACK)}</td>
        <td class="num mut">{r.playDuration/60:.0f}’</td>
      </tr>''')
    return "".join(rows)


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Leyton Orient — Season Tracker</title>
<style>
  :root {{
    --bg: #171E29; --panel: #1E2735; --ink: #FFFFFF; --muted: #8B97A8;
    --grid: #2E3949; --team: #E4002B; --good: #8FE388; --bad: #FF2D2D;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--ink);
    font-family: -apple-system, "Avenir Next", "Helvetica Neue", Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  header {{ padding: 48px 40px 28px; border-bottom: 1px solid var(--grid); }}
  header .kicker {{
    color: var(--team); font-weight: 700; font-size: 12px; letter-spacing: 0.14em;
    text-transform: uppercase; margin-bottom: 10px;
  }}
  header h1 {{ margin: 0 0 8px; font-size: 34px; font-weight: 800; letter-spacing: -0.01em; }}
  header p {{ margin: 0; color: var(--muted); font-size: 14.5px; max-width: 700px; line-height: 1.5; }}
  header a {{ color: var(--muted); }}
  main {{ max-width: 1200px; margin: 0 auto; padding: 36px 40px 80px; }}
  section {{ margin-bottom: 52px; }}
  section h2 {{
    font-size: 13px; font-weight: 700; letter-spacing: 0.10em; text-transform: uppercase;
    margin: 0 0 4px; display: flex; align-items: center; gap: 10px;
  }}
  section .desc {{ color: var(--muted); font-size: 13px; margin: 0 0 18px; max-width: 760px; line-height: 1.5; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 20px; max-width: 640px; }}
  .stat {{ background: var(--panel); border: 1px solid var(--grid); border-radius: 10px; padding: 14px 16px; }}
  .stat .n {{ font-size: 24px; font-weight: 800; }}
  .stat .l {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-top: 2px; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--grid); border-radius: 10px; overflow: hidden; }}
  .table-wrap {{ overflow-x: auto; border-radius: 10px; }}
  th, td {{ padding: 10px 14px; text-align: left; font-size: 13px; border-bottom: 1px solid var(--grid); white-space: nowrap; }}
  th {{ color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }}
  tr:last-child td {{ border-bottom: none; }}
  tr.us {{ background: #E4002B14; }}
  tr.us td {{ border-bottom-color: #E4002B33; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  th.num {{ text-align: right; }}
  .mut {{ color: var(--muted); }}
  .good {{ color: var(--good); font-weight: 700; }}
  .bad {{ color: var(--bad); font-weight: 700; }}
  .pill {{ display: inline-block; padding: 2px 8px; border-radius: 5px; font-size: 11px; font-weight: 800; }}
  .comp-tag {{ font-size: 10.5px; color: var(--muted); border: 1px solid var(--grid); border-radius: 5px; padding: 2px 6px; }}
  .bar {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; }}
  .split {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 800px) {{ .split {{ grid-template-columns: 1fr; }} }}
  .pctl-panel {{ background: var(--panel); border: 1px solid var(--grid); border-radius: 10px; padding: 6px 18px; max-width: 760px; }}
  .pctl-row {{ display: flex; align-items: center; gap: 14px; padding: 12px 0; border-bottom: 1px solid var(--grid); }}
  .pctl-row:last-child {{ border-bottom: none; }}
  .pctl-label {{ width: 150px; flex-shrink: 0; font-size: 13px; }}
  .pctl-track {{ flex: 1; height: 8px; background: #2E394980; border-radius: 4px; overflow: hidden; }}
  .pctl-fill {{ height: 100%; border-radius: 4px; background: var(--team); }}
  .pctl-val {{ width: 90px; text-align: right; font-size: 12.5px; font-variant-numeric: tabular-nums; }}
  .pctl-rank {{ width: 70px; text-align: right; font-size: 12px; font-weight: 700; flex-shrink: 0; }}
  @media (max-width: 640px) {{ .pctl-label {{ width: 110px; font-size: 12px; }} .pctl-val {{ width: 70px; }} }}
  .note {{
    margin-top: 10px; padding: 12px 16px; background: #1B2432; border: 1px solid var(--grid);
    border-radius: 8px; color: var(--muted); font-size: 12.5px; line-height: 1.6;
  }}
  footer {{ max-width: 1200px; margin: 0 auto; padding: 0 40px 50px; color: var(--muted); font-size: 11.5px; }}
</style>
</head>
<body>

<header>
  <div class="kicker">Leyton Orient FC</div>
  <h1>Season Tracker</h1>
  <p>{season} · {n_played} match{plural} played so far. Form and results come from
     the live fixture list; minutes and impact use whatever post-match data has been
     pulled so far. <a href="dashboard.html">← back to dashboard</a></p>
  <p class="mut" style="font-size:11.5px;margin-top:10px;">Generated {generated}</p>
</header>

<main>

  <section>
    <h2>Season Record <span class="mut" style="text-transform:none;font-weight:500;">league only</span></h2>
    <p class="desc">Cup results are shown in the match log below but don't count toward this record.</p>
    <div class="cards">
      <div class="stat"><div class="n">{lg_played}</div><div class="l">Played</div></div>
      <div class="stat"><div class="n good">{lg_w}</div><div class="l">Won</div></div>
      <div class="stat"><div class="n mut">{lg_d}</div><div class="l">Drawn</div></div>
      <div class="stat"><div class="n bad">{lg_l}</div><div class="l">Lost</div></div>
      <div class="stat"><div class="n">{lg_gd:+d}</div><div class="l">Goal Diff</div></div>
      <div class="stat"><div class="n">{lg_pts}</div><div class="l">Points</div></div>
    </div>
  </section>

  <section>
    <h2>Set Pieces &amp; Efficiency <span class="mut" style="text-transform:none;font-weight:500;">league only</span></h2>
    <p class="desc">Set-piece goals cover every restart Impect tracks - corners, free kicks and throw-ins -
       not corners alone.{sp_missing_note}</p>
    <div class="cards">
      <div class="stat"><div class="n good">{clean_sheets}</div><div class="l">Clean Sheets</div></div>
      <div class="stat"><div class="n">{ppg:.2f}</div><div class="l">Points / Game</div></div>
      <div class="stat"><div class="n good">{sp_for}</div><div class="l">Scored (set piece)</div></div>
      <div class="stat"><div class="n bad">{sp_against}</div><div class="l">Conceded (set piece)</div></div>
    </div>
  </section>

  <section>
    <h2>League Table</h2>
    <p class="desc">Built from every played match in {season}, not just Orient's — so it reflects the
       whole division. Won't reflect an external points deduction, since that isn't part of the match data.</p>
    <div class="table-wrap"><table>
      <thead><tr><th>Pos</th><th>Team</th><th class="num">P</th><th class="num">W</th><th class="num">D</th><th class="num">L</th><th class="num">GF</th><th class="num">GA</th><th class="num">GD</th><th class="num">Pts</th></tr></thead>
      <tbody>{table_rows}</tbody>
    </table></div>
  </section>

  <section>
    <h2>League-Wide Percentile Rankings</h2>
    <p class="desc">Orient's season-to-date average against every other League One squad's season-to-date average,
       same matches as the table above. Doesn't cover set-piece threat specifically - that would need its own pull.</p>
    <div class="pctl-panel">{pctl_rows}</div>
  </section>

  <section>
    <h2>Upcoming Fixtures</h2>
    <p class="desc">Next {n_upcoming} match{up_plural} on the calendar.</p>
    <div class="table-wrap"><table>
      <thead><tr><th>Date</th><th>Kickoff</th><th>Comp</th><th>Opponent</th><th>Matchday</th></tr></thead>
      <tbody>{upcoming_rows}</tbody>
    </table></div>{pending_note}
  </section>

  <section>
    <h2>Match Log</h2>
    <div class="table-wrap"><table>
      <thead><tr><th>Date</th><th>Comp</th><th>Opponent</th><th class="num">Score</th><th>Result</th><th class="num">League after</th></tr></thead>
      <tbody>{log_rows}</tbody>
    </table></div>
  </section>

  <section>
    <h2>Minutes &amp; Workload</h2>
    <p class="desc">Total minutes played and appearances across {n_player_matches} match{p_plural} with post-match data pulled.{missing_note}</p>
    <div class="table-wrap"><table>
      <thead><tr><th>Player</th><th>Pos</th><th class="num">Apps</th><th class="num">Mins</th><th class="num">Avg/game</th><th class="num">G</th><th class="num">A</th></tr></thead>
      <tbody>{minutes_rows}</tbody>
    </table></div>
  </section>

  <section>
    <h2>Most Impactful Players</h2>
    <p class="desc">Ranked by xT per 90 (Impect's attacking threat-added metric, the same one used on the Performance Review page) —
       minimum {min_roll} minutes so a single cameo can't top the list. Defensive xT is shown for context but isn't part of the ranking:
       this measures threat created in possession, not defensive value.</p>
    <div class="table-wrap"><table>
      <thead><tr><th></th><th>Player</th><th>Pos</th><th class="num">Apps</th><th class="num">Mins</th><th class="num">xT/90</th><th class="num">Total xT</th><th class="num">Def. xT</th><th class="num">G</th><th class="num">A</th></tr></thead>
      <tbody>{impact_rows}</tbody>
    </table></div>
  </section>

  <section>
    <h2>Best &amp; Worst Individual Performances</h2>
    <p class="desc">Single-match xT, minimum {min_single} minutes played.</p>
    <div class="split">
      <div>
        <p class="mut" style="font-size:12px;text-transform:uppercase;letter-spacing:0.05em;margin:0 0 8px;">Best</p>
        <div class="table-wrap"><table>
          <thead><tr><th>Player</th><th>Pos</th><th>Opponent</th><th>Date</th><th class="num">xT</th><th class="num">Mins</th></tr></thead>
          <tbody>{best_rows}</tbody>
        </table></div>
      </div>
      <div>
        <p class="mut" style="font-size:12px;text-transform:uppercase;letter-spacing:0.05em;margin:0 0 8px;">Worst</p>
        <div class="table-wrap"><table>
          <thead><tr><th>Player</th><th>Pos</th><th>Opponent</th><th>Date</th><th class="num">xT</th><th class="num">Mins</th></tr></thead>
          <tbody>{worst_rows}</tbody>
        </table></div>
      </div>
    </div>
  </section>

  <div class="note">
    Generated by build_season.py — rerun after run.py pulls a new match to bring this up to date.
    "Season Record" is Orient's own results only, not a full division table.
  </div>

</main>

<footer>
  Local page — open season-tracker.html directly in a browser. Nothing here is published or hosted.
</footer>

</body>
</html>
"""


def main():
    load_env()
    cfg = load_match()
    team = cfg.get("team", "Leyton Orient")
    state = load_state()

    matches, apis, problems, reached = discover(cfg, team)
    for p in problems:
        print(f"  ! {p}")
    if not len(matches):
        raise SystemExit("No fixtures visible - check .env and match.toml.")

    log, season = build_log(matches, team, cfg, state)
    upcoming, pending = build_upcoming(matches, team, season, n=5)
    players, missing = build_player_frames(log, team)
    extras = build_league_extras(log, team)

    fb_row = matches[matches.id == int(cfg.get("id", 0))]
    div_api, div_matches = division_matches(fb_row, apis)
    table = league_table(div_matches, team)
    pcts = league_percentiles(div_api, div_matches, team)

    league = [m for m in log if not m["is_cup"]]
    lg_w = sum(1 for m in league if m["result"] == "W")
    lg_d = sum(1 for m in league if m["result"] == "D")
    lg_l = sum(1 for m in league if m["result"] == "L")
    lg_gf = sum(m["gf"] for m in league)
    lg_ga = sum(m["ga"] for m in league)

    m_table = minutes_table(players)
    i_table = impact_table(players)
    best, worst = single_match_extremes(players)

    n_player_matches = players.matchId.nunique() if len(players) else 0
    missing_note = (f" No pulled data yet for: {', '.join(sorted(set(missing)))}."
                     if missing else "")

    html_out = PAGE.format(
        season=season, n_played=len(log), plural="" if len(log) == 1 else "es",
        generated=datetime.now(timezone.utc).strftime("%a %-d %b %Y, %H:%M UTC"),
        lg_played=len(league), lg_w=lg_w, lg_d=lg_d, lg_l=lg_l,
        lg_gd=lg_gf - lg_ga, lg_pts=lg_w * 3 + lg_d,
        clean_sheets=extras["clean_sheets"], ppg=extras["ppg"],
        sp_for=extras["sp_for"], sp_against=extras["sp_against"],
        sp_missing_note=(f" No pulled data yet for: {', '.join(sorted(set(extras['missing'])))}."
                        if extras["missing"] else ""),
        table_rows=league_table_rows(table, team) if len(table) else '<tr><td colspan="10" class="mut">Could not load the division table.</td></tr>',
        pctl_rows=percentile_rows(pcts) if pcts else '<p class="mut" style="padding:14px 0;">Could not load division-wide squad scores.</p>',
        n_upcoming=len(upcoming), up_plural="" if len(upcoming) == 1 else "es",
        upcoming_rows=upcoming_rows(upcoming) if upcoming else '<tr><td colspan="5" class="mut">No upcoming fixtures scheduled.</td></tr>',
        pending_note=pending_note(pending),
        log_rows=log_rows(log),
        n_player_matches=n_player_matches, p_plural="" if n_player_matches == 1 else "es",
        missing_note=missing_note,
        minutes_rows=minutes_rows(m_table) if len(m_table) else '<tr><td colspan="7" class="mut">No player data pulled yet.</td></tr>',
        min_roll=MIN_MINUTES_ROLLING,
        impact_rows=impact_rows(i_table) if len(i_table) else '<tr><td colspan="10" class="mut">Not enough minutes pulled yet.</td></tr>',
        min_single=MIN_MINUTES_SINGLE,
        best_rows=extreme_rows(best, True) if len(best) else '<tr><td colspan="6" class="mut">—</td></tr>',
        worst_rows=extreme_rows(worst, False) if len(worst) else '<tr><td colspan="6" class="mut">—</td></tr>',
    )

    OUT.write_text(html_out)
    print(f"wrote {OUT}  ({len(log)} matches, "
          f"{n_player_matches} with player data)")


if __name__ == "__main__":
    main()
