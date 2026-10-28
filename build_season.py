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
from build_corners import corner_phases, freekick_phases, throwin_phases
import manual_corrections as MC

REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
OUT = REPORTS_ROOT / "season-tracker.html"
STYLE_OUT = REPORTS_ROOT / "style-analysis.html"

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


# column, label, display format - every one a rate/style stat from Impect's
# squad match scores, deliberately excluding anything outcome-based (goals,
# points, xG difference). The goal of style_similarity() below is "plays
# like us", not "as good as us" - mixing in quality metrics would cluster
# teams by table position instead of by how they actually play.
STYLE_METRICS = [
    ("RATIO_BALL_POSSESSION", "Possession", "{:.0%}"),
    ("PPDA", "Pressing Intensity (PPDA)", "{:.1f}"),
    ("MEAN_PRESSURE_HEIGHT", "Press Height", "{:.0f}m"),
    ("RATIO_PASSING_ACCURACY", "Passing Accuracy", "{:.0%}"),
    ("PXT_POSITIVE", "Attacking Threat (xT)", "{:.2f}"),
    ("PRESSURE_DURING_OPP_BUILD_UP_PERCENT", "Press Opponent Build-Up", "{:.0%}"),
    ("OPP_OPENING_BALLS_PRESSED_PERCENT", "Press Opening Balls", "{:.0%}"),
    ("FORCED_HIGH_PASSES_PERCENT", "Force Long/High Passes", "{:.0%}"),
    # Both from getSquadMatchsums (style_similarity()'s second pull below),
    # not getSquadMatchScores like everything above - the only place this
    # project pulls raw BYPASSED_DEFENDERS/BYPASSED_OPPONENTS alongside
    # each other rather than at the pre-aggregated ratio level. Genuinely
    # different metrics despite the similar names: BYPASSED_OPPONENTS
    # counts the full outfield picture (defenders + midfielders caught out
    # of position), BYPASSED_DEFENDERS only the back line - confirmed on
    # real match data, ~5.5x apart for the same team/match.
    ("BYPASSED_DEFENDERS", "Bypassed Defenders", "{:.1f}"),
    ("BYPASSED_OPPONENTS", "Bypassed Opponents", "{:.1f}"),
]


def style_similarity(api, div_matches, team):
    """Every League One club's season-to-date style fingerprint, and how
    far each one sits from Orient's - Euclidean distance across z-scored
    STYLE_METRICS, so no single metric's raw scale (PPDA ~4-10 vs
    possession ~30-70%) dominates the comparison just because its numbers
    are bigger.

    Returns (ranked, our_values, n_clubs): `ranked` is every other club
    sorted closest-first, each with its raw metric values alongside the
    distance; `our_values` is Orient's own raw values for the profile
    panel; `n_clubs` is how many clubs had enough data to be compared
    (less than the full division if a club is missing a metric entirely).
    """
    if api is None or not len(div_matches):
        return [], {}, 0
    ids = [int(i) for i in div_matches.id.tolist()]
    try:
        sc = api.getSquadMatchScores(matches=ids)
    except Exception:
        return [], {}, 0
    # BYPASSED_DEFENDERS/BYPASSED_OPPONENTS aren't in getSquadMatchScores -
    # a second, equally lightweight squad-level pull (not per-match event
    # data - still one call for the whole division, ~40s for 143 matches)
    # rather than folding them into the same frame as the ratio metrics.
    try:
        sm = api.getSquadMatchsums(matches=ids)
    except Exception:
        sm = None

    cols = [c for c, *_ in STYLE_METRICS]
    sc_cols = [c for c in cols if c in sc.columns]
    sm_cols = [c for c in cols if sm is not None and c in sm.columns and c not in sc_cols]
    present = sc_cols + sm_cols
    if not present:
        return [], {}, 0
    agg = sc.groupby("squadName")[sc_cols].mean()
    if sm_cols:
        agg = agg.join(sm.groupby("squadName")[sm_cols].mean(), how="outer")
    agg = agg.dropna()
    if team not in agg.index or len(agg) < 3:
        return [], {}, 0

    std = agg.std(ddof=0).replace(0, 1)   # guard a zero-variance column, not expected in practice
    z = (agg - agg.mean()) / std
    us_z = z.loc[team]
    dist = ((z - us_z) ** 2).sum(axis=1).pow(0.5).drop(team).sort_values()

    ranked = []
    for opp, d in dist.items():
        row = {"opponent": opp, "distance": float(d)}
        for col in present:
            row[col] = float(agg.loc[opp, col])
        ranked.append(row)
    our_values = {col: float(agg.loc[team, col]) for col in present}
    return ranked, our_values, len(agg)


# Own label/format definitions, deliberately not reused from STYLE_METRICS -
# the "Recent Form" trend below still wants Field Tilt even though the Style
# Profile comparison above no longer does, and the two have been free to
# diverge since the Style Profile page is a comparison against the rest of
# the division while this is Orient's own trend against its own season.
TREND_METRICS = [
    ("PPDA", "Pressing Intensity (PPDA)", "{:.1f}"),
    ("MEAN_PRESSURE_HEIGHT", "Press Height", "{:.0f}m"),
    ("RATIO_BALL_POSSESSION", "Possession", "{:.0%}"),
    ("FIELD_TILT", "Field Tilt", "{:.0%}"),
    ("PXT_POSITIVE", "Attacking Threat (xT)", "{:.2f}"),
]
TREND_COLS = [c for c, _, _ in TREND_METRICS]


def season_trends(log, team, table=None, opp_ppda=None):
    """Everything below shares one pass over this season's already-cached
    match data (ss.parquet, squad_scores.parquet, setpieces.parquet per
    match) rather than several separate ones: xG for/against, a
    last-5-games style trend against the season average, our set-piece
    volume/threat, Impect's expected points against the real table, and
    three splits of the same numbers - home vs away, versus top-third vs
    mid-table vs bottom-third opponents (by the current table, `table` from
    league_table() above), and versus opponents who press us hard vs sit
    off (`opp_ppda`, each opponent's own season-average PPDA, from
    style_similarity()'s pull - reused rather than fetched again).

    League matches only, same as "Season Record" above - a cup
    opponent's level can be wildly different from the league, which would
    distort a trend that's supposed to describe one team's season.
    """
    league = [m for m in log if not m["is_cup"] and m["cache"]]
    if not league:
        return None

    top_third_squads = bottom_third_squads = set()
    if table is not None and len(table):
        n = len(table)
        cut = max(1, n // 3)
        top_third_squads = set(table[table.pos <= cut].squad)
        bottom_third_squads = set(table[table.pos > n - cut].squad)

    ppda_values = list(opp_ppda.values()) if opp_ppda else []
    ppda_median = float(pd.Series(ppda_values).median()) if ppda_values else None

    per_match = []
    xg_for_total = xg_against_total = xpts_total = 0.0
    setp_frames = []
    splits = {
        "home": {"pts": 0, "gf": 0, "ga": 0, "xgf": 0.0, "xga": 0.0, "n": 0},
        "away": {"pts": 0, "gf": 0, "ga": 0, "xgf": 0.0, "xga": 0.0, "n": 0},
        "top_third": {"pts": 0, "gf": 0, "ga": 0, "xgf": 0.0, "xga": 0.0, "n": 0},
        "mid_table": {"pts": 0, "gf": 0, "ga": 0, "xgf": 0.0, "xga": 0.0, "n": 0},
        "bottom_third": {"pts": 0, "gf": 0, "ga": 0, "xgf": 0.0, "xga": 0.0, "n": 0},
        "pressed": {"acc": [], "bpd": [], "n": 0},
        "sat_off": {"acc": [], "bpd": [], "n": 0},
    }
    pts_map = {"W": 3, "D": 1, "L": 0}

    for m in league:
        cache = m["cache"]
        try:
            ss = pd.read_parquet(cache / "ss.parquet")
        except Exception:
            ss = pd.DataFrame()
        try:
            sc = pd.read_parquet(cache / "squad_scores.parquet")
        except Exception:
            sc = pd.DataFrame()

        us_ss = ss[ss.squadName == team] if len(ss) else ss
        opp_ss = ss[ss.squadName == m["opponent"]] if len(ss) else ss
        us_sc = sc[sc.squadName == team] if len(sc) else sc

        xg_for = float(us_ss.SHOT_XG.sum()) if "SHOT_XG" in us_ss.columns and len(us_ss) else 0.0
        xg_against = float(opp_ss.SHOT_XG.sum()) if "SHOT_XG" in opp_ss.columns and len(opp_ss) else 0.0
        xg_for_total += xg_for
        xg_against_total += xg_against
        bpd = float(us_ss.BYPASSED_DEFENDERS.sum()) if "BYPASSED_DEFENDERS" in us_ss.columns and len(us_ss) else None
        pts = pts_map[m["result"]]

        row = {"date": m["date"], "opponent": m["opponent"]}
        pass_acc = None
        if len(us_sc):
            r = us_sc.iloc[0]
            for col in TREND_COLS:
                if col in r.index and pd.notna(r[col]):
                    row[col] = float(r[col])
            if "EXPECTED_POINTS_SHOT_XG" in r.index and pd.notna(r["EXPECTED_POINTS_SHOT_XG"]):
                xpts_total += float(r["EXPECTED_POINTS_SHOT_XG"])
            if "RATIO_PASSING_ACCURACY" in r.index and pd.notna(r["RATIO_PASSING_ACCURACY"]):
                pass_acc = float(r["RATIO_PASSING_ACCURACY"])
        per_match.append(row)

        side = "home" if m["home"] else "away"
        splits[side]["n"] += 1
        splits[side]["pts"] += pts
        splits[side]["gf"] += m["gf"]
        splits[side]["ga"] += m["ga"]
        splits[side]["xgf"] += xg_for
        splits[side]["xga"] += xg_against

        if top_third_squads or bottom_third_squads:
            if m["opponent"] in top_third_squads:
                bucket = "top_third"
            elif m["opponent"] in bottom_third_squads:
                bucket = "bottom_third"
            else:
                bucket = "mid_table"
            splits[bucket]["n"] += 1
            splits[bucket]["pts"] += pts
            splits[bucket]["gf"] += m["gf"]
            splits[bucket]["ga"] += m["ga"]
            splits[bucket]["xgf"] += xg_for
            splits[bucket]["xga"] += xg_against

        if ppda_median is not None and m["opponent"] in (opp_ppda or {}):
            # lower PPDA = more aggressive pressing
            bucket = "pressed" if opp_ppda[m["opponent"]] < ppda_median else "sat_off"
            splits[bucket]["n"] += 1
            if pass_acc is not None:
                splits[bucket]["acc"].append(pass_acc)
            if bpd is not None:
                splits[bucket]["bpd"].append(bpd)

        try:
            setp_frames.append(pd.read_parquet(cache / "setpieces.parquet"))
        except Exception:
            pass

    df = pd.DataFrame(per_match)
    last5 = df.tail(5)

    def _avg(frame, col):
        vals = frame[col].dropna() if col in frame.columns else pd.Series(dtype=float)
        return float(vals.mean()) if len(vals) else None

    trend = []
    label_fmt = {c: (l, f) for c, l, f in TREND_METRICS}
    for col in TREND_COLS:
        season_v, last5_v = _avg(df, col), _avg(last5, col)
        if season_v is None or last5_v is None or col not in label_fmt:
            continue
        label, fmt = label_fmt[col]
        trend.append(dict(label=label, fmt=fmt, season=season_v, last5=last5_v))

    goals_for = sum(m["gf"] for m in league)
    goals_against = sum(m["ga"] for m in league)
    actual_pts = sum(pts_map[m["result"]] for m in league)

    setp_all = pd.concat(setp_frames, ignore_index=True) if setp_frames else pd.DataFrame()
    setpieces = {}
    if len(setp_all):
        for key, fn in (("Corners", corner_phases), ("Free Kicks", freekick_phases),
                        ("Throw-Ins", throwin_phases)):
            d = fn(setp_all, attacking=team)
            setpieces[key] = dict(n=len(d), xg=float(d.xg.sum()) if len(d) else 0.0,
                                  goals=int(d.goals.sum()) if len(d) else 0)

    for bucket in ("pressed", "sat_off"):
        splits[bucket]["acc"] = (sum(splits[bucket]["acc"]) / len(splits[bucket]["acc"])
                                 if splits[bucket]["acc"] else None)
        splits[bucket]["bpd"] = (sum(splits[bucket]["bpd"]) / len(splits[bucket]["bpd"])
                                 if splits[bucket]["bpd"] else None)

    return dict(
        n_matches=len(league),
        xg_for=xg_for_total, xg_against=xg_against_total,
        goals_for=goals_for, goals_against=goals_against,
        trend=trend,
        xpts_total=xpts_total, actual_pts=actual_pts,
        setpieces=setpieces,
        splits=splits, has_table_split=bool(top_third_squads or bottom_third_squads),
        has_press_split=ppda_median is not None,
    )


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
    # column, label, higher-is-better, display format, source endpoint
    ("RATIO_BALL_POSSESSION", "Possession", True, "{:.0%}", "scores"),
    ("FIELD_TILT", "Field Tilt", True, "{:.0%}", "scores"),
    ("MEAN_PRESSURE_HEIGHT", "Press Height", True, "{:.1f}m", "scores"),
    ("PPDA", "Pressing Intensity", False, "{:.1f} PPDA", "scores"),
    ("PXT_POSITIVE", "Attacking Threat (xT)", True, "{:.2f}", "scores"),
    ("RATIO_PASSING_ACCURACY", "Passing Accuracy", True, "{:.0%}", "scores"),
    # Not in getSquadMatchScores - shots live in the matchsums endpoint
    # instead (see fetch_data.py's ss.parquet pull), so this needs its own
    # fetch rather than riding along with the metrics above.
    ("SHOT_AT_GOAL_NUMBER", "Shots", True, "{:.1f}", "matchsums"),
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

    # Group metrics by which endpoint they come from, so each endpoint is
    # fetched once regardless of how many metrics it covers.
    by_source = {}
    for col, _, _, _, source in PERCENTILE_METRICS:
        by_source.setdefault(source, []).append(col)
    fetchers = {"scores": api.getSquadMatchScores, "matchsums": api.getSquadMatchsums}

    agg = {}
    for source, cols in by_source.items():
        try:
            df = fetchers[source](matches=ids)
        except Exception:
            continue
        present = [c for c in cols if c in df.columns]
        if present:
            agg[source] = df.groupby("squadName")[present].mean()

    rows = []
    for col, label, higher_better, fmt, source in PERCENTILE_METRICS:
        a = agg.get(source)
        if a is None or col not in a.columns:
            continue
        vals = a[col].dropna()
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


def best_single_match_performances(players):
    if not len(players):
        return pd.DataFrame()
    s = players.copy()
    s["mins"] = s.playDuration / 60
    s = s[s.mins >= MIN_MINUTES_SINGLE]
    return s.sort_values("PXT_ATTACK", ascending=False).head(5)


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


# A handful of the headline metrics for the similarity table itself - all
# ten would make the table too wide to read; the full set still shows in
# Orient's own style-profile panel.
STYLE_TABLE_COLS = ["RATIO_BALL_POSSESSION", "PPDA", "MEAN_PRESSURE_HEIGHT", "PXT_POSITIVE"]


def style_table_rows(ranked):
    label_by_col = {c: l for c, l, _ in STYLE_METRICS}
    fmt_by_col = {c: f for c, _, f in STYLE_METRICS}
    rows = []
    for i, r in enumerate(ranked):
        cells = "".join(
            f'<td class="num">{fmt_by_col[c].format(r[c])}</td>' if c in r else '<td class="num mut">—</td>'
            for c in STYLE_TABLE_COLS)
        rows.append(f'''
      <tr>
        <td class="mut">{i + 1}</td>
        <td>{html.escape(r["opponent"])}</td>
        <td class="num">{r["distance"]:.2f}</td>
        {cells}
      </tr>''')
    return "".join(rows)


def style_profile_rows(our_values):
    rows = []
    for col, label, fmt in STYLE_METRICS:
        if col not in our_values:
            continue
        rows.append(f'''
      <div class="pctl-row">
        <div class="pctl-label">{html.escape(label)}</div>
        <div class="pctl-val" style="flex:1; text-align:left;">{fmt.format(our_values[col])}</div>
      </div>''')
    return "".join(rows)


def _diff_span(value, fmt="{:+.2f}", good_if_positive=True):
    cls = "good" if (value >= 0) == good_if_positive else "bad"
    return f'<span class="{cls}">{fmt.format(value)}</span>'


def xg_cards(trends):
    gf, xgf = trends["goals_for"], trends["xg_for"]
    ga, xga = trends["goals_against"], trends["xg_against"]
    return f'''
    <div class="cards">
      <div class="stat"><div class="n">{xgf:.2f}</div><div class="l">xG For</div></div>
      <div class="stat"><div class="n">{gf}</div><div class="l">Goals For</div></div>
      <div class="stat"><div class="n">{_diff_span(gf - xgf)}</div><div class="l">Goals vs xG</div></div>
      <div class="stat"><div class="n">{xga:.2f}</div><div class="l">xG Against</div></div>
      <div class="stat"><div class="n">{ga}</div><div class="l">Goals Against</div></div>
      <div class="stat"><div class="n">{_diff_span(xga - ga)}</div><div class="l">Conceded vs xG</div></div>
    </div>'''


def trend_rows(trend):
    rows = []
    for t in trend:
        season_v, last5_v = t["season"], t["last5"]
        delta = last5_v - season_v
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "–")
        cls = "mut" if abs(delta) < 1e-9 else ("good" if delta > 0 else "bad")
        rows.append(f'''
      <tr>
        <td>{html.escape(t["label"])}</td>
        <td class="num">{t["fmt"].format(season_v)}</td>
        <td class="num">{t["fmt"].format(last5_v)}</td>
        <td class="num {cls}">{arrow}</td>
      </tr>''')
    return "".join(rows)


def setpiece_cards(setpieces):
    cards = []
    for key in ("Corners", "Free Kicks", "Throw-Ins"):
        d = setpieces.get(key, {"n": 0, "xg": 0.0, "goals": 0})
        cards.append(f'''
      <div class="stat">
        <div class="n">{d["n"]}</div><div class="l">{html.escape(key)} Taken</div>
      </div>
      <div class="stat">
        <div class="n">{d["xg"]:.2f}</div><div class="l">{html.escape(key)} xG</div>
      </div>
      <div class="stat">
        <div class="n">{d["goals"]}</div><div class="l">{html.escape(key)} Goals</div>
      </div>''')
    return f'<div class="cards" style="grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));">{"".join(cards)}</div>'


def split_cards(split, label_fmt="{:.0f}"):
    """Home/away or top-half/bottom-half comparison cards: points per game,
    goals for/against per game, xG for/against per game - counts alone
    aren't comparable when the two buckets have different numbers of
    matches, so everything here is a per-game rate."""
    n = split["n"] or 1
    ppg = split["pts"] / n
    return f'''
      <div class="stat"><div class="n">{split["n"]}</div><div class="l">Played</div></div>
      <div class="stat"><div class="n">{split["pts"]}</div><div class="l">Points Gained</div></div>
      <div class="stat"><div class="n">{ppg:.2f}</div><div class="l">Points/Game</div></div>
      <div class="stat"><div class="n">{split["gf"] / n:.2f}</div><div class="l">Goals For/Game</div></div>
      <div class="stat"><div class="n">{split["ga"] / n:.2f}</div><div class="l">Goals Against/Game</div></div>
      <div class="stat"><div class="n">{split["xgf"] / n:.2f}</div><div class="l">xG For/Game</div></div>
      <div class="stat"><div class="n">{split["xga"] / n:.2f}</div><div class="l">xG Against/Game</div></div>'''


def press_split_cards(split, label):
    acc = f'{split["acc"] * 100:.0f}%' if split["acc"] is not None else "—"
    bpd = f'{split["bpd"]:.1f}' if split["bpd"] is not None else "—"
    return f'''
      <div class="stat"><div class="n">{split["n"]}</div><div class="l">{html.escape(label)} — Matches</div></div>
      <div class="stat"><div class="n">{acc}</div><div class="l">{html.escape(label)} — Pass Accuracy</div></div>
      <div class="stat"><div class="n">{bpd}</div><div class="l">{html.escape(label)} — Bypassed Defenders</div></div>'''


STYLE_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Leyton Orient — Playstyle Analysis</title>
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
  header p {{ margin: 0; color: var(--muted); font-size: 14.5px; max-width: 760px; line-height: 1.5; }}
  main {{ max-width: 1200px; margin: 0 auto; padding: 36px 40px 80px; }}
  section {{ margin-bottom: 52px; }}
  section h2 {{
    font-size: 13px; font-weight: 700; letter-spacing: 0.10em; text-transform: uppercase;
    margin: 0 0 4px;
  }}
  section .desc {{ color: var(--muted); font-size: 13px; margin: 0 0 18px; max-width: 760px; line-height: 1.5; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--grid); border-radius: 10px; overflow: hidden; }}
  .table-wrap {{ overflow-x: auto; border-radius: 10px; }}
  th, td {{ padding: 10px 14px; text-align: left; font-size: 13px; border-bottom: 1px solid var(--grid); white-space: nowrap; }}
  th {{ color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }}
  tr:last-child td {{ border-bottom: none; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  th.num {{ text-align: right; }}
  .mut {{ color: var(--muted); }}
  .good {{ color: var(--good); font-weight: 700; }}
  .bad {{ color: var(--bad); font-weight: 700; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 20px; max-width: 760px; }}
  .stat {{ background: var(--panel); border: 1px solid var(--grid); border-radius: 10px; padding: 14px 16px; }}
  .stat .n {{ font-size: 24px; font-weight: 800; }}
  .stat .l {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-top: 2px; }}
  .pctl-panel {{ background: var(--panel); border: 1px solid var(--grid); border-radius: 10px; padding: 6px 18px; max-width: 480px; }}
  .pctl-row {{ display: flex; align-items: center; gap: 14px; padding: 12px 0; border-bottom: 1px solid var(--grid); }}
  .pctl-row:last-child {{ border-bottom: none; }}
  .pctl-label {{ width: 190px; flex-shrink: 0; font-size: 13px; }}
  .pctl-val {{ font-size: 13px; font-variant-numeric: tabular-nums; font-weight: 700; }}
  .pctl-track {{ flex: 1; height: 8px; background: #2E394980; border-radius: 4px; overflow: hidden; }}
  .pctl-fill {{ height: 100%; border-radius: 4px; background: var(--team); }}
  .legend {{ display: flex; gap: 20px; margin-top: 10px; font-size: 12px; color: var(--muted); }}
  .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
  .legend i {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
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
  <h1>Playstyle Analysis</h1>
  <p>This page runs in two halves. The first three sections are pure identity: which League One
     clubs play most like Orient this season, and which play nothing alike - judged only on
     style metrics (possession, pressing, territory, passing), never on results or table
     position. Two clubs with completely different league positions can still play almost
     identically; that's deliberate - it's the "how", not the "how well". Everything below it
     flips the question: not how Orient plays, but how that style is actually paying off this
     season - xG, form, set pieces, points earned vs expected, and how results split by venue,
     opponent strength, and approach faced.</p>
  <p class="mut" style="font-size:11.5px;margin-top:10px;">Generated {generated} · {n_clubs} clubs compared</p>
</header>

<main>

  <section>
    <h2>Orient's Style Profile</h2>
    <p class="desc">Season-to-date average across every League One match played so far.</p>
    <div class="pctl-panel">{profile_rows}</div>
  </section>

  <section>
    <h2>Most Similar Clubs</h2>
    <p class="desc">Ranked by overall distance across ten style metrics (lower = more similar) -
       not just the four shown here, see the full metric list in the profile panel above.</p>
    <div class="table-wrap"><table>
      <thead><tr>
        <th>#</th><th>Club</th><th class="num">Distance</th>
        <th class="num">Possession</th><th class="num">PPDA</th>
        <th class="num">Press Height</th><th class="num">xT</th>
      </tr></thead>
      <tbody>{similar_rows}</tbody>
    </table></div>
  </section>

  <section>
    <h2>Least Similar Clubs</h2>
    <p class="desc">The other end of the table - clubs whose approach looks nothing like Orient's.</p>
    <div class="table-wrap"><table>
      <thead><tr>
        <th>#</th><th>Club</th><th class="num">Distance</th>
        <th class="num">Possession</th><th class="num">PPDA</th>
        <th class="num">Press Height</th><th class="num">xT</th>
      </tr></thead>
      <tbody>{dissimilar_rows}</tbody>
    </table></div>
    <div class="note">
      Every club has played between 9 and 13 matches so far this season - enough to spot a real
      pattern, not yet a fully settled identity. Style metrics here deliberately exclude goals,
      points and xG difference, so this ranks "plays like us", not "as good as us" - a club far
      apart on this list can still be above or below Orient in the table.
    </div>
  </section>

  <section>
    <h2>xG Over/Under-Performance</h2>
    <p class="desc">League matches only. A positive "Goals vs xG" means scoring more than the
       chances deserved; a positive "Conceded vs xG" means the defence has held up better than
       the chances faced suggested it should.</p>
    {xg_cards}
  </section>

  <section>
    <h2>Recent Form vs Season Average</h2>
    <p class="desc">Last 5 league matches against the full-season average - a season number can
       hide a real recent shift either direction.</p>
    <div class="table-wrap"><table>
      <thead><tr><th>Metric</th><th class="num">Season Avg</th><th class="num">Last 5</th><th class="num">Trend</th></tr></thead>
      <tbody>{trend_rows}</tbody>
    </table></div>
  </section>

  <section>
    <h2>Our Set-Piece Profile</h2>
    <p class="desc">Every corner, wide free kick and near-box throw-in Orient has taken in league
       play this season, combined - same definitions used to scout opponents, applied to us.</p>
    {setpiece_cards}
  </section>

  <section>
    <h2>Expected Points vs Actual Points</h2>
    <p class="desc">Impect's own expected-points model, summed across league matches, against the
       real league table - whether results have run ahead of or behind the underlying performances.</p>
    <div class="cards">
      <div class="stat"><div class="n">{xpts_total:.1f}</div><div class="l">Expected Points</div></div>
      <div class="stat"><div class="n">{actual_pts}</div><div class="l">Actual Points</div></div>
      <div class="stat"><div class="n">{pts_diff}</div><div class="l">Difference</div></div>
    </div>
  </section>

  <section>
    <h2>Home vs Away</h2>
    <p class="desc">Everything per game, not per total - the two splits won't have the same
       number of matches.</p>
    <div style="display:flex; gap:28px; flex-wrap:wrap;">
      <div>
        <p class="mut" style="font-size:12px; margin:0 0 8px; text-transform:uppercase; letter-spacing:0.06em;">Home</p>
        <div class="cards" style="grid-template-columns: repeat(2, minmax(110px,1fr)); max-width:340px;">{home_cards}</div>
      </div>
      <div>
        <p class="mut" style="font-size:12px; margin:0 0 8px; text-transform:uppercase; letter-spacing:0.06em;">Away</p>
        <div class="cards" style="grid-template-columns: repeat(2, minmax(110px,1fr)); max-width:340px;">{away_cards}</div>
      </div>
    </div>
  </section>

  <section>
    <h2>Vs Top Third vs Mid-Table vs Bottom Third</h2>
    <p class="desc">Opponents split into thirds by their current league position, not where they
       stood on the day Orient played them - a rough but reasonable read on performance against
       stronger, mid-table, and weaker sides.</p>
    {table_split_note}
    <div style="display:flex; gap:28px; flex-wrap:wrap;">
      <div>
        <p class="mut" style="font-size:12px; margin:0 0 8px; text-transform:uppercase; letter-spacing:0.06em;">Vs Top Third</p>
        <div class="cards" style="grid-template-columns: repeat(2, minmax(110px,1fr)); max-width:340px;">{top_third_cards}</div>
      </div>
      <div>
        <p class="mut" style="font-size:12px; margin:0 0 8px; text-transform:uppercase; letter-spacing:0.06em;">Vs Mid-Table</p>
        <div class="cards" style="grid-template-columns: repeat(2, minmax(110px,1fr)); max-width:340px;">{mid_table_cards}</div>
      </div>
      <div>
        <p class="mut" style="font-size:12px; margin:0 0 8px; text-transform:uppercase; letter-spacing:0.06em;">Vs Bottom Third</p>
        <div class="cards" style="grid-template-columns: repeat(2, minmax(110px,1fr)); max-width:340px;">{bottom_third_cards}</div>
      </div>
    </div>
  </section>

  <section>
    <h2>Vs High Press vs Vs Low Block</h2>
    <p class="desc">Opponents split by their own season-average PPDA (below the 24-club median =
       presses us hard, above = sits off) - whether Orient's passing actually holds up against
       teams that come to press, or only looks good against sides that don't bother.</p>
    {press_split_note}
    <div class="cards" style="grid-template-columns: repeat(3, minmax(140px,1fr)); max-width:620px;">
      {pressed_cards}
      {sat_off_cards}
    </div>
  </section>

</main>

</body>
</html>
"""


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


def best_rows_html(df):
    rows = []
    for _, r in df.iterrows():
        rows.append(f'''
      <tr>
        <td>{html.escape(r.playerName)}</td>
        <td class="mut">{pos(r.position)}</td>
        <td class="mut">v {html.escape(r.opponent)}</td>
        <td class="mut">{fmt_date(r.date)}</td>
        <td class="num good">{f2(r.PXT_ATTACK)}</td>
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
    <h2>Best Individual Performances</h2>
    <p class="desc">Single-match xT, minimum {min_single} minutes played.</p>
    <div class="table-wrap"><table>
      <thead><tr><th>Player</th><th>Pos</th><th>Opponent</th><th>Date</th><th class="num">xT</th><th class="num">Mins</th></tr></thead>
      <tbody>{best_rows}</tbody>
    </table></div>
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
    style_ranked, style_ours, style_n = style_similarity(div_api, div_matches, team)
    opp_ppda = {r["opponent"]: r["PPDA"] for r in style_ranked if "PPDA" in r}
    trends = season_trends(log, team, table=table, opp_ppda=opp_ppda)

    league = [m for m in log if not m["is_cup"]]
    lg_w = sum(1 for m in league if m["result"] == "W")
    lg_d = sum(1 for m in league if m["result"] == "D")
    lg_l = sum(1 for m in league if m["result"] == "L")
    lg_gf = sum(m["gf"] for m in league)
    lg_ga = sum(m["ga"] for m in league)

    m_table = minutes_table(players)
    i_table = impact_table(players)
    best = best_single_match_performances(players)

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
        best_rows=best_rows_html(best) if len(best) else '<tr><td colspan="6" class="mut">—</td></tr>',
    )

    OUT.write_text(html_out)
    print(f"wrote {OUT}  ({len(log)} matches, "
          f"{n_player_matches} with player data)")

    if style_ranked:
        if trends:
            pts_diff = trends["actual_pts"] - trends["xpts_total"]
            pts_diff_txt = f'<span class="{"good" if pts_diff >= 0 else "bad"}">{pts_diff:+.1f}</span>'
            s = trends["splits"]
            no_table_note = ('<p class="mut" style="margin:0 0 14px;">Could not load the division '
                             'table - top/mid/bottom-third split unavailable.</p>'
                             if not trends["has_table_split"] else "")
            no_press_note = ('<p class="mut" style="margin:0 0 14px;">Not enough division-wide '
                             'press data yet.</p>' if not trends["has_press_split"] else "")
            trend_section = dict(
                xg_cards=xg_cards(trends), trend_rows=trend_rows(trends["trend"]),
                setpiece_cards=setpiece_cards(trends["setpieces"]),
                xpts_total=trends["xpts_total"], actual_pts=trends["actual_pts"],
                pts_diff=pts_diff_txt,
                home_cards=split_cards(s["home"]), away_cards=split_cards(s["away"]),
                table_split_note=no_table_note,
                top_third_cards=split_cards(s["top_third"]),
                mid_table_cards=split_cards(s["mid_table"]),
                bottom_third_cards=split_cards(s["bottom_third"]),
                press_split_note=no_press_note,
                pressed_cards=press_split_cards(s["pressed"], "Vs High Press"),
                sat_off_cards=press_split_cards(s["sat_off"], "Vs Low Block"),
            )
        else:
            empty = '<p class="mut" style="padding:14px 0;">Not enough cached match data yet.</p>'
            trend_section = dict(
                xg_cards=empty, trend_rows="", setpiece_cards=empty,
                xpts_total=0.0, actual_pts=0, pts_diff="–",
                home_cards="", away_cards="", table_split_note="",
                top_third_cards="", mid_table_cards="", bottom_third_cards="",
                press_split_note="", pressed_cards="", sat_off_cards="",
            )
        style_html = STYLE_PAGE.format(
            generated=datetime.now(timezone.utc).strftime("%a %-d %b %Y, %H:%M UTC"),
            n_clubs=style_n,
            profile_rows=style_profile_rows(style_ours),
            similar_rows=style_table_rows(style_ranked[:5]),
            dissimilar_rows=style_table_rows(list(reversed(style_ranked[-5:]))),
            **trend_section,
        )
        STYLE_OUT.write_text(style_html)
        print(f"wrote {STYLE_OUT}  ({style_n} clubs compared)")
    else:
        print("  ! skipped style-analysis.html - not enough division-wide data yet")


if __name__ == "__main__":
    main()
