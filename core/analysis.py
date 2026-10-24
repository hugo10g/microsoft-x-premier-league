"""
Deeper analysis of the off-ball metrics: percentiles, per-match form, player profiles,
team comparisons, match intensity, validation against the hidden truth and the effect
of scanning on keeping the ball.

    percentile_table    0-100 percentile of every pillar, across the pool or within a
                        group (e.g. broad position), with "lower is better" flipped
    per_match_totals    core.pipeline.player_totals for each game on its own, plus that
                        match's Off-Ball Value among its outfielders
    player_profile      one player's numbers, percentiles, match-by-match series,
                        strengths and weaknesses, and the most similar players
    team_match_summary  Home v Away on every pillar in one game
    intensity_blocks    the same, per 15-minute block, to show when a team pressed,
                        sprinted, ran decoys and scanned
    validation          Spearman correlation between each metric and the hidden
                        attribute it should reflect, with a bootstrap 95% interval
    scanning_effect     ball retention after receiving by number of scans beforehand,
                        with Wilson 95% intervals

The synthetic league knows every player's true attributes (data/synthetic/
league_players.csv), so `validation` asks a question real data can't answer: does a
metric recover the trait that generated the movement? Correlations are rank-based
(Spearman) because the metrics are skewed and the link need only be monotonic. The
bootstrap resamples players with replacement; with a few dozen players the interval
is wide, which is the honest picture.

Every function tolerates missing tables and columns (e.g. no "scanning" table before
head-orientation data exists): the affected rows or columns are left out or NaN.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from core import pipeline, roster as roster_mod

# Pillar column -> (display label, higher is better).
PILLARS: dict[str, tuple[str, bool]] = {
    "decoy_p90": ("Decoy runs (xT/90)", True),
    "lane_cut_p90": ("Pressing lane cuts /90", True),
    "transition_score": ("Transition reaction", True),
    "scan_rate": ("Scanning (scans/s)", True),
}
# (hidden attribute, metric, expected sign): what each metric should recover.
# The five tactical metrics (core.pipeline.TACTICAL_COLUMNS), shown beside the pillars.
# Run types describe style rather than quality: "higher" just means "more often".
TACTICAL: dict[str, tuple[str, bool]] = {
    "pressure_intensity": ("Pressure intensity /90", True),
    "press_wins_p90": ("Pressures won /90", True),
    "receipts_in_space_pct": ("Receipts in space", True),
    "def_responsibility_p90": ("Defensive responsibility /90", False),
    "obso_p90": ("OBSO /90", True),
    "runs_in_behind_p90": ("Runs in behind /90", True),
    "pulling_wide_p90": ("Pulling wide /90", True),
    "dropping_deep_p90": ("Dropping deep /90", True),
}
VALIDATION_PAIRS: list[tuple[str, str, int]] = [
    ("work_rate", "decoy_p90", 1),
    ("pressing", "lane_cut_p90", 1),
    ("work_rate", "transition_score", 1),
    ("work_rate", "median_latency", -1),
    ("awareness", "scan_rate", 1),
]
SCAN_BUCKETS = ["0", "1", "2", "3+"]
EFFECT_COLUMNS = ["bucket", "n", "retained", "retention", "low", "high"]
VALIDATION_COLUMNS = ["attribute", "metric", "rho", "low", "high", "n", "verdict"]


# ── Percentiles ──────────────────────────────────────────────────────────────
def percentile_table(pool: pd.DataFrame, pillars: dict[str, tuple[str, bool]] = PILLARS,
                     within: str | None = None) -> pd.DataFrame:
    """0-100 percentile per player (rows aligned with `pool`) for each pillar.

    `within` ranks each player only against their own group (e.g. "broad_position").
    Lower-is-better pillars are flipped so 100 is always best. A pillar missing from
    `pool` comes back as NaN; players with no value for a pillar get NaN for it.
    """
    out = pool[[c for c in ("name", within) if c and c in pool]].copy()
    for column, (_, higher) in pillars.items():
        if column not in pool:
            out[column] = np.nan
            continue
        values = pool[column].astype(float)
        ranked = (values.groupby(pool[within]).rank(pct=True, ascending=higher) if within
                  else values.rank(pct=True, ascending=higher))
        out[column] = ranked * 100
    return out


# ── Per-match totals ─────────────────────────────────────────────────────────
def _game_order(data: dict[str, pd.DataFrame]) -> list[str]:
    """Games in the order they were scored (Matchday 10 after 9, not after 1)."""
    source = data.get("profiles", next(iter(data.values()), pd.DataFrame()))
    return list(dict.fromkeys(source["game"])) if "game" in source else []


def _with_obv(board: pd.DataFrame, min_minutes: float) -> pd.DataFrame:
    """Per-90 columns for everyone; Off-Ball Value among outfielders with `min_minutes`."""
    board = board.copy()
    board["decoy_p90"] = board["decoy_value"].fillna(0) / board["minutes"] * 90
    board["lane_cut_p90"] = board["lane_cut_seconds"].fillna(0) / board["minutes"] * 90
    for column in pipeline.PILLARS:          # pillars another stage hasn't produced yet
        if column not in board:
            board[column] = np.nan
    eligible = (board["minutes"] >= min_minutes) & (board["broad_position"] != "Goalkeeper")
    if eligible.any():
        scored = pipeline.off_ball_value(board[eligible])
        board["off_ball_value"] = scored["off_ball_value"]
    else:
        board["off_ball_value"] = np.nan
    return board


def per_match_totals(data: dict[str, pd.DataFrame], names: pd.DataFrame,
                     min_minutes: float = 20.0) -> pd.DataFrame:
    """One row per (game, name): player_totals for that game alone, plus its Off-Ball Value.

    `data` is core.pipeline.score_all's output and `names` maps uid -> name, club,
    position and value (app.league_names). Off-Ball Value is the mean pillar
    percentile among that match's outfielders with at least `min_minutes`, so it
    reads "how good was this performance among the players on the pitch that day".
    """
    rows = []
    for game in _game_order(data):
        events = {key: table[table["game"] == game] for key, table in data.items() if "game" in table}
        timeline = roster_mod.static_timeline(events["profiles"], names, key="uid")
        if timeline.empty:
            continue
        board = _with_obv(pipeline.player_totals(events, timeline, on="uid"), min_minutes)
        clubs = list(dict.fromkeys(board["real_team"].dropna()))
        board["opponent"] = board["real_team"].map(
            lambda club: next((c for c in clubs if c != club), None))
        rows.append(board.assign(game=game, matchday=len(rows) + 1))
    if not rows:
        return pd.DataFrame(columns=["game", "matchday", "name", "opponent", "minutes", "off_ball_value"])
    out = pd.concat(rows, ignore_index=True)
    first = ["game", "matchday", "name", "real_team", "opponent"]
    return out[first + [c for c in out.columns if c not in first]]


# ── Player profile ───────────────────────────────────────────────────────────
def comparable_players(name: str, pool: pd.DataFrame, pillars: dict[str, tuple[str, bool]] = PILLARS,
                       n: int = 5, group: str = "broad_position") -> pd.DataFrame:
    """The `n` players nearest to `name` on the standardised pillar vector, same `group`.

    Each pillar is z-scored within the group (lower-is-better flipped), missing values
    sit at the group mean, and distance is Euclidean, averaged per pillar so it
    doesn't grow with the number of pillars. `similarity` = 1 / (1 + distance).
    """
    columns = [c for c in pillars if c in pool and pool[c].notna().any()]
    target = pool[pool["name"] == name]
    if target.empty or not columns:
        return pd.DataFrame(columns=["name", "distance", "similarity"])
    peers = pool[pool[group] == target[group].iloc[0]] if group in pool else pool
    values = peers[columns].astype(float)
    signs = np.array([1.0 if pillars[c][1] else -1.0 for c in columns])
    std = values.std(ddof=0).replace(0, 1).fillna(1)
    z = ((values - values.mean()) / std * signs).fillna(0.0)
    me = z[peers["name"] == name].iloc[0].to_numpy()
    distance = np.sqrt(((z.to_numpy() - me) ** 2).mean(axis=1))
    result = peers.assign(distance=distance, similarity=1 / (1 + distance))
    result = result[result["name"] != name].sort_values("distance").head(n)
    keep = [c for c in ("name", "real_team", "broad_position", "position", "minutes") if c in result]
    return result[keep + columns + ["distance", "similarity"]].reset_index(drop=True)


def player_profile(name: str, pool: pd.DataFrame, per_match: pd.DataFrame | None = None,
                   pillars: dict[str, tuple[str, bool]] = PILLARS, within: str = "broad_position",
                   n_comparable: int = 5) -> dict:
    """Everything the profile page needs for one player in `pool` (KeyError if absent).

    Returns a dict:
        row          the player's pool row (Series)
        percentiles  one row per pillar: pillar, label, value, overall, within_position
        matches      the player's per-match rows: game, matchday, opponent, minutes,
                     each pillar and off_ball_value
        strengths    labels of the two pillars with the highest within-position percentile
        weaknesses   labels of the two lowest (never the same pillar as a strength)
        comparables  core.analysis.comparable_players
    """
    if name not in set(pool["name"]):
        raise KeyError(name)
    at = pool.index[pool["name"] == name][0]
    overall = percentile_table(pool, pillars).loc[at]
    grouped = percentile_table(pool, pillars, within if within in pool else None).loc[at]
    row = pool.loc[at]
    percentiles = pd.DataFrame([
        {"pillar": c, "label": label, "value": row.get(c, np.nan),
         "overall": overall[c], "within_position": grouped[c]}
        for c, (label, _) in pillars.items()])

    ranked = percentiles.dropna(subset=["within_position"]).sort_values("within_position", ascending=False)
    strengths = ranked["label"].head(2).tolist()
    weaknesses = [label for label in ranked["label"].tail(2).tolist()[::-1] if label not in strengths]

    matches = pd.DataFrame()
    if per_match is not None and not per_match.empty:
        mine = per_match[per_match["name"] == name]
        keep = [c for c in ("game", "matchday", "opponent", "minutes", *pillars, "off_ball_value") if c in mine]
        matches = mine[keep].reset_index(drop=True)
    return {"row": row, "percentiles": percentiles, "matches": matches, "strengths": strengths,
            "weaknesses": weaknesses,
            "comparables": comparable_players(name, pool, pillars, n_comparable, within)}


# ── Team comparisons ─────────────────────────────────────────────────────────
def _sides(match_roster: pd.DataFrame) -> dict[str, str]:
    """"Home"/"Away" -> club name, from the match roster (tracking id prefix otherwise)."""
    club_col = "club" if "club" in match_roster else "real_team"
    clubs = {}
    if club_col in match_roster and "player" in match_roster:
        side = match_roster["player"].astype(str).str.split("_").str[0]
        clubs = match_roster.groupby(side)[club_col].first().to_dict()
    return {s: clubs.get(s, s) for s in ("Home", "Away")}


def _side_of(table: pd.DataFrame) -> pd.Series:
    return table["player"].astype(str).str.split("_").str[0]


def _table(tables: dict[str, pd.DataFrame], key: str, needs: Sequence[str]) -> pd.DataFrame | None:
    table = tables.get(key)
    if table is None or not set(needs) <= set(table.columns) or "player" not in table:
        return None
    return table.assign(side=_side_of(table))


def _observed(scanning: pd.DataFrame) -> pd.DataFrame:
    """Receptions with enough head data to count scans (scan_rate not NaN).

    The pipeline writes n_scans=0 with scan_rate NaN when the head feed is missing
    or too sparse, so those rows are unknown, not zero (as in pipeline.player_totals).
    """
    return scanning[scanning["scan_rate"].notna()] if "scan_rate" in scanning else scanning


def team_match_summary(tables_for_game: dict[str, pd.DataFrame], roster: pd.DataFrame) -> pd.DataFrame:
    """Home v Away on every pillar for one game: index = metric label, columns = clubs.

    `tables_for_game` is score_game's output (or score_all's filtered to one game),
    `roster` that game's roster file. Rows whose table is missing are left out.
    """
    clubs = _sides(roster)
    rows: dict[str, dict[str, float]] = {}

    def add(label: str, per_side: pd.Series) -> None:
        rows[label] = {s: float(per_side.get(s, np.nan)) for s in clubs}

    if (d := _table(tables_for_game, "decoys", ["decoy_value"])) is not None:
        add("Decoy runs", d.groupby("side").size())
        add("xT opened by decoy runs", d.groupby("side")["decoy_value"].sum())
    if (p := _table(tables_for_game, "pressing", ["lane_cut", "seconds"])) is not None:
        add("Pressing lane-cut seconds", (p["lane_cut"] * p["seconds"]).groupby(p["side"]).sum())
    if (t := _table(tables_for_game, "transitions", ["reached_sprint", "latency"])) is not None:
        sprinted = t["reached_sprint"].astype(bool)
        add("Transitions sprinted", sprinted.groupby(t["side"]).sum())
        add("Transitions (total)", t.groupby("side").size())
        add("Transition sprint rate %", sprinted.groupby(t["side"]).mean() * 100)
        add("Median transition latency (s)", t[sprinted].groupby("side")["latency"].median())
    if (s := _table(tables_for_game, "scanning", ["n_scans", "retained"])) is not None and len(s):
        add("Receptions", s.groupby("side").size())
        seen = _observed(s)       # unobserved receptions still count for receptions/retention
        add("Scans per reception", seen.groupby("side")["n_scans"].mean())
        add("Ball retention after receiving %", s["retained"].astype(float).groupby(s["side"]).mean() * 100)
    if (pr := _table(tables_for_game, "pressures", ["intensity"])) is not None:
        add("Pressure intensity (m closed down)", pr.groupby("side")["intensity"].sum())
    if (o := _table(tables_for_game, "obso", ["obso"])) is not None:
        add("Off-ball scoring opportunity (xT·s)", o.groupby("side")["obso"].sum())
    if (r := _table(tables_for_game, "runs", ["run_type"])) is not None:
        add("Runs in behind", (r["run_type"] == "Run in Behind").groupby(r["side"]).sum())
    if (rc := _table(tables_for_game, "receipts", ["in_space"])) is not None and len(rc):
        add("Receipts in space %", rc["in_space"].astype(float).groupby(rc["side"]).mean() * 100)
    if (c := _table(tables_for_game, "control", ["control_pct", "minute"])) is not None and len(c):
        per_frame = c.groupby(["side", "minute"])["control_pct"].sum()
        add("Pitch control % in possession", per_frame.groupby(level="side").mean())
        frames = per_frame.groupby(level="side").size()
        add("Possession share % (tracked)", frames / frames.sum() * 100)
    table = pd.DataFrame.from_dict(rows, orient="index", columns=list(clubs))
    return table.rename(columns=clubs).round(2)


def intensity_blocks(tables_for_game: dict[str, pd.DataFrame], roster: pd.DataFrame,
                     block: int = 15) -> pd.DataFrame:
    """Each side's off-ball work per `block`-minute block of one game.

    One row per (block, side): block_start (match minute), block ("45-60"), club,
    side, lane_cut_seconds, sprints (transitions that reached a sprint), decoy_xT,
    scans (head checks before receiving, from receptions with head data only;
    NaN without a scanning table or for a block with no observed reception),
    pressure_intensity and obso (NaN when their table is missing). Match
    minutes run 0-45 then 45-90, so second-half blocks continue 45-60, 60-75...;
    second-half stoppage time is one "90+" block (when `block` divides 90). Blocks
    with no events are 0 (scans: NaN).
    """
    clubs = _sides(roster)
    sources = {"lane_cut_seconds": ("pressing", ["lane_cut", "seconds"], lambda t: t["lane_cut"] * t["seconds"]),
               "sprints": ("transitions", ["reached_sprint"], lambda t: t["reached_sprint"].astype(float)),
               "decoy_xT": ("decoys", ["decoy_value"], lambda t: t["decoy_value"]),
               "scans": ("scanning", ["n_scans"], lambda t: t["n_scans"].astype(float)),
               "pressure_intensity": ("pressures", ["intensity"], lambda t: t["intensity"]),
               "obso": ("obso", ["obso"], lambda t: t["obso"])}
    sums, last = {}, 0.0
    for metric, (key, needs, value) in sources.items():
        table = _table(tables_for_game, key, [*needs, "minute"])
        if table is None:
            continue
        starts = (np.floor(table["minute"].astype(float) / block) * block).astype(int)
        if 90 % block == 0:
            starts = starts.clip(upper=90)
        last = max(last, float(table["minute"].max()) if len(table) else 0.0)
        if metric == "scans":
            observed = table.index.isin(_observed(table).index)
            table, starts = table[observed], starts[observed]
            if table.empty:
                continue
        sums[metric] = value(table).groupby([starts, table["side"]]).sum()
    final = min(int(last // block) * block, 90) if 90 % block == 0 else int(last // block) * block
    grid = pd.MultiIndex.from_product([np.arange(0, final + 1, block), list(clubs)],
                                      names=["block_start", "side"])
    out = pd.DataFrame(index=grid)
    for metric in sources:
        if metric not in sums:
            out[metric] = np.nan
        else:
            values = sums[metric].reindex(grid)
            out[metric] = (values if metric == "scans" else values.fillna(0.0)).to_numpy()
    out = out.reset_index()
    out.insert(1, "block", out["block_start"].map(
        lambda s: "90+" if s == 90 and 90 % block == 0 else f"{s}-{s + block}"))
    out.insert(2, "club", out["side"].map(clubs))
    return out


# ── Validation against the hidden truth ──────────────────────────────────────
def attach_truth(pool: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """`pool` joined with league_players.csv's hidden attributes, on name (and club)."""
    truth = players.rename(columns={"club": "real_team"})
    on = ["name", "real_team"] if {"real_team"} <= set(pool) & set(truth) else ["name"]
    extra = [c for c in truth.columns if c not in pool.columns or c in on]
    return pool.merge(truth[extra], on=on, how="left")


def rank_within(df: pd.DataFrame, columns: Sequence[str], group: str = "broad_position") -> pd.DataFrame:
    """`df` with each of `columns` replaced by its percentile rank (0-1) within `group`.

    Correlating these compares players only with their own position: a metric that
    is mostly a forward's job isn't credited, or penalised, just for that.
    """
    present = [c for c in columns if c in df]
    return df.assign(**df.groupby(group)[present].rank(pct=True)) if group in df else df.copy()


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, resamples: int = 1000,
                       seed: int = 0) -> tuple[float, float, float]:
    """(rho, low, high): Spearman rho and its percentile-bootstrap 95% interval."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    if len(x) < 3:
        return np.nan, np.nan, np.nan

    def rho(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Row-wise Spearman: Pearson correlation of the ranks."""
        ra, rb = rankdata(a, axis=-1), rankdata(b, axis=-1)
        ra = ra - ra.mean(axis=-1, keepdims=True)
        rb = rb - rb.mean(axis=-1, keepdims=True)
        denom = np.sqrt((ra ** 2).sum(axis=-1) * (rb ** 2).sum(axis=-1))
        with np.errstate(invalid="ignore", divide="ignore"):
            return (ra * rb).sum(axis=-1) / denom

    picks = np.random.default_rng(seed).integers(0, len(x), size=(resamples, len(x)))
    samples = rho(x[picks], y[picks])
    low, high = np.nanpercentile(samples, [2.5, 97.5]) if np.isfinite(samples).any() else (np.nan, np.nan)
    return float(rho(x, y)), float(low), float(high)


def verdict(rho: float, sign: int = 1) -> str:
    """Plain-English strength of the expected relationship (sign -1: should fall)."""
    if np.isnan(rho):
        return "not enough data"
    r = rho * sign
    if r <= -0.3:              # strong link the wrong way: a sign or definition bug
        return "opposite: the metric moves against the trait"
    word = "strong" if r >= 0.5 else "moderate" if r >= 0.3 else "weak" if r >= 0.1 else "none"
    return {"strong": "strong: the metric clearly recovers the trait",
            "moderate": "moderate: the metric tracks the trait, with noise",
            "weak": "weak: only a faint link to the trait",
            "none": "none: no evidence the metric reflects the trait"}[word]


def validation(pool_with_truth: pd.DataFrame,
               pairs: Sequence[tuple[str, str] | tuple[str, str, int]] = VALIDATION_PAIRS,
               resamples: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Spearman rho between each hidden attribute and the metric meant to recover it.

    `pairs` holds (attribute, metric) or (attribute, metric, expected sign); pairs
    with a missing column are skipped. Columns: attribute, metric, rho, low, high
    (bootstrap 95% interval, `resamples` with a fixed `seed`), n and verdict.
    """
    rows = []
    for pair in pairs:
        attribute, metric, sign = (*pair, 1) if len(pair) == 2 else pair
        if attribute not in pool_with_truth or metric not in pool_with_truth:
            continue
        x, y = pool_with_truth[attribute].astype(float), pool_with_truth[metric].astype(float)
        rho, low, high = bootstrap_spearman(x.to_numpy(), y.to_numpy(), resamples, seed)
        rows.append({"attribute": attribute, "metric": metric, "rho": rho, "low": low, "high": high,
                     "n": int((x.notna() & y.notna()).sum()), "verdict": verdict(rho, sign)})
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


# ── Scanning and ball retention ──────────────────────────────────────────────
def wilson_interval(successes, n, z: float = 1.959964):
    """Wilson score interval for a binomial proportion (works on arrays; n=0 -> NaN)."""
    successes, n = np.asarray(successes, float), np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = successes / n
        centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
        half = z / (1 + z ** 2 / n) * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return centre - half, centre + half


def scanning_effect(scanning_table: pd.DataFrame | None) -> pd.DataFrame:
    """Ball retention after receiving, by scans in the 5 s before ("0", "1", "2", "3+").

    Columns: bucket, n (receptions), retained (count), retention (share), low and
    high (Wilson 95% interval). Only receptions with head data count (scan_rate not
    NaN); empty, missing or wholly unobserved input gives an empty frame.
    """
    if scanning_table is None or scanning_table.empty or \
            not {"n_scans", "retained"} <= set(scanning_table.columns):
        return pd.DataFrame(columns=EFFECT_COLUMNS)
    table = _observed(scanning_table).dropna(subset=["n_scans", "retained"])
    if table.empty:
        return pd.DataFrame(columns=EFFECT_COLUMNS)
    bucket = np.minimum(table["n_scans"].astype(int), 3).map(dict(enumerate(SCAN_BUCKETS)))
    grouped = table["retained"].astype(float).groupby(bucket)
    out = pd.DataFrame({"n": grouped.size(), "retained": grouped.sum()}).reindex(SCAN_BUCKETS, fill_value=0)
    out["retention"] = np.where(out["n"] > 0, out["retained"] / out["n"].replace(0, 1), np.nan)
    out["low"], out["high"] = wilson_interval(out["retained"], out["n"])
    out = out.rename_axis("bucket").reset_index()
    out["n"] = out["n"].astype(int)
    return out[EFFECT_COLUMNS]
