"""
Off-Ball Impact Index: the five tactical metrics, weighted by what they do for the team.

The Off-Ball Index weights its categories equally. This one lets the matches decide:
in which pillars does more off-ball work go with a better shot balance?

    1. Cut every match into `block`-minute windows, per team.
    2. In each, the team's total of each metric, normalised by the possession it
       applies to: pressure intensity and defensive responsibility per minute the
       opponent has the ball; OBSO and runs per minute of the team's own; receipts
       in space as a share of receptions. Responsibility is negated (less is better).
    3. Regress the window's shot difference (for − against) on the standardised
       metrics, after taking out possession share, by non-negative least squares:
       a weight is the shots one standard deviation of the metric is worth, and a
       metric with no detectable benefit gets 0 instead of counting against the
       players who are good at it.
    4. Bootstrap whole matches to see how stable each weight is.
    5. A pillar's weight is the sum of its metrics' weights. The player index is
       core.composite with those pillar weights: the same within-position
       percentiles, low-sample shrinkage and coverage, so 50 is still the
       position's middle and the radar shows each pillar's subscore.

Shots are the outcome, not xT, because defensive responsibility is itself the xT of
opponents' passes: fitting it to xT would be circular. Team windows, not players, are
regressed because a shot balance belongs to the whole team; the weights say how much
each pillar matters, and players are scored on their own pillar work. It is an
association, not causation: a team that is winning may sit back and press less.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from core import tracking

# Pillar -> (label, player metric columns). Keys are not pipeline or trait column names.
PILLARS: dict[str, tuple[str, list[str]]] = {
    "press": ("Pressure", ["pressure_intensity"]),
    "receive": ("Receipts in space", ["receipts_in_space_pct"]),
    "cover": ("Defensive cover", ["def_responsibility_p90"]),
    "threat": ("OBSO", ["obso_p90"]),
    "move": ("Run types", ["runs_in_behind_p90", "pulling_wide_p90", "dropping_deep_p90"]),
}
METRICS = [m for _, columns in PILLARS.values() for m in columns]
LOWER_IS_BETTER = {"def_responsibility_p90"}
SAMPLES = {"receipts_in_space_pct": ("receptions", 10)}          # anything else: (minutes, 90)
RUN_COLUMNS = {"Run in Behind": "runs_in_behind_p90", "Pulling Wide": "pulling_wide_p90",
               "Dropping Deep": "dropping_deep_p90"}


def _side(table: pd.DataFrame) -> pd.Series:
    return table["player"].str.split("_").str[0]


def game_blocks(tables: dict[str, pd.DataFrame], events: pd.DataFrame, block: float = 5.0) -> pd.DataFrame:
    """One row per (team, window) of one game: possession minutes, shots and each metric's team total.

    `tables` are core.pipeline.score_game's (one game; player ids "Home_n"/"Away_n"),
    `events` the game's Metrica events. Windows run on the tracking clock (seconds).
    """
    seconds = block * 60
    frames = np.arange(events["Start Frame"].min(), events["End Frame"].max() + 1)
    known = pd.concat([events[[f"{edge} Frame", f"{edge} Time [s]"]].set_axis(["frame", "time"], axis=1)
                       for edge in ("Start", "End")]).dropna().sort_values("frame")
    clock = np.interp(frames, known["frame"], known["time"])
    owner = tracking.possession_by_frame(events, frames)
    windows = (clock // seconds).astype(int)
    dt = np.diff(clock, append=clock[-1])
    rows = []
    for k in np.unique(windows):
        here = windows == k
        held = {team: dt[here & (owner == team)].sum() / 60 for team in ("Home", "Away")}
        for team, rival in (("Home", "Away"), ("Away", "Home")):
            rows.append({"block": k, "team": team, "own_min": held[team], "opp_min": held[rival]})
    out = pd.DataFrame(rows).set_index(["team", "block"])

    def total(key: str, value=None) -> pd.Series:
        table = tables.get(key)
        if table is None or table.empty:
            return pd.Series(0.0, index=out.index)
        grouped = (value(table) if value else pd.Series(1.0, index=table.index)).groupby(
            [_side(table), (table["time"] // seconds).astype(int)])
        return grouped.sum().reindex(out.index).fillna(0)

    shots = events[events["Type"] == "SHOT"]
    out["shots_for"] = shots.groupby(["Team", (shots["Start Time [s]"] // seconds).astype(int)]).size() \
        .reindex(out.index).fillna(0)
    rival = out.rename(index={"Home": "Away", "Away": "Home"}, level="team")
    out["shots_against"] = rival["shots_for"].reindex(out.index)
    out["pressure_intensity"] = total("pressures", lambda t: t["intensity"]) / out["opp_min"]
    out["def_responsibility_p90"] = total("responsibility", lambda t: t["xt_conceded"]) / out["opp_min"]
    out["obso_p90"] = total("obso", lambda t: t["obso"]) / out["own_min"]
    out["receipts_in_space_pct"] = total("receipts", lambda t: t["in_space"].astype(float)) / total("receipts")
    for kind, column in RUN_COLUMNS.items():
        out[column] = total("runs", lambda t, k=kind: (t["run_type"] == k).astype(float)) / out["own_min"]
    return out.reset_index()


def team_blocks(data: dict[str, pd.DataFrame], games: list, block: float = 5.0) -> pd.DataFrame:
    """`game_blocks` for every game in `games` (folders), from core.pipeline.score_all's tables."""
    frames = []
    for game in games:
        tables = {key: table[table["game"] == game.name] for key, table in data.items() if "game" in table}
        events = tracking.load_events(game / f"{game.name}_RawEventsData.csv")
        frames.append(game_blocks(tables, events, block).assign(game=game.name))
    return pd.concat(frames, ignore_index=True)


def fit_weights(blocks: pd.DataFrame, min_possession: float = 0.5, resamples: int = 500,
                seed: int = 0) -> pd.DataFrame:
    """Per metric: NNLS weight in shots per SD, its bootstrap 95% interval and how often it is > 0.

    Windows where either side had the ball under `min_possession` minutes are dropped;
    a missing receipts share takes the mean. Resamples draw whole games.
    """
    usable = blocks[(blocks["own_min"] >= min_possession) & (blocks["opp_min"] >= min_possession)].copy()
    usable[METRICS] = usable[METRICS].fillna(usable[METRICS].mean())
    for column in LOWER_IS_BETTER:
        usable[column] = -usable[column]
    share = usable["own_min"] / (usable["own_min"] + usable["opp_min"])

    def fit(rows: pd.DataFrame, control: pd.Series) -> np.ndarray:
        x = rows[METRICS].to_numpy(float)
        x = (x - x.mean(axis=0)) / np.where(x.std(axis=0) > 0, x.std(axis=0), 1)
        y = (rows["shots_for"] - rows["shots_against"]).to_numpy(float)
        basis = np.column_stack([np.ones(len(rows)), control.to_numpy(float)])
        residual = lambda v: v - basis @ np.linalg.lstsq(basis, v, rcond=None)[0]
        return nnls(np.column_stack([residual(c) for c in x.T]), residual(y))[0]

    weights = fit(usable, share)
    rng, games = np.random.default_rng(seed), usable["game"].unique()
    draws = []
    for _ in range(resamples):
        picked = pd.concat([usable[usable["game"] == g] for g in rng.choice(games, len(games))], ignore_index=True)
        draws.append(fit(picked, picked["own_min"] / (picked["own_min"] + picked["opp_min"])))
    draws = np.array(draws)
    return pd.DataFrame({"weight": weights, "low": np.percentile(draws, 2.5, axis=0),
                         "high": np.percentile(draws, 97.5, axis=0), "positive_share": (draws > 0).mean(axis=0)},
                        index=pd.Index(METRICS, name="metric"))


def index_config(weights: pd.Series, min_minutes: float = 90, min_coverage: float = 0.5) -> dict:
    """A core.composite config: pillar weight = sum of its metrics' weights; equal metrics inside.

    Inside a pillar the metrics count equally, so its subscore describes the player's
    work on it whatever the fit. If every weight is 0 the pillars count equally.
    """
    pillar = {name: float(weights.reindex(columns).fillna(0).sum()) for name, (_, columns) in PILLARS.items()}
    if not sum(pillar.values()):
        pillar = dict.fromkeys(pillar, 1.0)
    metric = lambda m: {"weight": 1, "higher_is_better": m not in LOWER_IS_BETTER,
                        "sample": SAMPLES.get(m, ("minutes", 90))[0], "min_sample": SAMPLES.get(m, ("minutes", 90))[1]}
    return {"group_by": ["season", "broad_position"], "exclude_positions": ["Goalkeeper"], "min_minutes": min_minutes,
            "min_coverage": min_coverage,
            "categories": {name: {"label": label, "weight": pillar[name], "metrics": {m: metric(m) for m in columns}}
                           for name, (label, columns) in PILLARS.items()}}
