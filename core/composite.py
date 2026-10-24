"""
Composite Off-Ball Index: one 0-100 number per player, plus category subscores for a radar.

    score_m   = 50 + (P_m - 50) · n / (n + k)      P_m = mid-rank percentile of metric m within
                                                  (season, broad position), flipped when lower is
                                                  better; n = the player's sample, k = min_sample
    sub_c     = Σ w_m · score_m / Σ w_m            over the metrics of category c the player has
    index     = Σ W_c · sub_c / Σ W_c              over the categories the player has

Weights, directions, samples and categories live in config/composite_index.json. A
player below `min_minutes` (or in an excluded position) is not ranked. A missing value
drops out of the sums and the weights are renormalised; `coverage` is the share of the
total weight actually observed, and the index is NaN below `min_coverage`. A grouping
column the pool doesn't have (e.g. no "season" in a one-season league) is one group.
See docs/composite_index.md for the method and its limits.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "composite_index.json"


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    """The index config; raises if a weight is negative or a category has no positive weight."""
    config = json.loads(Path(path).read_text())
    for name, category in config["categories"].items():
        weights = [category["weight"], *(m["weight"] for m in category["metrics"].values())]
        if min(weights) < 0 or not any(m["weight"] > 0 for m in category["metrics"].values()):
            raise ValueError(f"category {name!r} needs non-negative weights and one positive metric weight")
    return config


def percentile(values: pd.Series, groups: list[pd.Series], higher_is_better: bool = True) -> pd.Series:
    """0-100 mid-rank percentile within each group ((rank - 0.5) / n); NaN stays NaN."""
    ranks = values.groupby(groups).rank(ascending=higher_is_better)
    n = values.notna().groupby(groups).transform("sum")
    return (ranks - 0.5) / n * 100


def composite(pool: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """`pool` with eligible, every metric's score, each category's subscore, coverage and index.

    Scores are 0-100 with 50 the group's middle; ineligible players keep NaN throughout.
    """
    config = config or load_config()
    out = pool.copy()
    for column in config["group_by"]:
        if column not in out:
            out[column] = "all"
    out["eligible"] = (out["minutes"] >= config["min_minutes"]) & \
        ~out["broad_position"].isin(config.get("exclude_positions", []))
    ranked = out[out["eligible"]]
    groups = [ranked[c] for c in config["group_by"]]
    total = sum(c["weight"] for c in config["categories"].values())
    subscores, coverage = {}, pd.Series(0.0, index=out.index)
    for name, category in config["categories"].items():
        weighted, weights = pd.Series(0.0, index=out.index), pd.Series(0.0, index=out.index)
        metric_total = sum(m["weight"] for m in category["metrics"].values())
        for metric, spec in category["metrics"].items():
            values = ranked[metric].astype(float) if metric in ranked else pd.Series(np.nan, index=ranked.index)
            sample = ranked[spec["sample"]].astype(float).fillna(0)
            score = 50 + (percentile(values, groups, spec["higher_is_better"]) - 50) * \
                sample / (sample + spec["min_sample"])
            out[f"{metric}_score"] = score.reindex(out.index)
            seen = out[f"{metric}_score"].notna()
            weighted += out[f"{metric}_score"].fillna(0) * spec["weight"]
            weights += seen * spec["weight"]
            coverage += seen * spec["weight"] / metric_total * category["weight"] / total
        subscores[name] = out[name] = weighted / weights.where(weights > 0)
    w = pd.Series({name: c["weight"] for name, c in config["categories"].items()})
    table = pd.DataFrame(subscores)
    index = (table * w).sum(axis=1, min_count=1) / table.notna().mul(w).sum(axis=1).replace(0, np.nan)
    out["coverage"] = coverage.where(out["eligible"])
    out["index"] = index.where(out["coverage"] >= config["min_coverage"])
    return out


def polygon(scored: pd.DataFrame, config: dict | None = None,
            identity: tuple[str, ...] = ("name", "real_team", "broad_position", "minutes")) -> list[dict]:
    """Radar-ready records for the ranked players, best index first.

    Each: identity fields, index, coverage and `axes` - one {category, label, score,
    metrics: {metric: score}} per category, in config order (score None when unseen).
    """
    config = config or load_config()
    clean = lambda v: None if pd.isna(v) else round(float(v), 1)
    records = []
    for _, row in scored[scored["eligible"]].sort_values("index", ascending=False).iterrows():
        axes = [{"category": name, "label": c["label"], "score": clean(row[name]),
                 "metrics": {m: clean(row[f"{m}_score"]) for m in c["metrics"]}}
                for name, c in config["categories"].items()]
        records.append({**{k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in identity if k in row},
                        "index": clean(row["index"]), "coverage": round(float(row["coverage"]), 2), "axes": axes})
    return records
