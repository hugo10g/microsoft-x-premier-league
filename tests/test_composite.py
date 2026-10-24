import json

import numpy as np
import pandas as pd
import pytest

from core import composite, pipeline

BIG = 1e9           # a sample so large the shrinkage is negligible


def config(**overrides):
    base = {"group_by": ["season", "broad_position"], "exclude_positions": ["Goalkeeper"], "min_minutes": 90,
            "min_coverage": 0.5, "categories": {
                "attack": {"label": "Attack", "weight": 1, "metrics": {
                    "a": {"weight": 1, "higher_is_better": True, "sample": "n", "min_sample": 1}}},
                "defence": {"label": "Defence", "weight": 1, "metrics": {
                    "d": {"weight": 1, "higher_is_better": False, "sample": "n", "min_sample": 1}}}}}
    return base | overrides


def pool(a, d, minutes=None, position=None, n=BIG):
    k = len(a)
    return pd.DataFrame({"name": [f"P{i}" for i in range(k)], "real_team": "X", "minutes": minutes or [900] * k,
                         "broad_position": position or ["Midfielder"] * k, "a": a, "d": d, "n": n})


def test_percentiles_are_mid_rank_and_flip_when_lower_is_better():
    values, groups = pd.Series([1.0, 2.0, 3.0, 4.0]), [pd.Series(["M"] * 4)]
    np.testing.assert_allclose(composite.percentile(values, groups), [12.5, 37.5, 62.5, 87.5])
    np.testing.assert_allclose(composite.percentile(values, groups, higher_is_better=False), [87.5, 62.5, 37.5, 12.5])


def test_each_position_is_ranked_only_against_itself():
    out = composite.composite(pool([1, 2, 10, 20], [0, 0, 0, 0], position=["Defender"] * 2 + ["Forward"] * 2),
                              config())
    np.testing.assert_allclose(out["a_score"], [25, 75, 25, 75], atol=1e-6)


def test_low_samples_shrink_towards_the_middle():
    out = composite.composite(pool([1, 2], [0, 0], n=[1, BIG]), config())
    assert out["a_score"].iloc[0] == pytest.approx(50 + (25 - 50) * 0.5)    # n = min_sample: halfway
    assert out["a_score"].iloc[1] == pytest.approx(75, abs=1e-6)


def test_ineligible_players_are_not_ranked_and_do_not_move_anyone_else():
    out = composite.composite(pool([1, 2, 99, 99], [0, 0, 0, 0], minutes=[900, 900, 30, 900],
                                   position=["Midfielder"] * 3 + ["Goalkeeper"]), config())
    assert out["eligible"].tolist() == [True, True, False, False]
    assert out.loc[~out["eligible"], ["a_score", "index", "coverage"]].isna().all().all()
    np.testing.assert_allclose(out["a_score"].iloc[:2], [25, 75], atol=1e-6)


def test_missing_metrics_renormalise_and_low_coverage_gives_no_index():
    out = composite.composite(pool([1, 2], [np.nan, 5.0]), config())
    assert out["coverage"].tolist() == [0.5, 1.0]
    assert out["index"].iloc[0] == pytest.approx(out["attack"].iloc[0])     # only the category it has
    strict = composite.composite(pool([1, 2], [np.nan, 5.0]), config(min_coverage=0.75))
    assert np.isnan(strict["index"].iloc[0]) and strict["index"].notna().iloc[1]


def test_index_is_the_weighted_mean_of_the_subscores():
    cfg = config()
    cfg["categories"]["attack"]["weight"] = 3
    out = composite.composite(pool([1, 2, 3, 4], [4, 3, 2, 1]), cfg)
    expected = (3 * out["attack"] + out["defence"]) / 4
    np.testing.assert_allclose(out["index"], expected)
    np.testing.assert_allclose(out["defence"], out["attack"], atol=1e-6)       # lower d is better


def test_polygon_is_radar_ready_best_first():
    records = composite.polygon(composite.composite(pool([1, 2], [2, 1]), config()), config())
    assert [r["name"] for r in records] == ["P1", "P0"]
    assert [axis["category"] for axis in records[0]["axes"]] == ["attack", "defence"]
    assert set(records[0]["axes"][0]) == {"category", "label", "score", "metrics"}
    json.dumps(records)                                                          # serialisable as is


def test_default_config_uses_only_metrics_the_pipeline_produces():
    cfg = composite.load_config()
    produced = set(pipeline.PILLARS) | set(pipeline.TACTICAL_COLUMNS) | {"pitch_control_pct"}
    for category in cfg["categories"].values():
        for metric, spec in category["metrics"].items():
            assert metric in produced and spec["sample"] in {"minutes", "receptions", "transitions"}
            assert spec["weight"] == 1 and category["weight"] == 1                # equal by default
    # Subscores become columns beside the pipeline's and the hidden traits: a clash overwrites one.
    traits = {"work_rate", "pressing", "awareness", "speed", "passing", "finishing"}
    assert not set(cfg["categories"]) & (produced | traits)


def test_negative_weights_are_rejected(tmp_path):
    bad = config()
    bad["categories"]["attack"]["metrics"]["a"]["weight"] = -1
    (path := tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        composite.load_config(path)
