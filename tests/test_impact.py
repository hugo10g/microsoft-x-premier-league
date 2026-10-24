import numpy as np
import pandas as pd
import pytest

from core import composite, impact

FPS = 25


def event(team, kind, second):
    frame = round(second * FPS)
    return {"Team": team, "Type": kind, "Subtype": None, "Period": 1, "Start Frame": frame,
            "Start Time [s]": frame / FPS, "End Frame": frame, "End Time [s]": frame / FPS, "From": None, "To": None}


def test_windows_split_possession_mirror_shots_and_normalise_by_the_right_side():
    # Home have it for 0-150 s, Away from 150 s; Away shoot at 200 s; the game ends at 600 s.
    events = pd.DataFrame([event("Home", "PASS", 0.04), event("Away", "PASS", 150.04),
                           event("Away", "SHOT", 200.04), event("Away", "PASS", 600.0)])
    tables = {"pressures": pd.DataFrame({"player": ["Home_1"], "time": [210.0], "intensity": [3.0]}),
              "obso": pd.DataFrame({"player": ["Home_2", "Away_9"], "time": [10.0, 400.0], "obso": [1.0, 2.0]})}
    out = impact.game_blocks(tables, events, block=5).set_index(["team", "block"])
    assert out.loc[("Home", 0), "own_min"] == pytest.approx(2.5, abs=0.01)
    assert out.loc[("Away", 1), "own_min"] == pytest.approx(5.0, abs=0.01)
    assert out.loc[("Away", 0), "shots_for"] == 1 and out.loc[("Home", 0), "shots_against"] == 1
    assert out.loc[("Home", 0), "pressure_intensity"] == pytest.approx(3.0 / 2.5, rel=0.01)   # per opponent minute
    assert out.loc[("Home", 0), "obso_p90"] == pytest.approx(1.0 / 2.5, rel=0.01)             # per own minute
    assert out.loc[("Away", 1), "obso_p90"] == pytest.approx(2.0 / 5.0, rel=0.01)
    assert np.isnan(out.loc[("Home", 0), "receipts_in_space_pct"])                         # no receptions: unknown


def blocks(n=240, seed=1):
    rng = np.random.default_rng(seed)
    table = pd.DataFrame({m: rng.normal(size=n) for m in impact.METRICS})
    table["own_min"] = table["opp_min"] = 2.5
    table["game"] = np.repeat([f"g{k}" for k in range(6)], n // 6)
    swing = 2 * table["obso_p90"] - 1.5 * table["def_responsibility_p90"] + rng.normal(scale=0.3, size=n)
    return table.assign(shots_for=swing, shots_against=0.0)


def test_weights_find_what_moves_shots_flip_responsibility_and_never_go_negative():
    weights = impact.fit_weights(blocks(), resamples=50)
    assert weights.loc["obso_p90", "weight"] == pytest.approx(2.0, abs=0.2)
    assert weights.loc["def_responsibility_p90", "weight"] == pytest.approx(1.5, abs=0.2)   # less conceded = good
    others = weights.drop(index=["obso_p90", "def_responsibility_p90"])["weight"]
    assert (weights["weight"] >= 0).all() and others.max() < 0.2
    assert weights.loc["obso_p90", "low"] > 0 and weights.loc["obso_p90", "positive_share"] == 1


def test_pillar_weights_sum_their_metrics_and_drive_the_composite_index():
    weights = pd.Series(0.0, index=impact.METRICS)
    weights[["obso_p90", "runs_in_behind_p90", "pulling_wide_p90"]] = [3.0, 0.5, 0.5]
    config = impact.index_config(weights)
    assert config["categories"]["threat"]["weight"] == 3 and config["categories"]["move"]["weight"] == 1
    assert config["categories"]["press"]["weight"] == 0
    pool = pd.DataFrame({"name": ["A", "B"], "minutes": 900, "receptions": 1e9, "broad_position": "Forward",
                         **{m: [1.0, 2.0] for m in impact.METRICS}})
    pool["minutes"] = 1e9                                   # no shrinkage
    out = composite.composite(pool, config)
    np.testing.assert_allclose(out["index"], (3 * out["threat"] + out["move"]) / 4)
    assert not set(impact.PILLARS) & set(pool.columns)
    equal = impact.index_config(pd.Series(0.0, index=impact.METRICS))
    assert {c["weight"] for c in equal["categories"].values()} == {1.0}
