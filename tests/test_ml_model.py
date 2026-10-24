import numpy as np
import pandas as pd

from core.ml_model import FEATURES, ROLES, cluster_player_roles, present_features, role_names, train_valuation_model


def league(n=60, seed=0):
    """Players whose value depends on pressing, plus noise; four clear movement types."""
    rng = np.random.default_rng(seed)
    kind = np.arange(n) % 4
    df = pd.DataFrame({
        "decoy_p90": np.where(kind == 0, 0.2, 0.02) + rng.normal(0, 0.01, n),
        "lane_cut_p90": np.where(kind == 1, 120.0, 30.0) + rng.normal(0, 5, n),
        "median_latency": np.where(kind == 2, 0.8, 4.0) + rng.normal(0, 0.2, n),
        "pitch_control_pct": np.where(kind == 3, 9.0, 3.0) + rng.normal(0, 0.3, n),
        "broad_position": rng.choice(["Defender", "Midfielder", "Forward"], n)})
    df["market_value_m"] = np.exp(2.5 + df["lane_cut_p90"] / 60 + rng.normal(0, 0.1, n))
    return df


def test_valuation_is_out_of_fold_and_learns_real_signal():
    model, out = train_valuation_model(league())
    assert out["Predicted_Value"].notna().all()
    assert out.attrs["valuation"]["r2"] > 0.8                       # value really depends on pressing
    assert "Pressing Lane Cuts /90" in out.attrs["valuation"]["features"]


def test_valuation_without_signal_scores_no_better_than_average():
    noise = league().assign(market_value_m=np.exp(np.random.default_rng(3).normal(3, 1, 60)))
    _, out = train_valuation_model(noise, include_position=False)
    assert out.attrs["valuation"]["r2"] < 0.2                       # out-of-fold: no memorising


def test_clusters_find_the_four_types_and_name_them_uniquely():
    df = league()
    out = cluster_player_roles(df)
    assert out["Cluster_Role"].nunique() == 4
    kind = np.arange(len(df)) % 4
    expected = {0: "Space creator", 1: "Lane-cutting presser", 2: "Transition sprinter", 3: "Space occupier"}
    assert (out["Cluster_Role"] == pd.Series(kind).map(expected)).all()
    assert list(out.attrs["clusters"].columns) == [FEATURES[f] for f in present_features(df)]
    assert "Scanning rate (scans/s before receiving)" not in out.attrs["clusters"].columns   # absent feature


def with_scanners(n=75, seed=0):
    """Five clear movement types: the four above plus players who scan a lot."""
    rng = np.random.default_rng(seed)
    kind = np.arange(n) % 5
    df = pd.DataFrame({
        "decoy_p90": np.where(kind == 0, 0.2, 0.02) + rng.normal(0, 0.01, n),
        "lane_cut_p90": np.where(kind == 1, 120.0, 30.0) + rng.normal(0, 5, n),
        "median_latency": np.where(kind == 2, 0.8, 4.0) + rng.normal(0, 0.2, n),
        "pitch_control_pct": np.where(kind == 3, 9.0, 3.0) + rng.normal(0, 0.3, n),
        "scan_rate": np.where(kind == 4, 1.2, 0.3) + rng.normal(0, 0.03, n),
        "broad_position": rng.choice(["Defender", "Midfielder", "Forward"], n)})
    df["market_value_m"] = np.exp(2.5 + df["scan_rate"] + rng.normal(0, 0.1, n))
    return df, kind


def test_five_archetypes_with_scanning():
    df, kind = with_scanners()
    out = cluster_player_roles(df, k=5)
    expected = {0: "Space creator", 1: "Lane-cutting presser", 2: "Transition sprinter",
                3: "Space occupier", 4: "Scanner"}
    assert (out["Cluster_Role"] == pd.Series(kind).map(expected)).all()
    _, priced = train_valuation_model(df)
    assert "Scanning rate (scans/s before receiving)" in priced.attrs["valuation"]["features"]


def test_rectangular_role_naming_is_unique_for_any_k():
    df, _ = with_scanners()
    for k in (2, 4, 6, 7):
        out = cluster_player_roles(df, k=k)
        labels = out.groupby("Cluster_ID")["Cluster_Role"].first()
        assert labels.nunique() == k
        if k <= len(ROLES):                     # each cluster gets a plain archetype name
            assert set(labels) <= set(ROLES.values())
        else:
            assert set(ROLES.values()) <= set(labels)


def test_role_names_k4_with_five_archetypes_uses_hungarian_assignment():
    centres = pd.DataFrame({"decoy_p90": [2.0, 0, 0, 0], "lane_cut_p90": [1.9, 0, 0, 0],
                            "median_latency": [0, 1.0, 0, 0], "pitch_control_pct": [0, 0, 1.0, 0],
                            "scan_rate": [0, 0, 0, 1.0]})
    names = role_names(centres)
    # Cluster 0 leads on decoys and pressing; it takes the stronger, and with four clusters
    # for five archetypes, pressing is the one left unused. Every name is distinct.
    assert names == {0: "Space creator", 1: "Transition sprinter", 2: "Space occupier", 3: "Scanner"}


def test_blank_scan_rate_is_ignored():
    df = league().assign(scan_rate=np.nan)
    out = cluster_player_roles(df)
    assert out["Cluster_Role"].nunique() == 4 and "Scanner" not in set(out["Cluster_Role"])
    _, priced = train_valuation_model(df)
    assert "Scanning rate (scans/s before receiving)" not in priced.attrs["valuation"]["features"]
