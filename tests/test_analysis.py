import numpy as np
import pandas as pd
import pytest

from core import analysis


# ── Fixtures ─────────────────────────────────────────────────────────────────
def pool() -> pd.DataFrame:
    """Six outfielders, two per position, with all four pillars plus latency."""
    return pd.DataFrame({
        "name": ["Fwd A", "Fwd B", "Mid A", "Mid B", "Def A", "Def B"],
        "real_team": ["Harbourside FC", "Northvale Athletic"] * 3,
        "broad_position": ["Forward", "Forward", "Midfielder", "Midfielder", "Defender", "Defender"],
        "minutes": [180.0, 170, 160, 190, 180, 175],
        "decoy_p90": [0.30, 0.10, 0.08, 0.05, 0.02, 0.01],
        "lane_cut_p90": [40.0, 150, 90, 100, 20, 60],
        "transition_score": [0.5, 0.4, 0.7, 0.2, 0.3, 0.6],
        "scan_rate": [0.2, 0.3, 0.9, 0.8, 0.4, 0.1],
        "median_latency": [2.5, 3.0, 1.1, 2.0, 4.0, np.nan],
    })


def profiles(game: str) -> pd.DataFrame:
    """Two players a side; Home_2 is subbed off at 60' for Home_3."""
    return pd.DataFrame({
        "player": ["Home_1", "Home_2", "Home_3", "Away_1", "Away_2"],
        "team": ["Home", "Home", "Home", "Away", "Away"],
        "depth": [30.0, 60, 60, 30, 60],
        "role": ["Defender", "Forward", "Forward", "Defender", "Forward"],
        "minute_on": [np.nan, np.nan, 60.0, np.nan, np.nan],
        "minute_off": [np.nan, 60.0, np.nan, np.nan, np.nan],
        "last_minute": 90.0, "game": game,
    }).assign(uid=lambda t: game + ":" + t["player"])


def game_tables(game: str = "Matchday_1_Harbourside_v_Northvale") -> dict[str, pd.DataFrame]:
    tag = lambda t: t.assign(game=game, uid=game + ":" + t["player"])
    return {
        "profiles": profiles(game),
        "decoys": tag(pd.DataFrame({"player": ["Home_2", "Home_2", "Away_2"], "time": [100.0, 3000, 600],
                                    "minute": [1.7, 50.0, 10.0], "duration": 2.0,
                                    "decoy_value": [0.10, 0.05, 0.02]})),
        "pressing": tag(pd.DataFrame({"player": ["Home_1", "Away_1", "Away_1"], "time": [50.0, 70, 2900],
                                      "minute": [0.8, 1.2, 47.0], "lane_cut": [0.5, 1.0, 0.25],
                                      "seconds": 0.04})),
        "transitions": tag(pd.DataFrame({"player": ["Home_1", "Home_2", "Away_1", "Away_2"],
                                         "phase": "attack", "latency": [1.0, 10.0, 2.0, 3.0],
                                         "reached_sprint": [True, False, True, True],
                                         "score": [0.6, 0.0, 0.4, 0.2], "time": [10.0, 10, 2800, 2800],
                                         "minute": [0.2, 0.2, 46.0, 46.0]})),
        "control": tag(pd.DataFrame({"player": ["Home_1", "Home_2", "Home_1", "Away_1"],
                                     "control_pct": [10.0, 20, 15, 25], "minute": [1.0, 1.0, 2.0, 3.0]})),
        "scanning": tag(pd.DataFrame({"player": ["Home_2", "Home_2", "Away_2"], "time": [200.0, 300, 3000],
                                      "minute": [3.3, 5.0, 50.0], "n_scans": [4, 2, 1],
                                      "scan_rate": [0.8, 0.4, 0.2], "ball_fixation": 0.5,
                                      "retained": [True, False, True]})),
    }


def match_roster() -> pd.DataFrame:
    return pd.DataFrame({"player": ["Home_1", "Home_2", "Home_3", "Away_1", "Away_2"],
                         "name": ["H One", "H Two", "H Three", "A One", "A Two"],
                         "club": ["Harbourside FC"] * 3 + ["Northvale Athletic"] * 2,
                         "position": ["Defender", "Forward", "Forward", "Defender", "Forward"],
                         "detailed_position": ["Centre-Back", "Striker", "Striker", "Centre-Back", "Striker"],
                         "market_value_m": [10.0, 20, 5, 12, 25]})


def names(games: list[str]) -> pd.DataFrame:
    return pd.concat([match_roster().assign(uid=lambda t, g=g: g + ":" + t["player"]) for g in games],
                     ignore_index=True).rename(columns={"club": "real_team"}).drop(columns="player")


# ── Percentiles and profiles ─────────────────────────────────────────────────
def test_percentiles_flip_lower_is_better_and_rank_within_position():
    pillars = {"decoy_p90": ("Decoy", True), "median_latency": ("Latency", False), "absent": ("X", True)}
    overall = analysis.percentile_table(pool(), pillars)
    assert overall.loc[0, "decoy_p90"] == 100                         # most xT of all six
    assert overall.loc[2, "median_latency"] == 100                    # fastest = best
    assert overall["absent"].isna().all() and np.isnan(overall.loc[5, "median_latency"])
    within = analysis.percentile_table(pool(), pillars, within="broad_position")
    assert within.loc[[0, 2, 4], "decoy_p90"].tolist() == [100, 100, 100]
    assert within.loc[[1, 3, 5], "decoy_p90"].tolist() == [50, 50, 50]


def test_comparable_players_exclude_self_and_stay_in_position():
    league = pd.concat([pool(), pool().assign(name=lambda t: t["name"] + " twin",
                                              decoy_p90=lambda t: t["decoy_p90"] * 1.01)], ignore_index=True)
    near = analysis.comparable_players("Fwd A", league, n=5)
    assert "Fwd A" not in set(near["name"])
    assert set(near["broad_position"]) == {"Forward"} and len(near) == 3
    assert near.loc[0, "name"] == "Fwd A twin"                        # nearly identical vector
    assert near["distance"].is_monotonic_increasing


def test_player_profile_strengths_weaknesses_and_matches():
    per_match = pd.DataFrame({"game": ["g1", "g2", "g1"], "matchday": [1, 2, 1],
                              "name": ["Mid A", "Mid A", "Mid B"], "opponent": ["X", "Y", "X"],
                              "minutes": [90.0, 70, 90], "scan_rate": [1.0, 0.8, 0.7],
                              "off_ball_value": [70.0, 60, 40]})
    profile = analysis.player_profile("Mid A", pool(), per_match)
    # Within midfielders Mid A leads on decoys, transitions and scanning, trails on pressing.
    assert profile["weaknesses"][0] == "Pressing lane cuts /90"
    assert len(profile["strengths"]) == 2 and "Pressing lane cuts /90" not in profile["strengths"]
    assert list(profile["matches"]["game"]) == ["g1", "g2"]
    assert set(profile["percentiles"]["pillar"]) == set(analysis.PILLARS)
    assert "Mid A" not in set(profile["comparables"]["name"])
    with pytest.raises(KeyError):
        analysis.player_profile("Nobody", pool())


# ── Per-match totals ─────────────────────────────────────────────────────────
def test_per_match_totals_one_row_per_game_and_name():
    games = ["Matchday_1_Harbourside_v_Northvale", "Matchday_2_Harbourside_v_Northvale"]
    tables = [game_tables(g) for g in games]
    data = {key: pd.concat([t[key] for t in tables], ignore_index=True) for key in tables[0]}
    out = analysis.per_match_totals(data, names(games), min_minutes=20)
    assert len(out) == 2 * 5 and not out.duplicated(["game", "name"]).any()
    assert out["matchday"].tolist()[:5] == [1] * 5 and set(out["matchday"]) == {1, 2}
    h2 = out[(out["game"] == games[0]) & (out["name"] == "H Two")].iloc[0]
    assert h2["minutes"] == pytest.approx(60) and h2["decoy_runs"] == 2     # 1.7' and 50', before the 60' sub
    assert h2["opponent"] == "Northvale Athletic"
    assert out["off_ball_value"].between(0, 100).all()


# ── Team comparisons ─────────────────────────────────────────────────────────
def test_team_match_summary_numbers():
    table = analysis.team_match_summary(game_tables(), match_roster())
    assert list(table.columns) == ["Harbourside FC", "Northvale Athletic"]      # Home first
    home, away = table["Harbourside FC"], table["Northvale Athletic"]
    assert home["Decoy runs"] == 2 and away["Decoy runs"] == 1
    assert home["xT opened by decoy runs"] == pytest.approx(0.15)
    assert away["Pressing lane-cut seconds"] == pytest.approx(round(1.25 * 0.04, 2))
    assert home["Transitions sprinted"] == 1 and home["Transitions (total)"] == 2
    assert home["Transition sprint rate %"] == 50 and away["Median transition latency (s)"] == 2.5
    assert home["Scans per reception"] == 3 and home["Ball retention after receiving %"] == 50
    assert home["Pitch control % in possession"] == pytest.approx(22.5)        # (10+20, 15) per frame
    assert home["Possession share % (tracked)"] == pytest.approx(66.67)


def test_team_match_summary_without_scanning_table():
    tables = game_tables()
    del tables["scanning"]
    table = analysis.team_match_summary(tables, match_roster())
    assert "Scans per reception" not in table.index and "Decoy runs" in table.index


def test_intensity_blocks_continue_into_the_second_half():
    blocks = analysis.intensity_blocks(game_tables(), match_roster())
    assert list(blocks["block_start"].unique()) == [0, 15, 30, 45]
    second = blocks[(blocks["block_start"] == 45) & (blocks["side"] == "Away")].iloc[0]
    assert second["block"] == "45-60" and second["club"] == "Northvale Athletic"
    assert second["sprints"] == 2 and second["lane_cut_seconds"] == pytest.approx(0.01)
    assert second["scans"] == 1 and second["decoy_xT"] == 0
    first_home = blocks[(blocks["block_start"] == 0) & (blocks["side"] == "Home")].iloc[0]
    assert first_home["scans"] == 6 and first_home["decoy_xT"] == pytest.approx(0.10)
    tables = game_tables()
    del tables["scanning"]
    assert analysis.intensity_blocks(tables, match_roster())["scans"].isna().all()
    tables["transitions"].loc[0, "minute"] = 93.5                  # second-half stoppage time
    late = analysis.intensity_blocks(tables, match_roster())
    assert late["block"].iloc[-1] == "90+" and late["block_start"].max() == 90


# ── Validation ───────────────────────────────────────────────────────────────
def test_bootstrap_interval_contains_rho_and_verdicts_follow_sign():
    rng = np.random.default_rng(1)
    trait = rng.uniform(0, 1, 60)
    df = pd.DataFrame({"work_rate": trait, "decoy_p90": trait + rng.normal(0, 0.15, 60),
                       "median_latency": 5 - 3 * trait + rng.normal(0, 0.3, 60),
                       "noise": rng.normal(0, 1, 60)})
    table = analysis.validation(df, [("work_rate", "decoy_p90"), ("work_rate", "median_latency", -1),
                                     ("work_rate", "noise"), ("awareness", "scan_rate")])
    assert list(table["metric"]) == ["decoy_p90", "median_latency", "noise"]   # missing pair skipped
    assert ((table["low"] <= table["rho"]) & (table["rho"] <= table["high"])).all()
    assert table.loc[0, "verdict"].startswith("strong") and table.loc[1, "verdict"].startswith("strong")
    assert table.loc[1, "rho"] < 0 and table.loc[2, "verdict"].startswith(("none", "weak"))
    again = analysis.validation(df, [("work_rate", "noise")])
    assert again.loc[0, "low"] == table.loc[2, "low"]                          # fixed seed


def test_attach_truth_joins_hidden_attributes_by_name():
    truth = pd.DataFrame({"name": ["Fwd A", "Mid A"], "club": ["Harbourside FC", "Harbourside FC"],
                          "awareness": [0.9, 0.1], "minutes": [1, 1]})
    joined = analysis.attach_truth(pool(), truth)
    assert joined.loc[joined["name"] == "Fwd A", "awareness"].iloc[0] == 0.9
    assert joined["minutes"].tolist() == pool()["minutes"].tolist()            # pool's own columns kept


# ── Scanning effect ──────────────────────────────────────────────────────────
def test_wilson_interval_known_values():
    low, high = analysis.wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-4) and high == pytest.approx(0.7634, abs=1e-4)
    low, high = analysis.wilson_interval(0, 10)
    assert low == pytest.approx(0, abs=1e-12) and high == pytest.approx(0.2775, abs=1e-4)


def test_scanning_effect_buckets_and_empty_input():
    scans = pd.DataFrame({"n_scans": [0, 0, 1, 2, 3, 5, 4], "retained": [False, True, True, True, True, True, False]})
    effect = analysis.scanning_effect(scans)
    assert list(effect["bucket"]) == ["0", "1", "2", "3+"]
    assert effect["n"].tolist() == [2, 1, 1, 3]
    assert effect.loc[3, "retention"] == pytest.approx(2 / 3)
    assert ((effect["low"] <= effect["retention"]) & (effect["retention"] <= effect["high"])).all()
    for empty in (None, pd.DataFrame(), pd.DataFrame(columns=["n_scans", "retained"])):
        out = analysis.scanning_effect(empty)
        assert out.empty and {"bucket", "n", "retention", "low", "high"} <= set(out.columns)


def test_receptions_without_head_data_are_unknown_not_zero():
    """The pipeline writes n_scans=0, scan_rate NaN when the head feed is missing."""
    tables = game_tables()
    tables["scanning"] = tables["scanning"].assign(n_scans=0, scan_rate=np.nan)
    summary = analysis.team_match_summary(tables, match_roster())
    assert summary.loc["Receptions", "Harbourside FC"] == 2                  # receptions still count
    assert summary.loc["Ball retention after receiving %", "Harbourside FC"] == 50
    assert summary.loc["Scans per reception"].isna().all()
    assert analysis.intensity_blocks(tables, match_roster())["scans"].isna().all()
    assert analysis.scanning_effect(tables["scanning"]).empty
    # Partly observed: the unobserved reception leaves bucket "0" and the mean alone.
    tables = game_tables()
    tables["scanning"].loc[1, ["n_scans", "scan_rate"]] = [0, np.nan]
    summary = analysis.team_match_summary(tables, match_roster())
    assert summary.loc["Scans per reception", "Harbourside FC"] == 4
    blocks = analysis.intensity_blocks(tables, match_roster())
    home = blocks[blocks["side"] == "Home"].set_index("block_start")["scans"]
    assert home[0] == 4 and np.isnan(home[45])
    assert analysis.scanning_effect(tables["scanning"])["n"].tolist() == [0, 1, 0, 1]


def test_verdict_flags_a_strong_link_in_the_wrong_direction():
    rng = np.random.default_rng(2)
    trait = rng.uniform(0, 1, 60)
    df = pd.DataFrame({"a": trait, "m": -(trait + rng.normal(0, 0.1, 60))})
    assert analysis.validation(df, [("a", "m", 1)]).loc[0, "verdict"].startswith("opposite")
    assert analysis.validation(df, [("a", "m", -1)]).loc[0, "verdict"].startswith("strong")


def test_rank_within_compares_players_only_with_their_own_position():
    df = pd.DataFrame({"broad_position": ["Forward", "Forward", "Defender", "Defender"],
                       "decoy_p90": [0.9, 0.5, 0.2, 0.1], "work_rate": [0.8, 0.2, 0.9, 0.3]})
    ranked = analysis.rank_within(df, ["decoy_p90", "work_rate", "missing"])
    assert ranked["decoy_p90"].tolist() == [1.0, 0.5, 1.0, 0.5]       # best of their position = 1
    assert ranked["work_rate"].tolist() == [1.0, 0.5, 1.0, 0.5]
    assert analysis.validation(ranked, [("work_rate", "decoy_p90")]).loc[0, "rho"] == 1.0
