import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from ui import charts
from ui.charts import TEAM_COLOUR

PILLARS = {"decoy_p90": ("Decoy runs (xT /90)", True),
           "median_latency": ("Reaction time (s)", False),
           "scan_rate": ("Scanning (scans/s)", True)}


def texts(fig: go.Figure) -> list[str]:
    return [a.text for a in fig.layout.annotations]


def is_no_data(fig: go.Figure) -> bool:
    return "No data" in texts(fig) and len(fig.data) == 0


@pytest.fixture
def pool():
    return pd.DataFrame({
        "name": ["Ada", "Bo", "Cy", "Di", "Ed", "Fy"],
        "real_team": ["Red", "Red", "Blue", "Blue", "Green", "Green"],
        "broad_position": ["Defender", "Defender", "Midfielder", "Midfielder", "Forward", "Forward"],
        "decoy_p90": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        "median_latency": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "scan_rate": [0.2, 0.4, np.nan, 0.8, 1.0, 1.2],
        "market_value_m": [10.0, 20.0, 5.0, 40.0, 8.0, 0.0],
        "Predicted_Value": [12.0, 10.0, 15.0, 44.0, 32.0, 9.0],
        "awareness": [0.1, 0.3, 0.5, 0.6, 0.8, 0.9],
    }, index=["u1", "u2", "u3", "u4", "u5", "u6"])


@pytest.mark.parametrize("call", [
    lambda: charts.pillar_profile(pd.Series({"name": "X"}), pd.DataFrame(), PILLARS),
    lambda: charts.metric_distribution(pd.DataFrame(), "decoy_p90", "Decoy", []),
    lambda: charts.match_timeline(pd.DataFrame(), pd.DataFrame(), "Red", "Blue"),
    lambda: charts.team_comparison(pd.DataFrame()),
    lambda: charts.intensity_timeline(pd.DataFrame(), "lane_cut", "Lane cuts"),
    lambda: charts.scanning_effect(pd.DataFrame()),
    lambda: charts.validation_scatter(pd.DataFrame(), [("awareness", "scan_rate", "Scanning")]),
    lambda: charts.value_ladder(pd.DataFrame()),
])
def test_empty_input_draws_a_no_data_figure(call):
    assert is_no_data(call())


def test_pillar_profile_percentiles_invert_for_lower_is_better(pool):
    fig = charts.pillar_profile(pool.loc["u6"], pool, PILLARS)
    assert len(fig.data) == 1                                             # u6 has every pillar
    bars = fig.data[0]
    by_label = dict(zip(bars.y, bars.x))
    assert by_label["Decoy runs (xT /90)"] == pytest.approx(100.0)      # highest decoy value
    assert by_label["Reaction time (s)"] == pytest.approx(100 / 6)      # slowest reaction = worst
    assert list(bars.text) == ["0.060", "6.00", "1.20"]
    fast = charts.pillar_profile(pool.loc["u1"], pool, PILLARS).data[0]
    assert dict(zip(fast.y, fast.x))["Reaction time (s)"] == pytest.approx(100.0)
    assert any(s.x0 == 50 for s in fig.layout.shapes)


def test_pillar_profile_shows_missing_pillar_as_grey_bar(pool):
    fig = charts.pillar_profile(pool.loc["u3"], pool, PILLARS)
    assert len(fig.data) == 2
    assert list(fig.data[1].y) == ["Scanning (scans/s)"] and list(fig.data[1].text) == ["no data"]


def test_pillar_profile_ranks_an_outside_player_against_the_pool(pool):
    outsider = pd.Series({"name": "New", "decoy_p90": 1.0, "median_latency": 0.1, "scan_rate": 2.0}, name="u99")
    bars = charts.pillar_profile(outsider, pool, PILLARS).data[0]
    assert all(x == pytest.approx(100.0) for x in bars.x)


def test_metric_distribution_rows_highlights_and_deterministic_jitter(pool):
    fig = charts.metric_distribution(pool, "decoy_p90", "Decoy runs (xT /90)", ["Cy", "u6"])
    names = [t.name for t in fig.data]
    assert {"All players", "Group median", "Highlighted"} <= set(names)
    hot = next(t for t in fig.data if t.name == "Highlighted")
    assert sorted(hot.text) == ["Cy", "Fy"]
    assert list(fig.layout.yaxis.ticktext) == ["Defender", "Midfielder", "Forward"]
    again = charts.metric_distribution(pool, "decoy_p90", "Decoy runs (xT /90)", ["Cy", "u6"])
    assert np.array_equal(fig.data[0].y, again.data[0].y)
    # Jitter keeps every point within its own row.
    assert np.all(np.abs(np.asarray(fig.data[0].y) - np.round(fig.data[0].y)) < 0.5)


def test_match_timeline_lanes_team_colours_and_goals():
    moments = pd.DataFrame({
        "kind": ["scan", "decoy", "press", "decoy"],
        "label": ["Elite scan before receiving", "High-xT decoy run", "Lane-cutting press", "High-xT decoy run"],
        "minute": [10.5, 20.0, 50.25, 70.0], "team": ["Home", "Away", "Home", "Home"],
        "name": ["Ada", "Cy", "Bo", "Ada"], "detail": ["4 head checks", "0.1 xT", "70% lanes", "0.2 xT"]})
    goals = pd.DataFrame({"minute": [33.0, 36.0], "team": ["Away", "Home"], "name": ["Cy", "Ada"]})
    fig = charts.match_timeline(moments, goals, "Red", "Blue")
    home, away = fig.data
    assert (home.name, away.name) == ("Red", "Blue")
    assert home.marker.color == TEAM_COLOUR["Home"] and away.marker.color == TEAM_COLOUR["Away"]
    assert home.marker.symbol != away.marker.symbol
    assert list(fig.layout.yaxis.categoryarray) == [
        "High-xT decoy run", "Lane-cutting press", "Elite scan before receiving"]
    assert home.customdata[1][0] == "50:15"
    goal_tags = [a for a in fig.layout.annotations if a.text.startswith("GOAL")]
    assert [a.bordercolor for a in goal_tags] == [TEAM_COLOUR["Away"], TEAM_COLOUR["Home"]]
    line_colours = [s.line.color for s in fig.layout.shapes]
    assert TEAM_COLOUR["Away"] in line_colours and any(s.x0 == 45 for s in fig.layout.shapes)


def test_match_timeline_with_goals_only_still_draws():
    goals = pd.DataFrame({"time": [600.0], "team": ["Home"], "name": ["Ada"]})
    fig = charts.match_timeline(pd.DataFrame(), goals, "Red", "Blue")
    assert any(t.startswith("GOAL Ada 10:00") for t in texts(fig))


def test_team_comparison_shares_sum_to_100():
    teams = pd.DataFrame({"Red": [3.0, 10.0, 0.0], "Blue": [1.0, 30.0, 0.0]},
                         index=["Decoy xT", "Lane cuts (s)", "Scans"])
    fig = charts.team_comparison(teams)
    home, away = fig.data
    assert fig.layout.title.text == "Who won the off-ball battle"
    assert home.marker.color == TEAM_COLOUR["Home"] and away.marker.color == TEAM_COLOUR["Away"]
    assert list(home.x[:2]) == pytest.approx([-75.0, -25.0]) and list(away.x[:2]) == pytest.approx([25.0, 75.0])
    assert np.isnan(home.x[2])                       # 0 vs 0: no share to show
    assert list(home.text[:2]) == ["3.00", "10.0"]


def test_intensity_timeline_orders_home_first_and_labels_blocks():
    per_block = pd.DataFrame({"block_start": [0, 15, 30, 0, 15, 30], "club": ["Blue"] * 3 + ["Red"] * 3,
                              "side": ["Away"] * 3 + ["Home"] * 3, "value": [1, 2, 3, 4, 5, 6.0]})
    fig = charts.intensity_timeline(per_block, "lane_cut_p90", "Lane cuts (s)")
    assert [t.name for t in fig.data] == ["Red", "Blue"]
    assert fig.data[0].line.color == TEAM_COLOUR["Home"] and fig.data[1].line.color == TEAM_COLOUR["Away"]
    assert list(fig.layout.xaxis.ticktext) == ["0-15'", "15-30'", "30-45'"]
    assert {"Red", "Blue"} <= set(texts(fig))       # direct labels at the line ends


def test_intensity_timeline_filters_by_metric_column():
    per_block = pd.DataFrame({"block_start": [0, 15], "club": ["Red", "Red"], "value": [1.0, 2.0],
                              "metric": ["other", "other"]})
    assert is_no_data(charts.intensity_timeline(per_block, "lane_cut_p90", "Lane cuts"))


def test_scanning_effect_bars_with_confidence_intervals():
    effect = pd.DataFrame({"bucket": ["0", "1", "2", "3+"], "retention": [0.6, 0.7, 0.8, 0.9],
                           "low": [0.5, 0.6, 0.7, 0.8], "high": [0.7, 0.8, 0.85, 0.95], "n": [100, 80, 60, 40]})
    fig = charts.scanning_effect(effect)
    bars = fig.data[0]
    assert list(bars.x) == ["0", "1", "2", "3+"] and list(bars.y) == pytest.approx([60, 70, 80, 90])
    assert list(bars.error_y.array) == pytest.approx([10, 10, 5, 5])
    assert list(bars.text) == ["n=100", "n=80", "n=60", "n=40"]
    assert fig.layout.title.text == "Players who scan keep the ball"
    overall = (60 * 100 + 70 * 80 + 80 * 60 + 90 * 40) / 280
    assert any(t == f"All receptions {overall:.0f}%" for t in texts(fig))


def test_validation_scatter_panels_with_spearman_titles(pool):
    pairs = [("awareness", "scan_rate", "Scanning"), ("awareness", "decoy_p90", "Decoys")]
    fig = charts.validation_scatter(pool, pairs)
    assert len(fig.data) == 4                          # points + trend line per panel
    titles = texts(fig)
    assert any("Scanning" in t and "ρ = 1.00" in t and "n = 5" in t for t in titles)
    assert any("Decoys" in t and "n = 6" in t for t in titles)


def test_value_ladder_sorts_by_price_ratio(pool):
    fig = charts.value_ladder(pool, top=3)
    _, market, ml = fig.data
    assert list(market.y) == ["Ed", "Cy", "Ada"]       # 4.0x, 3.0x, 1.2x; Fy's €0m is skipped
    assert list(ml.x) == [32.0, 15.0, 12.0] and list(ml.text) == ["×4.0", "×3.0", "×1.2"]
    assert market.marker.symbol != ml.marker.symbol


def goal_tags(fig: go.Figure) -> list:
    return [a for a in fig.layout.annotations if a.text.startswith("GOAL")]


def test_match_timeline_stacks_long_goal_tags_that_would_overlap():
    goals = pd.DataFrame({"minute": [20.0, 30.0, 61.0, 89.0],
                          "team": ["Home", "Away", "Home", "Away"],
                          "name": ["Soren Jovanović", "Kwame Delacroix-Okafor", "Bo", "Umar Wexford"]})
    tags = goal_tags(charts.match_timeline(pd.DataFrame(), goals, "A", "B"))
    shifts = [a.yshift for a in tags]
    assert shifts[0] != shifts[1]                      # 10' apart with long names: stacked
    assert shifts[2] == shifts[0]                      # far enough from both: back on the base row
    assert tags[3].xanchor == "right"                  # late goal tag kept inside the axis


def test_match_timeline_goal_without_scorer_has_no_nan():
    goals = pd.DataFrame({"minute": [5.0], "team": ["Away"], "name": [np.nan]})
    assert [a.text for a in goal_tags(charts.match_timeline(pd.DataFrame(), goals, "A", "B"))] == ["GOAL 05:00"]


def test_percentile_tolerates_duplicate_index_and_empty_pool():
    dup = pd.DataFrame({"decoy_p90": [1.0, 2.0, 3.0]}, index=["u1", "u1", "u2"])
    fig = charts.pillar_profile(dup.iloc[0], dup, {"decoy_p90": ("D", True)})
    assert np.isfinite(fig.data[0].x[0])
    assert np.isnan(charts.percentile(0.1, pd.Series([np.nan] * 5), True))


def test_pillar_profile_pool_without_data_shows_no_data_bar():
    pool = pd.DataFrame({"scan_rate": [np.nan] * 5})
    fig = charts.pillar_profile(pd.Series({"scan_rate": 0.1}, name="zz"), pool, {"scan_rate": ("S", True)})
    assert len(fig.data[0].x) == 0                     # not a lone 100th percentile
    assert list(fig.data[1].text) == ["no data"]


def test_intensity_timeline_ignores_rows_without_club():
    per_block = pd.DataFrame({"block_start": [0, 15, 0, 15], "club": ["A", "A", np.nan, np.nan],
                              "side": ["Home", "Home", "Away", "Away"], "value": [1.0, 2.0, 3.0, 4.0]})
    fig = charts.intensity_timeline(per_block, "decoy_p90", "Decoys")
    assert [t.name for t in fig.data] == ["A"]
