import numpy as np
import pandas as pd
import pytest

from core import pipeline, roster, tracking
from core.pipeline import off_ball_value, player_totals, reception_scans
from core.synthetic import make_squad, value_players, write_match


def test_within_position_ranks_each_group_separately():
    board = pd.DataFrame({
        "broad_position": ["Defender", "Defender", "Forward", "Forward"], "minutes": 90.0,
        "decoy_value": [0.1, 0.2, 1.0, 2.0], "lane_cut_seconds": [10.0, 20, 100, 200],
        "transition_score": [0.1, 0.2, 0.5, 0.6]})
    pooled = off_ball_value(board).set_index("decoy_value")["off_ball_value"]
    grouped = off_ball_value(board, within="broad_position").set_index("decoy_value")["off_ball_value"]
    assert pooled[0.2] < pooled[1.0]                 # one pool: the better defender ranks below both forwards
    assert grouped[0.2] == grouped[2.0] == 100       # within position: each group has its own top player


def test_scan_rate_is_the_fourth_pillar():
    board = pd.DataFrame({"minutes": 90.0, "decoy_value": [1.0, 1.0], "lane_cut_seconds": [5.0, 5.0],
                          "transition_score": [0.5, 0.5], "scan_rate": [0.2, 0.6]})
    value = off_ball_value(board).set_index("scan_rate")["off_ball_value"]
    assert value[0.6] > value[0.2]                  # level on three pillars, the better scanner leads


def tiny_tracking(seconds: float = 12.0) -> tracking.TrackingData:
    t = np.arange(1, int(seconds * 25) + 1) / 25
    positions = np.tile([[30.0, 34.0], [50.0, 34.0], [60.0, 20.0]], (len(t), 1, 1))
    return tracking.TrackingData(frames=np.arange(1, len(t) + 1), timestamps=t, periods=np.ones(len(t), int),
                                 positions=positions, ball=np.tile([40.0, 34.0], (len(t), 1)),
                                 player_ids=["Home_1", "Home_2", "Away_9"],
                                 teams=np.array(["Home", "Home", "Away"]))


def test_receptions_record_scans_and_whether_the_ball_was_kept():
    tr = tiny_tracking()
    events = pd.DataFrame([
        ("PASS", 25, 6.0, "Home_1", "Home_2"),       # Home_2 receives, then passes on: kept
        ("PASS", 175, 8.0, "Home_2", "Home_1"),      # Home_1 receives, then loses it: not kept
        ("BALL LOST", 225, 9.0, "Home_1", None),
        ("PASS", 250, 10.5, "Away_9", None),         # incomplete: no reception
    ], columns=["Type", "Start Frame", "End Time [s]", "From", "To"])
    yaw = np.zeros((len(tr.frames), 3))
    t = tr.timestamps[:, None]
    # The ball is behind Home_2 (bearing 180 deg); one quick check away from it at 3 s.
    yaw[:, 1] = 180.0 + 120.0 * ((t[:, 0] > 3.0) & (t[:, 0] < 3.6))
    yaw[:, 0] = 0.0                                                    # Home_1 just watches the ball
    table = reception_scans(tr, events, yaw)
    assert table["player"].tolist() == ["Home_2", "Home_1"]
    assert table["retained"].tolist() == [True, False]
    assert table["n_scans"].tolist() == [1, 0]
    assert table["n_scans"].dtype.kind == "i" and table["retained"].dtype == bool
    np.testing.assert_allclose(table["scan_rate"], [1 / 5, 0.0], atol=0.01)

    blind = reception_scans(tr, events, np.full_like(yaw, np.nan))   # no head feed: rate unknown
    assert blind["scan_rate"].isna().all() and (blind["n_scans"] == 0).all()


@pytest.fixture(scope="module")
def tiny_game(tmp_path_factory):
    rng = np.random.default_rng(5)
    used: set = set()
    home, away = make_squad("Test A", "4-3-3", rng, used), make_squad("Test B", "4-4-2", rng, used)
    value_players(home + away, rng)
    folder = tmp_path_factory.mktemp("league") / "Test_Match"
    write_match(folder, home, away, seed=6, half_minutes=2)
    return folder


def test_score_game_builds_the_scanning_table(tiny_game):
    tables = pipeline.score_game(tiny_game, pipeline.learn_xt([tiny_game]))
    scans = tables["scanning"]
    assert list(scans.columns) == pipeline.SCANNING_COLUMNS + ["game"]
    _, events = tracking.load_sample_game(tiny_game)
    assert len(scans) == ((events["Type"] == "PASS") & events["To"].notna()).sum()
    assert scans["scan_rate"].notna().mean() > 0.8                    # the head feed covers most windows
    assert (scans["n_scans"] >= 0).all() and scans["ball_fixation"].between(0, 1).all()
    first_half = scans["minute"] < 45
    assert np.allclose(scans.loc[first_half, "minute"], scans.loc[first_half, "time"] / 60, atol=0.01)
    assert (scans.loc[~first_half, "minute"] >= 45).all()

    names = pipeline.load_roster(tiny_game).rename(columns={"club": "real_team"})
    timeline = roster.static_timeline(tables["profiles"], names[["player", "name", "real_team", "position",
                                                                 "detailed_position", "market_value_m"]])
    totals = player_totals(tables, timeline)
    assert {"receptions", "scan_rate", "pre_reception_scans", "retention_rate"} <= set(totals.columns)
    assert totals["receptions"].sum() == len(scans)
    assert totals["retention_rate"].dropna().between(0, 1).all()

    without = player_totals({k: v for k, v in tables.items() if k != "scanning"}, timeline)
    assert without["scan_rate"].isna().all()                          # an old cache still has the columns


def test_saved_scores_load_only_for_the_same_signature(tmp_path):
    path = tmp_path / "scored.pkl"
    tables = {"decoys": pd.DataFrame({"player": ["Home_1"], "decoy_value": [0.1]})}
    assert pipeline.load_scored((("pipeline", 1),), path) is None                 # nothing saved yet
    pipeline.save_scored((("pipeline", 1), ("g", 5)), tables, path)
    pd.testing.assert_frame_equal(pipeline.load_scored((("pipeline", 1), ("g", 5)), path)["decoys"],
                                  tables["decoys"])
    assert pipeline.load_scored((("pipeline", 2), ("g", 5)), path) is None        # VERSION bumped
    assert pipeline.load_scored((("pipeline", 1), ("g", 6)), path) is None        # a game regenerated
