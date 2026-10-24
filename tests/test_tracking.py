import numpy as np
import pytest

from core.tracking import (PITCH_LENGTH, PITCH_WIDTH, load_events, load_head_orientation, load_tracking,
                           possession_by_frame, receptions)


def write_team(path, team, jerseys, rows):
    """A tracking file in Metrica's three-header-row layout."""
    lines = [",,," + "".join(f"{team},," for _ in jerseys) + ",",
             ",,," + "".join(f"{j},," for j in jerseys) + ",",
             "Period,Frame,Time [s]," + "".join(f"Player{j},," for j in jerseys) + "Ball,"]
    lines += [",".join(map(str, row)) for row in rows]
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def tracking(tmp_path):
    # Frame 1: home on the left, away on the right. Frame 3: second half, sides swapped.
    write_team(tmp_path / "home.csv", "Home", [1, 2], [
        [1, 1, 0.04, 0.25, 0.5, 0.30, 0.25, 0.5, 0.5],
        [1, 2, 0.08, 0.26, 0.5, "NaN", "NaN", 0.5, 0.5],
        [2, 3, 0.12, 0.75, 0.5, 0.70, 0.25, 0.5, 0.5],
    ])
    write_team(tmp_path / "away.csv", "Away", [7], [
        [1, 1, 0.04, 0.75, 0.5, 0.5, 0.5],
        [1, 2, 0.08, 0.75, 0.5, 0.5, 0.5],
        [2, 3, 0.12, 0.25, 0.5, 0.5, 0.5],
    ])
    return load_tracking(tmp_path / "home.csv", tmp_path / "away.csv")


def test_parses_players_and_metres(tracking):
    assert tracking.player_ids == ["Home_1", "Home_2", "Away_7"]
    assert tracking.positions.shape == (3, 3, 2)
    # Metrica y = 0.25 (a quarter from the top) is 51 m up a 68 m pitch.
    np.testing.assert_allclose(tracking.positions[0, 1], [0.30 * PITCH_LENGTH, 0.75 * PITCH_WIDTH])
    assert np.isnan(tracking.positions[1, 1]).all()
    np.testing.assert_allclose(tracking.ball[0], [PITCH_LENGTH / 2, PITCH_WIDTH / 2])


def test_attacking_direction_flips_at_half_time(tracking):
    assert tracking.attacks_right("Home").tolist() == [True, True, False]
    assert tracking.attacks_right("Away").tolist() == [False, False, True]


def test_possession_and_receptions(tmp_path):
    (tmp_path / "events.csv").write_text(
        "Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],From,To,Start X,Start Y,End X,End Y\n"
        "Home,PASS,,1,2,0.08,4,0.16,Player1,Player2,0.5,0.5,0.6,0.5\n"
        "Away,RECOVERY,,1,6,0.24,6,0.24,Player7,,0.6,0.5,NaN,NaN\n"
        "Away,BALL OUT,,1,8,0.32,8,0.32,Player7,,0.9,0.5,NaN,NaN\n")
    events = load_events(tmp_path / "events.csv")
    assert possession_by_frame(events, np.arange(1, 10)).tolist() == \
        [None, "Home", "Home", "Home", "Home", "Away", "Away", None, None]
    assert receptions(events).to_dict("records") == [{"player": "Home_2", "frame": 4, "time": 0.16}]


def test_player_names_with_a_space_get_the_same_id(tmp_path):
    write_team(tmp_path / "home.csv", "Home", [1], [[1, 1, 0.04, 0.25, 0.5, 0.5, 0.5]])
    text = (tmp_path / "home.csv").read_text().replace("Player1,", "Player 1,")
    (tmp_path / "home.csv").write_text(text)
    write_team(tmp_path / "away.csv", "Away", [7], [[1, 1, 0.04, 0.75, 0.5, 0.5, 0.5]])
    assert load_tracking(tmp_path / "home.csv", tmp_path / "away.csv").player_ids == ["Home_1", "Away_7"]


def test_head_orientation_aligns_by_frame_and_player(tmp_path, tracking):
    game = tmp_path / "Game"
    game.mkdir()
    assert np.isnan(load_head_orientation(game, tracking)).all()        # no file: unknown, not zero
    # Columns in another order, one player missing, frames out of order and one not tracked.
    (game / "Game_HeadOrientation.csv").write_text(
        "Period,Frame,Time [s],Away_7,Home_1\n"
        "2,3,0.12,90.0,-45.5\n"
        "1,1,0.04,180.0,10.0\n"
        "1,9,0.36,0.0,0.0\n")
    yaw = load_head_orientation(game, tracking)
    assert yaw.shape == (3, 3)                                          # (T, P) in player_ids order
    np.testing.assert_allclose(yaw[0], [10.0, np.nan, 180.0])
    assert np.isnan(yaw[1]).all()                                       # frame 2 isn't in the file
    np.testing.assert_allclose(yaw[2], [-45.5, np.nan, 90.0])
