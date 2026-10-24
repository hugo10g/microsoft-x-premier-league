import re
import sys
from pathlib import Path

import numpy as np
import pytest

from core.tracking import TrackingData
from ui.payload import build_window

T = 4
BALL = np.array([50.0, 34.0])
HTML = Path(__file__).resolve().parents[1] / "ui" / "pitch_player.html"
ESPRIMA = Path("/private/tmp/claude-501/-Users-hugoteng-Desktop-Microsoft-x-Premier-League/"
               "090d1f4e-432c-4519-92c2-41db49e10e38/scratchpad/jspkg")


def tracking() -> TrackingData:
    # Home_1 holds the ball 2 m away, Home_2 stands 10 m behind it, Away_1 10 m ahead.
    spots = np.array([[52.0, 34.0], [40.0, 34.0], [60.0, 34.0]])
    return TrackingData(frames=np.arange(T) + 1, timestamps=np.arange(T) * 0.04,
                        periods=np.ones(T, dtype=int), positions=np.repeat(spots[None], T, axis=0),
                        ball=np.repeat(BALL[None], T, axis=0), player_ids=["Home_1", "Home_2", "Away_1"],
                        teams=np.array(["Home", "Home", "Away"]))


def window(head_yaw=None, possession=("Home",) * T) -> dict:
    tr = tracking()
    return build_window(tr, np.zeros_like(tr.positions), np.array(possession, dtype=object),
                        np.zeros((12, 16)), np.arange(T) / 60, 0, T, ["A", "B", "C"],
                        {"Home": "Home FC", "Away": "Away FC"}, {"Home": "#e63946", "Away": "#3a86ff"},
                        head_yaw=head_yaw)


def yaw() -> np.ndarray:
    # Holder looks sideways throughout; Home_2 watches the ball, then turns away on frames 2-3;
    # Away_1 watches the ball (bearing 180) and leaves head data on frame 3.
    return np.array([[90.0, 0.4, 179.6],
                     [90.0, -0.6, 179.6],
                     [90.0, 180.0, 179.6],
                     [90.0, 120.2, np.nan]])


def test_head_is_flat_whole_degrees_with_null_for_missing():
    head = window(yaw())["head"]
    assert len(head) == T * 3
    assert head[:3] == [90, 0, 180]
    assert head[3:6] == [90, -1, 180]
    assert head[-3:] == [90, 120, None]
    assert all(v is None or isinstance(v, int) for v in head)


def test_scanning_lists_players_looking_away_but_not_the_holder():
    assert window(yaw())["scanning"] == [[], [], [1], [1]]


def test_without_possession_there_is_no_holder_to_exclude():
    scanning = window(yaw(), possession=(None,) * T)["scanning"]
    assert scanning == [[0], [0], [0, 1], [0, 1]]


def test_players_at_the_ball_are_not_scanning():
    tr = tracking()
    tr.positions[:, 2] = BALL + [0.5, 0.0]               # Away_1 on the ball, looking any way
    payload = build_window(tr, np.zeros_like(tr.positions), np.array([None] * T, dtype=object),
                           np.zeros((12, 16)), np.arange(T) / 60, 0, T, ["A", "B", "C"], {},
                           {"Home": "#e63946", "Away": "#3a86ff"}, head_yaw=np.full((T, 3), 0.0))
    assert all(2 not in frame for frame in payload["scanning"])


def test_window_slices_the_full_match_array():
    tr = tracking()
    full = np.vstack([np.full((2, 3), 7.0), yaw()])         # two earlier frames outside the window
    tr.frames, tr.timestamps = np.arange(T + 2), np.arange(T + 2) * 0.04
    tr.periods = np.ones(T + 2, dtype=int)
    tr.positions = np.concatenate([tr.positions[:2], tr.positions])
    tr.ball = np.concatenate([tr.ball[:2], tr.ball])
    payload = build_window(tr, np.zeros_like(tr.positions), np.array(["Home"] * (T + 2), dtype=object),
                           np.zeros((12, 16)), np.arange(T + 2) / 60, 2, T + 2, ["A", "B", "C"], {},
                           {"Home": "#e63946", "Away": "#3a86ff"}, head_yaw=full)
    assert payload["head"][:3] == [90, 0, 180]
    assert payload["scanning"] == [[], [], [1], [1]]


@pytest.mark.parametrize("head_yaw", [None, np.full((T, 3), np.nan)])
def test_no_head_data_leaves_the_payload_without_head(head_yaw):
    data = window(head_yaw)
    assert "head" not in data and "scanning" not in data
    assert len(data["positions"]) == T * 3 * 2


def test_player_script_parses():
    if str(ESPRIMA) not in sys.path:
        sys.path.insert(0, str(ESPRIMA))
    esprima = pytest.importorskip("esprima")
    script = re.search(r"<script>(.*)</script>", HTML.read_text(), re.S).group(1)
    esprima.parseScript(script.replace("__DATA__", "{}"))
