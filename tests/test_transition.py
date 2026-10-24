import numpy as np
import pytest

from core.config import TRANSITION
from core.kinematics import speed_kmh
from core.pillars.transition import (
    detect_turnovers, latency_from_speeds, transition_latencies, transition_latency,
)

FPS = 25
T = np.arange(0, 15, 1 / FPS)
T0 = 5.0


def run(speed_profile: np.ndarray) -> np.ndarray:
    """Positions of a player running along x at the given speeds."""
    x = np.concatenate([[0], np.cumsum(speed_profile[1:] / FPS)])
    return np.column_stack([x, np.zeros_like(x)])


def test_latency_of_an_accelerating_player():
    speeds = np.interp(T, [0, T0, T0 + 2, 15], [1, 1, 7, 7])   # 1 m/s, then ramp to 7 m/s
    result = transition_latency(T, run(speeds), T0)
    expected = 2 * (TRANSITION.sprint_kmh / 3.6 - 1) / 6     # when the ramp crosses 20 km/h
    assert result.reached_sprint
    assert result.latency == pytest.approx(expected, abs=0.08)


def test_walker_is_censored_and_penalised():
    result = transition_latency(T, run(np.full(len(T), 1.2)), T0)
    assert not result.reached_sprint
    assert result.latency == 10.0
    assert result.walk_fraction == 1.0
    assert result.score == pytest.approx(np.exp(-5) - 0.5)


def test_already_sprinting_has_zero_latency():
    assert transition_latency(T, run(np.full(len(T), 7.0)), T0).latency == pytest.approx(0)


def test_short_spike_is_not_a_sprint():
    speeds = np.full(len(T), 7.0)                              # km/h
    speeds[(T > 6) & (T < 6.2)] = 30.0                         # 0.2 s spike
    speeds[(T >= 9) & (T < 10)] = 22.0                         # 1 s genuine sprint
    assert latency_from_speeds(T, speeds, T0).latency == pytest.approx(4.0, abs=0.05)


def test_detect_turnovers_ignores_restarts_by_default():
    possession = ["A", "A", "B", None, None, "A", "A"]
    t = np.arange(7.0)
    assert detect_turnovers(t, possession) == [(2.0, "B")]
    assert detect_turnovers(t, possession, across_dead_ball=True) == [(2.0, "B"), (5.0, "A")]


def test_batch_labels_phases():
    positions = np.stack([run(np.full(len(T), 7.0)), run(np.full(len(T), 1.0))], axis=1)
    table = transition_latencies(T, speed_kmh(positions, T), ["p1", "p2"], ["A", "B"], [(T0, "A")])
    assert table.set_index("player")["phase"].to_dict() == {"p1": "attacking", "p2": "defensive"}


def test_batch_skips_players_far_from_the_ball():
    positions = np.stack([run(np.full(len(T), 7.0))] * 3, axis=1)
    distances = np.array([[10.0, 45.0, np.nan]])
    table = transition_latencies(T, speed_kmh(positions, T), ["near", "far", "unknown"], ["A"] * 3,
                                 [(T0, "A")], ball_distances=distances)
    assert table["player"].tolist() == ["near"]
