import numpy as np

from core.pillars.space import counterfactual_defenders, decoy_value, find_markers, off_ball_runs
from core.spatial import make_grid
from core.tracking import TrackingData

F = 25
GRID, _ = make_grid(105, 68)
XT = np.where(GRID[:, 0] > 70, 0.1, 0.01)   # threat sits in the final third


def scenario(marker_follows: bool):
    """Runner drops from the box out to the flank; a teammate waits in the final third."""
    s = np.linspace(0, 1, F)[:, None]
    runner = np.array([80, 34]) + s * [-20, 26]
    teammate = np.tile([85, 30], (F, 1))
    marker = runner + [1.5, 0] if marker_follows else np.tile([81.5, 34], (F, 1))
    other = np.tile([100, 34], (F, 1))
    attackers = np.stack([runner, teammate], axis=1)
    defenders = np.stack([marker, other], axis=1)
    vel = lambda x: np.gradient(x, 1 / 25, axis=0)
    return attackers, defenders, vel(attackers), vel(defenders)


def test_following_marker_is_found_and_creates_value():
    att, dfn, av, dv = scenario(marker_follows=True)
    assert find_markers(att, dfn, av, dv, runner=0).tolist() == [True, False]
    result = decoy_value(att, dfn, av, dv, 0, GRID, XT, cell_weight=1.0)
    assert result.value > 0


def test_static_defender_is_not_a_marker():
    att, dfn, av, dv = scenario(marker_follows=False)
    result = decoy_value(att, dfn, av, dv, 0, GRID, XT, cell_weight=1.0)
    assert not result.markers.any() and result.value == 0


def test_run_that_pulls_marker_into_danger_is_clamped_at_zero():
    """A run in behind: the marker drops into the box, shrinking a teammate's high-xT space."""
    s = np.linspace(0, 1, F)[:, None]
    runner = np.array([60, 34]) + s * [30, 0]
    teammate = np.tile([88, 40], (F, 1))
    marker = runner + [1.5, 0]
    holding = np.tile([55, 34], (F, 1))                      # covers the space the marker left
    keeper = np.tile([103, 34], (F, 1))
    att = np.stack([runner, teammate], axis=1)
    dfn = np.stack([marker, holding, keeper], axis=1)
    box_xt = np.where(GRID[:, 0] > 88.5, 0.3, 0.01)            # the threat is in the penalty box
    vel = lambda x: np.gradient(x, 1 / 25, axis=0)
    result = decoy_value(att, dfn, vel(att), vel(dfn), 0, GRID, box_xt, cell_weight=1.0)
    assert result.markers[0] and result.net_value < 0
    assert result.value == 0


def test_counterfactual_keeps_block_shift():
    start = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    end = np.array([[5.0, 20.0], [12.0, 1.0], [22.0, 1.0]])   # marker ran, block slid 2 right, 1 up
    ghost = counterfactual_defenders(start, end, np.array([True, False, False]))
    np.testing.assert_allclose(ghost[0], [2.0, 1.0])
    np.testing.assert_allclose(ghost[1:], end[1:])


def test_off_ball_runs_drop_abandoned_bursts_and_transition_reactions():
    n, fps = 150, 25
    t = np.arange(n)
    holder = np.tile([50.0, 34.0], (n, 1))
    runner = np.tile([60.0, 20.0], (n, 1))                      # Home_2: completes a 10 m run (2 s at 5 m/s)
    runner[25:75, 0] = 60.0 + 5.0 * (t[25:75] - 25) / fps
    runner[75:, 0] = runner[74, 0]
    burst = np.tile([40.0, 50.0], (n, 1))                        # Home_3: gives up after 6 m (1.2 s)
    burst[25:55, 0] = 40.0 + 5.0 * (t[25:55] - 25) / fps
    burst[55:, 0] = burst[54, 0]
    away = np.tile([80.0, 34.0], (n, 1))
    positions = np.stack([holder, runner, burst, away], axis=1)
    tr = TrackingData(frames=t + 1, timestamps=(t + 1) / fps, periods=np.ones(n, dtype=int),
                      positions=positions, ball=holder + [0.3, 0], player_ids=["Home_1", "Home_2", "Home_3", "Away_4"],
                      teams=np.array(["Home", "Home", "Home", "Away"]))
    speeds = np.zeros((n, 4))
    speeds[25:75, 1] = speeds[25:55, 2] = 18.0                   # km/h, above run speed
    settled = np.full(n, "Home", dtype=object)
    assert off_ball_runs(tr, speeds, settled)["player"].tolist() == [1]          # the burst is dropped
    just_won = settled.copy()
    just_won[:20] = "Away"                                       # Home win the ball 0.2 s before the run
    assert off_ball_runs(tr, speeds, just_won).empty                              # a transition, not a decoy
    behind = positions.copy()                                    # the same run, but 30 m behind the ball
    behind[:, 1, 0] -= 30.0
    tr_behind = TrackingData(frames=tr.frames, timestamps=tr.timestamps, periods=tr.periods, positions=behind,
                             ball=tr.ball, player_ids=tr.player_ids, teams=tr.teams)
    assert tr_behind.attacks_right("Home")[0]
    assert off_ball_runs(tr_behind, speeds, settled).empty                         # the line moving, not a decoy
