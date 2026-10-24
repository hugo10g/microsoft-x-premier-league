import numpy as np

from core.pillars.pressing import lane_cut, lanes_blocked, project

CARRIER = np.array([50.0, 34.0])
RECEIVERS = np.array([[70.0, 34.0], [50.0, 54.0]])    # straight ahead, square to the left


def test_shadow_blocks_only_the_lane_behind_the_presser():
    assert lanes_blocked(CARRIER, np.array([55.0, 34.5]), RECEIVERS, reach=1.5).tolist() == [True, False]


def test_presser_beyond_the_receiver_blocks_nothing():
    assert not lanes_blocked(CARRIER, np.array([75.0, 34.0]), RECEIVERS, reach=1.5).any()


def test_projection_reveals_a_curved_run():
    presser, velocity = np.array([55.0, 30.0]), np.array([0.0, 8.0])   # 4 m short of the lane, closing fast
    np.testing.assert_allclose(project(presser, velocity, 0.5), [55.0, 34.0])
    now, projected = lane_cut(CARRIER, presser, velocity, RECEIVERS, np.array([3.0, 1.0]))
    assert now == 0 and projected == 0.75


def test_vectorised_over_frames():
    carriers = np.stack([CARRIER, CARRIER])
    pressers = np.array([[55.0, 34.5], [50.0, 40.0]])
    receivers = np.stack([RECEIVERS, RECEIVERS])
    assert lanes_blocked(carriers, pressers, receivers, 1.5).tolist() == [[True, False], [False, True]]
