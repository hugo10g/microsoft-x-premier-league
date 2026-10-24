import numpy as np

from core.pose import body_orientation_from_velocity


def test_orientation_follows_travel_then_faces_ball():
    velocity = np.array([[3.0, 0.0], [0.0, 3.0], [0.1, 0.0]])     # last frame: standing
    player = np.zeros((3, 2))
    ball = np.tile([0.0, -10.0], (3, 1))
    heading = body_orientation_from_velocity(velocity, player, ball)
    np.testing.assert_allclose(heading, [[1, 0], [0, 1], [0, -1]])


def test_orientation_holds_last_heading_without_ball():
    velocity = np.array([[0.0, 3.0], [0.1, 0.0], [0.0, 0.1], [np.nan, np.nan]])
    heading = body_orientation_from_velocity(velocity)
    np.testing.assert_allclose(heading[:3], [[0, 1]] * 3)
    assert np.isnan(heading[3]).all()
