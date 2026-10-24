"""
Direction helpers and an estimated body orientation for players.

Pitch coordinates are taken as x along the length, y across, y pointing "up"
(counter-clockwise from x). Directions are unit vectors, NaN where unknown.
"""

import numpy as np
import pandas as pd

from core.config import POSE, PoseConfig


def unit(vectors: np.ndarray) -> np.ndarray:
    """Normalise vectors along the last axis; zero-length vectors become NaN."""
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norm > 1e-9, vectors / norm, np.nan)


def angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unsigned angle in degrees between two arrays of vectors."""
    cos = np.sum(unit(a) * unit(b), axis=-1)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def body_orientation_from_velocity(velocity: np.ndarray, player_pos: np.ndarray | None = None,
                                   ball_pos: np.ndarray | None = None,
                                   config: PoseConfig = POSE) -> np.ndarray:
    """Estimated body orientation: the direction of travel while moving.

    Shapes: velocity and player_pos (T, 2) or (T, P, 2) in m/s and metres; ball_pos (T, 2).

    Below `min_moving_speed_kmh` the direction of travel is noise, so a near-stationary
    player is assumed to face the ball if its position is given, and otherwise keeps
    the last heading they had while moving. Known failure: a player backpedalling or
    side-stepping (defenders do both constantly) is recorded facing the wrong way.
    """
    heading = unit(velocity)
    slow = np.linalg.norm(velocity, axis=-1) * 3.6 < config.min_moving_speed_kmh
    if ball_pos is not None and player_pos is not None:
        ball = ball_pos if velocity.ndim == 2 else ball_pos[:, None]
        heading = np.where(slow[..., None], unit(ball - player_pos), heading)
    else:
        heading[slow] = np.nan
        flat = pd.DataFrame(heading.reshape(len(heading), -1)).ffill().to_numpy(copy=True)
        heading = flat.reshape(heading.shape)
    heading[np.isnan(velocity).any(axis=-1)] = np.nan       # don't invent frames with no data
    return heading
