"""
Pitch geometry and pitch control.

Pitch control answers "which team owns each zone?", from where every player is
standing. xT is learned on a 120 x 80 frame (each action drawn as if its team
attacks left to right) and looked up on the metric tracking pitch with `xt_lookup`.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit

from core.config import PITCH_CONTROL, PitchControlConfig

XT_LENGTH, XT_WIDTH = 120, 80  # the frame xT is learned in (core.pipeline.xt_training_events)


def make_grid(length: float, width: float, cell: float = 1.0):
    """Cell centres covering a pitch, as ((N, 2) points, (rows, cols) shape)."""
    gx, gy = np.meshgrid(np.arange(cell / 2, length, cell), np.arange(cell / 2, width, cell))
    return np.column_stack([gx.ravel(), gy.ravel()]), gx.shape


def cell_index(points: np.ndarray, nx: int, ny: int, length: float = XT_LENGTH,
               width: float = XT_WIDTH) -> np.ndarray:
    """Flat index (row * nx + col) of the nx x ny zone each point falls in."""
    ix = np.clip((points[..., 0] / length * nx).astype(int), 0, nx - 1)
    iy = np.clip((points[..., 1] / width * ny).astype(int), 0, ny - 1)
    return iy * nx + ix


def xt_lookup(xt: np.ndarray, points: np.ndarray, attacks_right, length: float,
              width: float) -> np.ndarray:
    """xT of each point on a y-up metric pitch, for a team attacking right or left.

    The xT grid is learned with the attack towards +x and y = 0 on the top
    touchline. On a y-up pitch that means flipping y when the team attacks right,
    and flipping x when it attacks left (a 180° rotation of the first).
    `attacks_right` is a bool, or an array broadcastable to points[..., 0].
    """
    ny, nx = xt.shape
    x, y = points[..., 0], points[..., 1]
    along = np.where(attacks_right, x, length - x)
    across = np.where(attacks_right, width - y, y)
    oriented = np.stack([along, across], axis=-1)
    index = cell_index(np.nan_to_num(oriented), nx, ny, length, width)
    return np.where(np.isnan(oriented).any(axis=-1), np.nan, xt.ravel()[index])


def nearest_distance(points: np.ndarray, others: np.ndarray) -> np.ndarray:
    """(..., N) distance from each of `points` (..., N, 2) to the nearest of `others` (..., M, 2).

    Missing players (NaN) are skipped; a point with nobody to measure to gets inf.
    """
    gaps = np.linalg.norm(points[..., :, None, :] - others[..., None, :, :], axis=-1)
    return np.where(np.isnan(gaps), np.inf, gaps).min(axis=-1)


def attacking_control(attackers: np.ndarray, defenders: np.ndarray, grid: np.ndarray,
                      config: PitchControlConfig = PITCH_CONTROL) -> np.ndarray:
    """Probability the attacking team controls each grid point (a 'soft Voronoi').

    Compare how far the nearest attacker and the nearest defender are from a point.
    A hard Voronoi gives the point to whoever is closer. Passing the gap through
    a logistic curve instead makes control fade smoothly across the boundary.
    As `softness` shrinks towards 0, this turns back into a Voronoi diagram.
    Pass any (N, 2) points as `grid`, e.g. a metric pitch grid or player positions.
    """
    nearest = lambda players: (cdist(grid, players).min(axis=1) if len(players)
                               else np.full(len(grid), np.inf))
    return expit((nearest(defenders) - nearest(attackers)) / config.softness)
