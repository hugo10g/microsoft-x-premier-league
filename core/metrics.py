"""
Expected Threat (xT): how valuable each zone of the pitch is to the team on the ball.

Learned from the league's own passes, carries and shots (core.pipeline.learn_xt
builds the training table). The off-ball pillars weight space and passing lanes by it.
"""

import numpy as np
import pandas as pd

from core.config import XT
from core.spatial import cell_index, xt_lookup


def xy(locations: pd.Series) -> np.ndarray:
    """Turn a column of [x, y] lists into an (n, 2) float array."""
    return np.array([loc[:2] for loc in locations], dtype=float).reshape(-1, 2)


def build_xt(events: pd.DataFrame, nx: int = XT.bins[0], ny: int = XT.bins[1]) -> np.ndarray:
    """Karun Singh's Expected Threat, learned from passes, carries and shots.

    xT(zone) = P(shoot) * P(goal | shot)  +  P(move) * sum over zones z' of P(z -> z') * xT(z')

    The value of a zone is what you can score from there directly, plus the value
    of wherever you usually move the ball next. It refers to itself, so we start
    at zero and repeat the update until the numbers stop changing.

    `events` uses StatsBomb's layout: a 120 x 80 pitch, every action drawn as if its
    team attacks left to right, with columns type ("Pass" / "Carry" / "Shot"),
    location, pass_end_location, pass_outcome (NaN when completed),
    carry_end_location and shot_outcome ("Goal" or anything else).
    """
    n = nx * ny
    passes = events[events["type"] == "Pass"]
    completed = passes[passes["pass_outcome"].isna()]
    carries = events[events["type"] == "Carry"]
    shots = events[events["type"] == "Shot"]
    goals = shots[shots["shot_outcome"] == "Goal"]

    cells = lambda s: cell_index(xy(s), nx, ny)
    move_starts = np.concatenate([cells(passes["location"]), cells(carries["location"])])
    good_starts = np.concatenate([cells(completed["location"]), cells(carries["location"])])
    good_ends = np.concatenate([cells(completed["pass_end_location"]),
                                cells(carries["carry_end_location"])])

    n_moves = np.bincount(move_starts, minlength=n)
    n_shots = np.bincount(cells(shots["location"]), minlength=n)
    n_goals = np.bincount(cells(goals["location"]), minlength=n)

    safe_div = lambda a, b: np.divide(a, b, out=np.zeros(a.shape), where=b > 0)
    p_shot = safe_div(n_shots, n_moves + n_shots)
    p_move = safe_div(n_moves, n_moves + n_shots)
    p_goal = safe_div(n_goals, n_shots)

    # transition[a, b] = share of ALL moves from zone a that successfully reached zone b.
    # Failed moves sit in the denominator but lead nowhere, so they count as zero value.
    transition = np.zeros((n, n))
    np.add.at(transition, (good_starts, good_ends), 1)
    transition = safe_div(transition, n_moves[:, None].repeat(n, axis=1))

    xt = np.zeros(n)
    for _ in range(100):
        updated = p_shot * p_goal + p_move * (transition @ xt)
        if np.abs(updated - xt).max() < 1e-7:
            break
        xt = updated
    return updated.reshape(ny, nx)   # rows = y bins, columns = x bins


def xt_added(xt: np.ndarray, start: np.ndarray, end: np.ndarray, attacks_right, length: float,
             width: float) -> np.ndarray:
    """xT gained by moving the ball from `start` to `end` ((N, 2) metres), for the team on the ball.

    Negative when the ball goes backwards into less dangerous space; `attacks_right`
    is a bool or (N,) array, as in core.spatial.xt_lookup.
    """
    right = np.asarray(attacks_right)
    return xt_lookup(xt, end, right, length, width) - xt_lookup(xt, start, right, length, width)
