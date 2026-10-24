"""
Run Types: what each off-ball run was for, from its direction of travel.

Runs are the sustained off-ball sprints of core.pillars.space.off_ball_runs. Each is
read relative to the runner's team: their velocity minus the mean velocity of their
outfield teammates, so a whole block retreating or pushing up isn't anyone's run.
That relative movement is read in the runner's attacking frame (+along = towards
the opponent's goal, +out = towards the nearer touchline) from its mean direction
and its length:

    Run in Behind   along holds ≥ dominance of the direction, and it ends within
                    behind_margin of the opponent's last outfield defender or past him
    Pulling Wide    out holds ≥ dominance and it ends nearer the touchline
    Dropping Deep   -along holds ≥ dominance: coming back towards the ball
    Other           anything else (diagonals, forward runs short of the line)

Runs that move less than min_distance against the team stay unclassified. Thresholds live in
core.config.RunTypeConfig.
"""

import warnings

import numpy as np
import pandas as pd

from core.config import RUN_TYPES, SPACE, RunTypeConfig, SpaceConfig
from core.kinematics import sample_interval
from core.pillars.space import off_ball_runs
from core.tracking import PITCH_WIDTH, TrackingData

COLUMNS = ["player", "time", "duration", "distance", "run_type"]


def classify(start: np.ndarray, end: np.ndarray, relative_velocity: np.ndarray, dt: float,
             attacks_right: bool, last_line: float, config: RunTypeConfig = RUN_TYPES) -> str | None:
    """The type of one run from its start/end (2,), velocity relative to the team (F, 2) and the line.

    `last_line` is the x of the opponent's last outfield defender at the end of the run.
    None when the run moves less than min_distance against the team.
    """
    s = 1.0 if attacks_right else -1.0
    if np.linalg.norm(np.nansum(relative_velocity, axis=0) * dt) < config.min_distance:
        return None
    mean = np.nanmean(relative_velocity, axis=0)
    direction = mean / max(np.linalg.norm(mean), 1e-9)
    side = 1.0 if start[1] >= PITCH_WIDTH / 2 else -1.0     # +y is towards the near touchline
    along, out = direction[0] * s, direction[1] * side
    if along >= config.dominance and (end[0] - last_line) * s >= -config.behind_margin:
        return "Run in Behind"
    if out >= config.dominance and abs(end[1] - PITCH_WIDTH / 2) > abs(start[1] - PITCH_WIDTH / 2):
        return "Pulling Wide"
    if -along >= config.dominance:
        return "Dropping Deep"
    return "Other"


def run_types(tracking: TrackingData, possession: np.ndarray, velocities: np.ndarray | None = None,
              outfield: np.ndarray | None = None, config: RunTypeConfig = RUN_TYPES,
              space: SpaceConfig = SPACE) -> pd.DataFrame:
    """Every classified off-ball run in a match (columns as COLUMNS).

    `outfield` (P,) bool marks who counts as a defender for the line (default: everyone).
    """
    velocities = tracking.velocities() if velocities is None else velocities
    outfield = np.ones(len(tracking.player_ids), dtype=bool) if outfield is None else outfield
    speeds = np.linalg.norm(velocities, axis=-1) * 3.6
    dt = sample_interval(tracking.timestamps)
    rows = []
    for run in off_ball_runs(tracking, speeds, possession, space).itertuples():
        team = tracking.teams[run.player]
        last = run.end - 1
        right = bool(tracking.attacks_right(team)[run.start])
        rivals = tracking.positions[last, (tracking.teams != team) & outfield, 0]
        rivals = rivals[~np.isnan(rivals)]
        if not len(rivals):
            continue
        line = rivals.max() if right else rivals.min()        # the deepest defender, nearest their goal
        start, end = tracking.positions[run.start, run.player], tracking.positions[last, run.player]
        mates = (tracking.teams == team) & outfield
        mates[run.player] = False
        window = velocities[run.start:run.end]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # a frame with no teammate tracked
            relative = window[:, run.player] - np.nanmean(window[:, mates], axis=1)
        kind = classify(start, end, relative, dt, right, line, config)
        if kind is not None:
            rows.append((tracking.player_ids[run.player], tracking.timestamps[run.start],
                         tracking.timestamps[last] - tracking.timestamps[run.start],
                         float(np.linalg.norm(end - start)), kind))
    return pd.DataFrame(rows, columns=COLUMNS)
