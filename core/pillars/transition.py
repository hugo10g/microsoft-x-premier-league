"""
Pillar 4, Transition Intensity: how fast a player hits a sprint after the ball changes hands.

    s(t)  = speed in km/h from smoothed tracking (core.kinematics.speed_kmh)
    t*    = earliest t ≥ t0 with s(u) ≥ v_sprint for every u in [t, t + Δ_min]
    L     = t* - t0, or the horizon H if no sustained sprint starts by t0 + H (censored)
    W     = share of [t0, t0 + H] spent below walking pace
    score = exp(-L / τ) - λ · W

Requiring the sprint to be sustained for Δ_min stops a single noisy frame, or a
player coasting out of a sprint they were already in, from counting as a reaction.
A player already sprinting at the turnover, and staying there, scores L = 0.
Thresholds live in core.config.TransitionConfig.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from core.config import TRANSITION, TransitionConfig
from core.kinematics import runs, sample_interval, speed_kmh


@dataclass(frozen=True)
class TransitionResult:
    latency: float           # seconds to a sustained sprint; the horizon if never reached
    reached_sprint: bool
    walk_fraction: float     # share of the horizon spent walking
    peak_speed_kmh: float    # within the horizon
    valid_fraction: float    # share of the horizon with tracking
    score: float


def latency_from_speeds(timestamps: np.ndarray, speeds_kmh: np.ndarray, t_turnover: float,
                        config: TransitionConfig = TRANSITION) -> TransitionResult:
    """Transition latency from a precomputed (T,) speed series in km/h."""
    timestamps = np.asarray(timestamps, dtype=float)
    dt = sample_interval(timestamps)
    horizon = config.horizon
    window = np.flatnonzero((timestamps >= t_turnover) & (timestamps <= t_turnover + horizon))
    if len(window) == 0:
        raise ValueError(f"No tracking frames after t_turnover={t_turnover}")
    first, last = window[0], window[-1]

    # Look past the horizon so a sprint starting just before it can prove it was sustained.
    fast = np.nan_to_num(speeds_kmh[first:], nan=0.0) >= config.sprint_kmh
    needed = int(np.ceil(config.min_sprint_duration / dt - 1e-9))
    latency, reached = horizon, False
    for start, end in runs(fast):
        if first + start > last:
            break
        if end - start >= needed:
            latency, reached = float(timestamps[first + start] - t_turnover), True
            break

    in_horizon = speeds_kmh[first:last + 1]
    observed = in_horizon[~np.isnan(in_horizon)]
    walk_fraction = float(np.mean(observed < config.walk_kmh)) if len(observed) else np.nan
    return TransitionResult(
        latency=latency,
        reached_sprint=reached,
        walk_fraction=walk_fraction,
        peak_speed_kmh=float(observed.max()) if len(observed) else np.nan,
        valid_fraction=len(observed) / (round(horizon / dt) + 1),
        score=float(np.exp(-latency / config.tau) - config.walk_penalty * walk_fraction),
    )


def transition_latency(timestamps: np.ndarray, positions: np.ndarray, t_turnover: float,
                       config: TransitionConfig = TRANSITION) -> TransitionResult:
    """Transition latency for one player from raw (T, 2) positions in metres."""
    return latency_from_speeds(timestamps, speed_kmh(positions, timestamps), t_turnover, config)


def detect_turnovers(timestamps: np.ndarray, possession, across_dead_ball: bool = False):
    """(time, team that won the ball) for every change of possession.

    `possession` holds the team in control at each frame, None/NaN when the ball is
    dead or loose. By default a change only counts when one team's frame is directly
    followed by the other's, which keeps open-play turnovers and drops restarts
    (the ball going out for a throw-in, say). Set across_dead_ball=True to count those too.
    """
    team = pd.Series(possession, dtype=object)
    if across_dead_ball:
        team = team.ffill()
    before = team.shift()
    changed = team.notna() & before.notna() & (team != before)
    return [(float(timestamps[i]), team.iloc[i]) for i in np.flatnonzero(changed.to_numpy())]


def transition_latencies(timestamps: np.ndarray, speeds_kmh: np.ndarray, player_ids, player_teams,
                         turnovers, ball_distances: np.ndarray | None = None,
                         config: TransitionConfig = TRANSITION) -> pd.DataFrame:
    """Every player's transition after every turnover.

    speeds_kmh (T, P), e.g. from `speed_kmh(positions, timestamps)` or, for Metrica,
    the norm of `TrackingData.velocities()` × 3.6. player_ids and player_teams have
    length P; turnovers is what `detect_turnovers` returns. Players on the team that
    won the ball are in an attacking transition, everyone else in a defensive one.
    Players with no tracking after the turnover (not on the pitch) are skipped.

    ball_distances (len(turnovers), P) is each player's distance to the ball at each
    turnover. When given, players further than `config.max_ball_distance` (or with
    no distance) are skipped: a centre-back 60 m away isn't expected to sprint, and
    scoring them would bury real reactions under the 10 s horizon cap.
    """
    rows = []
    for n, (t_turnover, winner) in enumerate(turnovers):
        for p, (player, team) in enumerate(zip(player_ids, player_teams)):
            if ball_distances is not None and not ball_distances[n, p] <= config.max_ball_distance:
                continue
            result = latency_from_speeds(timestamps, speeds_kmh[:, p], t_turnover, config)
            if result.valid_fraction == 0:
                continue
            rows.append({"t_turnover": t_turnover, "player": player, "team": team,
                         "phase": "attacking" if team == winner else "defensive",
                         "ball_distance": np.nan if ball_distances is None else ball_distances[n, p],
                         **asdict(result)})
    return pd.DataFrame(rows)
