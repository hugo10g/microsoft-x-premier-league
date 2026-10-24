"""
Pressure Intensity: how hard a defender closes down the ball carrier in open play.

    carrier    the attacker nearest the ball, if within control_distance of it
    d          defender's distance to the carrier
    closing    (v_def - v_carrier) · û,  û = unit vector defender -> carrier  (m/s)
    pressure   a spell of frames with d ≤ radius and closing ≥ min_closing, held
               for min_duration
    intensity  Σ closing · (1 - d / radius) · Δt   over the spell (metres)

Intensity is the ground made up on the carrier, weighted by how close it was made
up: sprinting at a carrier from 9 m counts less than the same speed from 2 m.
Lane cuts (core.pillars.pressing) say whether a press is *angled* well; this says
how *aggressive* it is. Thresholds live in core.config.PressureConfig.
"""

import numpy as np
import pandas as pd

from core.config import PRESSURE, PressureConfig
from core.kinematics import runs, sample_interval
from core.tracking import TrackingData

COLUMNS = ["player", "carrier", "start_time", "duration", "min_distance", "peak_closing_kmh", "intensity"]


def closing_speed(defenders: np.ndarray, defender_vel: np.ndarray, carrier: np.ndarray,
                  carrier_vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(distance, closing speed in m/s) of each defender to the carrier.

    defenders and defender_vel (..., D, 2); carrier and carrier_vel (..., 2). Closing
    speed is how fast the gap shrinks: positive when the defender gains on the carrier.
    """
    offset = carrier[..., None, :] - defenders
    distance = np.linalg.norm(offset, axis=-1)
    towards = np.divide(offset, distance[..., None], out=np.zeros_like(offset), where=distance[..., None] > 1e-6)
    closing = np.sum((defender_vel - carrier_vel[..., None, :]) * towards, axis=-1)
    return distance, closing


def pressures(tracking: TrackingData, possession: np.ndarray, velocities: np.ndarray | None = None,
              config: PressureConfig = PRESSURE) -> pd.DataFrame:
    """Every pressure in a match, one row per (defender, spell); columns as COLUMNS."""
    velocities = tracking.velocities() if velocities is None else velocities
    dt = sample_interval(tracking.timestamps)
    needed = int(np.ceil(config.min_duration / dt - 1e-9))
    ids = np.array(tracking.player_ids)
    rows = []
    for team in ("Home", "Away"):
        attackers = np.flatnonzero(tracking.teams == team)
        defenders = np.flatnonzero(tracking.teams != team)
        frames = np.flatnonzero(possession == team)
        if not len(frames) or not len(defenders):
            continue
        to_ball = np.linalg.norm(tracking.positions[frames][:, attackers] - tracking.ball[frames][:, None], axis=-1)
        to_ball = np.where(np.isnan(to_ball), np.inf, to_ball)
        nearest = to_ball.argmin(axis=1)
        controlled = to_ball[np.arange(len(frames)), nearest] <= config.control_distance
        frames, nearest = frames[controlled], nearest[controlled]
        carrier = attackers[nearest]
        distance, closing = closing_speed(tracking.positions[frames][:, defenders],
                                          velocities[frames][:, defenders],
                                          tracking.positions[frames, carrier], velocities[frames, carrier])
        with np.errstate(invalid="ignore"):
            pressing = (distance <= config.radius) & (closing * 3.6 >= config.min_closing_kmh)
        weight = np.where(pressing, closing * (1 - distance / config.radius) * dt, 0.0)
        for d, defender in enumerate(defenders):
            for start, end in runs(pressing[:, d]):
                # A spell must be one unbroken stretch of play: consecutive frames, same carrier.
                span = frames[start:end]
                breaks = np.flatnonzero((np.diff(span) != 1) | (np.diff(carrier[start:end]) != 0)) + 1
                for a, b in zip(np.r_[0, breaks], np.r_[breaks, len(span)]):
                    if b - a < needed:
                        continue
                    k = slice(start + a, start + b)
                    rows.append((ids[defender], ids[carrier[start + a]], tracking.timestamps[span[a]],
                                 (b - a) * dt, float(distance[k, d].min()),
                                 float(closing[k, d].max() * 3.6), float(weight[k, d].sum())))
    return pd.DataFrame(rows, columns=COLUMNS).sort_values("start_time", ignore_index=True)
