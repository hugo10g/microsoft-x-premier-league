"""
Kinematics from raw tracking arrays: gap filling, velocity and speed.

Conventions used by every tracking-based module:
    timestamps  (T,)          seconds, uniformly sampled (e.g. 25 Hz).
                              A missing sample is a row of NaN, never a dropped row.
    positions   (T, 2)        pitch coordinates in metres for one player (or the ball),
             or (T, P, 2)     for P players at once.
"""

import numpy as np
from scipy.signal import savgol_filter

from core.config import KINEMATICS, KinematicsConfig

KMH = 1 / 3.6  # multiply km/h by this to get m/s


def sample_interval(timestamps: np.ndarray, tolerance: float = 0.05) -> float:
    """The frame interval dt, after checking the clock really is uniform."""
    steps = np.diff(timestamps)
    dt = float(np.median(steps))
    if dt <= 0 or np.abs(steps - dt).max() > tolerance * dt:
        raise ValueError("Timestamps must be uniformly sampled. "
                         "Mark missing frames as NaN rows instead of dropping them.")
    return dt


def runs(mask: np.ndarray) -> np.ndarray:
    """(start, end) frame pairs, end exclusive, for every run of True in a 1-D mask."""
    padded = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    return np.flatnonzero(np.diff(padded.astype(int))).reshape(-1, 2)


def fill_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate NaN gaps of at most `max_gap` frames; longer gaps stay NaN.

    Tracking and pose feeds drop the odd frame (occlusion, a player leaving the
    camera). Short gaps are safe to bridge; inventing a long one is not.
    """
    values = np.array(values, dtype=float)
    flat = values.reshape(len(values), -1)          # a view, so edits land in `values`
    for start, end in runs(np.isnan(flat).any(axis=1)):
        if end - start > max_gap or start == 0 or end == len(flat):
            continue
        weight = (np.arange(1, end - start + 1) / (end - start + 1))[:, None]
        flat[start:end] = flat[start - 1] + weight * (flat[end] - flat[start - 1])
    return values


def velocity(positions: np.ndarray, timestamps: np.ndarray,
             config: KinematicsConfig = KINEMATICS) -> np.ndarray:
    """Velocity in m/s, same shape as `positions`, from smoothed frame-to-frame deltas.

        v_raw(t) = (p(t) - p(t-1)) / dt            frame-by-frame displacement
        v_raw(t) = NaN  where |v_raw| > max speed  tracking glitches (ID swaps, jumps)
        v(t)     = Savitzky-Golay(v_raw)           local polynomial fit over the window

    Raw deltas turn centimetre jitter into speed spikes of several m/s at 25 Hz;
    the filter keeps genuine accelerations while killing the jitter. Each unbroken
    stretch of data is filtered separately, so it never smooths across a gap.
    """
    if positions.ndim == 3:
        return np.stack([velocity(positions[:, p], timestamps, config)
                         for p in range(positions.shape[1])], axis=1)

    dt = sample_interval(timestamps)
    gap = round(config.max_gap / dt)
    positions = fill_gaps(positions, gap)

    raw = np.full(positions.shape, np.nan)
    raw[1:] = np.diff(positions, axis=0) / dt
    raw[np.linalg.norm(raw, axis=1) > config.max_speed_kmh * KMH] = np.nan
    raw = fill_gaps(raw, gap)

    frames = max(round(config.smoothing_window / dt), config.polyorder + 1)
    frames += frames % 2 == 0                        # the filter needs an odd window
    vel = np.full(raw.shape, np.nan)
    for start, end in runs(~np.isnan(raw).any(axis=1)):
        segment = raw[start:end]
        vel[start:end] = (savgol_filter(segment, frames, config.polyorder, axis=0)
                          if len(segment) >= frames else segment)
    return vel


def speed(positions: np.ndarray, timestamps: np.ndarray,
          config: KinematicsConfig = KINEMATICS) -> np.ndarray:
    """Speed in m/s: (T,) for one player, (T, P) for several."""
    return np.linalg.norm(velocity(positions, timestamps, config), axis=-1)


def speed_kmh(positions: np.ndarray, timestamps: np.ndarray,
              config: KinematicsConfig = KINEMATICS) -> np.ndarray:
    """Speed in km/h: (T,) for one player, (T, P) for several."""
    return speed(positions, timestamps, config) * 3.6
