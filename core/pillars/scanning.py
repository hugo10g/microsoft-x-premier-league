"""
Pillar 1, Scanning & Awareness: head checks made off the ball before receiving it.

Following Geir Jordet's "visual exploratory action" (a body or head movement in
which the player's face is actively and temporarily directed away from the ball,
to look for teammates, opponents or space), a scan is detected from head yaw,
the player's position and the ball's position:

    h(t)   = unit head direction from yaw; NaN gaps ≤ max_gap interpolated, then a
             vector average over `smoothing` seconds (averaging angles directly
             would break at ±180°)
    b(t)   = bearing from the player to the ball; undefined within min_ball_distance,
             where a player on or at the ball has no "away from it" to look
    φ(t)   = angle between h(t) and b(t), 0..180°
    ω(t)   = head angular speed, °/s

    scan   = a maximal interval that starts when φ rises above on_angle and ends when
             φ falls back below off_angle (hysteresis, so jitter around one threshold
             is not a string of scans), lasting min_duration..max_duration, with a
             peak ω ≥ min_peak_speed over the excursion (from where φ last left
             off_angle). Longer is facing away, not a check; slower is drift.

Before a reception at t_r, over [t_r - window, t_r):
    n_scans        scans whose onset falls in the window
    scan_rate      n_scans per observed second (φ defined), NaN when less than
                   min_observed of the window has head data
    ball_fixation  share of observed frames with φ < fixation_angle

Thresholds live in core.config.ScanningConfig.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config import SCANNING, ScanningConfig
from core.kinematics import fill_gaps, runs

SCAN_COLUMNS = ["start", "end", "duration", "peak_speed", "peak_angle"]


@dataclass(frozen=True)
class ScanWindow:
    n_scans: int            # head checks with onset in the window
    scan_rate: float        # per observed second; NaN with too little head data
    ball_fixation: float    # 0..1 share of observed frames looking at the ball
    observed: float         # 0..1 share of the window with a defined head-to-ball angle


def _interval(timestamps: np.ndarray) -> float:
    return float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 0.04


def head_direction(yaw_deg: np.ndarray, dt: float, config: ScanningConfig = SCANNING) -> np.ndarray:
    """(T, 2) smoothed unit head vectors from yaw in degrees; NaN where unknown."""
    rad = np.radians(np.asarray(yaw_deg, dtype=float))
    vec = fill_gaps(np.stack([np.cos(rad), np.sin(rad)], axis=1), round(config.max_gap / dt))
    width = max(1, round(config.smoothing / dt))
    width += width % 2 == 0
    if width > 1:
        missing = np.isnan(vec[:, 0])
        kernel = np.ones(width)
        total = np.stack([np.convolve(np.nan_to_num(vec[:, k]), kernel, "same") for k in (0, 1)], axis=1)
        count = np.convolve((~missing).astype(float), kernel, "same")
        vec = np.where(missing[:, None], np.nan, total / np.maximum(count, 1)[:, None])
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return np.divide(vec, norm, out=np.full_like(vec, np.nan), where=norm > 1e-9)


def angle_to_ball(head: np.ndarray, player_pos: np.ndarray, ball_pos: np.ndarray,
                  config: ScanningConfig = SCANNING) -> np.ndarray:
    """(T,) φ: degrees between the head direction and the bearing to the ball."""
    to_ball = np.asarray(ball_pos, dtype=float) - np.asarray(player_pos, dtype=float)
    distance = np.linalg.norm(to_ball, axis=1)
    cosine = np.einsum("ij,ij->i", head, to_ball) / np.where(distance > 0, distance, np.nan)
    phi = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    phi[~(distance >= config.min_ball_distance)] = np.nan
    return phi


def angular_speed(head: np.ndarray, dt: float) -> np.ndarray:
    """(T,) head turn speed in degrees per second (NaN on the first frame and across gaps)."""
    speed = np.full(len(head), np.nan)
    if len(head) > 1:
        cross = head[1:, 0] * head[:-1, 1] - head[1:, 1] * head[:-1, 0]
        dot = np.einsum("ij,ij->i", head[1:], head[:-1])
        speed[1:] = np.degrees(np.abs(np.arctan2(cross, dot))) / dt
    return speed


def detect_scans(timestamps: np.ndarray, yaw_deg: np.ndarray, player_pos: np.ndarray,
                 ball_pos: np.ndarray, config: ScanningConfig = SCANNING) -> pd.DataFrame:
    """Every scan in a (T,) yaw series: start/end times (s), duration, peak speed and angle.

    An interval still open where the data begins is skipped (its onset is unknown);
    one cut short by missing data or the ball arriving ends there.
    """
    timestamps = np.asarray(timestamps, dtype=float)
    dt = _interval(timestamps)
    head = head_direction(yaw_deg, dt, config)
    return find_scans(timestamps, angle_to_ball(head, player_pos, ball_pos, config),
                      angular_speed(head, dt), config)


def find_scans(timestamps: np.ndarray, phi: np.ndarray, speed: np.ndarray,
               config: ScanningConfig = SCANNING) -> pd.DataFrame:
    """Scans from a precomputed φ (degrees off the ball) and head speed (°/s) series."""
    dt = _interval(timestamps)
    # Hysteresis: above on_angle switches a scan on, below off_angle (or no data) off;
    # in between, the previous state carries forward.
    index = np.arange(len(phi))
    off = ~(phi >= config.off_angle)                   # NaN counts as off
    switch = np.where((phi > config.on_angle) | off, index, -1)
    latest = np.maximum.accumulate(switch) if len(switch) else switch
    state = (latest >= 0) & (phi[np.maximum(latest, 0)] > config.on_angle)
    last_off = np.maximum.accumulate(np.where(off, index, -1)) if len(off) else off

    rows = []
    for start, end in runs(state):
        if start == 0:
            continue
        duration = (end - start) * dt
        if not config.min_duration <= duration <= config.max_duration:
            continue
        excursion = last_off[start - 1] + 1
        peak = np.nanmax(speed[excursion:end]) if np.isfinite(speed[excursion:end]).any() else 0.0
        if peak < config.min_peak_speed:
            continue
        rows.append((timestamps[start], timestamps[start] + duration, duration, peak,
                     float(np.nanmax(phi[start:end]))))
    return pd.DataFrame(rows, columns=SCAN_COLUMNS)


def scans_before_reception(timestamps: np.ndarray, yaw_deg: np.ndarray, player_pos: np.ndarray,
                           ball_pos: np.ndarray, t_receive: float, window: float | None = None,
                           config: ScanningConfig = SCANNING) -> ScanWindow:
    """Scanning in the `window` seconds before a reception at `t_receive`.

    Only a slice around the window is analysed: max_duration of context before it, so
    the hysteresis state is settled, and after it, so a scan running into the
    reception is measured in full.
    """
    window = config.window if window is None else window
    timestamps = np.asarray(timestamps, dtype=float)
    lo = np.searchsorted(timestamps, t_receive - window - config.max_duration)
    hi = np.searchsorted(timestamps, t_receive + config.max_duration, side="right")
    ts = timestamps[lo:hi]
    if len(ts) < 2:
        return ScanWindow(0, np.nan, np.nan, 0.0)
    dt = _interval(ts)
    head = head_direction(yaw_deg[lo:hi], dt, config)
    phi = angle_to_ball(head, player_pos[lo:hi], ball_pos[lo:hi], config)
    scans = find_scans(ts, phi, angular_speed(head, dt), config)
    n_scans = int(((scans["start"] >= t_receive - window) & (scans["start"] < t_receive)).sum())

    phi = phi[(ts >= t_receive - window - 1e-9) & (ts < t_receive - 1e-9)]
    seen = np.isfinite(phi)
    observed = min(1.0, seen.sum() * dt / window)
    rate = n_scans / (seen.sum() * dt) if observed >= config.min_observed else np.nan
    fixation = float((phi[seen] < config.fixation_angle).mean()) if seen.any() else np.nan
    return ScanWindow(n_scans, float(rate), fixation, float(observed))
