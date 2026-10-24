"""
View model for the pitch player: one playback window as plain JSON-ready arrays.

Everything analytical happens here, in Python, from the core modules' outputs;
the browser only draws. Per frame it carries positions, velocity-based body
orientation (core.pose proxy), speed, the ball, who has it, and the cover-shadow
result from core.pillars.pressing (which lanes each press is cutting). Per window
it carries xT surfaces for both teams and the constants the overlays need.

With head-orientation data (core.tracking.load_head_orientation) it also carries
head yaw per player and, per frame, who is mid head check: a player whose head
points more than SCAN_ANGLE away from the ball is looking around, not at play.
"""

import numpy as np

from core.config import PITCH_CONTROL, PRESSING, SCANNING
from core.pillars.pressing import lanes_blocked
from core.pose import body_orientation_from_velocity
from core.spatial import xt_lookup
from core.tracking import PITCH_LENGTH, PITCH_WIDTH, TrackingData

# Shared with the scanning pillar so the cones and the metric use one definition.
SCAN_ANGLE = SCANNING.on_angle            # degrees off the ball bearing that counts as a head check
NEAR_BALL = SCANNING.min_ball_distance    # metres; this close to the ball its bearing is undefined


def flat(values: np.ndarray, decimals: int) -> list:
    """Row-major list with NaN as null (JSON has no NaN)."""
    rounded = np.round(values.astype(float), decimals).ravel()
    return [None if np.isnan(v) else float(v) for v in rounded]


def press_geometry(positions: np.ndarray, ball: np.ndarray, possession: np.ndarray,
                   teams: np.ndarray) -> tuple[list, list, list]:
    """Per frame: ball carrier, defenders pressing them, and receivers whose lane is cut.

    Same rules as the pressing pillar: the carrier is the attacker nearest the ball,
    pressers are defenders within max_press_distance of the carrier, and a lane is
    cut when its receiver sits inside a presser's cover shadow, beyond the presser.
    """
    carriers, pressers, cut = [], [], []
    for frame, owner in enumerate(possession):
        on = ~np.isnan(positions[frame]).any(axis=1)
        if owner is None or np.isnan(ball[frame]).any():
            carriers.append(-1), pressers.append([]), cut.append([])
            continue
        attackers = np.flatnonzero(on & (teams == owner))
        defenders = np.flatnonzero(on & (teams != owner))
        carrier = attackers[np.linalg.norm(positions[frame, attackers] - ball[frame], axis=1).argmin()]
        receivers = attackers[attackers != carrier]
        near = defenders[np.linalg.norm(positions[frame, defenders] - positions[frame, carrier], axis=1)
                         <= PRESSING.max_press_distance]
        blocked = (lanes_blocked(positions[frame, carrier], positions[frame, near],
                                 positions[frame, receivers], PRESSING.reach).any(axis=0)
                   if len(near) else np.zeros(len(receivers), dtype=bool))
        carriers.append(int(carrier))
        pressers.append(near.tolist())
        cut.append(receivers[blocked].tolist())
    return carriers, pressers, cut


def head_checks(yaw: np.ndarray, positions: np.ndarray, ball: np.ndarray,
                carriers: list[int]) -> list[list[int]]:
    """Per frame, indices of players whose head points over SCAN_ANGLE away from the ball.

    yaw: (T, P) degrees, counter-clockwise from +x. The ball carrier and anyone within
    NEAR_BALL of the ball are skipped, as are frames without head data or a ball.
    """
    offset = ball[:, None, :] - positions
    bearing = np.degrees(np.arctan2(offset[..., 1], offset[..., 0]))
    gap = np.abs((yaw - bearing + 180.0) % 360.0 - 180.0)          # smallest angle, 0..180
    with np.errstate(invalid="ignore"):
        away = (gap > SCAN_ANGLE) & (np.linalg.norm(offset, axis=-1) > NEAR_BALL)
    rows = np.arange(len(carriers))
    holder = np.asarray(carriers)
    away[rows[holder >= 0], holder[holder >= 0]] = False
    return [np.flatnonzero(frame).tolist() for frame in away]


def xt_surfaces(xt: np.ndarray, tr: TrackingData, row: int) -> dict:
    """xT on the window's pitch for each team, rows from y = 0 upwards, in their attacking direction."""
    ny, nx = xt.shape
    gx, gy = np.meshgrid((np.arange(nx) + 0.5) * PITCH_LENGTH / nx, (np.arange(ny) + 0.5) * PITCH_WIDTH / ny)
    cells = np.stack([gx, gy], axis=-1)
    return {team: np.round(xt_lookup(xt, cells, bool(tr.attacks_right(team)[row]),
                                     PITCH_LENGTH, PITCH_WIDTH), 4).tolist()
            for team in ("Home", "Away")}


def build_window(tr: TrackingData, velocities: np.ndarray, possession: np.ndarray, xt: np.ndarray,
                 minutes: np.ndarray, start: int, end: int, names: list[str], team_names: dict,
                 team_colours: dict, highlight: dict | None = None, autoplay: bool = False,
                 start_offset: int = 0, head_yaw: np.ndarray | None = None) -> dict:
    """Everything the pitch player needs for tracking rows [start, end).

    names: display name per player (same order as tr.player_ids). highlight:
    {"player": id, "start": row, "end": row, "label": text} to ring a player through
    a key moment. start_offset: frame within the window where playback begins.
    head_yaw: the full match's (T, P) head yaw in degrees, or None without head data;
    adds "head" (whole degrees, null when unknown) and "scanning" (head checks per frame).
    """
    rows = slice(start, end)
    positions, vel, ball = tr.positions[rows], velocities[rows], tr.ball[rows]
    heading = body_orientation_from_velocity(vel, positions, ball)
    carriers, pressers, cut = press_geometry(positions, ball, possession[rows], tr.teams)
    mark = None
    if highlight:
        mark = {"player": tr.index(highlight["player"]), "label": highlight["label"],
                "from": max(0, highlight["start"] - start), "to": min(end, highlight["end"]) - start}
    yaw = None if head_yaw is None else np.asarray(head_yaw, dtype=float)[rows]
    head = {}
    if yaw is not None and np.isfinite(yaw).any():
        head = {"head": [None if v is None else int(v) for v in flat(yaw, 0)],
                "scanning": head_checks(yaw, positions, ball, carriers)}
    return {
        "fps": round(1 / float(np.median(np.diff(tr.timestamps[:50])))),
        "frames": end - start, "start": int(start_offset), "autoplay": autoplay,
        "players": [{"id": pid, "team": team, "number": pid.split("_")[1], "name": name}
                    for pid, team, name in zip(tr.player_ids, tr.teams, names)],
        "teams": {team: {"name": team_names.get(team, team), "colour": team_colours[team]}
                  for team in ("Home", "Away")},
        "positions": flat(positions, 2), "heading": flat(heading, 2),
        "speed": flat(np.linalg.norm(vel, axis=-1) * 3.6, 1), "ball": flat(ball, 2),
        "minute": np.round(minutes[rows], 4).tolist(), "period": tr.periods[rows].tolist(),
        "possession": [o if isinstance(o, str) else None for o in possession[rows]],
        "carrier": carriers, "pressers": pressers, "cut": cut,
        "xt": xt_surfaces(xt, tr, start),
        "attacks_right": {team: bool(tr.attacks_right(team)[start]) for team in ("Home", "Away")},
        "pitch": {"length": PITCH_LENGTH, "width": PITCH_WIDTH},
        "softness": PITCH_CONTROL.softness, "reach": PRESSING.reach,
        "highlight": mark,
        **head,
    }
