"""
Pillar 3, Pressing Mechanics: does the presser's run cut off the valuable passing lanes?

    p̂_d     = p_d + Δt · v_d                         presser projected Δt = 0.5 s ahead
    shadow  = cone from the carrier c through p̂_d, half-angle α = arctan(r / |p̂_d - c|)
    w_k     = xT(p_k) · C(p_k)                        lane value: threat × how open k is
    LaneCut = Σ_k w_k · 1[lane c→k inside the shadow] / Σ_k w_k

Every lane starts at the carrier, the cone's apex, so "the lane crosses the
shadow" reduces to two numpy comparisons per receiver: the receiver lies within α
of the presser's bearing, and further from the carrier than the presser is. No
polygon intersection needed, so it vectorises across frames and receivers at once.

A straight sprint at the carrier scores low; a curved run that screens the best
lane scores high. Comparing LaneCut at p_d and at p̂_d shows whether the approach
is *heading* somewhere useful. Thresholds live in core.config.PressingConfig.
"""

import numpy as np
import pandas as pd
from scipy.special import expit

from core.config import PITCH_CONTROL, PRESSING, PitchControlConfig, PressingConfig
from core.pose import angle_between
from core.spatial import nearest_distance, xt_lookup
from core.tracking import PITCH_LENGTH, PITCH_WIDTH, TrackingData


def project(position: np.ndarray, velocity: np.ndarray, seconds: float) -> np.ndarray:
    """Where a player will be after `seconds` at constant velocity."""
    return position + seconds * velocity


def lanes_blocked(carrier: np.ndarray, defender: np.ndarray, receivers: np.ndarray,
                  reach: float) -> np.ndarray:
    """(..., K) bool: which carrier -> receiver lanes fall in the defender's cover shadow.

    carrier and defender (..., 2), receivers (..., K, 2). The defender's distance
    is floored at `reach`, which caps the shadow at 45° either side when they are
    right on top of the carrier.
    """
    to_defender = defender - carrier
    distance = np.maximum(np.linalg.norm(to_defender, axis=-1), reach)
    half_angle = np.degrees(np.arctan(reach / distance))
    to_receiver = receivers - carrier[..., None, :]
    bearing = angle_between(to_receiver, to_defender[..., None, :])
    beyond = np.linalg.norm(to_receiver, axis=-1) > distance[..., None]
    return (bearing <= half_angle[..., None]) & beyond


def lane_weights(receivers: np.ndarray, defenders: np.ndarray, receiver_xt: np.ndarray,
                 control: PitchControlConfig = PITCH_CONTROL) -> np.ndarray:
    """(..., K) value of each lane: receiver's xT × their pitch control where they stand.

    The receiver is the nearest attacker to their own position, so control reduces
    to expit(distance to nearest defender / softness). That runs from 0.5 when marked
    tight to 1 when wide open. Missing receivers weigh 0.
    """
    return np.nan_to_num(receiver_xt * expit(nearest_distance(receivers, defenders) / control.softness))


def lane_cut(carrier: np.ndarray, presser: np.ndarray, presser_vel: np.ndarray,
             receivers: np.ndarray, weights: np.ndarray, config: PressingConfig = PRESSING):
    """(now, projected): share of threat-weighted lanes blocked, from p_d and from p̂_d.

    Shapes as in `lanes_blocked`, weights (..., K). NaN where no lane has any value.
    """
    total = weights.sum(axis=-1)

    def share(position):
        blocked = (weights * lanes_blocked(carrier, position, receivers, config.reach)).sum(axis=-1)
        return np.divide(blocked, total, out=np.full(total.shape, np.nan), where=total > 0)

    return share(presser), share(project(presser, presser_vel, config.projection))


def pressing_rows(tracking: TrackingData, presser_id: str, possession: np.ndarray,
                  config: PressingConfig = PRESSING) -> np.ndarray:
    """Rows where the opponent has the ball and the presser is within press distance of it."""
    p = tracking.index(presser_id)
    team = tracking.teams[p]
    distance = np.linalg.norm(tracking.positions[:, p] - tracking.ball, axis=-1)
    opponent_ball = (possession != team) & pd.notna(possession)
    return np.flatnonzero(opponent_ball & (distance <= config.max_press_distance))


def pressing_lane_cut(tracking: TrackingData, presser_id: str, rows: np.ndarray, xt: np.ndarray,
                      velocities: np.ndarray | None = None, config: PressingConfig = PRESSING,
                      control: PitchControlConfig = PITCH_CONTROL) -> pd.DataFrame:
    """LaneCut for one presser on the given rows (e.g. from `pressing_rows`).

    The carrier on each row is the opponent nearest the ball; every other opponent
    on the pitch is a candidate receiver.
    """
    p = tracking.index(presser_id)
    team = tracking.teams[p]
    attacking, defending = np.flatnonzero(tracking.teams != team), np.flatnonzero(tracking.teams == team)
    positions = tracking.positions[rows]
    attackers, defenders = positions[:, attacking], positions[:, defending]

    to_ball = np.linalg.norm(attackers - tracking.ball[rows][:, None], axis=-1)
    carrier_k = np.where(np.isnan(to_ball), np.inf, to_ball).argmin(axis=1)
    frame = np.arange(len(rows))
    carrier = attackers[frame, carrier_k]
    receivers = attackers.copy()
    receivers[frame, carrier_k] = np.nan

    attacks_right = tracking.attacks_right(tracking.teams[attacking[0]])[rows]
    receiver_xt = xt_lookup(xt, receivers, attacks_right[:, None], PITCH_LENGTH, PITCH_WIDTH)
    weights = lane_weights(receivers, defenders, receiver_xt, control)

    velocity = (tracking.velocities() if velocities is None else velocities)[rows, p]
    now, projected = lane_cut(carrier, positions[:, p], velocity, receivers, weights, config)
    return pd.DataFrame({
        "time": tracking.timestamps[rows],
        "carrier": np.array(tracking.player_ids)[attacking[carrier_k]],
        "distance_to_carrier": np.linalg.norm(positions[:, p] - carrier, axis=-1),
        "lane_cut_now": now,
        "lane_cut_projected": projected,
        "lane_cut_gain": projected - now,
    })
