"""
Off-Ball Scoring Opportunity (OBSO): the threat an attacker offers by where they stand.

On every sampled frame their team has the ball, for each attacker not on it:

    threat   = xT(p)                                 how dangerous their spot is
    control  = tanh(d_opp / (2 · softness))          0 when marked tight, → 1 when free
    reach    = exp(-|p - ball| / reach)              how likely the ball can get there
    OBSO     = threat · control · reach · Δt

summed over the match: xT-seconds of opportunity, whether or not the ball ever
came. A striker who holds a free spot between the centre-backs piles it up; one
standing marked or out of range of the ball doesn't. After Spearman's (2018)
OBSO = P(score) · P(control) · P(ball reaches), with xT for P(score) and a
distance model for the other two. Thresholds live in core.config.OBSOConfig.
"""

import numpy as np
import pandas as pd

from core.config import OBSO, PITCH_CONTROL, OBSOConfig, PitchControlConfig
from core.kinematics import sample_interval
from core.spatial import nearest_distance, xt_lookup
from core.tracking import PITCH_LENGTH, PITCH_WIDTH, TrackingData

COLUMNS = ["player", "time", "obso"]


def frame_obso(attackers: np.ndarray, defenders: np.ndarray, ball: np.ndarray, attacker_xt: np.ndarray,
               config: OBSOConfig = OBSO, control: PitchControlConfig = PITCH_CONTROL) -> np.ndarray:
    """(..., A) per-second OBSO of each attacker: xT x control x reach (NaN players give 0).

    attackers (..., A, 2), defenders (..., D, 2), ball (..., 2), attacker_xt (..., A).
    """
    owned = np.tanh(nearest_distance(attackers, defenders) / (2 * control.softness))
    reach = np.exp(-np.linalg.norm(attackers - ball[..., None, :], axis=-1) / config.reach)
    return np.nan_to_num(attacker_xt * owned * reach)


def obso(tracking: TrackingData, possession: np.ndarray, xt: np.ndarray, config: OBSOConfig = OBSO,
         control: PitchControlConfig = PITCH_CONTROL) -> pd.DataFrame:
    """OBSO of every off-ball attacker on every sampled in-possession frame (columns as COLUMNS).

    Each row is worth `config.every` seconds; the player nearest the ball (on it) is skipped.
    """
    step = max(1, round(config.every / sample_interval(tracking.timestamps)))
    sampled = np.arange(0, len(tracking.frames), step)
    ids = np.array(tracking.player_ids)
    tables = []
    for team in ("Home", "Away"):
        frames = sampled[(possession[sampled] == team) & ~np.isnan(tracking.ball[sampled]).any(axis=1)]
        if not len(frames):
            continue
        members, rivals = np.flatnonzero(tracking.teams == team), np.flatnonzero(tracking.teams != team)
        attackers, ball = tracking.positions[frames][:, members], tracking.ball[frames]
        right = tracking.attacks_right(team)[frames]
        threat = xt_lookup(xt, attackers, right[:, None], PITCH_LENGTH, PITCH_WIDTH)
        value = frame_obso(attackers, tracking.positions[frames][:, rivals], ball, threat, config, control)
        on_ball = np.linalg.norm(attackers - ball[:, None], axis=-1)
        on_ball = np.where(np.isnan(on_ball), np.inf, on_ball).argmin(axis=1)
        present = ~np.isnan(attackers).any(axis=-1)
        present[np.arange(len(frames)), on_ball] = False
        f, a = np.nonzero(present)
        tables.append(pd.DataFrame({"player": ids[members[a]], "time": tracking.timestamps[frames[f]],
                                    "obso": value[f, a] * step * sample_interval(tracking.timestamps)}))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=COLUMNS)
