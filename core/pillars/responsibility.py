"""
Defensive Responsibility: the threat opponents' passes move into each defender's zone.

    base_k     defender k's average offset from their team's outfield centroid while
               defending, per period (where k stands in the shape)
    slot_k(t)  centroid(t) + base_k: where k's zone is at the moment of a pass
    share_k    ∝ exp(-|end - slot_k|² / 2σ²): the pass's end point is shared between
               the slots around it, so a ball into the gap between two defenders is
               split between them
    conceded_k = Σ over the opponent's completed passes  max(0, xT(end) - xT(start)) · share_k

Responsibility follows the shape, not where the defender happened to be: a full-back
caught upfield is still charged for the ball into the space they should be holding.
Lower is better. Goalkeepers are left out. Thresholds live in
core.config.ResponsibilityConfig.
"""

import warnings

import numpy as np
import pandas as pd

from core.config import RESPONSIBILITY, ResponsibilityConfig
from core.metrics import xt_added
from core.tracking import PITCH_LENGTH, PITCH_WIDTH, TrackingData

COLUMNS = ["player", "time", "passer", "xt_added", "share", "xt_conceded"]


def base_offsets(positions: np.ndarray) -> np.ndarray:
    """(P, 2) each player's mean offset from the group's centroid over the given frames ((F, P, 2))."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)        # all-NaN columns: absent players
        centroid = np.nanmean(positions, axis=1, keepdims=True)
        return np.nanmean(positions - centroid, axis=0)


def zone_shares(end: np.ndarray, slots: np.ndarray, spread: float) -> np.ndarray:
    """(D,) share of a ball played to `end` charged to each slot ((D, 2), NaN = absent)."""
    gaps = np.linalg.norm(slots - end, axis=-1)
    weight = np.where(np.isnan(gaps), 0.0, np.exp(-0.5 * (np.nan_to_num(gaps) / spread) ** 2))
    if weight.sum() < 1e-12:                    # far from everyone: the nearest slot owns it
        weight = np.where(gaps == np.nanmin(gaps), 1.0, 0.0) if np.isfinite(gaps).any() else weight
    total = weight.sum()
    return weight / total if total > 0 else weight


def defensive_responsibility(tracking: TrackingData, events: pd.DataFrame, possession: np.ndarray,
                             xt: np.ndarray, outfield: np.ndarray | None = None,
                             config: ResponsibilityConfig = RESPONSIBILITY) -> pd.DataFrame:
    """One row per (opponent completed pass, responsible defender); columns as COLUMNS.

    `outfield` (P,) bool marks players who can be charged (default: everyone); pass
    the roster's non-goalkeepers. Passes that lose xT charge nobody.
    """
    outfield = np.ones(len(tracking.player_ids), dtype=bool) if outfield is None else outfield
    ids = np.array(tracking.player_ids)
    bases = {}
    for team in ("Home", "Away"):
        members = np.flatnonzero((tracking.teams == team) & outfield)
        for period in np.unique(tracking.periods):
            defending = np.flatnonzero((tracking.periods == period) & (possession != team) & pd.notna(possession))
            bases[team, period] = (members, base_offsets(tracking.positions[defending][:, members])
                                   if len(defending) else np.full((len(members), 2), np.nan))

    passes = events[(events["Type"] == "PASS") & events["To"].notna()]
    rows = []
    for _, ball in passes.iterrows():
        row = min(int(tracking.frame_index([ball["End Frame"]])[0]), len(tracking.frames) - 1)
        team = "Away" if ball["Team"] == "Home" else "Home"           # the defending side
        start = np.array([ball["Start X"], ball["Start Y"]])
        end = np.array([ball["End X"], ball["End Y"]])
        gained = float(xt_added(xt, start, end, bool(tracking.attacks_right(ball["Team"])[row]),
                                PITCH_LENGTH, PITCH_WIDTH))
        if not gained > 0:
            continue
        members, offsets = bases[team, tracking.periods[row]]
        here = tracking.positions[row, members]
        on = ~np.isnan(here).any(axis=1)
        if not on.any():
            continue
        slots = np.where(on[:, None], here[on].mean(axis=0) + offsets, np.nan)
        share = zone_shares(end, slots, config.spread)
        for k in np.flatnonzero(share > 1e-3):
            rows.append((ids[members[k]], float(ball["End Time [s]"]), ball["From"], gained,
                         float(share[k]), gained * float(share[k])))
    return pd.DataFrame(rows, columns=COLUMNS)
