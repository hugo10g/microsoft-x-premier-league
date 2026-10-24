"""
Ball Receipts in Space: how much room a player has at the moment they receive a pass.

    space = distance (m) from the receiver to the nearest opponent on the reception frame

A receipt is "in space" when that distance is at least ReceiptConfig.in_space.
Finding space before the ball arrives is off-ball work: the receiver chose where
to be. Thresholds live in core.config.ReceiptConfig.
"""

import numpy as np
import pandas as pd

from core.config import RECEIPTS, ReceiptConfig
from core.tracking import TrackingData, receptions

COLUMNS = ["player", "time", "space", "in_space"]


def nearest_opponent(tracking: TrackingData, player: str, row: int) -> float:
    """Distance (m) from `player` to the nearest opponent on tracking row `row` (NaN if unknown)."""
    p = tracking.index(player)
    opponents = tracking.positions[row, tracking.teams != tracking.teams[p]]
    gaps = np.linalg.norm(opponents - tracking.positions[row, p], axis=-1)
    return float(np.nanmin(gaps)) if np.isfinite(gaps).any() else np.nan


def receipts_in_space(tracking: TrackingData, events: pd.DataFrame,
                      config: ReceiptConfig = RECEIPTS) -> pd.DataFrame:
    """One row per completed pass reception: the space the receiver had (columns as COLUMNS)."""
    received = receptions(events)
    received = received[received["player"].isin(tracking.player_ids)]
    rows = np.clip(tracking.frame_index(received["frame"].to_numpy()), 0, len(tracking.frames) - 1)
    space = [nearest_opponent(tracking, player, row) for player, row in zip(received["player"], rows)]
    table = pd.DataFrame({"player": received["player"].to_numpy(), "time": received["time"].to_numpy(),
                          "space": space})
    table["in_space"] = table["space"] >= config.in_space
    return table[COLUMNS]
