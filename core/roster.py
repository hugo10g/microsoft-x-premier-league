"""
Roles and identities for tracked players.

    match_minutes           the match clock for every tracking frame
    infer_positions         each player's average depth and side, when they were on,
                            and a broad position from how deep they play
    tracking_substitutions  who came on for whom, from when players appear and vanish
    static_timeline         who each tracked player is, from when to when, with their
                            name, club, position and value (from the match roster)
    attribute               credit each event to the name in force at its minute
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from core.config import ROSTER, RosterConfig
from core.tracking import PITCH_LENGTH, PITCH_WIDTH, TrackingData

POSITIONS = ["Goalkeeper", "Defender", "Midfielder", "Forward"]   # back to front


def match_minutes(tracking: TrackingData) -> np.ndarray:
    """(T,) match clock in minutes: 0-45 in the first half, from 45 in the second."""
    minutes = np.zeros(len(tracking.timestamps))
    for period in np.unique(tracking.periods):
        rows = tracking.periods == period
        start = tracking.timestamps[rows][0]
        minutes[rows] = 45 * (period - 1) + (tracking.timestamps[rows] - start) / 60
    return minutes


def infer_positions(tracking: TrackingData, config: RosterConfig = ROSTER) -> pd.DataFrame:
    """Each player's average depth and side, when they were on, and a broad position.

    Depth is distance from their own goal line and lateral runs from -1 (right
    touchline) to +1 (left touchline) in the direction they attack, both using
    `attacks_right` so they're comparable across halves. Per team, the deepest
    player is the goalkeeper; the outfielders are ranked by depth and split
    defender / midfielder / forward by `config` shares.
    """
    on_pitch = ~np.isnan(tracking.positions).any(axis=-1)
    minutes = match_minutes(tracking)
    rows = []
    for team in np.unique(tracking.teams):
        right = tracking.attacks_right(team)[:, None]
        x, y = tracking.positions[..., 0], tracking.positions[..., 1]
        depth = np.where(right, x, PITCH_LENGTH - x)
        lateral = np.where(right, 1, -1) * (y - PITCH_WIDTH / 2) / (PITCH_WIDTH / 2)
        for p in np.flatnonzero(tracking.team_mask(team)):
            frames = np.flatnonzero(on_pitch[:, p])
            if len(frames):
                rows.append({"player": tracking.player_ids[p], "team": team,
                             "depth": float(np.nanmean(depth[:, p])),
                             "lateral": float(np.nanmean(lateral[:, p])),
                             "starter": bool(on_pitch[0, p]),
                             "minute_on": np.nan if on_pitch[0, p] else minutes[frames[0]],
                             "minute_off": np.nan if on_pitch[-1, p] else minutes[frames[-1]]})
    table = pd.DataFrame(rows)

    table["role"] = "Midfielder"
    for _, team_rows in table.groupby("team"):
        order = team_rows["depth"].rank(method="first").astype(int)     # 1 = deepest
        outfield = len(team_rows) - 1
        defenders = round(config.defender_share * outfield)
        forwards = round(config.forward_share * outfield)
        table.loc[order.index[order == 1], "role"] = "Goalkeeper"
        table.loc[order.index[(order > 1) & (order <= 1 + defenders)], "role"] = "Defender"
        table.loc[order.index[order > len(team_rows) - forwards], "role"] = "Forward"
    return table


def tracking_substitutions(profiles: pd.DataFrame) -> pd.DataFrame:
    """(team, off, on, minute) for every substitution visible in the tracking data.

    A substitute appears on the frame their teammate disappears, so each player
    coming on is paired with the player of the same team whose exit is closest in time.
    """
    pairs = []
    for team, rows in profiles.groupby("team"):
        on = rows[rows["minute_on"].notna()]
        off = rows[rows["minute_off"].notna()]
        if on.empty or off.empty:
            continue
        gap = np.abs(on["minute_on"].to_numpy()[:, None] - off["minute_off"].to_numpy()[None])
        for i, j in zip(*linear_sum_assignment(gap)):
            pairs.append({"team": team, "off": off["player"].iloc[j], "on": on["player"].iloc[i],
                          "minute": on["minute_on"].iloc[i]})
    return pd.DataFrame(pairs, columns=["team", "off", "on", "minute"])


def formation(positions) -> str:
    """'4-3-3' style string from the broad positions of the players on the pitch."""
    counts = pd.Series(list(positions)).value_counts()
    return "-".join(str(counts.get(pos, 0)) for pos in POSITIONS[1:])


def static_timeline(profiles: pd.DataFrame, names: pd.DataFrame | None = None,
                    key: str = "player") -> pd.DataFrame:
    """A timeline where each tracked player keeps one identity while on the pitch.

    `names` (joined on `key`) supplies name, club, position and value, e.g. from
    each match's roster; without it the tracking id stands in for the name. A player still
    on at the final whistle ends at the profile's `last_minute` when it's known.
    """
    table = profiles.copy()
    final = table["last_minute"] if "last_minute" in table else np.inf
    table["start"] = table["minute_on"].fillna(0.0)
    table["end"] = table["minute_off"].fillna(final)
    if names is None:
        table["name"], table["market_value_m"], table["real_team"] = table[key], np.nan, None
        table["position"] = table["detailed_position"] = table["role"]
    else:
        table = table.merge(names, on=key)
    table["formation"] = None
    keep = ["player", "team", "start", "end", "name", "position", "detailed_position",
            "market_value_m", "real_team", "formation", "depth"]
    return table[keep + [c for c in ("game", "uid") if c in table and c not in keep]]


def active(timeline: pd.DataFrame, minute: float) -> pd.DataFrame:
    """Timeline rows in force at a match minute."""
    return timeline[(timeline["start"] <= minute) & (minute < timeline["end"])]


def attribute(events: pd.DataFrame, timeline: pd.DataFrame, on: str = "player") -> pd.DataFrame:
    """Add the name in force for each event's (`on`, minute), dropping unmatched events.

    `on` is "player" within one game, or "uid" when events span several games.
    """
    joined = events.merge(timeline[[on, "start", "end", "name"]], on=on)
    return joined[(joined["start"] <= joined["minute"]) & (joined["minute"] < joined["end"])] \
        .drop(columns=["start", "end"])
