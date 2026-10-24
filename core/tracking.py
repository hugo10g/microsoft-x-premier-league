"""
Metrica Sports tracking and event data -> the arrays the tracking metrics use.

Metrica's CSV format (sample data: github.com/metrica-sports/sample-data):
    one tracking file per team, three header rows (team, jersey, column name),
    then Period, Frame, Time [s] and an x, y column pair per player and the ball.
    Coordinates are fractions of the pitch with the origin at the top-left corner.

We convert to metres with the origin at the bottom-left corner and y pointing up,
the convention core.pose assumes. Metrica doesn't flip coordinates at half-time,
so which way a team attacks depends on the period (see `attacks_right`).

Head orientation, where a feed has it, comes in a separate <name>_HeadOrientation.csv:
Period, Frame, Time [s], then one column per tracking id (e.g. Home_7) holding head
yaw in degrees in the same pitch frame (0 = facing +x, counter-clockwise positive).
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import KINEMATICS, KinematicsConfig
from core.kinematics import velocity

PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0   # metres; Metrica's sample data assumes 105 x 68

# Events after which a team is in control of the ball, and ones after which nobody is.
POSSESSION_EVENTS = {"PASS", "SHOT", "RECOVERY", "BALL LOST", "SET PIECE"}
DEAD_BALL_EVENTS = {"BALL OUT", "FAULT RECEIVED", "OFFSIDE"}   # the referee stops play


@dataclass
class TrackingData:
    """One match of tracking, every array aligned on the frame axis."""
    frames: np.ndarray        # (T,) Metrica frame numbers
    timestamps: np.ndarray    # (T,) seconds
    periods: np.ndarray       # (T,) 1, 2, ...
    positions: np.ndarray     # (T, P, 2) metres, NaN while a player is off the pitch
    ball: np.ndarray          # (T, 2) metres
    player_ids: list[str]     # (P,) e.g. "Home_11"
    teams: np.ndarray         # (P,) "Home" / "Away"

    def team_mask(self, team: str) -> np.ndarray:
        return self.teams == team

    def index(self, player_id: str) -> int:
        return self.player_ids.index(player_id)

    def period(self, period: int) -> "TrackingData":
        """The frames of one period only."""
        keep = self.periods == period
        return TrackingData(self.frames[keep], self.timestamps[keep], self.periods[keep],
                            self.positions[keep], self.ball[keep], self.player_ids, self.teams)

    def frame_index(self, frames) -> np.ndarray:
        """Row index of each Metrica frame number."""
        return np.searchsorted(self.frames, frames)

    def attacks_right(self, team: str) -> np.ndarray:
        """(T,) True on frames where `team` attacks towards x = PITCH_LENGTH.

        At each period's kick-off both teams stand in their own half, so a team
        whose players start on the left is attacking to the right.
        """
        right = np.zeros(len(self.frames), dtype=bool)
        players = self.team_mask(team)
        for period in np.unique(self.periods):
            rows = np.flatnonzero(self.periods == period)
            start_x = np.nanmean(self.positions[rows[0], players, 0])
            right[rows] = start_x < PITCH_LENGTH / 2
        return right

    def velocities(self, config: KinematicsConfig = KINEMATICS) -> np.ndarray:
        """(T, P, 2) m/s, computed per period so nothing is smoothed across half-time."""
        out = np.full(self.positions.shape, np.nan)
        for period in np.unique(self.periods):
            rows = self.periods == period
            out[rows] = velocity(self.positions[rows], self.timestamps[rows], config)
        return out


def to_metres(xy: np.ndarray) -> np.ndarray:
    """Metrica fractions (origin top-left, y down) -> metres (origin bottom-left, y up)."""
    return np.stack([xy[..., 0] * PITCH_LENGTH, (1 - xy[..., 1]) * PITCH_WIDTH], axis=-1)


def read_team(path: Path) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    """One team's tracking file: (frame columns, player ids, (T, P, 2) positions, (T, 2) ball)."""
    with open(path, newline="") as f:
        team_row, _, name_row = (next(csv.reader([f.readline()])) for _ in range(3))
    team = next(name for name in team_row if name)

    data = pd.read_csv(path, skiprows=3, header=None).to_numpy(dtype=float)
    # Each name sits above its x column; the y column next to it is unnamed.
    columns = {name: i for i, name in enumerate(name_row) if i >= 3 and name}
    players = [name for name in columns if name != "Ball"]
    xy = np.stack([data[:, [columns[p], columns[p] + 1]] for p in players], axis=1)
    ball = data[:, [columns["Ball"], columns["Ball"] + 1]]

    meta = pd.DataFrame({"period": data[:, 0].astype(int), "frame": data[:, 1].astype(int),
                         "time": data[:, 2]})
    # Metrica occasionally writes "Player 26" with a space; ids must not depend on it.
    ids = [f"{team}_{p.removeprefix('Player').strip()}" for p in players]
    return meta, ids, to_metres(xy), to_metres(ball)


def load_tracking(home_path: Path, away_path: Path) -> TrackingData:
    """Both teams' tracking files merged into one TrackingData."""
    home_meta, home_ids, home_xy, ball = read_team(home_path)
    away_meta, away_ids, away_xy, _ = read_team(away_path)
    if not np.array_equal(home_meta["frame"], away_meta["frame"]):
        raise ValueError("Home and away tracking files cover different frames")
    return TrackingData(
        frames=home_meta["frame"].to_numpy(),
        timestamps=home_meta["time"].to_numpy(),
        periods=home_meta["period"].to_numpy(),
        positions=np.concatenate([home_xy, away_xy], axis=1),
        ball=ball,
        player_ids=home_ids + away_ids,
        teams=np.array([i.split("_")[0] for i in home_ids + away_ids]),
    )


def load_sample_game(game_dir: Path) -> tuple[TrackingData, pd.DataFrame]:
    """Tracking and events for a game folder in Metrica's layout (e.g. a data/synthetic/ match)."""
    name = game_dir.name
    tracking = load_tracking(game_dir / f"{name}_RawTrackingData_Home_Team.csv",
                             game_dir / f"{name}_RawTrackingData_Away_Team.csv")
    return tracking, load_events(game_dir / f"{name}_RawEventsData.csv")


def load_head_orientation(game_dir: Path, tracking: TrackingData) -> np.ndarray:
    """(T, P) head yaw in degrees, aligned to `tracking` by frame and ordered like its player_ids.

    All NaN when the game has no head-orientation file; NaN for any frame or player
    the file doesn't cover.
    """
    yaw = np.full((len(tracking.frames), len(tracking.player_ids)), np.nan)
    path = game_dir / f"{game_dir.name}_HeadOrientation.csv"
    if not path.exists():
        return yaw
    table = pd.read_csv(path)
    if table.empty:
        return yaw
    file_frames = table["Frame"].to_numpy()
    order = np.argsort(file_frames, kind="stable")
    at = np.clip(np.searchsorted(file_frames[order], tracking.frames), 0, len(file_frames) - 1)
    rows = order[at]
    matched = file_frames[rows] == tracking.frames
    for p, player in enumerate(tracking.player_ids):
        if player in table:
            yaw[matched, p] = table[player].to_numpy(dtype=float)[rows[matched]]
    return yaw


def load_events(path: Path) -> pd.DataFrame:
    """Metrica events with player ids matching TrackingData and locations in metres."""
    events = pd.read_csv(path)
    for column in ("From", "To"):
        events[column] = np.where(events[column].notna(),
                                  events["Team"] + "_" + events[column].str.removeprefix("Player").str.strip(),
                                  None)
    for end in ("Start", "End"):
        events[[f"{end} X", f"{end} Y"]] = to_metres(events[[f"{end} X", f"{end} Y"]].to_numpy())
    return events


def possession_by_frame(events: pd.DataFrame, frames: np.ndarray) -> np.ndarray:
    """(T,) team in possession at each frame; None before the first event and while the ball is out.

    Possession belongs to whoever made the most recent on-ball event, until the ball
    goes out of play.
    """
    relevant = events[events["Type"].isin(POSSESSION_EVENTS | DEAD_BALL_EVENTS)]
    relevant = relevant.sort_values("Start Frame", kind="stable")
    owner = np.where(relevant["Type"].isin(DEAD_BALL_EVENTS), None, relevant["Team"]).astype(object)
    latest = np.searchsorted(relevant["Start Frame"].to_numpy(), frames, side="right") - 1
    return np.where(latest >= 0, owner[np.clip(latest, 0, None)], None)


def receptions(events: pd.DataFrame) -> pd.DataFrame:
    """Every completed pass's receiver and the frame/time they received it."""
    passes = events[(events["Type"] == "PASS") & events["To"].notna()]
    return pd.DataFrame({"player": passes["To"], "frame": passes["End Frame"],
                         "time": passes["End Time [s]"]}).reset_index(drop=True)
