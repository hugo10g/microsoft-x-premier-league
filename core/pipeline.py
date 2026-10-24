"""
Batch scoring: every game in data/synthetic/ (Metrica's CSV layout) through the tracking pillars.

Results are kept at event level (one row per decoy run, pressing frame,
transition, pass reception, pressure, classified run and zone charge, plus OBSO
in 10-second buckets, each with its game, Metrica player and match minute), so
they can be
credited to whichever name a player carried at that minute and then totalled per
player, per game or across all games.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from core import metrics, roster, tracking
from core.spatial import attacking_control, make_grid
from core.kinematics import sample_interval
from core.pillars import obso, pressing, pressure, receipts, responsibility, run_types, scanning, space, transition

GAMES_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"
# Bump when the shape of score_game's output changes, so cached results are rebuilt.
VERSION = 8
PILLARS = ["decoy_p90", "lane_cut_p90", "transition_score", "scan_rate"]
SCANNING_COLUMNS = ["player", "time", "minute", "n_scans", "scan_rate", "ball_fixation", "retained"]
OBSO_BUCKET = 10.0         # seconds of OBSO summed into one row
# The five tactical metrics player_totals adds (per 90 where it says so).
TACTICAL_COLUMNS = ["pressures", "pressure_intensity", "press_closing_kmh", "receipt_space",
                    "receipts_in_space_pct", "def_responsibility_p90", "obso_p90",
                    "runs_in_behind_p90", "pulling_wide_p90", "dropping_deep_p90"]
RUN_TYPE_COLUMNS = {"Run in Behind": "runs_in_behind_p90", "Pulling Wide": "pulling_wide_p90",
                    "Dropping Deep": "dropping_deep_p90"}


SCORED_FILE = GAMES_DIR / "scored.pkl"   # score_all's tables, saved by scripts/precompute_scores.py


def games_signature(games: list[Path]) -> tuple:
    """What scored tables depend on: VERSION plus each game's name and newest CSV time (whole seconds).

    Whole seconds survive copying the league into a container image; a regenerated
    game or a VERSION bump changes the signature, so stale tables are never loaded.
    """
    return (("pipeline", VERSION),) + tuple(
        (game.name, int(max(f.stat().st_mtime for f in game.glob("*.csv")))) for game in games)


def save_scored(signature: tuple, tables: dict[str, pd.DataFrame], path: Path = SCORED_FILE) -> None:
    pd.to_pickle((signature, tables), path)


def load_scored(signature: tuple, path: Path = SCORED_FILE) -> dict[str, pd.DataFrame] | None:
    """The saved score_all tables if they were made for `signature`, else None (score afresh)."""
    if not path.exists():
        return None
    saved, tables = pd.read_pickle(path)
    return tables if saved == signature else None


def available_games(root: Path = GAMES_DIR) -> list[Path]:
    """Game folders in Metrica's CSV layout (tracking per team plus events)."""
    def complete(folder: Path) -> bool:
        name = folder.name
        return all((folder / f"{name}_{suffix}.csv").exists() for suffix in
                   ("RawTrackingData_Home_Team", "RawTrackingData_Away_Team", "RawEventsData"))
    return sorted(folder for folder in root.iterdir() if folder.is_dir() and complete(folder)) \
        if root.exists() else []


def load_roster(game_dir: Path) -> pd.DataFrame:
    """A game's <name>_roster.csv: tracking id -> name, club, position, market value."""
    return pd.read_csv(game_dir / f"{game_dir.name}_roster.csv")


def xt_training_events(game_dir: Path) -> pd.DataFrame:
    """One game's passes, carries and shots in the layout core.metrics.build_xt learns from.

    That layout is StatsBomb's: a 120 x 80 pitch, y down, every action drawn as if
    its team attacks left to right. Carries are inferred: from where a player
    receives a pass to where they make their next action.
    """
    tr, events = tracking.load_sample_game(game_dir)
    starts = {p: np.flatnonzero(tr.periods == p)[0] for p in np.unique(tr.periods)}
    right = {(team, p): bool(tr.attacks_right(team)[row]) for team in ("Home", "Away")
             for p, row in starts.items()}

    def frame_of(row, end: str):
        r = right[(row["Team"], row["Period"])]
        x, y = row[f"{end} X"], row[f"{end} Y"]
        return [(x if r else tracking.PITCH_LENGTH - x) / tracking.PITCH_LENGTH * 120,
                ((tracking.PITCH_WIDTH - y) if r else y) / tracking.PITCH_WIDTH * 80]

    rows = []
    actions = events[events["Type"].isin(["PASS", "BALL LOST", "SHOT"])].reset_index(drop=True)
    for k, row in actions.iterrows():
        if row["Type"] == "SHOT":
            rows.append({"type": "Shot", "location": frame_of(row, "Start"), "shot_type": "Open Play",
                         "shot_outcome": "Goal" if "GOAL" in str(row["Subtype"]) else "Saved"})
            continue
        if row["Type"] == "BALL LOST" and row["Subtype"] != "INTERCEPTION":
            continue
        completed = row["Type"] == "PASS" and pd.notna(row["To"])
        rows.append({"type": "Pass", "location": frame_of(row, "Start"), "pass_end_location": frame_of(row, "End"),
                     "pass_outcome": np.nan if completed else "Incomplete", "pass_type": np.nan})
        following = actions.iloc[k + 1] if k + 1 < len(actions) else None
        if completed and following is not None and following["From"] == row["To"]:
            carry_end = frame_of(following, "Start")
            rows.append({"type": "Carry", "location": frame_of(row, "End"), "carry_end_location": carry_end})
    columns = ["type", "location", "pass_end_location", "pass_outcome", "pass_type",
               "carry_end_location", "shot_type", "shot_outcome"]
    return pd.DataFrame(rows).reindex(columns=columns)


def learn_xt(games: list[Path] | None = None) -> np.ndarray:
    """Expected Threat learned from the games' own passes, carries and shots."""
    games = available_games() if games is None else games
    return metrics.build_xt(pd.concat([xt_training_events(g) for g in games], ignore_index=True))


def space_control(tr: tracking.TrackingData, possession: np.ndarray, every: float = 1.0,
                  cell: float = 2.0) -> pd.DataFrame:
    """Pitch Control %: the share of the pitch each attacker controls, sampled every `every` s.

    On each sampled in-possession frame, the attacking team's soft-Voronoi control
    (the same model as core.spatial.attacking_control) is split among its players:
    each grid cell goes to the attacker nearest to it. A player's sample is the
    area they control as a percentage of the whole pitch.
    """
    grid, _ = make_grid(tracking.PITCH_LENGTH, tracking.PITCH_WIDTH, cell)
    step = max(1, round(every / sample_interval(tr.timestamps)))
    rows = []
    for frame in range(0, len(tr.timestamps), step):
        owner = possession[frame]
        if not isinstance(owner, str):
            continue
        on = ~np.isnan(tr.positions[frame]).any(axis=1)
        attackers = np.flatnonzero(on & (tr.teams == owner))
        defenders = np.flatnonzero(on & (tr.teams != owner))
        if not len(attackers) or not len(defenders):
            continue
        control = attacking_control(tr.positions[frame, attackers], tr.positions[frame, defenders], grid)
        nearest = cdist(grid, tr.positions[frame, attackers]).argmin(axis=1)
        share = np.bincount(nearest, weights=control, minlength=len(attackers))
        rows.append(pd.DataFrame({"player": np.array(tr.player_ids)[attackers], "frame": frame,
                                  "control_pct": 100 * share / len(grid)}))
    return pd.concat(rows, ignore_index=True)


def open_play_turnovers(tr: tracking.TrackingData, possession: np.ndarray):
    """Turnovers detected one period at a time, so half-time never counts as one."""
    found = []
    for period in np.unique(tr.periods):
        rows = tr.periods == period
        found += transition.detect_turnovers(tr.timestamps[rows], possession[rows])
    return found


def reception_scans(tr: tracking.TrackingData, events: pd.DataFrame, yaw: np.ndarray) -> pd.DataFrame:
    """One row per completed pass reception: the receiver's scanning before it, and whether they kept the ball.

    `retained` is True when the next on-ball action (pass, shot or ball lost) is the
    receiver's own completed pass or shot; a loss, a pass out of play or anyone
    else's action is False. `yaw` is (T, P) head yaw aligned with `tr`.
    """
    actions = events[events["Type"].isin(["PASS", "SHOT", "BALL LOST"])] \
        .sort_values("Start Frame", kind="stable").reset_index(drop=True)
    following = actions.shift(-1)
    kept = (following["From"] == actions["To"]) & (
        ((following["Type"] == "PASS") & following["To"].notna()) | (following["Type"] == "SHOT"))
    received = (actions["Type"] == "PASS") & actions["To"].isin(tr.player_ids)
    rows = []
    for player, group in actions[received].groupby("To", sort=False):
        p = tr.index(player)
        for k, t in zip(group.index, group["End Time [s]"]):
            w = scanning.scans_before_reception(tr.timestamps, yaw[:, p], tr.positions[:, p], tr.ball, float(t))
            rows.append((k, player, float(t), w.n_scans, w.scan_rate, w.ball_fixation, bool(kept[k])))
    table = pd.DataFrame(rows, columns=["order", "player", "time", "n_scans", "scan_rate",
                                        "ball_fixation", "retained"])
    table["n_scans"] = table["n_scans"].astype(int)
    table["retained"] = table["retained"].astype(bool)
    return table.sort_values("order").drop(columns="order").reset_index(drop=True)


def score_game(game_dir: Path, xt: np.ndarray) -> dict[str, pd.DataFrame]:
    """Every scored event in one game, plus each player's profile and minutes played.

    Returns tables keyed "profiles", "decoys", "pressing", "transitions", "control",
    "scanning", "pressures", "receipts", "responsibility", "obso" and "runs", each with
    `game` and `player` columns; event tables also carry the
    tracking `time` (seconds) and the match `minute`. Without a head-orientation file
    the scanning rows are still there, with no scans and a NaN rate.
    """
    tr, events = tracking.load_sample_game(game_dir)
    vel = tr.velocities()
    possession = tracking.possession_by_frame(events, tr.frames)
    minutes = roster.match_minutes(tr)
    clock = lambda seconds: np.interp(seconds, tr.timestamps, minutes)
    game = game_dir.name

    profiles = roster.infer_positions(tr)
    on_pitch = ~np.isnan(tr.positions).any(axis=-1)
    played = dict(zip(tr.player_ids, on_pitch.sum(axis=0) * sample_interval(tr.timestamps) / 60))
    profiles["minutes"] = profiles["player"].map(played)
    profiles["last_minute"] = minutes[-1]

    decoys = space.decoy_runs(tr, possession, xt, vel)
    decoys = pd.DataFrame({"player": decoys["player"], "time": decoys["start_time"],
                           "minute": clock(decoys["start_time"]), "duration": decoys["duration"],
                           "decoy_value": decoys["decoy_value"]})

    press = []
    for player in tr.player_ids:
        rows = pressing.pressing_rows(tr, player, possession)
        if len(rows):
            cuts = pressing.pressing_lane_cut(tr, player, rows, xt, vel)
            press.append(pd.DataFrame({"player": player, "time": cuts["time"].to_numpy(),
                                       "minute": clock(cuts["time"]),
                                       "lane_cut": cuts["lane_cut_projected"].to_numpy(),
                                       "seconds": sample_interval(tr.timestamps)}))

    found = open_play_turnovers(tr, possession)
    at = np.searchsorted(tr.timestamps, [t for t, _ in found])
    ball_distances = np.linalg.norm(tr.positions[at] - tr.ball[at][:, None], axis=-1)
    speeds = np.linalg.norm(vel, axis=-1) * 3.6
    moves = transition.transition_latencies(tr.timestamps, speeds, tr.player_ids, tr.teams,
                                            found, ball_distances)
    moves = moves[["player", "phase", "latency", "reached_sprint", "score"]].assign(
        time=moves["t_turnover"], minute=clock(moves["t_turnover"]))

    control = space_control(tr, possession)
    control = control.assign(minute=minutes[control["frame"]]).drop(columns="frame")

    scans = reception_scans(tr, events, tracking.load_head_orientation(game_dir, tr))
    scans = scans.assign(minute=clock(scans["time"]))[SCANNING_COLUMNS]

    keepers = set(load_roster(game_dir).query("position == 'Goalkeeper'")["player"])
    outfield = np.array([pid not in keepers for pid in tr.player_ids])
    pressed = pressure.pressures(tr, possession, vel)
    pressed = pressed.rename(columns={"start_time": "time"}).assign(minute=clock(pressed["start_time"]))
    received = receipts.receipts_in_space(tr, events)
    received = received.assign(minute=clock(received["time"]))
    charged = responsibility.defensive_responsibility(tr, events, possession, xt, outfield)
    charged = charged.assign(minute=clock(charged["time"]))
    offered = obso.obso(tr, possession, xt)
    offered = offered.groupby(["player", (offered["time"] // OBSO_BUCKET) * OBSO_BUCKET], as_index=False)["obso"].sum()
    offered = offered.assign(minute=clock(offered["time"]))
    typed = run_types.run_types(tr, possession, vel, outfield)
    typed = typed.assign(minute=clock(typed["time"]))

    tables = {"profiles": profiles, "decoys": decoys,
              "pressing": pd.concat(press, ignore_index=True), "transitions": moves, "control": control,
              "scanning": scans, "pressures": pressed, "receipts": received, "responsibility": charged,
              "obso": offered, "runs": typed}
    return {key: table.assign(game=game) for key, table in tables.items()}


def score_all(xt: np.ndarray, games: list[Path] | None = None, progress=None) -> dict[str, pd.DataFrame]:
    """`score_game` for every available game, concatenated into global tables.

    Every table gains a `uid` column ("<game>:<player>"): Metrica's ids repeat
    across games and nothing says Home_9 in one game is Home_9 in the next, so each
    game's players are treated as different anonymous players.
    """
    games = available_games() if games is None else games
    per_game = []
    for n, game_dir in enumerate(games, 1):
        per_game.append(score_game(game_dir, xt))
        if progress:
            progress(n, len(games))
    return {key: pd.concat([g[key] for g in per_game], ignore_index=True)
            .assign(uid=lambda t: t["game"] + ":" + t["player"]) for key in per_game[0]}


def player_totals(events: dict[str, pd.DataFrame], timeline: pd.DataFrame,
                  on: str = "player") -> pd.DataFrame:
    """Per-name totals for each pillar, crediting every event to the name in force.

    `timeline` (core.roster) says which name each tracked player carried when; `on`
    is the column linking events to it ("player" within one game, "uid" across games).
    """
    spans = timeline.assign(minutes=timeline["end"] - timeline["start"])
    table = spans.groupby("name").agg(
        team=("team", "first"), real_team=("real_team", "first"), minutes=("minutes", "sum"),
        market_value_m=("market_value_m", "first"),
        metrica_ids=(on, lambda ids: ", ".join(dict.fromkeys(ids))))
    # The position a name spent longest in (it can change with the formation).
    table["position"] = spans.sort_values("minutes").groupby("name")["detailed_position"].last()
    table["broad_position"] = spans.sort_values("minutes").groupby("name")["position"].last()

    credit = lambda key: roster.attribute(events[key], timeline, on=on).groupby("name")
    decoys = credit("decoys")
    table = table.join(decoys.agg(decoy_runs=("decoy_value", "size"), decoy_value=("decoy_value", "sum")))
    press = roster.attribute(events["pressing"], timeline, on=on)
    press = press.assign(cut_seconds=press["lane_cut"] * press["seconds"]).groupby("name")
    table = table.join(press.agg(lane_cut_seconds=("cut_seconds", "sum"), mean_lane_cut=("lane_cut", "mean")))
    moves = roster.attribute(events["transitions"], timeline, on=on)
    table = table.join(moves.groupby("name").agg(
        transitions=("latency", "size"), sprint_rate=("reached_sprint", "mean"),
        transition_score=("score", "mean")))
    # Most turnovers never draw a sprint, so a median over all of them is just the
    # horizon cap. Latency is reported over the turnovers that did; sprint rate and
    # the score carry the ones that didn't.
    table = table.join(moves[moves["reached_sprint"]].groupby("name")["latency"].median()
                       .rename("median_latency"))
    if "control" in events:
        table = table.join(credit("control")["control_pct"].mean().rename("pitch_control_pct"))
    # Scanning: rates only where the head was seen through the window, so a missing
    # feed reads as unknown (NaN), never as a player who doesn't scan.
    scan_columns = ["receptions", "scan_rate", "pre_reception_scans", "retention_rate"]
    if "scanning" in events and len(events["scanning"]):
        scans = roster.attribute(events["scanning"], timeline, on=on)
        seen = scans[scans["scan_rate"].notna()]
        table = table.join(scans.groupby("name").agg(
            receptions=("n_scans", "size"), retention_rate=("retained", "mean")))
        table = table.join(seen.groupby("name").agg(
            scan_rate=("scan_rate", "mean"), pre_reception_scans=("n_scans", "mean")))
    table = table.reindex(columns=list(table.columns) + [c for c in scan_columns if c not in table])
    table = tactical_totals(table, events, timeline, on)
    return table.reset_index()


def tactical_totals(table: pd.DataFrame, events: dict[str, pd.DataFrame], timeline: pd.DataFrame,
                    on: str) -> pd.DataFrame:
    """The five tactical metrics, per name (index of `table`), as TACTICAL_COLUMNS.

    Volumes are per 90 of the minutes in `table`. A missing table leaves its columns
    NaN; a player with none of an event in a present table gets 0 for counts.
    """
    per90 = lambda values: values.reindex(table.index).fillna(0) / table["minutes"].where(table["minutes"] > 0) * 90
    credit = lambda key: roster.attribute(events[key], timeline, on=on) if key in events else None
    if (pressed := credit("pressures")) is not None:
        grouped = pressed.groupby("name")
        table["pressures"] = grouped.size().reindex(table.index).fillna(0)
        table["pressure_intensity"] = per90(grouped["intensity"].sum())
        table["press_closing_kmh"] = grouped["peak_closing_kmh"].mean()
    if (received := credit("receipts")) is not None:
        grouped = received.groupby("name")
        table["receipt_space"] = grouped["space"].median()
        table["receipts_in_space_pct"] = grouped["in_space"].mean()
    if (charged := credit("responsibility")) is not None:
        table["def_responsibility_p90"] = per90(charged.groupby("name")["xt_conceded"].sum())
    if (offered := credit("obso")) is not None:
        table["obso_p90"] = per90(offered.groupby("name")["obso"].sum())
    if (typed := credit("runs")) is not None:
        counts = typed.groupby(["name", "run_type"]).size().unstack(fill_value=0)
        for kind, column in RUN_TYPE_COLUMNS.items():
            table[column] = per90(counts[kind] if kind in counts else pd.Series(dtype=float))
    return table.reindex(columns=list(table.columns) + [c for c in TACTICAL_COLUMNS if c not in table])


def off_ball_value(board: pd.DataFrame, within: str | None = None) -> pd.DataFrame:
    """Per-90 pillar columns and Off-Ball Value: the mean percentile across the pillars.

    Percentiles are taken within `board`, so filter it to the pool you want to
    compare (one game, a minimum of minutes, outfielders only) before calling. Pass
    `within="broad_position"` to rank each player only against their own position:
    decoy runs and pressing near the ball are mostly midfield and forward work, so
    a single pool ranks defenders low for their role, not their quality.
    """
    board = board.copy()
    board["decoy_p90"] = board["decoy_value"].fillna(0) / board["minutes"] * 90
    board["lane_cut_p90"] = board["lane_cut_seconds"].fillna(0) / board["minutes"] * 90
    board = board.reindex(columns=list(board.columns) + [c for c in PILLARS if c not in board])
    ranks = (board.groupby(within)[PILLARS].rank(pct=True) if within
             else board[PILLARS].rank(pct=True))
    board["off_ball_value"] = ranks.mean(axis=1, skipna=True) * 100
    return board.sort_values("off_ball_value", ascending=False)
