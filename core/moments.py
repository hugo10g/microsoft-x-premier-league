"""
Key Moments: the standout off-ball events in a game, ready to jump to.

Reads the event tables core.pipeline.score_game produces and picks, per kind, the
highest-scoring events, spread out in time:

    decoy                   decoy runs with the most xT created for teammates
    press                   spells of pressing that keep a large share of the
                            threat-weighted passing lanes cut
    defensive_transition    sprints into a defensive transition within `elite_latency`
                            (but not under `min_reaction`: that's a player already
                            sprinting when the ball changed hands, not a reaction)
    attacking_transition    the same after winning the ball
    scan                    receptions the player kept after at least `scan_min_checks`
                            head checks in the window before the ball arrived (from
                            head orientation; a game without it has none)

A scan moment starts at the beginning of that window, so playback shows the checks.
"""

import pandas as pd

from core.config import MOMENTS, SCANNING, MomentsConfig

KINDS = {
    "decoy": "High-xT decoy run",
    "press": "Lane-cutting press",
    "defensive_transition": "Elite defensive transition",
    "attacking_transition": "Explosive attacking transition",
    "scan": "Elite scan before receiving",
}
COLUMNS = ["kind", "player", "time", "end", "minute", "score", "detail"]


def decoy_moments(decoys: pd.DataFrame) -> pd.DataFrame:
    runs = decoys[decoys["decoy_value"] > 0]
    return pd.DataFrame({
        "kind": "decoy", "player": runs["player"], "time": runs["time"],
        "end": runs["time"] + runs["duration"], "minute": runs["minute"],
        "score": runs["decoy_value"],
        "detail": runs["decoy_value"].map(lambda v: f"{v:.3f} xT of space opened for teammates")})


def press_moments(pressing: pd.DataFrame, config: MomentsConfig = MOMENTS) -> pd.DataFrame:
    """Unbroken spells where a player keeps at least `press_min_cut` of the lanes cut."""
    frames = pressing.sort_values(["player", "time"])
    strong = frames["lane_cut"] >= config.press_min_cut
    # A new spell starts after any weak frame, a gap in time, or a change of player.
    gap = frames["time"].diff() > 1.5 * frames["seconds"]
    spell = (~strong | gap | (frames["player"] != frames["player"].shift())).cumsum()
    spells = frames[strong].groupby(spell[strong]).agg(
        player=("player", "first"), time=("time", "first"), end=("time", "last"),
        minute=("minute", "first"), cut=("lane_cut", "mean"), seconds=("seconds", "first"))
    spells["duration"] = spells["end"] - spells["time"] + spells["seconds"]
    spells = spells[spells["duration"] >= config.press_min_seconds]
    return pd.DataFrame({
        "kind": "press", "player": spells["player"], "time": spells["time"], "end": spells["end"],
        "minute": spells["minute"], "score": spells["cut"] * spells["duration"],
        "detail": [f"{cut:.0%} of threat-weighted lanes cut for {duration:.1f} s"
                   for cut, duration in zip(spells["cut"], spells["duration"])]})


def transition_moments(transitions: pd.DataFrame, config: MomentsConfig = MOMENTS) -> pd.DataFrame:
    quick = transitions[transitions["reached_sprint"]
                        & transitions["latency"].between(config.min_reaction, config.elite_latency)]
    return pd.DataFrame({
        "kind": quick["phase"] + "_transition", "player": quick["player"], "time": quick["time"],
        "end": quick["time"] + quick["latency"] + 2.0, "minute": quick["minute"],
        "score": quick["score"],
        "detail": quick["latency"].map(lambda s: f"sprinting (20+ km/h) {s:.1f} s after the turnover")})


def scan_moments(scanning: pd.DataFrame, config: MomentsConfig = MOMENTS,
                 window: float = SCANNING.window) -> pd.DataFrame:
    """Kept receptions after at least `scan_min_checks` head checks; more checks score higher."""
    best = scanning[scanning["retained"].astype(bool) & (scanning["n_scans"] >= config.scan_min_checks)]
    return pd.DataFrame({
        "kind": "scan", "player": best["player"], "time": best["time"] - window,
        "end": best["time"] + 1.0, "minute": best["minute"] - window / 60,
        "score": best["n_scans"] + 0.01 * best["scan_rate"].fillna(0).clip(upper=10),
        "detail": best["n_scans"].map(
            lambda n: f"{n} head checks in the {window:g} s before receiving, then kept the ball")})


def spread_out(moments: pd.DataFrame, per_kind: int, min_gap: float) -> pd.DataFrame:
    """The best `per_kind` moments of each kind, at least `min_gap` seconds apart."""
    kept = []
    for _, group in moments.sort_values("score", ascending=False).groupby("kind", sort=False):
        times: list[float] = []
        for row in group.itertuples():
            if len(times) == per_kind:
                break
            if all(abs(row.time - t) >= min_gap for t in times):
                times.append(row.time)
                kept.append(row.Index)
    return moments.loc[kept]


def key_moments(events: dict[str, pd.DataFrame], config: MomentsConfig = MOMENTS) -> pd.DataFrame:
    """Key moments for one game's event tables, in match order."""
    parts = [decoy_moments(events["decoys"]), press_moments(events["pressing"], config),
             transition_moments(events["transitions"], config)]
    if "scanning" in events and len(events["scanning"]):
        parts.append(scan_moments(events["scanning"], config))
    found = pd.concat([part for part in parts if len(part)], ignore_index=True) \
        if any(len(part) for part in parts) else pd.DataFrame(columns=COLUMNS)
    if found.empty:
        return pd.DataFrame(columns=COLUMNS)
    found = spread_out(found, config.per_kind, config.min_gap)
    return found[COLUMNS].sort_values("time").reset_index(drop=True)
