import numpy as np
import pandas as pd

from core.config import MomentsConfig
from core.moments import KINDS, key_moments, press_moments, scan_moments


def press_frames(player, start, cuts, step=0.04):
    times = start + step * np.arange(len(cuts))
    return pd.DataFrame({"player": player, "time": times, "minute": times / 60,
                         "lane_cut": cuts, "seconds": step})


def test_press_spells_need_strong_frames_held_long_enough():
    frames = pd.concat([
        press_frames("A", 10.0, [0.8] * 50),                 # 2 s strong: a moment
        press_frames("A", 30.0, [0.8] * 10),                 # 0.4 s: too short
        press_frames("B", 10.0, [0.8] * 20 + [0.1] + [0.8] * 20),   # broken by a weak frame
    ])
    spells = press_moments(frames, MomentsConfig(press_min_seconds=1.0))
    assert spells["player"].tolist() == ["A"]
    assert spells["time"].iloc[0] == 10.0


def test_key_moments_keep_the_best_per_kind_spread_out_in_time():
    decoys = pd.DataFrame({"player": ["A", "B", "C"], "time": [100.0, 105.0, 400.0],
                           "minute": [1.7, 1.75, 6.7], "duration": 2.0,
                           "decoy_value": [0.10, 0.05, 0.08]})
    transitions = pd.DataFrame({"player": ["D", "E", "G"], "phase": ["defensive", "attacking", "attacking"],
                                "latency": [0.8, 3.0, 0.0], "reached_sprint": True,
                                "score": [0.6, 0.2, 1.0], "time": [200.0, 300.0, 500.0],
                                "minute": [3.3, 5.0, 8.3]})
    events = {"decoys": decoys, "transitions": transitions,
              "pressing": press_frames("F", 0.0, [0.1] * 10)}
    moments = key_moments(events, MomentsConfig(per_kind=2, min_gap=15.0))
    # B is within 15 s of the better run by A; E's 3 s latency isn't elite; G was
    # already sprinting (0 s), which isn't a reaction.
    assert moments[["kind", "player"]].values.tolist() == [
        ["decoy", "A"], ["defensive_transition", "D"], ["decoy", "C"]]


def receptions(rows):
    return pd.DataFrame(rows, columns=["player", "time", "minute", "n_scans", "scan_rate",
                                       "ball_fixation", "retained"])


def test_scan_moments_need_enough_checks_and_the_ball_kept():
    scanning = receptions([
        ("A", 100.0, 1.67, 4, 0.8, 0.5, True),        # elite: 4 checks, kept it
        ("B", 200.0, 3.33, 5, 1.0, 0.4, False),       # more checks, but lost the ball
        ("C", 300.0, 5.00, 2, 0.4, 0.7, True),        # kept, but only 2 checks
        ("D", 400.0, 6.67, 4, 0.9, 0.5, True),        # ties A on checks, higher rate
    ])
    moments = scan_moments(scanning, MomentsConfig(scan_min_checks=3))
    assert moments["player"].tolist() == ["A", "D"]
    assert moments.set_index("player")["score"]["D"] > moments.set_index("player")["score"]["A"]
    a = moments.iloc[0]
    assert a["time"] == 95.0 and a["end"] > 100.0                     # playback starts with the checks
    assert a["detail"] == "4 head checks in the 5 s before receiving, then kept the ball"
    assert KINDS["scan"] == "Elite scan before receiving"


def test_key_moments_include_scans_spread_out_and_capped():
    empty = {"decoys": pd.DataFrame(columns=["player", "time", "minute", "duration", "decoy_value"]),
             "transitions": pd.DataFrame(columns=["player", "phase", "latency", "reached_sprint",
                                                  "score", "time", "minute"]),
             "pressing": press_frames("F", 0.0, [0.1] * 10)}
    scanning = receptions([("A", 100.0, 1.7, 5, 1.0, 0.5, True), ("B", 105.0, 1.75, 4, 0.8, 0.5, True),
                           ("C", 400.0, 6.7, 3, 0.6, 0.5, True), ("E", 600.0, 10.0, 3, 0.5, 0.5, True)])
    moments = key_moments(empty | {"scanning": scanning}, MomentsConfig(per_kind=2, min_gap=15.0))
    assert moments[["kind", "player"]].values.tolist() == [["scan", "A"], ["scan", "C"]]
    assert key_moments(empty).empty                                   # no scanning table: still fine
