import numpy as np
import pandas as pd

from core.roster import active, attribute, infer_positions, static_timeline, tracking_substitutions
from core.tracking import TrackingData


def test_roles_follow_depth_and_flip_at_half_time():
    # Six home players at increasing depth; second half mirrored, so depth is unchanged.
    xs = np.array([5.0, 30, 35, 50, 55, 80])
    first = np.stack([xs, np.full(6, 34.0)], axis=-1)
    second = np.stack([105 - xs, np.full(6, 34.0)], axis=-1)
    tr = TrackingData(frames=np.arange(2), timestamps=np.array([0.0, 0.04]), periods=np.array([1, 2]),
                      positions=np.stack([first, second]), ball=np.zeros((2, 2)),
                      player_ids=[f"Home_{i}" for i in range(6)], teams=np.array(["Home"] * 6))
    roles = infer_positions(tr).set_index("player")["role"]
    assert roles.tolist() == ["Goalkeeper", "Defender", "Defender", "Midfielder", "Midfielder", "Forward"]


def substitution_match():
    """One team, 90 minutes at 1 Hz: a keeper, and a striker replaced at 60'."""
    t = np.arange(0, 90 * 60, 1.0)
    positions = np.array([np.tile([5.0, 34.0], (len(t), 1)), np.tile([70.0, 34.0], (len(t), 1)),
                          np.tile([70.0, 34.0], (len(t), 1))]).transpose(1, 0, 2).copy()
    positions[t >= 3600, 1] = np.nan
    positions[t < 3600, 2] = np.nan
    return TrackingData(frames=np.arange(len(t)), timestamps=t, periods=np.ones(len(t), dtype=int),
                        positions=positions, ball=np.zeros((len(t), 2)),
                        player_ids=["Home_1", "Home_9", "Home_19"], teams=np.array(["Home"] * 3))


def test_substitutions_come_from_appearances():
    subs = tracking_substitutions(infer_positions(substitution_match()))
    assert subs[["off", "on"]].values.tolist() == [["Home_9", "Home_19"]]
    assert abs(subs["minute"].iloc[0] - 60) < 0.1


def test_events_are_credited_to_the_name_in_force():
    tr = substitution_match()
    names = pd.DataFrame({"player": tr.player_ids, "name": ["Keeper", "Nine", "Super Sub"],
                          "real_team": "Test FC", "position": ["Goalkeeper", "Forward", "Forward"],
                          "detailed_position": ["Goalkeeper", "Centre-Forward", "Centre-Forward"],
                          "market_value_m": [10.0, 50.0, 30.0]})
    timeline = static_timeline(infer_positions(tr), names)
    assert set(active(timeline, 30)["name"]) == {"Keeper", "Nine"}
    assert set(active(timeline, 75)["name"]) == {"Keeper", "Super Sub"}
    events = pd.DataFrame({"player": ["Home_9", "Home_19", "Home_19"], "minute": [10.0, 70.0, 80.0],
                           "value": [1, 2, 4]})
    credited = attribute(events, timeline).groupby("name")["value"].sum().to_dict()
    assert credited == {"Nine": 1, "Super Sub": 6}
