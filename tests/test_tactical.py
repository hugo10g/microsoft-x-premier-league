import numpy as np
import pandas as pd

from core import pipeline
from core.pillars.obso import frame_obso
from core.pillars.pressure import closing_speed, pressures
from core.pillars.receipts import receipts_in_space
from core.pillars.responsibility import base_offsets, zone_shares
from core.pillars.run_types import classify
from core.tracking import TrackingData

FPS = 25


def match(positions: np.ndarray, ball: np.ndarray, teams: list[str]) -> TrackingData:
    """A one-period match from (T, P, 2) positions; player ids Home_1, Away_2, ..."""
    n = len(positions)
    return TrackingData(frames=np.arange(1, n + 1), timestamps=np.arange(1, n + 1) / FPS,
                        periods=np.ones(n, dtype=int), positions=positions, ball=ball,
                        player_ids=[f"{team}_{k}" for k, team in enumerate(teams, 1)], teams=np.array(teams))


# ── Pressure Intensity ───────────────────────────────────────────────────────
def test_closing_speed_is_the_rate_the_gap_shrinks():
    carrier, still = np.array([50.0, 34.0]), np.zeros(2)
    defenders = np.array([[45.0, 34.0], [50.0, 40.0]])
    velocity = np.array([[5.0, 0.0], [5.0, 0.0]])                   # both run along +x
    distance, closing = closing_speed(defenders, velocity, carrier, still)
    np.testing.assert_allclose(distance, [5.0, 6.0])
    np.testing.assert_allclose(closing, [5.0, 0.0], atol=1e-9)      # head-on vs across the carrier


def test_a_sprint_at_the_carrier_is_one_pressure_with_intensity():
    t = np.arange(50)                                                # 2 s
    carrier = np.tile([50.0, 34.0], (50, 1))
    presser = np.column_stack([42.0 + 6.0 * t / FPS, np.full(50, 34.0)])    # 6 m/s from 8 m away
    idle = np.tile([20.0, 10.0], (50, 1))                            # an away player far from everything
    tr = match(np.stack([carrier, presser, idle], axis=1), carrier + [0.3, 0], ["Home", "Away", "Away"])
    found = pressures(tr, np.full(50, "Home", dtype=object))
    assert found["player"].tolist() == ["Away_2"] and found["carrier"].tolist() == ["Home_1"]
    row = found.iloc[0]
    assert row["peak_closing_kmh"] > 18 and row["min_distance"] < 2
    assert 0 < row["intensity"] < 8                                  # metres closed, weighted by proximity


# ── Ball receipts in space ───────────────────────────────────────────────────
def test_space_is_the_distance_to_the_nearest_opponent_at_reception():
    positions = np.array([[[30.0, 30.0], [50.0, 30.0], [56.0, 30.0], [50.0, 39.0]]] * 3)
    tr = match(positions, np.tile([30.0, 30.0], (3, 1)), ["Home", "Home", "Away", "Away"])
    events = pd.DataFrame({"Type": ["PASS"], "To": ["Home_2"], "End Frame": [3], "End Time [s]": [0.12]})
    table = receipts_in_space(tr, events)
    assert table["space"].tolist() == [6.0] and table["in_space"].tolist() == [True]


# ── Defensive responsibility ─────────────────────────────────────────────────
def test_base_offsets_follow_the_shape_not_the_block():
    frames = np.array([[[10.0, 10.0], [20.0, 10.0]], [[30.0, 10.0], [40.0, 10.0]]])   # block slides 20 m
    np.testing.assert_allclose(base_offsets(frames), [[-5.0, 0.0], [5.0, 0.0]])


def test_a_pass_between_two_slots_is_split_and_a_missing_player_is_never_charged():
    slots = np.array([[40.0, 30.0], [40.0, 40.0], [np.nan, np.nan]])
    np.testing.assert_allclose(zone_shares(np.array([40.0, 35.0]), slots, spread=8.0), [0.5, 0.5, 0.0])
    near_first = zone_shares(np.array([40.0, 31.0]), slots, spread=8.0)
    assert near_first[0] > 0.6 and near_first.sum() == 1


# ── OBSO ─────────────────────────────────────────────────────────────────────
def test_obso_rewards_free_dangerous_spots_within_reach_of_the_ball():
    ball = np.array([60.0, 34.0])
    attackers = np.array([[85.0, 34.0], [85.0, 34.0], [62.0, 10.0]])
    defenders_free = np.array([[70.0, 50.0]])
    value = frame_obso(attackers[:1], defenders_free, ball, np.array([0.2]))
    marked = frame_obso(attackers[1:2], np.array([[85.5, 34.0]]), ball, np.array([0.2]))
    assert value[0] > 5 * marked[0]                                  # free beats marked tight
    far = frame_obso(np.array([[85.0, 34.0]]), defenders_free, np.array([10.0, 34.0]), np.array([0.2]))
    assert far[0] < value[0]                                         # the ball can't easily get there
    assert frame_obso(np.array([[np.nan, np.nan]]), defenders_free, ball, np.array([np.nan]))[0] == 0


# ── Run types ────────────────────────────────────────────────────────────────
def kind(start, end, line=80.0, right=True):
    start, end = np.array(start, float), np.array(end, float)
    velocity = np.tile((end - start) / 2.0, (50, 1))                 # 2 s at constant speed
    return classify(start, end, velocity, 1 / FPS, right, line)


def test_runs_are_classified_by_direction_and_the_defensive_line():
    assert kind([70, 34], [84, 36]) == "Run in Behind"               # past the line at x = 80
    assert kind([60, 34], [72, 35]) == "Other"                       # forward, but short of the line
    assert kind([60, 50], [63, 63]) == "Pulling Wide"                # towards the near (top) touchline
    assert kind([60, 20], [62, 5]) == "Pulling Wide"                 # ...or the bottom one
    assert kind([70, 34], [58, 33]) == "Dropping Deep"
    assert kind([70, 34], [72, 35]) is None                          # too short to be a run
    assert kind([35, 34], [21, 36], line=25.0, right=False) == "Run in Behind"   # mirrored second half


def test_tactical_totals_are_per_90_and_zero_for_a_player_without_events():
    timeline = pd.DataFrame({"player": ["Home_1", "Home_2"], "name": ["A", "B"], "start": [0.0, 0.0],
                             "end": [45.0, 90.0]})
    table = pd.DataFrame({"minutes": [45.0, 90.0]}, index=pd.Index(["A", "B"], name="name"))
    events = {"obso": pd.DataFrame({"player": ["Home_1"], "minute": [10.0], "obso": [3.0]}),
              "runs": pd.DataFrame({"player": ["Home_1", "Home_1"], "minute": [5.0, 6.0],
                                    "run_type": ["Run in Behind", "Dropping Deep"]})}
    out = pipeline.tactical_totals(table, events, timeline, "player")
    assert out.loc["A", "obso_p90"] == 6.0 and out.loc["B", "obso_p90"] == 0.0
    assert out.loc["A", "runs_in_behind_p90"] == 2.0 and out.loc["A", "pulling_wide_p90"] == 0.0
    assert out["pressure_intensity"].isna().all()                    # no pressures table: unknown, not zero
