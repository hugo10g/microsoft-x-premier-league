import json

import numpy as np
import pandas as pd

from core import live
from core.match_tools import MatchFacts
from core.tracking import TrackingData

FPS = 25
N = 30 * FPS                                       # 30 s of a match


def tracking() -> TrackingData:
    """Home_1 on the ball at (50, 34); Home_2 open at (60, 34); Away_3 on the line to Home_4 at (60, 10)."""
    spots = np.array([[50.0, 34.0], [60.0, 34.0], [55.0, 22.0], [60.0, 10.0], [70.0, 50.0]])
    teams = ["Home", "Home", "Away", "Home", "Away"]
    t = np.arange(1, N + 1) / FPS
    ball = np.tile([50.0, 34.0], (N, 1))
    ball[500:510] = np.linspace([50.0, 34.0], [104.0, 34.0], 10)          # a shot at 20 s: 54 m in 0.4 s
    return TrackingData(frames=np.arange(1, N + 1), timestamps=t, periods=np.ones(N, dtype=int),
                        positions=np.tile(spots, (N, 1, 1)), ball=ball,
                        player_ids=[f"{team}_{k}" for k, team in enumerate(teams, 1)], teams=np.array(teams))


def event(kind, sub, team, second, src, dst=None, end=None, start=(50.0, 34.0), to=(60.0, 34.0)):
    frame = round(second * FPS)
    return {"Team": team, "Type": kind, "Subtype": sub, "Period": 1, "Start Frame": frame,
            "Start Time [s]": frame / FPS, "End Frame": round((end or second) * FPS), "End Time [s]": end or second,
            "From": src, "To": dst, "Start X": start[0], "Start Y": start[1], "End X": to[0], "End Y": to[1]}


EVENTS = pd.DataFrame([
    event("PASS", None, "Home", 2.0, "Home_1", "Home_2", 2.6),                                 # open, square
    event("BALL LOST", "INTERCEPTION", "Home", 5.0, "Home_1", None, 5.4, to=(55.0, 22.0)),     # through Away_3
    event("FAULT RECEIVED", None, "Away", 8.0, "Away_5"),
    event("CARD", "YELLOW", "Home", 8.0, "Home_4"),
    event("PASS", None, "Home", 18.0, "Home_2", "Home_1", 18.6, start=(60.0, 34.0), to=(50.0, 34.0)),
    event("SHOT", "ON TARGET-GOAL", "Home", 20.0, "Home_1", None, 20.4),
]).astype({"Subtype": object, "To": object})
NAMES = {"Home_1": "Ada Runner", "Home_2": "Bo Open", "Away_3": "Cy Block", "Home_4": "Di Wide", "Away_5": "Ed Fouled"}
CLUBS = {"Home": "Harbourside FC", "Away": "Northvale Athletic"}


def feed():
    tr = tracking()
    no_moments = pd.DataFrame(columns=["kind", "player", "time", "end", "minute", "score", "detail"])
    return live.event_feed(EVENTS, tr, tr.timestamps / 60, no_moments, MatchFacts("T", *CLUBS.values(), (1, 0)),
                           NAMES, CLUBS)


def test_a_pass_through_a_defender_is_harder_and_an_intercepted_pass_is_not_completed():
    quality = live.pass_quality(tracking(), EVENTS)
    open_pass, intercepted = quality.iloc[0], quality.iloc[1]
    assert open_pass["completed"] and not intercepted["completed"]
    assert intercepted["lane"] < 1.0 < open_pass["lane"] and intercepted["difficulty"] > open_pass["difficulty"]
    assert open_pass["distance"] == 10.0


def test_shot_speed_is_the_balls_peak_speed():
    assert live.shot_speeds(tracking(), EVENTS).iloc[0] > 400        # 6 m a frame = 540 km/h in this toy


def test_the_feed_scores_the_goal_names_the_assist_and_says_who_was_fouled():
    rows = feed().set_index("kind")
    assert rows.loc["goal", "text"].startswith("GOAL! Ada Runner scores for Harbourside FC: Harbourside FC 1-0")
    assert "Bo Open's 10 m pass" in rows.loc["goal", "text"]
    assert rows.loc["card", "why"] == "For a foul on Ed Fouled." and rows.loc["card", "tier"] == "key"
    items = live.overlay_items(feed(), "T")
    assert json.loads(json.dumps(items))[0]["type"] == "card" and {"t", "clock", "text", "show_for_s"} <= set(items[0])


def test_live_state_only_counts_what_has_happened_by_then():
    tr = tracking()
    possession = np.where(tr.timestamps < 15, "Home", "Away").astype(object)
    early = live.state_at(EVENTS, tr, tr.timestamps / 60, possession, 10 / 60)
    late = live.state_at(EVENTS, tr, tr.timestamps / 60, possession, 0.5)
    assert early["Home"]["goals"] == 0 and late["Home"]["goals"] == 1
    assert early["Home"]["passes"] == 2 and early["Home"]["pass_accuracy"] == 50
    assert early["Home"]["yellow_cards"] == 1 and early["Home"]["fouls"] == 1      # Away were fouled (FAULT RECEIVED)
    assert early["Home"]["possession"] == 100 and late["Home"]["possession"] == 50


def test_rhythm_shows_who_had_the_ball_and_how_often_it_changed():
    tr = tracking()
    possession = np.where((np.arange(N) // FPS) % 2 == 0, "Home", "Away").astype(object)    # swaps every second
    spells = live.rhythm(tr, (tr.timestamps - 1 / FPS) / 60, possession, spell=0.25)        # two 15 s spells
    assert spells["home_share"].between(45, 55).all() and spells["changes"].sum() == 29
