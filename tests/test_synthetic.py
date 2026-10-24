import numpy as np
import pytest

from core import pipeline, tracking
from core.synthetic import DT, Match, make_squad, scan_effect, value_players, write_match


def test_synthetic_match_loads_with_full_teams(tmp_path):
    rng = np.random.default_rng(0)
    used: set = set()
    home, away = make_squad("Test A", "4-3-3", rng, used), make_squad("Test B", "3-5-2", rng, used)
    value_players(home + away, rng)
    for squad in (home, away):
        assert len({p.jersey for p in squad}) == len(squad)          # shirt numbers never repeat
    folder = tmp_path / "Test_Match"
    write_match(folder, home, away, seed=1, half_minutes=2)

    tr, events = tracking.load_sample_game(folder)
    on_pitch = ~np.isnan(tr.positions).any(axis=-1)
    assert on_pitch[0].sum() == 22                                  # 11 a side at kick-off
    assert (on_pitch.sum(axis=1) == 22).all()                       # never short: subs swap one for one
    assert set(events["Type"]) >= {"PASS", "SET PIECE"}
    roster = pipeline.load_roster(folder)
    assert set(roster["player"]) == set(tr.player_ids)              # every tracked player is named
    assert roster["name"].is_unique


def test_xt_training_events_are_on_the_statsbomb_pitch(tmp_path):
    rng = np.random.default_rng(1)
    used: set = set()
    home, away = make_squad("Test A", "4-3-3", rng, used), make_squad("Test B", "4-4-2", rng, used)
    value_players(home + away, rng)
    folder = tmp_path / "Test_Match"
    write_match(folder, home, away, seed=2, half_minutes=5)
    rows = pipeline.xt_training_events(folder)
    assert {"Pass", "Carry"} <= set(rows["type"])
    starts = np.array(rows["location"].tolist())
    assert (starts[:, 0] >= -3).all() and (starts[:, 0] <= 123).all()
    assert (starts[:, 1] >= -3).all() and (starts[:, 1] <= 83).all()


def test_head_orientation_is_written_for_every_tracked_player(tmp_path):
    rng = np.random.default_rng(3)
    used: set = set()
    home, away = make_squad("Test A", "4-3-3", rng, used), make_squad("Test B", "4-4-2", rng, used)
    value_players(home + away, rng)
    assert all(0 < p.awareness < 1 for p in home + away)
    assert len({round(p.awareness, 6) for p in home}) == len(home)    # an attribute, not a constant
    folder = tmp_path / "Test_Match"
    write_match(folder, home, away, seed=4, half_minutes=2)

    tr, _ = tracking.load_sample_game(folder)
    yaw = tracking.load_head_orientation(folder, tr)
    assert yaw.shape == (len(tr.frames), len(tr.player_ids))
    on_pitch = ~np.isnan(tr.positions).any(axis=-1)
    assert (np.isnan(yaw) == ~on_pitch).all()                         # NaN exactly while off the pitch
    assert (np.abs(yaw[on_pitch]) <= 180).all()
    turn = np.abs((np.diff(yaw, axis=0) + 180) % 360 - 180)
    assert np.nanpercentile(turn, 99) * 25 < 700                      # heads turn, they don't teleport


def test_scanning_before_receiving_helps_the_next_action():
    assert scan_effect(None) == (1.0, 0.0)                            # turnovers and restarts: neutral
    lose_0, pass_0 = scan_effect(0)
    lose_1, pass_1 = scan_effect(1)
    lose_3, pass_3 = scan_effect(5)
    assert lose_0 > 1 > lose_1 > lose_3 and pass_0 < 0 < pass_1 < pass_3      # more checks, better


@pytest.mark.skipif(not pipeline.available_games(), reason="synthetic league not generated")
def test_league_xt_rises_towards_goal():
    # A single short match can have no goals (and so a flat, all-zero xT);
    # the full league has enough for the expected shape.
    xt = pipeline.learn_xt()
    assert xt[:, -4:].mean() > 5 * xt[:, :4].mean()
    assert np.unravel_index(xt.argmax(), xt.shape)[1] >= xt.shape[1] - 2   # peak next to goal


def test_the_ball_never_teleports_and_longer_passes_are_struck_harder():
    rng = np.random.default_rng(2)
    used: set = set()
    home, away = make_squad("Test A", "4-3-3", rng, used), make_squad("Test B", "4-4-2", rng, used)
    value_players(home + away, rng)
    match = Match(home, away, seed=3, half_minutes=6)
    frames, positions, ball, events = match.run()
    same_half = frames["Period"].to_numpy()[1:] == frames["Period"].to_numpy()[:-1]
    speed = np.linalg.norm(np.diff(ball, axis=0), axis=1)[same_half] / DT
    assert speed.max() < 40                                          # m/s: a hard shot, never a jump
    passes = events[events["Type"] == "PASS"]
    flight = passes["End Time [s]"] - passes["Start Time [s]"]
    length = np.hypot((passes["End X"] - passes["Start X"]) * 105, (passes["End Y"] - passes["Start Y"]) * 68)
    assert flight[length > 30].median() > flight[length < 10].median()   # long balls take longer...
    assert (length / flight)[length > 30].median() > (length / flight)[length < 10].median()   # ...but are struck harder


def kicked_off_match():
    rng = np.random.default_rng(4)
    used: set = set()
    home, away = make_squad("Test A", "4-3-3", rng, used), make_squad("Test B", "4-4-2", rng, used)
    value_players(home + away, rng)
    match = Match(home, away, seed=5)
    match.period, match.possession = 1, 0                           # home attack +x (goal at x = 105)
    match.place_for_kickoff()
    match.holder = 9
    return match


def test_a_foul_in_the_box_is_a_penalty_and_outside_it_a_free_kick():
    match = kicked_off_match()
    match.ball = np.array([97.0, 34.0])
    match.foul(fouler=20, frame=10)
    assert match.events[0]["Type"] == "FAULT RECEIVED" and match.restart["kind"] == "PENALTY"
    np.testing.assert_allclose(match.restart["spot"], [94.0, 34.0])   # 11 m out
    match = kicked_off_match()
    match.ball = np.array([60.0, 30.0])
    match.foul(fouler=20, frame=10)
    assert match.restart["kind"] == "FREE KICK" and match.restart["side"] == 0
    np.testing.assert_allclose(match.restart["spot"], [60.0, 30.0])


def test_offside_is_beyond_the_second_last_defender_and_the_ball_and_corners_go_to_the_flag():
    match = kicked_off_match()
    away = np.flatnonzero(match.active & (match.team == 1))
    match.pos[away] = [50.0, 34.0]
    match.pos[away[:2]] = [[95.0, 34.0], [90.0, 30.0]]              # keeper and last defender
    match.ball = np.array([60.0, 34.0])
    match.pos[3], match.pos[4] = [92.0, 20.0], [89.0, 20.0]
    assert match.offside(3) and not match.offside(4)                 # beyond the line at 90 m, or level-ish
    match.ball = np.array([93.0, 34.0])
    assert not match.offside(3)                                      # never offside behind the ball
    match.corner(0, frame=10, near=np.array([105.0, 5.0]))
    assert match.restart["kind"] == "CORNER KICK"
    np.testing.assert_allclose(match.restart["spot"], [104.7, 0.3])
