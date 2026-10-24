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


def test_a_defender_in_the_shooting_lane_blocks_and_one_off_it_never_does():
    match = kicked_off_match()
    away = np.flatnonzero(match.active & (match.team == 1) & ~match.keeper)
    match.pos[away] = [20.0, 5.0]                                     # everyone far away...
    here, line = np.array([80.0, 34.0]), np.array([25.0, 0.0])        # ...a shot straight at goal
    assert all(match.blocker(here, line) is None for _ in range(50))
    match.pos[away[0]] = [86.0, 34.0]                                 # one defender right in the lane
    blocks = [match.blocker(here, line) for _ in range(50)]
    assert sum(b is not None for b in blocks) > 30 and {b[0] for b in blocks if b} == {int(away[0])}
    match.pos[away[0]] = [86.0, 39.0]                                 # 5 m off the line: no block
    assert all(match.blocker(here, line) is None for _ in range(50))


def test_a_decoy_runs_space_is_taken_by_the_teammate_nearest_it():
    match = kicked_off_match()
    home = np.flatnonzero(match.active & (match.team == 0) & ~match.keeper)
    away = np.flatnonzero(match.active & (match.team == 1) & ~match.keeper)
    match.pos[home] = [10.0, 5.0]
    match.pos[away] = [30.0, 60.0]
    runner, near = home[0], home[1]
    match.pos[runner], match.pos[away[0]], match.pos[near] = [70.0, 34.0], [72.0, 36.0], [62.0, 38.0]
    space, taker = match.space_taker(int(runner), 0)
    np.testing.assert_allclose(space, [72.0, 36.0])
    assert taker == near                                              # the marker's spot goes to the nearest mate


def test_the_bench_goalkeeper_never_comes_on_so_each_side_has_one_keeper():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        used: set = set()
        home, away = make_squad("Test A", "4-3-3", rng, used), make_squad("Test B", "4-4-2", rng, used)
        match = Match(home, away, seed=seed)
        for side in (0, 1):
            squad = match.team == side
            assert (squad & match.keeper).sum() == 1                    # only the starting keeper is in the 14


def test_at_a_penalty_everyone_but_the_taker_and_keeper_is_outside_the_box_and_the_d():
    from core.synthetic import BOX_DEPTH, FPS, WALL_DISTANCE
    match = kicked_off_match()
    match.last_turnover = -10_000
    match.ball = np.array([97.0, 34.0])
    match.foul(fouler=20, frame=10)
    spot, taker = match.restart["spot"], match.restart["taker"]
    for frame in range(11, 11 + 60 * FPS):                             # the minute it takes to set up
        if frame >= match.restart["moves"]:
            match.ball = match.toward(match.ball, spot, 10.0)
        match.steer(frame, *match.frame_targets(frame))
    others = match.active & ~match.keeper
    others[taker] = False
    assert (np.linalg.norm(match.pos[others] - spot, axis=1) > WALL_DISTANCE).all()
    assert (match.pos[others, 0] < 105.0 - BOX_DEPTH).all()


def test_two_players_sent_to_the_same_spot_stand_apart_instead_of_on_top_of_each_other():
    from core.synthetic import FPS, MATE_GAP
    match = kicked_off_match()
    match.last_turnover = -10_000
    home = np.flatnonzero(match.active & (match.team == 0) & ~match.keeper)
    a, b = home[0], home[1]
    match.pos[a], match.pos[b] = [60.0, 30.0], [60.0, 30.0]           # exactly on top of each other
    for frame in range(3 * FPS):
        targets = match.pos.copy()
        targets[[a, b]] = [60.0, 30.0]                                  # and both told to stay there
        match.steer(frame, targets, np.full(len(match.pos), 0.5))
    assert np.linalg.norm(match.pos[a] - match.pos[b]) > 0.8 * MATE_GAP
