import numpy as np
import pytest

from core import pipeline, tracking
from core.synthetic import make_squad, scan_effect, value_players, write_match


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
