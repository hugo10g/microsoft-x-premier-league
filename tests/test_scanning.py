import numpy as np

from core.pillars.scanning import detect_scans, scans_before_reception

FPS = 25


def scene(seconds: float = 10.0):
    """A player at the centre spot looking at a ball 10 m away along +x (bearing 0 deg)."""
    t = np.arange(1, int(seconds * FPS) + 1) / FPS
    player = np.tile([52.5, 34.0], (len(t), 1))
    ball = np.tile([62.5, 34.0], (len(t), 1))
    return t, np.zeros(len(t)), player, ball


def look(yaw, t, at, angle=120.0, hold=0.4, speed=600.0):
    """Turn the head `angle` deg away from the ball at `speed` deg/s, hold, and come back."""
    turn = abs(angle) / speed
    rise = np.clip((t - at) / turn, 0, 1)
    fall = np.clip((t - (at + turn + hold)) / turn, 0, 1)
    return yaw + angle * (rise - fall)


def test_three_quick_head_checks_are_three_scans():
    t, yaw, player, ball = scene()
    for at in (1.0, 3.0, 5.0):
        yaw = look(yaw, t, at, angle=110 if at != 3.0 else -130)
    scans = detect_scans(t, yaw, player, ball)
    assert len(scans) == 3
    np.testing.assert_allclose(scans["start"], [1.12, 3.16, 5.12], atol=0.1)
    assert (scans["peak_speed"] > 300).all()


def test_slow_drift_and_facing_away_are_not_scans():
    t, yaw, player, ball = scene()
    drift = look(yaw, t, 1.0, angle=80, hold=0.3, speed=40)          # under 90 deg/s
    away = look(yaw, t, 1.0, angle=150, hold=4.0)                    # away for over 2 s
    assert detect_scans(t, drift, player, ball).empty
    assert detect_scans(t, away, player, ball).empty


def test_jitter_around_the_threshold_is_one_scan_and_short_gaps_are_bridged():
    t, yaw, player, ball = scene()
    jitter = yaw.copy()
    span = (t >= 2.0) & (t < 2.6)
    jitter[span] = np.where(np.arange(span.sum()) % 2, 52, 70)       # hysteresis holds it on
    assert len(detect_scans(t, jitter, player, ball)) == 1
    gappy = look(yaw, t, 2.0, angle=120, hold=0.5)
    gappy[np.flatnonzero(t >= 2.4)[:3]] = np.nan                      # 0.12 s dropout mid-check
    assert len(detect_scans(t, gappy, player, ball)) == 1


def test_window_counts_only_recent_onsets():
    t, yaw, player, ball = scene()
    for at in (1.0, 6.0, 8.0):
        yaw = look(yaw, t, at)
    window = scans_before_reception(t, yaw, player, ball, t_receive=9.6, window=5.0)
    assert window.n_scans == 2                                        # the check at 1 s is too early
    assert np.isclose(window.scan_rate, 2 / 5.0, atol=0.02)
    assert 0.5 < window.ball_fixation < 1.0
    assert window.observed > 0.99


def test_missing_head_data_gives_no_rate():
    t, yaw, player, ball = scene()
    yaw = look(yaw, t, 6.0)
    yaw[t < 8.0] = np.nan
    window = scans_before_reception(t, yaw, player, ball, t_receive=9.6)
    assert window.n_scans == 0
    assert np.isnan(window.scan_rate)
    nothing = scans_before_reception(t, np.full(len(t), np.nan), player, ball, t_receive=9.6)
    assert nothing.n_scans == 0 and np.isnan(nothing.scan_rate) and np.isnan(nothing.ball_fixation)


def test_ball_at_the_feet_has_no_bearing():
    t, yaw, player, _ = scene()
    on_ball = player + [0.5, 0.0]                                     # within 1 m
    assert detect_scans(t, look(yaw, t, 2.0), player, on_ball).empty
    window = scans_before_reception(t, yaw, player, on_ball, t_receive=9.6)
    assert window.observed == 0 and np.isnan(window.scan_rate)
