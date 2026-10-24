"""
Pillar 2, Space Exploitation: the threat a decoy run creates for teammates.

A decoy run is worth the space it opens for *other* attackers by dragging markers
away. Removing the runner and recomputing pitch control isn't enough on its own:
the defenders would still stand where the run pulled them. The counterfactual
has to put the markers back too:

    M_i  = defenders who had runner i as their nearest attacker for ≥ marker_share
           of the run and moved with them (mean velocity cosine ≥ marker_alignment)
    D̃    = defenders at the end of the run, except each marker m sits at
           p_m(t0) + (shift of the rest of the defensive block over the run),
           i.e. "followed the team, not the runner"
    net_i   = Σ_z xT(z) · [C_{-i}(z; D_t1) - C_{-i}(z; D̃)] · ΔA
    Decoy_i = max(0, net_i)

C_{-i} is attacking pitch control with the runner removed in both terms, so the
runner's own space never counts; only teammates' gains do. ΔA converts grid cells
into xT-zone units.

Each run is clamped at zero before any aggregation. A run in behind drags
centre-backs back into the box, which is high-xT space, so the raw net change
there is negative even though the run did its job of stretching the line. The
metric therefore credits space created and never penalises a run; `net_value`
keeps the unclamped figure for inspection. Thresholds live in core.config.SpaceConfig.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config import PITCH_CONTROL, SPACE, PitchControlConfig, SpaceConfig
from core.kinematics import runs, sample_interval
from core.pillars.transition import detect_turnovers
from core.pose import unit
from core.spatial import attacking_control, make_grid, xt_lookup
from core.tracking import PITCH_LENGTH, PITCH_WIDTH, TrackingData


@dataclass(frozen=True)
class DecoyResult:
    markers: np.ndarray      # (D,) bool, defenders who followed the runner
    value: float             # xT zones' worth of control teammates gained, clamped at 0
    net_value: float         # the same before clamping; negative if markers moved into danger


def find_markers(attackers: np.ndarray, defenders: np.ndarray, attacker_vel: np.ndarray,
                 defender_vel: np.ndarray, runner: int,
                 config: SpaceConfig = SPACE) -> np.ndarray:
    """(D,) bool: which defenders tracked the runner during the run.

    attackers (F, A, 2), defenders (F, D, 2) and their velocities in m/s over the
    run's F frames; `runner` indexes the attacker axis.
    """
    gaps = np.linalg.norm(defenders[:, :, None] - attackers[:, None], axis=-1)    # (F, D, A)
    nearest = np.where(np.isnan(gaps), np.inf, gaps).argmin(axis=2)
    share = (nearest == runner).mean(axis=0)

    runner_vel = attacker_vel[:, runner]
    min_speed = config.alignment_min_speed_kmh / 3.6
    moving = ((np.linalg.norm(runner_vel, axis=-1) > min_speed)[:, None]
              & (np.linalg.norm(defender_vel, axis=-1) > min_speed))
    cosine = np.sum(unit(defender_vel) * unit(runner_vel)[:, None], axis=-1)
    counted = moving & ~np.isnan(cosine)
    alignment = np.divide(np.where(counted, cosine, 0).sum(axis=0), counted.sum(axis=0),
                          out=np.full(defenders.shape[1], -np.inf), where=counted.any(axis=0))
    return (share >= config.marker_share) & (alignment >= config.marker_alignment)


def counterfactual_defenders(start: np.ndarray, end: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """(D, 2) defender positions at the end of the run had the markers ignored the runner."""
    rest = ~markers
    shift = end[rest].mean(axis=0) - start[rest].mean(axis=0) if rest.any() else 0.0
    ghost = end.copy()
    ghost[markers] = start[markers] + shift
    return ghost


def decoy_value(attackers: np.ndarray, defenders: np.ndarray, attacker_vel: np.ndarray,
                defender_vel: np.ndarray, runner: int, grid: np.ndarray, xt_surface: np.ndarray,
                cell_weight: float, config: SpaceConfig = SPACE,
                control: PitchControlConfig = PITCH_CONTROL) -> DecoyResult:
    """Decoy value of one run. Arrays cover the run's F frames, as in `find_markers`.

    grid (N, 2) cell centres, xt_surface (N,) the attacking team's xT at each cell,
    cell_weight the cell area in xT-zone units.
    """
    markers = find_markers(attackers, defenders, attacker_vel, defender_vel, runner, config)
    if not markers.any():
        return DecoyResult(markers, 0.0, 0.0)
    teammates = np.delete(attackers[-1], runner, axis=0)
    actual = attacking_control(teammates, defenders[-1], grid, control)
    ghost = attacking_control(teammates, counterfactual_defenders(defenders[0], defenders[-1], markers),
                              grid, control)
    net = float(np.nansum((actual - ghost) * xt_surface) * cell_weight)
    return DecoyResult(markers, max(0.0, net), net)


def off_ball_runs(tracking: TrackingData, speeds_kmh: np.ndarray, possession: np.ndarray,
                  config: SpaceConfig = SPACE) -> pd.DataFrame:
    """(player index, start row, end row) of every completed run made without the ball.

    A frame counts when the player's team is in possession, the ball is tracked,
    the player isn't their team's closest to the ball, and they're above run speed.
    A run must hold for min_run_duration and cover min_run_distance (shorter is a
    burst the player gave up on), and end at least ahead_of_ball metres goal-side of the
    ball: a centre-back stepping up behind play drags no one off a teammate, it is the
    line moving. Runs starting within transition_window of their team winning the ball
    are transition reactions, scored by the transition pillar, so they are left out
    here and never count twice.
    """
    dt = sample_interval(tracking.timestamps)
    to_ball = np.linalg.norm(tracking.positions - tracking.ball[:, None], axis=-1)   # (T, P)
    to_ball = np.where(np.isnan(to_ball), np.inf, to_ball)
    ball_known = ~np.isnan(tracking.ball).any(axis=1)

    candidate = np.zeros(to_ball.shape, dtype=bool)
    for team in np.unique(tracking.teams):
        members = tracking.team_mask(team)
        on_ball = np.zeros(to_ball.shape, dtype=bool)
        on_ball[np.arange(len(to_ball)), np.where(members, to_ball, np.inf).argmin(axis=1)] = True
        in_possession = (possession == team) & ball_known
        candidate[:, members] = (in_possession[:, None] & ~on_ball
                                 & (np.nan_to_num(speeds_kmh) >= config.run_speed_kmh))[:, members]

    needed = int(np.ceil(config.min_run_duration / dt - 1e-9))
    won = {}                                                # team -> times it won the ball
    for time, team in detect_turnovers(tracking.timestamps, possession):
        won.setdefault(team, []).append(time)

    def transition(p: int, start: int) -> bool:
        times = np.asarray(won.get(tracking.teams[p], []))
        since = tracking.timestamps[start] - times
        return bool(((since >= 0) & (since < config.transition_window)).any())

    def ahead(p: int, last: int) -> bool:
        sign = 1 if tracking.attacks_right(tracking.teams[p])[last] else -1
        return sign * (tracking.positions[last, p, 0] - tracking.ball[last, 0]) >= config.ahead_of_ball

    rows = [(p, start, end) for p in range(candidate.shape[1]) for start, end in runs(candidate[:, p])
            if end - start >= needed and tracking.periods[start] == tracking.periods[end - 1]
            and np.linalg.norm(tracking.positions[end - 1, p] - tracking.positions[start, p]) >= config.min_run_distance
            and ahead(p, end - 1) and not transition(p, start)]
    return pd.DataFrame(rows, columns=["player", "start", "end"])


def decoy_runs(tracking: TrackingData, possession: np.ndarray, xt: np.ndarray,
               velocities: np.ndarray | None = None, config: SpaceConfig = SPACE,
               control: PitchControlConfig = PITCH_CONTROL) -> pd.DataFrame:
    """Decoy value of every off-ball run in a match, one row per run.

    possession from `core.tracking.possession_by_frame`, xt from `core.metrics.build_xt`,
    velocities from `tracking.velocities()` (computed if not given).
    """
    velocities = tracking.velocities() if velocities is None else velocities
    speeds_kmh = np.linalg.norm(velocities, axis=-1) * 3.6
    grid, _ = make_grid(PITCH_LENGTH, PITCH_WIDTH, control.cell_size)
    ny, nx = xt.shape
    cell_weight = control.cell_size ** 2 / ((PITCH_LENGTH / nx) * (PITCH_WIDTH / ny))
    surfaces = {}

    results = []
    for run in off_ball_runs(tracking, speeds_kmh, possession, config).itertuples():
        team = tracking.teams[run.player]
        window = slice(run.start, run.end)
        on_pitch = ~np.isnan(tracking.positions[window]).any(axis=(0, 2))
        attacking = np.flatnonzero(on_pitch & (tracking.teams == team))
        defending = np.flatnonzero(on_pitch & (tracking.teams != team))
        if run.player not in attacking:
            continue
        direction = bool(tracking.attacks_right(team)[run.start])
        if (team, direction) not in surfaces:
            surfaces[team, direction] = xt_lookup(xt, grid, direction, PITCH_LENGTH, PITCH_WIDTH)

        result = decoy_value(
            tracking.positions[window][:, attacking], tracking.positions[window][:, defending],
            velocities[window][:, attacking], velocities[window][:, defending],
            int(np.flatnonzero(attacking == run.player)[0]), grid, surfaces[team, direction],
            cell_weight, config, control)
        results.append({
            "player": tracking.player_ids[run.player], "team": team,
            "start_time": tracking.timestamps[run.start],
            "duration": tracking.timestamps[run.end - 1] - tracking.timestamps[run.start],
            "markers": [tracking.player_ids[d] for d in defending[result.markers]],
            "decoy_value": result.value,
            "net_xt_change": result.net_value,
        })
    return pd.DataFrame(results)
