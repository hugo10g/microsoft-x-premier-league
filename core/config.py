"""
Every tunable threshold in one place, grouped by module.

Try other values without editing this file by passing a modified copy to any metric:

    from dataclasses import replace
    from core.config import TRANSITION
    latency_from_speeds(..., config=replace(TRANSITION, sprint_kmh=22.0))

Speeds are in km/h, angles in degrees, times in seconds, distances in metres.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class KinematicsConfig:
    smoothing_window: float = 0.3        # Savitzky-Golay window over frame-to-frame velocity
    polyorder: int = 1
    max_speed_kmh: float = 43.0          # ~12 m/s: anything faster is a tracking glitch
    max_gap: float = 0.5                 # longest missing stretch that may be interpolated


@dataclass(frozen=True)
class PitchControlConfig:
    softness: float = 3.0                # distance gap that shifts control from 50% to ~73%
    cell_size: float = 1.0               # grid resolution for tracking-based control


@dataclass(frozen=True)
class XTConfig:
    bins: tuple[int, int] = (16, 12)     # xT grid: columns along the length x rows across


@dataclass(frozen=True)
class PoseConfig:
    min_moving_speed_kmh: float = 5.0    # slower than this, velocity says nothing about facing


@dataclass(frozen=True)
class TransitionConfig:
    sprint_kmh: float = 20.0
    walk_kmh: float = 7.0
    min_sprint_duration: float = 1.0     # a sprint is held this long (a half-second burst isn't one)
    horizon: float = 10.0
    tau: float = 2.0                     # latency at which the reward falls to 1/e
    walk_penalty: float = 0.5
    max_ball_distance: float = 30.0      # only judge players this close to the ball at the turnover


@dataclass(frozen=True)
class SpaceConfig:
    run_speed_kmh: float = 15.0          # an off-ball run is at least this fast...
    min_run_duration: float = 1.0        # ...for at least this long
    min_run_distance: float = 8.0        # ...covering at least this far (shorter is an abandoned burst)
    transition_window: float = 3.0       # runs starting this soon after the team wins the ball are transitions
    ahead_of_ball: float = 0.0           # a run must end at least this far goal-side of the ball (m)
    marker_share: float = 0.6            # share of the run a defender has the runner as nearest attacker
    marker_alignment: float = 0.5        # mean cosine between marker and runner velocity
    alignment_min_speed_kmh: float = 5.0 # frames slower than this don't count towards alignment


@dataclass(frozen=True)
class PressingConfig:
    projection: float = 0.5              # seconds the presser is projected ahead
    reach: float = 1.5                   # half-width a defender can block
    max_press_distance: float = 15.0     # further from the carrier than this isn't a press


@dataclass(frozen=True)
class ScanningConfig:
    on_angle: float = 60.0               # head this far off the ball bearing starts a scan...
    off_angle: float = 40.0              # ...which ends when it comes back inside this (hysteresis)
    min_duration: float = 0.1            # a scan lasts at least this...
    max_duration: float = 2.0            # ...and at most this (longer is facing away, not a check)
    min_peak_speed: float = 90.0         # deg/s: the head must turn quickly, not drift
    smoothing: float = 0.12              # vector-average window on the head direction
    max_gap: float = 0.2                 # longest missing stretch of yaw that is interpolated
    min_ball_distance: float = 1.0       # closer to the ball than this, its bearing is undefined
    window: float = 5.0                  # seconds before a reception that count
    min_observed: float = 0.6            # share of the window needing head data for a rate
    fixation_angle: float = 30.0         # within this of the ball bearing is looking at the ball


@dataclass(frozen=True)
class PressureConfig:
    radius: float = 10.0                 # a defender this close to the carrier can pressure them
    min_closing_kmh: float = 5.0         # ...while closing them down faster than this
    min_duration: float = 0.4            # a pressure lasts at least this
    control_distance: float = 2.0        # the carrier is the attacker within this of the ball


@dataclass(frozen=True)
class ReceiptConfig:
    in_space: float = 5.0                # nearest opponent at least this far: received in space


@dataclass(frozen=True)
class ResponsibilityConfig:
    spread: float = 8.0                  # Gaussian width sharing a zone between neighbouring slots


@dataclass(frozen=True)
class OBSOConfig:
    every: float = 0.2                   # seconds between sampled frames
    reach: float = 25.0                  # metres: the ball reaching a player falls off as exp(-d / reach)


@dataclass(frozen=True)
class RunTypeConfig:
    min_distance: float = 4.0            # shorter runs are left unclassified
    dominance: float = 0.6               # share of the run's direction one axis must hold
    behind_margin: float = 2.0           # a run in behind ends within this of the last defender, or past him


@dataclass(frozen=True)
class RosterConfig:
    defender_share: float = 0.4          # deepest share of a team's outfielders labelled defenders
    forward_share: float = 0.25          # most advanced share labelled forwards; the rest midfielders


@dataclass(frozen=True)
class MomentsConfig:
    per_kind: int = 8                    # moments kept per kind and game
    min_gap: float = 15.0                # seconds between two moments of the same kind
    press_min_cut: float = 0.5           # lane-cut share that counts as a strong press frame...
    press_min_seconds: float = 1.0       # ...held for at least this long
    elite_latency: float = 1.5           # seconds to a sprint for an elite transition...
    min_reaction: float = 0.2            # ...but not quicker: below this they were already sprinting
    scan_min_checks: int = 3             # head checks before a kept reception for an elite scan
    lead_in: float = 3.0                 # seconds of build-up shown before a moment


KINEMATICS = KinematicsConfig()
PITCH_CONTROL = PitchControlConfig()
XT = XTConfig()
POSE = PoseConfig()
TRANSITION = TransitionConfig()
SPACE = SpaceConfig()
PRESSING = PressingConfig()
SCANNING = ScanningConfig()
PRESSURE = PressureConfig()
RECEIPTS = ReceiptConfig()
RESPONSIBILITY = ResponsibilityConfig()
OBSO = OBSOConfig()
RUN_TYPES = RunTypeConfig()
ROSTER = RosterConfig()
MOMENTS = MomentsConfig()
