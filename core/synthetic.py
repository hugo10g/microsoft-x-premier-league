"""
Synthetic, football-realistic matches, written in Metrica Sports' CSV layout.

Every club, player, name and value here is fictional. Each player has hidden
attributes (speed, passing, work rate, pressing, finishing, awareness) that drive
a 25 Hz simulation, so the off-ball metrics can be checked against a known ground
truth: a player with a high work rate should, for example, react faster in
transitions, and an aware player should scan more before receiving.

The match engine, frame by frame:
    ball        held by a player, in flight (pass or shot), or dead (out of play,
                goal) until a restart; the holder decides when to pass, shoot or carry,
                sooner under pressure. A struck ball slows as it travels; it rolls into
                the receiver's stride, a save ends in the keeper's hands, a goal in the
                net, and a dead ball is brought back to its restart spot (never teleported)
    rules       fouls (free kicks, penalties in the box, yellow cards), offside at the
                moment of the pass, corners from saves parried and shots deflected
                behind, and opponents 9.15 m from a free kick or corner
    shape       each team holds its formation slots, a block that slides with the
                ball: higher and wider in possession, deeper and narrower without it
    roles       the nearest defenders press the ball (and close the shooting lane near
                goal), attackers make forward runs and markers follow them, a teammate
                moves into the space a dragged marker leaves, the pass receiver attacks
                the landing spot
    movement    every player steers towards their target under a top speed and an
                acceleration limit; after a turnover, low-work-rate players react late
    heads       every player's head follows the ball under a turn-rate limit; off the
                ball they make head checks (quick turns away from the ball and back)
                at a rate set by their awareness, more often when they may receive:
                the holder's likeliest pass options, and a pass's receiver in flight.
                A receiver who checked in the 5 s before the ball arrived is less
                likely to be dispossessed and passes better; one who didn't, worse

Output (one folder per match, named like the folder):
    <name>_RawTrackingData_Home_Team.csv / _Away_Team.csv / _RawEventsData.csv
    <name>_roster.csv   tracking id -> fictional name, club, position, market value
    <name>_HeadOrientation.csv   head yaw per frame and player, degrees in the pitch
                        frame (0 = facing +x, counter-clockwise), NaN while off the pitch
"""

import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 25
DT = 1 / FPS
L, W = 105.0, 68.0

# Formation slots: (detailed position, depth 0 = own goal .. 1 = most advanced,
# lateral +1 = left .. -1 = right in the attacking direction).
FORMATIONS = {
    "4-3-3": [("Goalkeeper", 0, 0), ("Left-Back", .25, .9), ("Centre-Back", .2, .35),
              ("Centre-Back", .2, -.35), ("Right-Back", .25, -.9), ("Defensive Midfield", .45, 0),
              ("Central Midfield", .6, .45), ("Central Midfield", .6, -.45),
              ("Left Winger", .88, .75), ("Right Winger", .88, -.75), ("Centre-Forward", 1.0, 0)],
    "4-4-2": [("Goalkeeper", 0, 0), ("Left-Back", .25, .9), ("Centre-Back", .2, .35),
              ("Centre-Back", .2, -.35), ("Right-Back", .25, -.9), ("Left Midfield", .62, .8),
              ("Central Midfield", .55, .3), ("Central Midfield", .55, -.3),
              ("Right Midfield", .62, -.8), ("Centre-Forward", .95, .25), ("Centre-Forward", .95, -.25)],
    "3-5-2": [("Goalkeeper", 0, 0), ("Centre-Back", .2, .6), ("Centre-Back", .18, 0),
              ("Centre-Back", .2, -.6), ("Left Midfield", .5, .9), ("Defensive Midfield", .45, 0),
              ("Central Midfield", .6, .4), ("Central Midfield", .6, -.4), ("Right Midfield", .5, -.9),
              ("Centre-Forward", .95, .25), ("Centre-Forward", .95, -.25)],
}
BENCH = ["Goalkeeper", "Centre-Back", "Central Midfield", "Left Winger", "Centre-Forward"]
BROAD = {"Goalkeeper": "Goalkeeper", "Left-Back": "Defender", "Right-Back": "Defender",
         "Centre-Back": "Defender", "Defensive Midfield": "Midfielder", "Central Midfield": "Midfielder",
         "Left Midfield": "Midfielder", "Right Midfield": "Midfielder", "Left Winger": "Forward",
         "Right Winger": "Forward", "Centre-Forward": "Forward"}

FIRST = ["Aiden", "Bastian", "Callum", "Dario", "Elias", "Felix", "Gael", "Hugo", "Ilan", "Jonas",
         "Kai", "Luca", "Mateo", "Nils", "Oskar", "Pavel", "Quinn", "Rafa", "Soren", "Teo",
         "Umar", "Viktor", "Wes", "Xavi", "Yann", "Zane", "Arlo", "Bram", "Cyrus", "Dane",
         "Emre", "Finn", "Goran", "Hamza", "Ivo", "Jude", "Kofi", "Lior", "Milo", "Noah"]
LAST = ["Abara", "Brenner", "Castell", "Dorval", "Eskildsen", "Falcao", "Gruber", "Hollis", "Ibarra",
        "Jansen", "Kovac", "Lindqvist", "Moreau", "Nakamura", "Okafor", "Petrov", "Quaresma", "Rocha",
        "Sander", "Toivonen", "Ulloa", "Varga", "Whitlock", "Xhaka", "Yilmaz", "Zielinski", "Achterberg",
        "Bellamy", "Cardoso", "Delacroix", "Ekström", "Fontaine", "Galván", "Hartmann", "Iversen",
        "Jovanović", "Kruger", "Lemaire", "Mbeki", "Novak", "Oyelaran", "Pellegrini", "Quist", "Ravel",
        "Szabo", "Tanaka", "Uribe", "Vidal", "Wexford", "Yamada"]

# The goal model the engine samples from: a shot's scoring chance by distance and
# angle to goal (scaled by the shooter's finishing). xT is later *learned* from the
# resulting events, not copied from here.
def goal_probability(distance: float, angle: float) -> float:
    return float(1 / (1 + np.exp(0.2 * distance - 0.4 - 1.1 * angle)))


# Head model. A head check turns away from the ball at a fast turn rate to look at
# someone who matters - a nearby opponent, or a teammate to pass to -
# follows it for a moment, then comes back. Only targets 70+ deg off the ball count
# (nearer ones are already in view), and with nothing worth checking there is no
# check. Glances are smaller looks (well under the 60 deg a scan detector needs) so
# not every head movement is a scan.
HEAD_TRACK_RATE = 360.0           # deg/s while following the ball
HEAD_CHECK_RATE = 600.0           # deg/s turning into and out of a check
CHECK_BASE, CHECK_PER_AWARENESS = 0.12, 0.6   # checks per second off the ball
CHECK_NEAR_BOOST = 1.5            # ...when their team has the ball within CHECK_NEAR_DISTANCE
CHECK_NEAR_DISTANCE = 30.0
# Before receiving: while a teammate has the ball, each one's checks rise by up to
# RECEIVE_BOOST (scaled by how likely the holder is to pass to them, from the same
# model choose_action uses); a pass's receiver keeps that boost while the ball is
# travelling, if at least FLIGHT_CHECK_MARGIN s remain to look back before it arrives.
# Aware players anticipate more: the boost is x(0.5 + awareness), 1 on average.
RECEIVE_BOOST = 2.0
FLIGHT_CHECK_MARGIN = 0.6
KEEPER_CHECK_SHARE = 0.25         # keepers scan rarely
GLANCE_RATE = 0.3                 # small looks per second
CHECK_MIN_ANGLE = 70.0            # deg off the ball bearing for a target to need a check
CHECK_RANGE = 35.0                # m: players further away aren't checked
CHECK_FALLOFF = 12.0              # m: nearer targets are likelier (weight exp(-d / falloff))
# Relevance of each target kind, (team in possession, team defending).
CHECK_WEIGHT = {"opponent": (1.5, 1.0), "teammate": (1.0, 0.6)}
OFFSET_TARGET = -1                # look_target value: a glance at an offset from the ball
HEAD_NOISE = 1.0                  # deg, measurement noise on the recorded yaw
# Off-ball runs by role: how often a player sets off (x work rate) and which kind of
# run they make. A run in behind aims past the opponent's last outfield defender, a
# run pulling wide heads for the near touchline, dropping deep comes back towards the
# ball. Centre-backs and keepers don't make runs.
RUN_STYLE = {   # role: (rate multiplier, (in behind, pulling wide, dropping deep))
    "Centre-Forward": (1.3, (0.6, 0.15, 0.25)),
    "Left Winger": (1.2, (0.35, 0.5, 0.15)), "Right Winger": (1.2, (0.35, 0.5, 0.15)),
    "Left Midfield": (1.0, (0.3, 0.55, 0.15)), "Right Midfield": (1.0, (0.3, 0.55, 0.15)),
    "Central Midfield": (0.9, (0.3, 0.15, 0.55)), "Defensive Midfield": (0.5, (0.1, 0.1, 0.8)),
    "Left-Back": (0.8, (0.25, 0.75, 0.0)), "Right-Back": (0.8, (0.25, 0.75, 0.0)),
}
RUN_KINDS = ("in_behind", "wide", "deep")
# Pressing by role: how eagerly a player steps out to press (their distance to the
# ball is divided by this when choosing pressers) and how hard they close down.
PRESS_DRIVE = {"Defensive Midfield": 1.8, "Central Midfield": 1.2, "Centre-Forward": 0.85,
               "Left Winger": 0.9, "Right Winger": 0.9, "Left Midfield": 1.0, "Right Midfield": 1.0,
               "Left-Back": 0.9, "Right-Back": 0.9, "Centre-Back": 0.75, "Goalkeeper": 0.5}
# Shooting, tuned so a match averages about 25 shots, a third on target, and
# 2.8 goals (a Premier League season's rates): how readily a player shoots from a
# given distance, and how often an on-target shot beats the keeper.
SHOT_TENDENCY = 1.5
FINISH_FACTOR = 5.0
# After a turnover, players near the ball react after REACTION_BASE + REACTION_SLOW *
# (1 - work rate) seconds, plus a random REACTION_JITTER (mean, s) per turnover, then sprint
# for TRANSITION_BURST s: attackers forward into space, defenders back goal-side of the ball.
REACTION_BASE, REACTION_SLOW, REACTION_JITTER = 0.15, 0.9, 0.4
TRANSITION_BURST, TRANSITION_RADIUS = 2.5, 30.0
SCAN_MEMORY = 5.0                 # seconds of checks a receiver benefits from
# Checks in that memory (0, 1, 2, 3+) -> (dispossession multiplier, pass-success logit shift).
SCAN_EFFECTS = {0: (1.4, -0.3), 1: (0.6, 0.25), 2: (0.45, 0.45), 3: (0.35, 0.65)}
# Ball physics. A struck ball slows at a constant rate (grass and air), so a flight covers
# s(t) = v0·t - a·t²/2: passes are struck harder the longer they are, shots harder still.
# A held ball moves to its holder's feet at up to CONTROL_SPEED, so a pass rolls into the
# receiver's stride instead of snapping to them.
PASS_DECEL, SHOT_DECEL = 1.6, 3.0          # m/s²
CONTROL_SPEED = 12.0                       # m/s
NET_DEPTH, MISS_RUN_ON = 1.5, 4.0          # m a goal carries into the net; a miss runs past the line
RESTART_PAUSE, RETURN_SPEED = 1.5, 10.0    # s a dead ball rests, then m/s back to its restart spot
# Release: time on the ball grows with the room around the holder (pressed players play
# early, free ones carry) and shrinks for better passers. ONE_TOUCH of pressed receptions
# are first-time passes, and a holder closed down to PRESSURE_RELEASE m plays within a
# reaction time.
ONE_TOUCH, PRESSURE_RELEASE = 0.25, 2.0
# Rules. A challenge at tackling distance is a foul FOUL_CHANCE x (0.6 + 0.8 pressing) of the
# time (x BOX_CARE inside the defender's own box, where it is a penalty); YELLOW_SHARE of fouls
# are booked, once per player (no second yellows or reds: it stays 11 v 11). A receiver beyond
# the second-last opponent, the ball and halfway when the pass is played is offside; passers
# mostly see it (OFFSIDE_AVOID off the option's score). Some saves are parried and some wide
# shots deflected behind for a corner.
FOUL_CHANCE, BOX_CARE, YELLOW_SHARE = 0.11, 1.0, 0.17
WALL_DISTANCE = 9.15                         # m opponents keep from a free kick or corner
SET_MARGIN = 3.0                             # m beyond a line players aim for (they stop short and drift)
PENALTY_SPOT, BOX_DEPTH, BOX_HALF_WIDTH = 11.0, 16.5, 20.16
PENALTY_ON_TARGET = 0.9
PENALTY_SCORES = (0.97, 0.7)    # scored when on target: keeper dived the wrong way, the right way
# A keeper facing a shot dives: a burst at DIVE_ACCEL m/s² up to DIVE_SPEED m/s, with no
# slowing into the spot (a dive, not a jog). At a penalty they guess a side and dive at the kick.
DIVE_ACCEL, DIVE_SPEED = 20.0, 6.5
# Bodies take up space: players closer than MATE_GAP (teammates) or RIVAL_GAP (opponents, who
# can be in contact in a duel) are pushed apart at SEPARATION_PUSH m/s per metre of overlap.
MATE_GAP, RIVAL_GAP, SEPARATION_PUSH = 2.0, 0.8, 8.0
SAVE_TO_CORNER, MISS_TO_CORNER = 0.4, 0.4
# Blocking: an outfield defender within BLOCK_REACH m of a shot's path, in the first 90% of
# it, blocks it BLOCK_CHANCE x (1 - distance / reach) of the time (nearest the shooter first).
# A block goes behind for a corner MISS_TO_CORNER of the time, else the blocker has it. Within
# SHOOTING_RANGE of goal, pressing defenders close the shooting lane instead of the pass forward.
BLOCK_REACH, BLOCK_CHANCE, SHOOTING_RANGE = 2.2, 0.85, 30.0
OVERHIT = 0.12                               # share of failed passes that run out of play
# Seconds the ball is dead before each restart can be taken (a match has ~55-60 min in play).
RESTART_WAIT = {"THROW IN": 16.0, "GOAL KICK": 24.0, "FREE KICK": 28.0, "CORNER KICK": 32.0,
                "PENALTY": 60.0, "KICK OFF": 55.0}
OFFSIDE_AVOID = 4.4


def wrap_degrees(angle):
    """Angles wrapped to [-180, 180)."""
    return (np.asarray(angle) + 180.0) % 360.0 - 180.0


def scan_effect(checks: int | None) -> tuple[float, float]:
    """(dispossession multiplier, pass-success logit shift) for a receiver's next decision.

    None (not just received a pass: a turnover or a restart) is neutral. No check
    in the last SCAN_MEMORY seconds makes the next action worse; checks make it
    better, more so the more of them (up to three), as in studies of elite scanning.
    Around two checks, the typical count, the effect is a halved dispossession risk
    and +0.45 on the pass-success logit.
    """
    if checks is None:
        return 1.0, 0.0
    return SCAN_EFFECTS[min(checks, 3)]


@dataclass
class Player:
    name: str
    club: str
    jersey: int
    detailed: str
    depth: float
    lateral: float
    speed: float        # top speed, m/s
    passing: float      # 0..1
    work_rate: float    # 0..1: how quickly and hard they react off the ball
    pressing: float     # 0..1: how readily they press, and how well they angle the press
    finishing: float    # 0..1
    awareness: float = 0.5  # 0..1: how often they scan (head checks) off the ball
    value: float = 0.0  # market value, €m (set by `value_players`)

    @property
    def position(self) -> str:
        return BROAD[self.detailed]


def make_squad(club: str, formation: str, rng: np.random.Generator, used: set) -> list[Player]:
    """Eleven starters in the club's formation, then five substitutes."""
    def name() -> str:
        while True:
            candidate = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
            if candidate not in used:
                used.add(candidate)
                return candidate
    slots = FORMATIONS[formation] + [(position, None, None) for position in BENCH]
    numbers = rng.permutation([n for n in range(2, 40) if n != 12])    # 1 and 12 are the keepers
    squad = []
    for k, (detailed, depth, lateral) in enumerate(slots):
        attacking = BROAD[detailed] in ("Forward", "Midfielder")
        squad.append(Player(
            name=name(), club=club, jersey=1 if k == 0 else 12 if k == 11 else int(numbers[k]),
            detailed=detailed, depth=depth if depth is not None else 0.5, lateral=lateral or 0.0,
            speed=float(rng.uniform(7.3, 9.4) if detailed != "Goalkeeper" else rng.uniform(6.2, 7.2)),
            passing=float(rng.beta(4, 3)), work_rate=float(rng.beta(3, 3)),
            pressing=float(rng.beta(3, 3) * (1.1 if attacking else 0.9)),
            finishing=float(rng.beta(4, 3) if BROAD[detailed] == "Forward" else rng.beta(2, 4)),
            # Its own stream (per club and slot), so adding it left every other draw unchanged.
            awareness=float(np.random.default_rng([zlib.crc32(club.encode()), k]).beta(3, 3))))
    return squad


def value_players(players: list[Player], rng: np.random.Generator) -> None:
    """Market values: mostly ability and position, plus a reputation term the metrics can't see.

    The noise term is what creates undervalued players for the Value Finder to spot.
    """
    position_bonus = {"Goalkeeper": -0.6, "Defender": -0.2, "Midfielder": 0.0, "Forward": 0.35}
    for p in players:
        ability = 0.27 * p.work_rate + 0.22 * p.pressing + 0.22 * p.passing + 0.18 * p.finishing \
            + 0.13 * (p.speed - 7.3) / 2.1 + 0.13 * p.awareness
        p.value = float(np.round(np.exp(1.0 + 3.2 * ability + position_bonus[p.position]
                                        + rng.normal(0, 0.45)), 1))


class Match:
    """One 25 Hz match between two squads. `run()` returns tracking and events."""

    def __init__(self, home: list[Player], away: list[Player], seed: int, half_minutes: float = 45.0):
        self.rng = np.random.default_rng(seed)
        self.half_minutes = half_minutes
        # 11 starters + 3 substitutes per side appear in the tracking.
        self.home, self.away = home, away
        # Three outfield substitutes per side: the bench keeper never comes on (only a keeper
        # can replace a keeper, and keepers aren't substituted here).
        subs = lambda squad: [squad[k] for k in self.rng.choice(
            [k for k in range(11, len(squad)) if squad[k].detailed != "Goalkeeper"], 3, replace=False)]
        self.players = home[:11] + subs(home) + away[:11] + subs(away)
        n = len(self.players)
        self.team = np.array([0] * 14 + [1] * 14)
        self.depth = np.array([p.depth for p in self.players])
        self.lateral = np.array([p.lateral for p in self.players])
        self.top_speed = np.array([p.speed for p in self.players])
        self.work = np.array([p.work_rate for p in self.players])
        self.press = np.array([p.pressing for p in self.players])
        self.keeper = np.array([p.detailed == "Goalkeeper" for p in self.players])
        self.awareness = np.array([p.awareness for p in self.players])
        self.drive = np.array([PRESS_DRIVE.get(p.detailed, 1.0) for p in self.players])
        self.reaction_jitter = np.zeros(n)
        self.active = np.array([k % 14 < 11 for k in range(n)])
        self.pos = np.zeros((n, 2))
        self.vel = np.zeros((n, 2))
        # Head: current yaw, the offset from the ball a look aims for, and until when.
        self.yaw = np.zeros(n)
        self.look_offset = np.zeros(n)
        self.look_target = np.full(n, OFFSET_TARGET)   # the player being checked, or OFFSET_TARGET
        self.look_until = np.zeros(n, dtype=int)      # end of the hold
        self.busy_until = np.zeros(n, dtype=int)      # end of the return to the ball
        self.check_onsets: list[list[int]] = [[] for _ in range(n)]
        self.holder_checks: int | None = None         # the holder's checks before receiving
        self.next_decision, self.rushed = 0, False    # when the holder acts; already closed down?
        self.booked: set[int] = set()                 # players with a yellow card
        self.runs: dict = {}                          # runner -> (aim, until, space they open, who takes it)
        self.flight = self.holder = self.receiver = self.restart = None
        self.diving: int | None = None               # the keeper diving at a shot in flight
        self.events: list[dict] = []
        self.schedule_substitutions()

    # ── Helpers ──────────────────────────────────────────────────────────────
    def schedule_substitutions(self) -> None:
        """Three changes per side in the second half, like for like where possible."""
        self.subs = []
        for side in (0, 1):
            base = 14 * side
            starters = [k for k in range(base + 1, base + 11)]
            for k, minute in zip(range(base + 11, base + 14), sorted(self.rng.uniform(55, 85, 3))):
                group = self.players[k].position
                same = [s for s in starters if self.players[s].position == group] or starters
                out = int(self.rng.choice(same))
                starters.remove(out)
                self.subs.append((minute, out, k))

    def attacks_right(self, side: int) -> bool:
        return (side == 0) == (self.period == 1)

    def side_sign(self, side: int) -> float:
        return 1.0 if self.attacks_right(side) else -1.0

    def goal_of(self, side: int) -> np.ndarray:
        """The goal `side` attacks."""
        return np.array([L if self.attacks_right(side) else 0.0, W / 2])

    def log(self, side: int, kind: str, subtype: str, start: int, end: int, who, to, a, b) -> None:
        jersey = lambda k: f"Player{self.players[k].jersey}" if k is not None else None
        self.events.append({
            "Team": "Home" if side == 0 else "Away", "Type": kind, "Subtype": subtype,
            "Period": self.period, "Start Frame": start + 1, "Start Time [s]": round((start + 1) * DT, 2),
            "End Frame": end + 1, "End Time [s]": round((end + 1) * DT, 2),
            "From": jersey(who), "To": jersey(to),
            "Start X": a[0] / L, "Start Y": 1 - a[1] / W, "End X": b[0] / L, "End Y": 1 - b[1] / W})

    # ── Shape ────────────────────────────────────────────────────────────────
    def shape_targets(self, kickoff: bool = False) -> np.ndarray:
        """Where every player would stand in their team's shape, given the ball."""
        target = np.zeros_like(self.pos)
        for side in (0, 1):
            s = self.side_sign(side)
            own = 0.0 if s > 0 else L
            ball_rel = (self.ball[0] - own) * s
            has_ball = side == self.possession
            if kickoff:
                back, span, width = 10.0, 38.0, 26.0
            elif has_ball:
                back, span, width = np.clip(ball_rel - 32, 8, 52), 46.0, 30.0
            else:
                back, span, width = np.clip(ball_rel - 24, 6, 44), 32.0, 21.0
            mine = self.team == side
            rel = np.where(self.keeper, 4.0, back + self.depth * span)
            if kickoff:
                rel = np.minimum(rel, L / 2 - 1.5)
            target[mine, 0] = own + s * rel[mine]
            target[mine, 1] = W / 2 + s * self.lateral[mine] * width + 0.3 * (self.ball[1] - W / 2)
            target[mine & self.keeper, 1] = W / 2 + 0.25 * (self.ball[1] - W / 2)
        target[:, 0] = np.clip(target[:, 0], 0.5, L - 0.5)
        target[:, 1] = np.clip(target[:, 1], 0.5, W - 0.5)
        return target

    def place_for_kickoff(self) -> None:
        self.ball = np.array([L / 2, W / 2])
        self.pos = self.shape_targets(kickoff=True)
        self.vel[:] = 0
        taker = self.nearest(self.possession, self.ball, outfield=True)
        self.pos[taker] = self.ball + np.array([-0.8 * self.side_sign(self.possession), 0])

    def strike(self, kind: str, here: np.ndarray, to: np.ndarray, v0: float, decel: float, frame: int,
               **fields) -> dict:
        """A flight from `here` to `to`, struck at v0 m/s and slowing at `decel` m/s² (raised if
        needed so it always arrives)."""
        distance = float(np.linalg.norm(to - here))
        v0 = max(v0, 1.05 * np.sqrt(2 * decel * distance))
        seconds = (v0 - np.sqrt(v0 ** 2 - 2 * decel * distance)) / decel
        return {"kind": kind, "from": here, "to": to, "start": frame, "end": frame + max(2, int(np.ceil(seconds * FPS))),
                "v0": v0, "decel": decel, **fields}

    def in_box(self, point: np.ndarray, side: int) -> bool:
        """Is `point` inside the penalty area `side` attacks?"""
        goal = self.goal_of(side)
        return abs(point[0] - goal[0]) <= BOX_DEPTH and abs(point[1] - W / 2) <= BOX_HALF_WIDTH

    def offside_line(self, side: int) -> float:
        """`side`'s offside line as a depth (x times the attacking direction): the second-last
        opponent, but never behind halfway or the ball."""
        s = self.side_sign(side)
        rivals = np.sort(self.pos[self.active & (self.team != side), 0] * s)
        return max(rivals[-2], L / 2 * s, self.ball[0] * s)

    def offside(self, k: int) -> bool:
        side = int(self.team[k])
        return bool(self.pos[k, 0] * self.side_sign(side) > self.offside_line(side) + 0.3)

    def blocker(self, here: np.ndarray, line: np.ndarray) -> tuple[int, float] | None:
        """(defender, fraction of the way) of whoever blocks a shot along `line`, if anyone."""
        defenders = np.flatnonzero(self.active & (self.team != self.possession) & ~self.keeper)
        along = ((self.pos[defenders] - here) @ line) / max(line @ line, 1e-6)
        gap = np.linalg.norm(self.pos[defenders] - (here + along[:, None] * line), axis=1)
        near = np.flatnonzero((along > 0.02) & (along < 0.9) & (gap < BLOCK_REACH))
        for k in near[np.argsort(along[near])]:                 # nearest the shooter first
            if self.rng.random() < BLOCK_CHANCE * (1 - gap[k] / BLOCK_REACH):
                return int(defenders[k]), float(along[k])
        return None

    def shoot(self, frame: int, aim: np.ndarray, on_target: bool, scores: bool, blockable: bool = True,
              dive: np.ndarray | None = None) -> None:
        """The holder strikes at `aim` on the goal line: blocked, into the net, to the keeper, or wide.

        The keeper dives at `dive` (a penalty's guess) or, by default, where the ball is going.
        """
        holder, side, here = self.holder, self.possession, self.ball.copy()
        keepers = np.flatnonzero(self.keeper & self.active & (self.team == 1 - side))
        keeper = int(keepers[0]) if len(keepers) else self.nearest(1 - side, aim)
        line = aim - here
        heading = line / max(np.linalg.norm(line), 1e-6)
        block = self.blocker(here, line) if blockable else None
        if block is not None:                          # into a defender
            to, on_target, scores = here + block[1] * line, False, False
        elif scores:                                   # on into the net
            to = aim + heading * NET_DEPTH / max(abs(heading[0]), 0.2)
        elif on_target:                                # where the keeper meets it
            catch = self.pos[keeper] if dive is None else dive
            reach = np.clip(((catch - here) @ line) / max(line @ line, 1e-6), 0.7, 1.0)
            to = here + reach * line
        else:                                          # wide, on past the line
            to = aim + heading * MISS_RUN_ON / max(abs(heading[0]), 0.2)
        to[1] = np.clip(to[1], -2.0, W + 2.0)
        power = 22 + 8 * self.players[holder].finishing + self.rng.normal(0, 2)
        dive = np.clip(to if dive is None else dive, [0.5, 0.5], [L - 0.5, W - 0.5])
        self.flight = self.strike("shot", here, to, power, SHOT_DECEL, frame, on_target=on_target,
                                  goal=scores, shooter=holder, keeper=keeper, dive=dive,
                                  blocker=block[0] if block is not None else None)
        self.diving = keeper if block is None else None
        self.holder = None

    def foul(self, fouler: int, frame: int) -> None:
        """`fouler` brings down the holder: a free kick there, or a penalty inside the box."""
        side, spot = self.possession, self.ball.copy()
        self.log(side, "FAULT RECEIVED", "", frame, frame, self.holder, None, spot, spot)
        if fouler not in self.booked and self.rng.random() < YELLOW_SHARE:
            self.booked.add(fouler)
            self.log(1 - side, "CARD", "YELLOW", frame, frame, fouler, None, spot, spot)
        if self.in_box(spot, side):
            mates = np.flatnonzero(self.active & (self.team == side) & ~self.keeper)
            taker = int(mates[np.argmax([self.players[k].finishing for k in mates])])
            penalty = self.goal_of(side) - np.array([self.side_sign(side) * PENALTY_SPOT, 0])
            self.dead_ball(side, penalty, frame, "PENALTY", taker=taker)
        else:
            self.dead_ball(side, np.clip(spot, [1.0, 1.0], [L - 1.0, W - 1.0]), frame, "FREE KICK")

    def take_penalty(self, frame: int) -> None:
        """The taker picks a side, the keeper guesses one and dives at the kick."""
        goal, s = self.goal_of(self.possession), self.side_sign(self.possession)
        side, guess = self.rng.choice([-1.0, 1.0], 2)
        on_target = self.rng.random() < PENALTY_ON_TARGET
        corner = side * (self.rng.uniform(0.8, 3.3) if on_target else self.rng.uniform(3.9, 5.5))
        scores = on_target and self.rng.random() < PENALTY_SCORES[int(guess == side)]
        self.shoot(frame, goal + np.array([0, corner]), on_target, scores, blockable=False,
                   dive=goal + np.array([-s * 0.6, guess * 2.8]))

    def corner(self, side: int, frame: int, near: np.ndarray) -> None:
        """A corner to `side` at the flag nearest `near`."""
        goal = self.goal_of(side)
        flag = np.array([np.clip(goal[0], 0.3, L - 0.3), 0.3 if near[1] < W / 2 else W - 0.3])
        self.dead_ball(side, flag, frame, "CORNER KICK")

    def hold_frames(self, k: int) -> int:
        """Frames player k keeps a ball they've just won or received (see ONE_TOUCH)."""
        rivals = self.active & (self.team != self.team[k])
        room = float(np.min(np.linalg.norm(self.pos[rivals] - self.pos[k], axis=1)))
        if room < 6 and self.rng.random() < ONE_TOUCH:
            seconds = self.rng.uniform(0.2, 0.45)
        else:
            mean = (0.55 + 0.15 * min(room, 12)) * (1.3 - 0.6 * self.players[k].passing)
            seconds = self.rng.gamma(2.0, mean / 2)
        return int(seconds * FPS)

    def nearest(self, side: int, point: np.ndarray, outfield: bool = False, exclude=None) -> int:
        mask = self.active & (self.team == side)
        if outfield:
            mask &= ~self.keeper
        if exclude is not None:
            mask[exclude] = False
        d = np.linalg.norm(self.pos - point, axis=1)
        d[~mask] = np.inf
        return int(np.argmin(d))

    # ── Ball actions ─────────────────────────────────────────────────────────
    def choose_action(self, frame: int) -> None:
        holder, side = self.holder, self.possession
        p = self.players[holder]
        here = self.ball.copy()                            # at the holder's feet
        goal = self.goal_of(side)
        to_goal = goal - here
        distance = float(np.linalg.norm(to_goal))
        opponents = np.flatnonzero(self.active & (self.team != side))
        pressure = float(np.min(np.linalg.norm(self.pos[opponents] - here, axis=1)))
        lose_factor, pass_shift = scan_effect(self.holder_checks)
        self.holder_checks = None

        in_front = to_goal[0] * self.side_sign(side) > 0.5          # not from on or behind the goal line
        if in_front and distance < 32 and self.rng.random() < SHOT_TENDENCY * np.clip(0.36 - distance / 90, 0.03, 0.3):
            angle = float(np.arctan2(7.32 * abs(to_goal[0]), to_goal @ to_goal - 3.66 ** 2))
            chance = goal_probability(distance, angle) * (0.6 + 0.8 * p.finishing)
            # Aim error grows with distance and shrinks with finishing: about a third of
            # shots hit the target. Only those can score, at a rate that puts the overall
            # conversion near a real league's (about one shot in ten).
            aim = goal + np.array([0, self.rng.normal(0, (4.0 + distance / 5) * (1.3 - 0.6 * p.finishing))])
            on_target = abs(aim[1] - W / 2) < 3.66
            self.shoot(frame, aim, on_target, on_target and self.rng.random() < min(0.9, chance * FINISH_FACTOR))
            return
        if pressure < 1.6:                                 # at tackling distance: foul, tackle or neither
            winner = self.nearest(1 - side, here)
            fouls = FOUL_CHANCE * (0.6 + 0.8 * self.press[winner]) * (BOX_CARE if self.in_box(here, side) else 1)
            roll = self.rng.random()
            if roll < fouls:
                self.foul(winner, frame)
                return
            if roll < fouls + 0.35 * (1.1 - p.passing) * lose_factor:
                self.log(side, "BALL LOST", "THEFT", frame, frame, holder, None, here, here)
                self.log(1 - side, "RECOVERY", "THEFT", frame, frame, winner, None, here, here)
                self.turnover(winner, frame)
                return

        mates, lead, length, weights = self.pass_options(holder)
        pick = int(self.rng.choice(len(mates), p=weights))
        receiver, end = int(mates[pick]), lead[pick]
        end = np.clip(end, [-2, -2], [L + 2, W + 2])

        # The lane: how close any opponent stands to the ball's path.
        seg = end - here
        t = np.clip(((self.pos[opponents] - here) @ seg) / max(seg @ seg, 1e-6), 0.05, 0.95)
        closest = here + t[:, None] * seg
        gaps = np.linalg.norm(self.pos[opponents] - closest, axis=1)
        lane = float(gaps.min())
        success = 1 / (1 + np.exp(-(1.4 + 1.6 * (lane - 1.1) + 2.2 * (p.passing - 0.5) - length[pick] / 30
                                    + pass_shift)))
        v0 = min(9 + 0.3 * length[pick], 26.0)
        flight = self.strike("pass", here, end, v0, PASS_DECEL, frame, passer=holder, receiver=receiver)
        if self.offside(receiver):                         # flagged when the receiver plays it
            flight.update(kind="offside", offside_at=self.pos[receiver].copy())
        elif self.rng.random() > success:
            if self.rng.random() < OVERHIT:                        # out of play
                out = here + seg * 1.6
                out[1] = np.clip(out[1], -1.5, W + 1.5)
                if 0 < out[1] < W:
                    out[1] = -1.0 if out[1] < W / 2 else W + 1.0
                flight = self.strike("out", here, out, 1.15 * v0, PASS_DECEL, frame, passer=holder,
                                     receiver=receiver)
            else:
                k = int(np.argmin(gaps))
                flight = self.strike("intercepted", here, closest[k], v0, PASS_DECEL, frame, passer=holder,
                                     receiver=receiver, interceptor=int(opponents[k]))
        self.flight, self.holder = flight, None
        self.receiver = flight.get("receiver")

    def pass_options(self, holder: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The holder's teammates and how likely each is to get the next pass.

        Returns (mates, lead point, pass length, probability): open, forward and short
        options score higher, keepers and passes over 45 m much lower.
        """
        side, here = int(self.team[holder]), self.pos[holder]
        opponents = np.flatnonzero(self.active & (self.team != side))
        mates = np.flatnonzero(self.active & (self.team == side))
        mates = mates[mates != holder]
        lead = self.pos[mates] + self.vel[mates] * 0.6
        span = lead - here
        length = np.linalg.norm(span, axis=1)
        forward = span[:, 0] * self.side_sign(side)
        openness = np.linalg.norm(lead[:, None, :] - self.pos[opponents][None, :, :], axis=2).min(axis=1)
        score = 0.06 * forward + 0.35 * np.minimum(openness, 8) - 0.025 * length \
            - 3.0 * self.keeper[mates] - 4.0 * (length > 45) \
            - OFFSIDE_AVOID * (self.pos[mates, 0] * self.side_sign(side) > self.offside_line(side) + 0.3)
        weights = np.exp((score - score.max()) / 0.8)
        return mates, lead, length, weights / weights.sum()

    def expecting_ball(self, frame: int) -> np.ndarray:
        """Per player, 0..1: how strongly they expect to receive the ball next.

        The likeliest pass option is 1 and others scale with their pass probability;
        a pass's receiver is 1 while it travels and there is time to look back.
        """
        expect = np.zeros(len(self.players))
        if self.holder is not None and self.restart is None:
            mates, _, _, chance = self.pass_options(self.holder)
            expect[mates] = chance / chance.max()
        elif self.flight is not None and self.flight["kind"] == "pass" \
                and (self.flight["end"] - frame) * DT >= FLIGHT_CHECK_MARGIN:
            expect[self.flight["receiver"]] = 1.0
        return expect

    def land_ball(self, frame: int) -> None:
        f, side = self.flight, self.possession
        self.flight, self.diving = None, None
        if f["kind"] == "pass":
            self.log(side, "PASS", "", f["start"], frame, f["passer"], f["receiver"], f["from"], f["to"])
            self.holder = f["receiver"]
            self.holder_checks = self.recent_checks(f["receiver"], frame)
            self.next_decision, self.rushed = frame + self.hold_frames(f["receiver"]), False
        elif f["kind"] == "offside":
            self.log(side, "PASS", "", f["start"], frame, f["passer"], None, f["from"], f["to"])
            self.log(side, "OFFSIDE", "", frame, frame, f["receiver"], None, f["offside_at"], f["offside_at"])
            self.dead_ball(1 - side, np.clip(f["offside_at"], [1.0, 1.0], [L - 1.0, W - 1.0]), frame, "FREE KICK")
        elif f["kind"] == "intercepted":
            self.log(side, "BALL LOST", "INTERCEPTION", f["start"], frame, f["passer"], None, f["from"], f["to"])
            self.log(1 - side, "RECOVERY", "INTERCEPTION", frame, frame, f["interceptor"], None, f["to"], f["to"])
            self.turnover(f["interceptor"], frame)
        elif f["kind"] == "out":
            self.log(side, "PASS", "", f["start"], frame, f["passer"], None, f["from"], f["to"])
            self.log(side, "BALL OUT", "", frame, frame, f["passer"], None, f["to"], f["to"])
            spot = np.array([np.clip(f["to"][0], 1, L - 1), np.clip(f["to"][1], 0, W)])
            self.dead_ball(1 - side, spot, frame, "THROW IN")
        else:                                                      # shot
            blocker = f["blocker"]
            subtype = "BLOCKED" if blocker is not None else "ON TARGET-GOAL" if f["goal"] \
                else "ON TARGET-SAVED" if f["on_target"] else "OFF TARGET-OUT"
            self.log(side, "SHOT", subtype, f["start"], frame, f["shooter"], None, f["from"], f["to"])
            keeper = f["keeper"]
            if blocker is not None and self.rng.random() < MISS_TO_CORNER:   # deflected behind
                self.log(1 - side, "BALL OUT", "", frame, frame, blocker, None, f["to"], f["to"])
                self.corner(side, frame, f["to"])
            elif blocker is not None:
                self.log(1 - side, "RECOVERY", "BLOCK", frame, frame, blocker, None, f["to"], f["to"])
                self.turnover(blocker, frame)
            elif f["goal"]:
                self.score[side] += 1
                self.possession = 1 - side
                self.dead_ball(1 - side, np.array([L / 2, W / 2]), frame, "KICK OFF", kickoff=True)
            elif f["on_target"] and self.rng.random() < SAVE_TO_CORNER:     # parried behind
                self.log(1 - side, "BALL OUT", "", frame, frame, keeper, None, f["to"], f["to"])
                self.corner(side, frame, f["to"])
            elif f["on_target"]:
                self.log(1 - side, "RECOVERY", "SAVED SHOT", frame, frame, keeper, None, f["to"], f["to"])
                self.turnover(keeper, frame)
            elif self.rng.random() < MISS_TO_CORNER:                         # deflected behind
                blocker = self.nearest(1 - side, f["to"], outfield=True)
                self.log(1 - side, "BALL OUT", "", frame, frame, blocker, None, f["to"], f["to"])
                self.corner(side, frame, f["to"])
            else:
                self.log(side, "BALL OUT", "", frame, frame, f["shooter"], None, f["to"], f["to"])
                spot = np.array([5.5 if f["to"][0] < L / 2 else L - 5.5, W / 2])
                self.dead_ball(1 - side, spot, frame, "GOAL KICK", taker=keeper)

    def turnover(self, new_holder: int, frame: int) -> None:
        self.possession = int(self.team[new_holder])
        self.holder = new_holder
        self.holder_checks = None
        self.last_turnover = frame
        self.reaction_jitter = self.rng.exponential(REACTION_JITTER, len(self.players))
        self.runs.clear()
        self.next_decision, self.rushed = frame + int(0.2 * FPS) + self.hold_frames(new_holder), False

    def dead_ball(self, side: int, spot: np.ndarray, frame: int, kind: str, wait: float | None = None,
                  kickoff: bool = False, taker: int | None = None) -> None:
        wait = RESTART_WAIT[kind] if wait is None else wait
        self.possession = side
        self.holder, self.flight = None, None             # the ball stays where it went dead for now
        self.restart = {"side": side, "spot": spot, "kind": kind, "ready": frame + int(wait * FPS),
                        "moves": frame + int(RESTART_PAUSE * FPS),
                        "taker": taker if taker is not None else self.nearest(side, spot, outfield=True),
                        "kickoff": kickoff}
        self.runs.clear()

    # ── Movement ─────────────────────────────────────────────────────────────
    def separation(self) -> np.ndarray:
        """(P, 2) m/s pushing apart players standing closer than MATE_GAP / RIVAL_GAP."""
        on = np.flatnonzero(self.active)
        gap = self.pos[on][:, None] - self.pos[on][None]                 # from each other player to me
        distance = np.linalg.norm(gap, axis=-1)
        np.fill_diagonal(distance, np.inf)
        need = np.where(self.team[on][:, None] == self.team[on][None], MATE_GAP, RIVAL_GAP)
        overlap = np.clip(need - distance, 0, None)
        # Two players on exactly the same spot step apart sideways, by index, rather than not at all.
        apart = np.where(distance[..., None] > 1e-6, gap / np.maximum(distance, 1e-6)[..., None],
                         np.stack([np.sign(on[:, None] - on[None]), np.zeros((len(on), len(on)))], axis=-1))
        push = np.zeros_like(self.pos)
        push[on] = (apart * overlap[..., None]).sum(axis=1) * SEPARATION_PUSH
        return push

    def steer(self, frame: int, targets: np.ndarray, urgency: np.ndarray) -> None:
        since = (frame - self.last_turnover) * DT
        slow = since < REACTION_BASE + REACTION_SLOW * (1 - self.work) + self.reaction_jitter   # late to react
        urgency = np.where(slow & ~self.keeper, urgency * 0.3, urgency)
        offset = targets - self.pos
        distance = np.linalg.norm(offset, axis=1, keepdims=True)
        speed = np.minimum(self.top_speed[:, None] * urgency[:, None], 0.7 * np.maximum(distance - 1.5, 0))
        desired = np.divide(offset, distance, out=np.zeros_like(offset), where=distance > 1e-6) * speed
        desired += self.separation()
        limit = np.full((len(self.pos), 1), 5.0)
        if self.diving is not None:                    # a dive: full speed into the spot
            k = self.diving
            desired[k] = offset[k] / max(float(distance[k, 0]), 1e-6) * min(DIVE_SPEED, distance[k, 0] / DT)
            limit[k] = DIVE_ACCEL
        accel = (desired - self.vel) / 0.45
        norm = np.linalg.norm(accel, axis=1, keepdims=True)
        accel = np.where(norm > limit, accel / np.maximum(norm, 1e-9) * limit, accel)
        self.vel += accel * DT + self.rng.normal(0, 0.05, self.vel.shape)
        self.pos += self.vel * DT

    def frame_targets(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        kickoff = self.restart is not None and self.restart["kickoff"]
        targets = self.shape_targets(kickoff=kickoff)
        far = np.linalg.norm(targets - self.pos, axis=1) > 14
        urgency = np.where(far, 0.5 + 0.5 * self.work, 0.3)
        side = self.possession
        s = self.side_sign(side)

        if self.restart is not None:
            return self.restart_targets(targets, urgency)
        ball_target = self.flight["to"] if self.flight else self.ball
        line = self.offside_line(side) - 0.8                  # attackers hold the line unless running
        receiver = self.flight.get("receiver") if self.flight else None
        holding = self.active & (self.team == side) & ~self.keeper & (targets[:, 0] * s > line)
        holding[[k for k in (self.holder, receiver) if k is not None]] = False
        targets[holding, 0] = line * s
        if self.holder is not None:                            # carry forward when there's room
            room = np.min(np.linalg.norm(self.pos[self.active & (self.team != side)] - self.pos[self.holder], axis=1))
            carry = self.pos[self.holder] + np.array([s * (6 if room > 4 else 1.5), self.rng.normal(0, 1)])
            carry = np.clip(carry, [1.0, 1.0], [L - 1.0, W - 1.0])            # never dribble off the pitch
            targets[self.holder], urgency[self.holder] = carry, 0.45 + 0.04 * min(room, 6)
        if self.holder is not None and np.linalg.norm(self.ball - self.pos[self.holder]) > 1.2:
            targets[self.holder], urgency[self.holder] = self.ball, 0.9      # meet the ball
        if self.flight and self.flight.get("receiver") is not None and self.flight["kind"] in ("pass", "offside"):
            targets[self.flight["receiver"]], urgency[self.flight["receiver"]] = self.flight["to"], 0.9
        if self.flight and self.flight["kind"] == "shot" and self.flight["blocker"] is None:   # the keeper dives
            targets[self.flight["keeper"]], urgency[self.flight["keeper"]] = self.flight["dive"], 1.0

        # Off-ball runs by role, and the markers who follow them.
        attackers = np.flatnonzero(self.active & (self.team == side) & ~self.keeper)
        for k in attackers:
            style = RUN_STYLE.get(self.players[k].detailed)
            if k == self.holder or k in self.runs or style is None:
                continue
            if self.rng.random() < DT * 0.045 * style[0] * (0.15 + 1.7 * self.work[k]):
                kind = RUN_KINDS[int(self.rng.choice(3, p=style[1]))]
                length = 1.5 + 2.5 * self.work[k] + self.rng.uniform(0, 1)      # hard workers run longer
                self.runs[k] = (self.run_aim(k, kind, side), frame + int(length * FPS)) + self.space_taker(k, side)
        for k, (aim, until, space, taker) in list(self.runs.items()):
            if frame > until or k == self.holder:
                del self.runs[k]
                continue
            targets[k], urgency[k] = aim, 0.7 + 0.3 * self.work[k]
            marker = self.nearest(1 - side, self.pos[k], outfield=True)
            if np.linalg.norm(self.pos[marker] - self.pos[k]) < 7 + 8 * self.work[k]:   # committed runs drag markers further
                goal_side = self.pos[k] - np.array([s * 1.5, 0])
                targets[marker] = 0.75 * goal_side + 0.25 * targets[marker]
                urgency[marker] = 0.6 + 0.35 * self.work[marker]
                if taker is not None and taker != self.holder and taker not in self.runs:   # into the space left
                    targets[taker] = [min(space[0] * s, line) * s, space[1]]
                    urgency[taker] = 0.3 + 0.25 * self.awareness[taker]       # a drift into it, not a sprint

        # Transition burst: players near the ball sprint once they've reacted (see steer).
        if (frame - self.last_turnover) * DT < TRANSITION_BURST:
            near = self.active & ~self.keeper & (np.linalg.norm(self.pos - self.ball, axis=1) < TRANSITION_RADIUS)
            for k in np.flatnonzero(near):
                if k != self.holder:
                    attacking = self.team[k] == side
                    targets[k] = self.pos[k] + [s * 15, 0] if attacking else self.ball + [s * 12, 0]
                    urgency[k] = 0.8 + 0.2 * self.work[k]

        # Pressing: the nearest defender (two if the team presses hard) closes the ball,
        # angling the run to screen the most dangerous forward option if they press well.
        defenders = np.flatnonzero(self.active & (self.team != side) & ~self.keeper)
        reach = np.linalg.norm(self.pos[defenders] - ball_target, axis=1)
        eager = self.drive[defenders] * (0.5 + self.press[defenders])
        order = defenders[np.argsort(reach / eager)]
        count = 2 if self.press[defenders].mean() > 0.55 else 1
        options = np.flatnonzero(self.active & (self.team == side) & ~self.keeper)
        forward_option = options[np.argmax(self.pos[options, 0] * s)]
        shooting = np.linalg.norm(self.goal_of(side) - ball_target) < SHOOTING_RANGE
        count = 2 if shooting else count
        for k in order[:count]:
            if not shooting and np.linalg.norm(self.pos[k] - ball_target) > 18 * min(self.drive[k], 1.4) * (0.6 + 0.8 * self.press[k]):
                continue
            screen = (self.goal_of(side) if shooting else self.pos[forward_option]) - ball_target
            screen = screen / max(np.linalg.norm(screen), 1e-6) * 1.6 * self.press[k]
            targets[k] = ball_target + screen
            urgency[k] = min(1.0, (0.45 + 0.45 * self.press[k]) * self.drive[k])
        return targets, urgency

    def restart_targets(self, targets: np.ndarray, urgency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Where players set up for a restart: the taker on the ball; for a penalty everyone else
        outside the box and WALL_DISTANCE from the spot, and the keeper on the line; for a corner
        both teams in the box; and opponents WALL_DISTANCE from a free kick or corner."""
        r = self.restart
        side, spot, kind = r["side"], r["spot"], r["kind"]
        s, goal = self.side_sign(side), self.goal_of(side)
        outfield = self.active & ~self.keeper
        if kind == "PENALTY":
            edge = goal[0] - s * (BOX_DEPTH + SET_MARGIN)
            inside = outfield & ((targets[:, 0] - edge) * s > 0)
            targets[inside, 0] = edge
            keeper = self.active & self.keeper & (self.team != side)
            targets[keeper] = [goal[0] - s * 0.3, W / 2]
        elif kind == "CORNER KICK":
            attack, defend = outfield & (self.team == side) & (self.depth > 0.35), outfield & (self.team != side)
            targets[attack, 0] = goal[0] - s * (6 + 6 * self.depth[attack])
            targets[attack, 1] = W / 2 + 14 * self.lateral[attack]
            targets[defend, 0] = goal[0] - s * (3 + 8 * (1 - self.depth[defend]))
            targets[defend, 1] = W / 2 + 12 * self.lateral[defend]
        if kind in ("FREE KICK", "CORNER KICK", "PENALTY"):
            away = targets - spot
            gap = np.linalg.norm(away, axis=1)
            # A penalty clears every outfielder (the taker is put back on the spot below).
            kept = self.active & ~self.keeper if kind == "PENALTY" else self.active & (self.team != side)
            close = kept & (gap < WALL_DISTANCE + SET_MARGIN)
            reach = WALL_DISTANCE + SET_MARGIN
            if kind == "PENALTY":                      # straight back from the spot, keeping their line
                across = targets[close, 1] - spot[1]
                targets[close, 0] = spot[0] - s * np.sqrt(np.clip(reach ** 2 - across ** 2, 0, None))
            else:
                direction = np.where(gap[:, None] > 1e-6, away / np.maximum(gap, 1e-6)[:, None], [-s, 0.0])
                targets[close] = spot + direction[close] * reach
        targets[:] = np.clip(targets, [0.5, 0.5], [L - 0.5, W - 0.5])
        targets[r["taker"]], urgency[r["taker"]] = spot, 0.8
        return targets, urgency

    def space_taker(self, runner: int, side: int) -> tuple[np.ndarray, int | None]:
        """(the spot the runner's marker stands on, the teammate nearest it within 18 m) at
        the start of a run: if the marker follows, that teammate moves into the space."""
        space = self.pos[self.nearest(1 - side, self.pos[runner], outfield=True)].copy()
        mates = np.flatnonzero(self.active & (self.team == side) & ~self.keeper)
        mates = mates[(mates != runner) & (mates != (self.holder if self.holder is not None else -1))]
        gaps = np.linalg.norm(self.pos[mates] - space, axis=1)
        return space, int(mates[np.argmin(gaps)]) if len(mates) and gaps.min() < 18 else None

    def run_aim(self, k: int, kind: str, side: int) -> np.ndarray:
        """Where player k's off-ball run of `kind` ("in_behind", "wide" or "deep") heads.

        Aims sit beyond where the run will end (runs stop on a timer), so the runner
        holds their speed through the space instead of braking onto a spot.
        """
        s = self.side_sign(side)
        here = self.pos[k]
        if kind == "in_behind":
            rivals = np.flatnonzero(self.active & (self.team != side) & ~self.keeper)
            line = (self.pos[rivals, 0] * s).max() * s if len(rivals) else here[0]  # last outfield defender
            depth = max((line - here[0]) * s + self.rng.uniform(8, 14), self.rng.uniform(16, 22))
            aim = here + np.array([s * depth, self.rng.uniform(-4, 4)])
        elif kind == "wide":
            touchline = W - self.rng.uniform(1.5, 4) if here[1] > W / 2 else self.rng.uniform(1.5, 4)
            aim = np.array([here[0] + s * self.rng.uniform(2, 8), touchline])
        else:                                                   # drop towards the ball to receive
            back = self.rng.uniform(14, 22)
            aim = here + np.array([-s * back, 0.3 * (self.ball[1] - here[1]) + self.rng.uniform(-3, 3)])
        return np.clip(aim, [2, 2], [L - 2, W - 2])

    # ── Heads ────────────────────────────────────────────────────────────────
    @staticmethod
    def toward(ball: np.ndarray, point: np.ndarray, speed: float) -> np.ndarray:
        """The ball moved towards `point` by at most `speed` m/s for one frame."""
        gap = point - ball
        step = np.linalg.norm(gap)
        return point.copy() if step <= speed * DT else ball + gap / step * speed * DT

    def recent_checks(self, k: int, frame: int) -> int:
        """Head checks player k started in the last SCAN_MEMORY seconds."""
        onsets, since, count = self.check_onsets[k], frame - SCAN_MEMORY * FPS, 0
        for onset in reversed(onsets):
            if onset <= since:
                break
            count += 1
        return count

    def face_ball(self) -> None:
        to_ball = self.ball - self.pos
        self.yaw = np.degrees(np.arctan2(to_ball[:, 1], to_ball[:, 0]))
        self.look_until[:] = self.busy_until[:] = 0

    def update_heads(self, frame: int) -> None:
        """Turn every head towards its target under a turn-rate limit; start new looks."""
        to_ball = self.ball - self.pos
        bearing = np.degrees(np.arctan2(to_ball[:, 1], to_ball[:, 0]))
        target = np.where(frame < self.look_until, bearing + self.look_offset, bearing)
        for k in np.flatnonzero((frame < self.look_until) & (self.look_target != OFFSET_TARGET)):
            point = self.pos[self.look_target[k]]
            target[k] = np.degrees(np.arctan2(point[1] - self.pos[k, 1], point[0] - self.pos[k, 0]))
        if self.holder is not None:                     # on the ball: look where they're going
            h = self.holder
            speed = np.linalg.norm(self.vel[h])
            target[h] = np.degrees(np.arctan2(self.vel[h, 1], self.vel[h, 0])) if speed > 1.0 \
                else (0.0 if self.attacks_right(int(self.team[h])) else 180.0)
        rate = np.where(frame < self.busy_until, HEAD_CHECK_RATE * DT, HEAD_TRACK_RATE * DT)
        turn = (target - self.yaw + 180.0) % 360.0 - 180.0
        self.yaw = (self.yaw + np.minimum(np.maximum(turn, -rate), rate) + 180.0) % 360.0 - 180.0

        free = self.active & (frame >= self.busy_until)
        if self.holder is not None:
            free[self.holder] = False
        near = (self.team == self.possession) & (np.hypot(to_ball[:, 0], to_ball[:, 1]) < CHECK_NEAR_DISTANCE)
        check_rate = (CHECK_BASE + CHECK_PER_AWARENESS * self.awareness) \
            * np.where(near, CHECK_NEAR_BOOST, 1.0) * np.where(self.keeper, KEEPER_CHECK_SHARE, 1.0) \
            * (1.0 + RECEIVE_BOOST * (0.5 + self.awareness) * self.expecting_ball(frame))
        draw = self.rng.random(len(self.players)) / DT
        checks = np.flatnonzero(free & (draw < check_rate))
        glances = np.flatnonzero(free & (draw >= check_rate) & (draw < check_rate + GLANCE_RATE))
        for k, is_check in [(k, True) for k in checks] + [(k, False) for k in glances]:
            if is_check:
                picked = self.check_target(k, bearing[k])
                if picked is None:                      # nothing out there worth a look
                    continue
                self.look_target[k], offset = picked
            else:
                offset = self.rng.uniform(15, 40)
                self.look_target[k] = OFFSET_TARGET
                # Glance to whichever side of the ball faces into the pitch, never out of it.
                centre = np.degrees(np.arctan2(W / 2 - self.pos[k, 1], L / 2 - self.pos[k, 0]))
                inward = abs(wrap_degrees(bearing[k] + offset - centre)) < abs(wrap_degrees(bearing[k] - offset - centre))
                self.look_offset[k] = offset if inward else -offset
            turn = offset / HEAD_CHECK_RATE
            self.look_until[k] = frame + int(round((turn + self.rng.uniform(0.2, 0.6)) * FPS))
            self.busy_until[k] = self.look_until[k] + int(round((turn + 0.1) * FPS))
            if is_check:
                self.check_onsets[k].append(frame)

    def check_target(self, k: int, ball_bearing: float) -> tuple[int, float] | None:
        """Who player k's head check looks at: (player index, degrees off the ball), or None.

        Candidates are other players on the pitch within CHECK_RANGE who sit
        CHECK_MIN_ANGLE or more off the ball bearing (nearer ones are already in view).
        Each is weighted by CHECK_WEIGHT (opponents matter more when k's team has the
        ball: they are who might close k down) and by distance; one is drawn at random.
        Only people are checked, never a goal or empty space, so a keeper with nobody
        beside or behind him doesn't check at all.
        """
        side = int(self.team[k])
        phase = 0 if self.possession == side else 1            # CHECK_WEIGHT column
        others = np.flatnonzero(self.active & (np.arange(len(self.players)) != k))
        offset = self.pos[others] - self.pos[k]
        distance = np.hypot(offset[:, 0], offset[:, 1])
        gap = np.abs(wrap_degrees(np.degrees(np.arctan2(offset[:, 1], offset[:, 0])) - ball_bearing))
        keep = (gap >= CHECK_MIN_ANGLE) & (distance <= CHECK_RANGE) & (distance > 1.0)
        if not keep.any():
            return None
        relevance = np.where(self.team[others] == side, CHECK_WEIGHT["teammate"][phase],
                             CHECK_WEIGHT["opponent"][phase])
        weight = np.where(keep, relevance * np.exp(-distance / CHECK_FALLOFF), 0.0)
        pick = int(self.rng.choice(len(others), p=weight / weight.sum()))
        return int(others[pick]), float(gap[pick])

    # ── Main loop ────────────────────────────────────────────────────────────
    def run(self) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
        """Simulate both halves. Returns (frame table, positions (T, 28, 2), ball (T, 2), events).

        Head yaw per frame, (T, 28) degrees with NaN while off the pitch, is kept in `self.head_yaw`.
        """
        stoppage = self.rng.uniform(1, 4, 2)
        lengths = [int((self.half_minutes + extra) * 60 * FPS) for extra in stoppage]
        total = sum(lengths)
        positions = np.full((total, len(self.players), 2), np.nan)
        ball = np.zeros((total, 2))
        head = np.full((total, len(self.players)), np.nan)
        periods = np.zeros(total, dtype=int)
        self.score = [0, 0]
        frame = 0
        for period, length in enumerate(lengths, 1):
            self.period = period
            self.possession = 0 if period == 1 else 1
            self.runs, self.flight, self.holder, self.receiver = {}, None, None, None
            self.last_turnover = frame
            self.place_for_kickoff()
            self.face_ball()
            self.holder_checks = None
            self.restart = None
            self.dead_ball(self.possession, np.array([L / 2, W / 2]), frame, "KICK OFF", 0.4, kickoff=True)
            for _ in range(length):
                minute = (self.half_minutes * (period - 1)) + (frame - sum(lengths[:period - 1])) * DT / 60
                for sub in [s for s in self.subs if s[0] <= minute]:
                    if self.holder != sub[1] and (self.flight is None or self.flight.get("receiver") != sub[1]):
                        _, out, inn = sub
                        self.active[out], self.active[inn] = False, True
                        self.pos[inn], self.vel[inn] = self.pos[out], 0
                        self.yaw[inn] = self.yaw[out]
                        self.depth[inn], self.lateral[inn] = self.depth[out], self.lateral[out]
                        self.runs.pop(out, None)
                        self.subs.remove(sub)

                landed = False
                if self.restart is not None:                   # waiting for a restart
                    r = self.restart
                    if frame >= r["moves"]:                    # the ball is brought back to the spot
                        self.ball = self.toward(self.ball, r["spot"], RETURN_SPEED)
                    if frame >= r["ready"] and np.linalg.norm(self.pos[r["taker"]] - r["spot"]) < 1.5 \
                            and np.allclose(self.ball, r["spot"]):
                        self.restart = None
                        self.holder, self.holder_checks = r["taker"], None
                        self.log(r["side"], "SET PIECE", r["kind"], frame, frame, r["taker"], None, r["spot"], r["spot"])
                        self.next_decision, self.rushed = frame + int(0.3 * FPS), True
                        if r["kind"] == "PENALTY":
                            self.take_penalty(frame)
                elif self.flight is not None:
                    f, landed = self.flight, True                # no control step on top this frame
                    line, tau = f["to"] - f["from"], (frame - f["start"]) * DT
                    covered = min(f["v0"] * tau - f["decel"] * tau ** 2 / 2, np.linalg.norm(line))
                    self.ball = f["to"].copy() if frame >= f["end"] else \
                        f["from"] + line * covered / max(np.linalg.norm(line), 1e-6)
                    if frame >= f["end"]:
                        self.land_ball(frame)
                elif self.holder is not None:
                    pressed = np.min(np.linalg.norm(self.pos[self.active & (self.team != self.possession)]
                                                    - self.pos[self.holder], axis=1)) < PRESSURE_RELEASE
                    if pressed and not self.rushed:            # closed down: play within a reaction time
                        self.next_decision = min(self.next_decision, frame + int(self.rng.uniform(0.15, 0.5) * FPS))
                        self.rushed = True
                    if frame >= self.next_decision and np.linalg.norm(self.ball - self.pos[self.holder]) < 1.0:
                        self.choose_action(frame)

                targets, urgency = self.frame_targets(frame)
                self.steer(frame, targets, urgency)
                self.pos = np.clip(self.pos, [-1.5, -1.5], [L + 1.5, W + 1.5])
                if self.holder is not None and not landed:     # at the feet, or rolling there
                    heading = self.vel[self.holder] / max(np.linalg.norm(self.vel[self.holder]), 0.5)
                    self.ball = self.toward(self.ball, self.pos[self.holder] + 0.4 * heading, CONTROL_SPEED)
                self.update_heads(frame)
                positions[frame, self.active] = self.pos[self.active] + self.rng.normal(0, 0.03, (self.active.sum(), 2))
                head[frame, self.active] = self.yaw[self.active]
                ball[frame], periods[frame] = self.ball, period
                frame += 1
        frames = pd.DataFrame({"Period": periods, "Frame": np.arange(1, total + 1),
                               "Time [s]": np.round(np.arange(1, total + 1) * DT, 2)})
        self.head_yaw = wrap_degrees(head + self.rng.normal(0, HEAD_NOISE, head.shape))
        return frames, positions, ball, pd.DataFrame(self.events)


# ── Writing in Metrica's layout ──────────────────────────────────────────────
def write_team_file(path: Path, side: str, players: list[Player], frames: pd.DataFrame,
                    positions: np.ndarray, ball: np.ndarray) -> None:
    jerseys = [p.jersey for p in players]
    header = [",,," + "".join(f"{side},," for _ in jerseys) + ",",
              ",,," + "".join(f"{j},," for j in jerseys) + ",",
              "Period,Frame,Time [s]," + "".join(f"Player{j},," for j in jerseys) + "Ball,"]
    columns = [frames["Period"].to_numpy(), frames["Frame"].to_numpy(), frames["Time [s]"].to_numpy()]
    for k in range(len(players)):
        columns += [positions[:, k, 0] / L, 1 - positions[:, k, 1] / W]
    columns += [ball[:, 0] / L, 1 - ball[:, 1] / W]
    table = pd.DataFrame(np.column_stack(columns))
    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        table.to_csv(f, header=False, index=False, float_format="%.5f", na_rep="NaN")


def write_head_file(path: Path, ids: list[str], frames: pd.DataFrame, yaw: np.ndarray) -> None:
    """Head yaw per frame, one column per tracking id, degrees to 1 decimal (NaN off the pitch)."""
    table = pd.concat([frames, pd.DataFrame(np.round(yaw, 1), columns=ids)], axis=1)
    table.to_csv(path, index=False, na_rep="NaN")


def write_match(folder: Path, home: list[Player], away: list[Player], seed: int,
                half_minutes: float = 45.0) -> dict:
    """Simulate one match and write it as a Metrica-style game folder."""
    sim = Match(home, away, seed, half_minutes)
    frames, positions, ball, events = sim.run()
    folder.mkdir(parents=True, exist_ok=True)
    name = folder.name
    for side, slice_ in (("Home", slice(0, 14)), ("Away", slice(14, 28))):
        write_team_file(folder / f"{name}_RawTrackingData_{side}_Team.csv", side, sim.players[slice_],
                        frames, positions[:, slice_], ball)
    events.to_csv(folder / f"{name}_RawEventsData.csv", index=False)
    ids = [f"{'Home' if k < 14 else 'Away'}_{p.jersey}" for k, p in enumerate(sim.players)]
    write_head_file(folder / f"{name}_HeadOrientation.csv", ids, frames, sim.head_yaw)
    roster = pd.DataFrame([{
        "player": ids[k], "name": p.name, "club": p.club,
        "jersey": p.jersey, "position": p.position, "detailed_position": p.detailed,
        "market_value_m": p.value} for k, p in enumerate(sim.players)])
    roster.to_csv(folder / f"{name}_roster.csv", index=False)
    return {"match": name, "home": home[0].club, "away": away[0].club,
            "score": f"{sim.score[0]}-{sim.score[1]}", "events": len(events)}
