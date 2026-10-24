"""
Verified match facts: the ground truth the match-report agents work from and are checked against.

Everything here is plain pandas, no language model. `build_match_facts` turns one
match's events, roster, key moments, per-player metrics and pipeline event tables
into a list of numbered facts (F1, F2, ...), each with its match clock when it
happened at a moment in time. The facts are about off-ball work first:

    score, team          the result and each team's shots and passing (context)
    team_offball         each team's off-ball totals: decoy runs and xT opened,
                         pressing lane-cut seconds, transition sprint rate, scans
                         per reception and receptions kept
    goal, build_up       each goal, and the scoring team's off-ball work in the same
                         possession (at most BUILD_UP_SECONDS before the shot):
                         decoy runs, the press that forced the regain, sprints into
                         the attack, the scorer's head checks before receiving
    <moment kinds>       core.moments key moments (decoy, press, transitions, scan)
    pillar_leader        the match's top player in each pillar
    performer            the highest Off-Ball Index scores in the match

The agents may only use these facts; `check_report` verifies that a written report
cites real moments, credits only the players in the cited facts, gives each of
them only the off-ball actions those facts record (a decoy run, press, sprint,
scan or ball regain, each next to the right player's name), and uses no numbers the facts don't contain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from core.config import MOMENTS
from core.moments import KINDS

# A number, with thousands separators read as one ("1,000" is 1000, not 1 and 0).
NUMBER = re.compile(r"(?<![\w.:,])\d{1,3}(?:,\d{3})+(?!\d)|(?<![\w.:])\d+(?:\.\d+)?")
CITATION = re.compile(r"\[(\d{1,3}:\d{2})\]")

BUILD_UP_SECONDS = 25.0     # longest build-up credited to a goal
PRESS_LEAD = 6.0            # seconds of pressing before a regain that count towards it
PRESS_MIN_CUT = 0.2         # ...and the lane-cut share that makes it worth crediting
SPRINT_MAX_LATENCY = 4.0    # slower than this, a sprint isn't a reaction to the turnover
TOP_PER_CLAUSE = 2          # decoy runs / sprints named per build-up
MIN_SPRINTS = 3             # sprints needed to lead the transition pillar
MIN_RECEPTIONS = 5          # receptions needed to lead the scanning pillar

POSSESSION_TYPES = {"PASS", "SHOT", "RECOVERY", "BALL LOST", "SET PIECE", "BALL OUT"}
REGAIN = {"INTERCEPTION": "with an interception", "THEFT": "with a tackle", "SAVED SHOT": "from a save"}
RESTART = {"THROW IN": "a throw-in", "GOAL KICK": "a goal kick", "KICK OFF": "the kick-off",
           "FREE KICK": "a free kick", "CORNER KICK": "a corner"}

# The pipeline tables each fact kind reads, with the columns it needs.
TABLE_COLUMNS = {
    "decoys": ["player", "time", "decoy_value"],
    "pressing": ["player", "time", "lane_cut", "seconds"],
    "transitions": ["player", "phase", "latency", "reached_sprint", "time"],
    "scanning": ["player", "time", "n_scans", "retained"],
    "pressures": ["player", "time", "intensity"],
    "obso": ["player", "time", "obso"],
    "runs": ["player", "time", "run_type"],
}


# Off-ball actions a report may credit, and the words that claim them.
ACTIONS = {
    "decoy": re.compile(r"\bdecoy", re.I),
    "press": re.compile(r"\bpress(?:ed|es|ing)?\b", re.I),
    "sprint": re.compile(r"\bsprint", re.I),
    "scan": re.compile(r"\bscan\w*|\b(?:head|shoulder)[- ]checks?\b|\bcheck(?:ed|ing|s)? (?:over )?"
                       r"(?:his|her|their|the) shoulders?\b|\blook(?:ed|ing|s)? (?:a)?round\b", re.I),
    "regain": re.compile(r"\bwon (?:the ball |it |possession )?back\b|\bregain\w*|\bintercept\w*"
                         r"|\brecover(?:ed|s|ing)? (?:the ball|possession|it)\b|\bball recover\w*"
                         r"|\btackl\w*", re.I),
}
MOMENT_ACTION = {"decoy": "decoy", "press": "press", "defensive_transition": "sprint",
                 "attacking_transition": "sprint", "scan": "scan"}


def _number(value: str) -> float:
    return float(value.replace(",", ""))


def clock(minute: float) -> str:
    """Match minute (second half counting on from 45) -> 'mm:ss', whole seconds truncated.

    Rounding to microseconds first absorbs float error (27 + 55/60 minutes is 27:55, not 27:54).
    """
    seconds = int(round(minute * 60, 6))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


@dataclass(frozen=True)
class Fact:
    id: str                       # "F7"
    kind: str                     # "score", "team", "goal", "build_up", ... or a core.moments kind
    text: str                     # one sentence, numbers included
    minute: float | None = None   # match minute, for facts that happened at a moment
    time: float | None = None     # tracking clock in seconds (to jump the pitch player)
    end: float | None = None
    player: str | None = None     # tracking id of the player involved
    name: str | None = None       # that player's name, for checking who a report credits
    names: tuple[str, ...] = ()   # other players the fact names (e.g. a goal's build-up)
    credits: tuple[tuple[str, str], ...] = ()   # (name, action) pairs a build-up credits

    @property
    def actions(self) -> set[tuple[str, str]]:
        """(name, action) for every off-ball action this fact credits (name None if unnamed)."""
        own = {(self.name, MOMENT_ACTION[self.kind])} if self.kind in MOMENT_ACTION else set()
        return own | set(self.credits)

    @property
    def clock(self) -> str | None:
        return clock(self.minute) if self.minute is not None else None

    @property
    def everyone(self) -> set[str]:
        """Every player name this fact credits."""
        return ({self.name} if self.name else set()) | set(self.names)

    def line(self) -> str:
        """How the fact is shown to an agent: 'F7 [23:14] text' (or without a clock)."""
        return f"{self.id} [{self.clock}] {self.text}" if self.clock else f"{self.id} {self.text}"


@dataclass
class MatchFacts:
    game: str
    home: str
    away: str
    score: tuple[int, int]
    facts: list[Fact] = field(default_factory=list)

    def by_id(self) -> dict[str, Fact]:
        return {f.id: f for f in self.facts}

    def of_kind(self, *kinds: str) -> list[Fact]:
        return [f for f in self.facts if f.kind in kinds]

    def lines(self, facts: list[Fact] | None = None) -> str:
        return "\n".join(f.line() for f in (self.facts if facts is None else facts))

    def allowed_numbers(self) -> set[float]:
        """Every number a report may use: those in the facts, small counts, and the minutes
        of real moments (a report may say "the 28th minute" for an event at 27:55)."""
        numbers = {_number(n) for f in self.facts for n in NUMBER.findall(f.text)}
        minutes = {float(int(f.minute) + k) for f in self.facts if f.minute is not None for k in (0, 1)}
        return numbers | minutes | {float(n) for n in range(0, 12)} | {45.0, 90.0}

    def player_names(self) -> set[str]:
        return set().union(*(f.everyone for f in self.facts)) if self.facts else set()


def _event_minutes(events: pd.DataFrame) -> pd.Series:
    """Match minute of each event, from its period's first event (the kick-off)."""
    start = events.groupby("Period")["Start Time [s]"].transform("min")
    return 45 * (events["Period"] - 1) + (events["Start Time [s]"] - start) / 60


def _table(tables: dict[str, pd.DataFrame] | None, key: str) -> pd.DataFrame | None:
    """A pipeline table with the columns its facts need, or None if it's missing or empty."""
    table = (tables or {}).get(key)
    if table is None or table.empty or not set(TABLE_COLUMNS[key]) <= set(table.columns):
        return None
    table = table.assign(side=table["player"].astype(str).str.split("_").str[0])
    if key == "scanning":       # receptions with too little head data count for nothing
        table = table[table["n_scans"].notna()
                      & (table["scan_rate"].notna() if "scan_rate" in table else True)]
        return table if len(table) else None
    return table


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _possession_start(events: pd.DataFrame, shot_index) -> pd.Series:
    """The first event of the possession that ended with the shot at `shot_index`.

    Walks back through the scoring team's on-ball events until the ball was last
    out of play, with the other team, or lost; a regain or a restart opens a possession.
    """
    ordered = events[events["Type"].isin(POSSESSION_TYPES)].sort_values("Start Time [s]", kind="stable")
    side = ordered.at[shot_index, "Team"]
    start = ordered.index.get_loc(shot_index)
    while start > 0 and ordered.iloc[start]["Type"] not in ("RECOVERY", "SET PIECE"):
        previous = ordered.iloc[start - 1]
        if previous["Team"] != side or previous["Type"] in ("BALL OUT", "BALL LOST"):
            break
        start -= 1
    return ordered.iloc[start]


def _press_before(pressing: pd.DataFrame | None, side: str, regain: float) -> pd.Series | None:
    """The scoring team's strongest press in the PRESS_LEAD seconds before a regain."""
    if pressing is None:
        return None
    frames = pressing[(pressing["side"] == side) & pressing["time"].between(regain - PRESS_LEAD, regain)]
    if frames.empty:
        return None
    spells = frames.assign(cut_seconds=frames["lane_cut"] * frames["seconds"]).groupby("player").agg(
        cut=("lane_cut", "mean"), duration=("seconds", "sum"), cut_seconds=("cut_seconds", "sum"),
        first=("time", "min"))
    best = spells.sort_values("cut_seconds", ascending=False).iloc[0]
    return best if best["duration"] >= 0.5 and best["cut"] >= PRESS_MIN_CUT else None


def build_up(goal: pd.Series, events: pd.DataFrame, tables: dict[str, pd.DataFrame] | None,
             name_of: dict[str, str]) -> tuple[list[str], list[tuple[str, str]], float]:
    """The scoring team's off-ball work in the possession that ended in a goal.

    Returns (clauses, (name, action) credits, playback start in tracking seconds). The window
    for decoys, sprints and scans runs from the possession's first event (at most
    BUILD_UP_SECONDS back) to the shot; playback starts earlier when a credited press began
    before that, so Watch shows the whole press.
    """
    side, shot = goal["Team"], float(goal["Start Time [s]"])
    name = lambda player: name_of.get(player, player)
    ago = lambda t: f"{shot - t:.0f} s before the shot"
    opened = _possession_start(events, goal.name)
    opened_at = float(opened["Start Time [s]"])
    start = max(shot - BUILD_UP_SECONDS, opened_at - 1.0)
    playback = start
    clauses: list[str] = []
    credits: list[tuple[str, str]] = []

    if opened["Type"] == "RECOVERY" and opened_at >= shot - BUILD_UP_SECONDS \
            and pd.notna(opened["From"]):
        winner = name(opened["From"])
        how = REGAIN.get(str(opened["Subtype"]), "")
        won = f"won the ball back{' ' + how if how else ''} {ago(opened_at)}"
        press = _press_before(_table(tables, "pressing"), side, opened_at)
        if press is None:
            clauses.append(f"{winner} {won}")
        elif press.name == opened["From"]:
            clauses.append(f"{winner} pressed, cutting {press['cut']:.0%} of the threat-weighted passing lanes "
                           f"for {press['duration']:.1f} s, and {won}")
        else:
            clauses.append(f"{name(press.name)}'s press cut {press['cut']:.0%} of the threat-weighted passing "
                           f"lanes for {press['duration']:.1f} s before {winner} {won}")
        if press is not None:
            credits.append((name(press.name), "press"))
            playback = min(start, float(press["first"]))
        credits.append((winner, "regain"))
    elif opened["Type"] == "SET PIECE" and opened_at >= shot - BUILD_UP_SECONDS:
        clauses.append(f"the move began from {RESTART.get(str(opened['Subtype']), 'a restart')} {ago(opened_at)}")

    decoys = _table(tables, "decoys")
    if decoys is not None:
        runs = decoys[(decoys["side"] == side) & decoys["time"].between(start, shot)
                      & (decoys["decoy_value"] >= 0.0005)]
        for run in runs.nlargest(TOP_PER_CLAUSE, "decoy_value").sort_values("time").itertuples():
            clauses.append(f"{name(run.player)}'s decoy run {ago(run.time)} opened {run.decoy_value:.3f} xT "
                           "of space for teammates")
            credits.append((name(run.player), "decoy"))

    moves = _table(tables, "transitions")
    if moves is not None:
        sprints = moves[(moves["side"] == side) & (moves["phase"] == "attacking") & moves["reached_sprint"]
                        & moves["latency"].between(MOMENTS.min_reaction, SPRINT_MAX_LATENCY)
                        & moves["time"].between(start, shot)]
        for move in sprints.nsmallest(TOP_PER_CLAUSE, "latency").itertuples():
            clauses.append(f"{name(move.player)} sprinted (20+ km/h) into the attack {move.latency:.1f} s "
                           "after the turnover")
            credits.append((name(move.player), "sprint"))

    scans = _table(tables, "scanning")
    if scans is not None:
        mine = scans[(scans["player"] == goal["From"]) & scans["time"].between(start, shot + 0.5)]
        if len(mine):
            last = mine.sort_values("time").iloc[-1]
            clauses.append(f"{name(goal['From'])} made {_plural(int(last['n_scans']), 'head check')} "
                           "in the 5 s before receiving")
            credits.append((name(goal["From"]), "scan"))
    return clauses, list(dict.fromkeys(credits)), playback


def team_offball_text(side: str, club: str, tables: dict[str, pd.DataFrame] | None) -> str | None:
    """One team's off-ball totals for the match, or None without any pipeline tables."""
    parts = []
    decoys = _table(tables, "decoys")
    if decoys is not None:
        mine = decoys[decoys["side"] == side]
        parts.append(f"{_plural(len(mine), 'decoy run')} opening {mine['decoy_value'].sum():.3f} xT for teammates")
    pressing = _table(tables, "pressing")
    if pressing is not None:
        mine = pressing[pressing["side"] == side]
        parts.append(f"{(mine['lane_cut'] * mine['seconds']).sum():.0f} s of threat-weighted passing lanes "
                     "cut while pressing")
    moves = _table(tables, "transitions")
    if moves is not None and (moves["side"] == side).any():
        rate = moves.loc[moves["side"] == side, "reached_sprint"].mean()
        parts.append(f"a sprint (20+ km/h) in {rate:.0%} of their players' transition reactions")
    scans = _table(tables, "scanning")
    if scans is not None:
        mine = scans[scans["side"] == side]
        if len(mine):
            parts.append(f"{mine['n_scans'].mean():.1f} head checks per reception and "
                         f"{mine['retained'].astype(float).mean():.0%} of receptions kept")
    runs = _table(tables, "runs")
    if runs is not None:
        mine = runs[runs["side"] == side]
        parts.append(f"{_plural(int((mine['run_type'] == 'Run in Behind').sum()), 'run')} in behind")
    pressure = _table(tables, "pressures")
    if pressure is not None:
        parts.append(f"{pressure.loc[pressure['side'] == side, 'intensity'].sum():.0f} m of ground closed "
                     "down on the ball carrier")
    return f"{club} off the ball: {', '.join(parts)}." if parts else None


def pillar_leaders(tables: dict[str, pd.DataFrame] | None) -> list[tuple[str, str]]:
    """(player, text without the name) for the match's top player in each pillar."""
    leaders = []
    decoys = _table(tables, "decoys")
    if decoys is not None:
        per = decoys.groupby("player")["decoy_value"].agg(["size", "sum"])
        best = per["sum"].idxmax()
        if per.at[best, "sum"] > 0:
            leaders.append((best, f"Decoy-run leader: {{}}: {_plural(int(per.at[best, 'size']), 'decoy run')} "
                                  f"opening {per.at[best, 'sum']:.3f} xT for teammates."))
    pressing = _table(tables, "pressing")
    if pressing is not None:
        per = (pressing["lane_cut"] * pressing["seconds"]).groupby(pressing["player"]).sum()
        if per.max() > 0:
            leaders.append((per.idxmax(), f"Pressing leader: {{}}: {per.max():.1f} s of threat-weighted "
                                          "passing lanes cut."))
    moves = _table(tables, "transitions")
    if moves is not None:
        sprints = moves[moves["reached_sprint"] & (moves["latency"] >= MOMENTS.min_reaction)]
        per = sprints.groupby("player")["latency"].agg(["size", "median"])
        per = per[per["size"] >= MIN_SPRINTS]
        if len(per):
            best = per["median"].idxmin()
            leaders.append((best, f"Transition leader: {{}}: a median {per.at[best, 'median']:.1f} s to "
                                  f"sprint over {int(per.at[best, 'size'])} sprints into transitions."))
    scans = _table(tables, "scanning")
    if scans is not None:
        per = scans.groupby("player").agg(receptions=("n_scans", "size"), scans=("n_scans", "mean"),
                                         kept=("retained", lambda r: r.astype(float).mean()))
        per = per[per["receptions"] >= MIN_RECEPTIONS]
        if len(per):
            best = per["scans"].idxmax()
            row = per.loc[best]
            leaders.append((best, f"Scanning leader: {{}}: {row['scans']:.1f} head checks per reception over "
                                  f"{int(row['receptions'])} receptions, {row['kept']:.0%} of them kept."))
    pressure = _table(tables, "pressures")
    if pressure is not None:
        per = pressure.groupby("player")["intensity"].agg(["size", "sum"])
        best = per["sum"].idxmax()
        leaders.append((best, f"Pressure-intensity leader: {{}}: closed down the ball carrier "
                              f"{_plural(int(per.at[best, 'size']), 'time')}, making up {per.at[best, 'sum']:.0f} m "
                              "of ground on them."))
    runs = _table(tables, "runs")
    if runs is not None:
        behind = runs[runs["run_type"] == "Run in Behind"].groupby("player").size()
        if len(behind):
            leaders.append((behind.idxmax(), f"Runs-in-behind leader: {{}}: {_plural(int(behind.max()), 'run')} "
                                             "past the last defender."))
    offered = _table(tables, "obso")
    if offered is not None:
        per = offered.groupby("player")["obso"].sum()
        if per.max() > 0:
            leaders.append((per.idxmax(), f"Off-ball scoring opportunity (OBSO) leader: {{}}: {per.max():.1f} "
                                          "xT-seconds of threat offered from free, reachable positions."))
    return leaders


def build_match_facts(game: str, events: pd.DataFrame, roster: pd.DataFrame,
                      moments: pd.DataFrame, board: pd.DataFrame, top: int = 3,
                      tables: dict[str, pd.DataFrame] | None = None) -> MatchFacts:
    """Number every fact worth reporting about one match.

    events: core.tracking.load_events output; roster: the match's <name>_roster.csv;
    moments: core.moments.key_moments for the match; board: per-player totals for the
    match after core.pipeline.off_ball_value, optionally with an in-match Off-Ball Index
    ("index"), which then ranks the performers (name, real_team, index, off_ball_value, ...);
    tables: the match's core.pipeline.score_game tables (decoys, pressing, transitions
    and, when present, scanning, pressures, runs and obso). Without tables the off-ball facts that need them
    (team_offball, pillar_leader and most of build_up) are left out.
    """
    name_of = dict(zip(roster["player"], roster["name"]))
    club_of_side = {side: roster.loc[roster["player"].str.startswith(side), "club"].iloc[0]
                    for side in ("Home", "Away")}
    club_of = lambda player: club_of_side.get(str(player).split("_")[0], str(player).split("_")[0])
    home, away = club_of_side["Home"], club_of_side["Away"]
    events = events.reset_index(drop=True)
    events = events.assign(minute=_event_minutes(events))
    facts: list[Fact] = []
    add = lambda kind, text, **kw: facts.append(Fact(f"F{len(facts) + 1}", kind, text, **kw))

    shots = events[events["Type"] == "SHOT"]
    goals = shots[shots["Subtype"].astype(str).str.contains("GOAL")].sort_values("Start Time [s]")
    score = (int((goals["Team"] == "Home").sum()), int((goals["Team"] == "Away").sum()))
    add("score", f"Final score: {home} {score[0]}-{score[1]} {away}.")

    for side, club in club_of_side.items():
        mine = events[events["Team"] == side]
        passes = int((mine["Type"] == "PASS").sum())
        completed = int(((mine["Type"] == "PASS") & mine["To"].notna()).sum())
        side_shots = shots[shots["Team"] == side]
        on_target = int(side_shots["Subtype"].astype(str).str.startswith("ON TARGET").sum())
        all_passes = int((events["Type"] == "PASS").sum())
        share = round(100 * passes / all_passes) if all_passes else 0
        rival = events[events["Team"] != side]
        fouls = int((rival["Type"] == "FAULT RECEIVED").sum())          # fouls this side committed
        yellows = int(((mine["Type"] == "CARD") & (mine["Subtype"] == "YELLOW")).sum())
        corners = int(((mine["Type"] == "SET PIECE") & (mine["Subtype"] == "CORNER KICK")).sum())
        offsides = int((mine["Type"] == "OFFSIDE").sum())
        add("team", f"{club}: {len(side_shots)} shots ({on_target} on target), {completed} of {passes} "
                    f"passes completed ({round(100 * completed / passes) if passes else 0}%), "
                    f"{share}% of the match's passes; {fouls} fouls, {yellows} yellow cards, "
                    f"{corners} corners, {offsides} offsides.")
    for side, club in club_of_side.items():
        text = team_offball_text(side, club, tables)
        if text:
            add("team_offball", text)

    running = {"Home": 0, "Away": 0}
    for index, goal in goals.iterrows():
        running[goal["Team"]] += 1
        scorer = name_of.get(goal["From"], goal["From"])
        tally = f"{home} {running['Home']}-{running['Away']} {away}"
        shot_end = float(goal["End Time [s]"]) + 3.0
        add("goal", f"Goal: {scorer} ({club_of_side[goal['Team']]}) scores; {tally}.",
            minute=goal["minute"], time=float(goal["Start Time [s]"]), end=shot_end, player=goal["From"],
            name=scorer)
        clauses, credits, start = build_up(goal, events, tables, name_of)
        if clauses:
            text = "; ".join(clauses)
        else:
            text = (f"no decoy run, pressing regain or transition sprint by {club_of_side[goal['Team']]} "
                    "was measured in the move")
        add("build_up", f"Build-up to {scorer}'s goal for {club_of_side[goal['Team']]} ({tally}): "
                        f"{text[0].upper() + text[1:]}.",
            minute=goal["minute"], time=start, end=shot_end, player=goal["From"], name=scorer,
            names=tuple(dict.fromkeys(n for n, _ in credits if n != scorer)), credits=tuple(credits))

    for row in moments.sort_values("time").itertuples():
        add(row.kind, f"{KINDS.get(row.kind, row.kind)}: {name_of.get(row.player, row.player)} "
                      f"({club_of(row.player)}): {row.detail}.",
            minute=float(row.minute), time=float(row.time), end=float(row.end), player=row.player,
            name=name_of.get(row.player, row.player))

    for player, text in pillar_leaders(tables):
        who = name_of.get(player, player)
        add("pillar_leader", text.format(f"{who} ({club_of(player)})"), player=player, name=who)

    if not board.empty and "off_ball_value" in board:
        by_index = "index" in board and board["index"].notna().any()
        ranked = board.sort_values("index" if by_index else "off_ball_value", ascending=False, na_position="last")
        for rating, row in zip(ranked["index"] if by_index else ranked["off_ball_value"], ranked.head(top).itertuples()):
            parts = [f"Off-Ball Index {rating:.0f}" if by_index and pd.notna(rating)
                     else f"pillar score {row.off_ball_value:.0f}"]
            if pd.notna(getattr(row, "decoy_p90", None)):
                parts.append(f"Decoy Run Value {row.decoy_p90:.3f} per 90")
            if pd.notna(getattr(row, "lane_cut_p90", None)):
                parts.append(f"Pressing Lane Cuts {row.lane_cut_p90:.0f} s per 90")
            if pd.notna(getattr(row, "scan_rate", None)):
                parts.append(f"Scanning {row.scan_rate:.2f} head checks per second before receiving")
            add("performer", f"Top off-ball performer: {row.name} ({row.real_team}), {', '.join(parts)}.",
                name=row.name)
    return MatchFacts(game, home, away, score, facts)


def normalise_citations(text: str, match: MatchFacts) -> str:
    """Put real event times into the [mm:ss] citation format.

    Models sometimes write "at 51:24" or "(51:24)" instead of "[51:24]". A time that
    matches a fact's clock (to within a second) is wrapped in brackets; anything else
    is left as written, so a wrong time still fails the check. Formatting only: no
    content changes.
    """
    known = {int(c.split(":")[0]) * 60 + int(c.split(":")[1]) for c in (f.clock for f in match.facts) if c}

    def bracket(found: re.Match) -> str:
        value = found.group(1)
        total = int(value.split(":")[0]) * 60 + int(value.split(":")[1])
        return f"[{value}]" if any(abs(total - k) <= 1 for k in known) else found.group(0)

    text = re.sub(r"\s*\[F\d+(?:\s*,\s*F\d+)*\]", "", text)                     # drop fact IDs like [F4, F9]
    text = re.sub(r"\((\d{1,3}:\d{2})\)", bracket, text)                     # (51:24) -> [51:24]
    return re.sub(r"(?<![\[\d:])(\d{1,3}:\d{2})(?![\]\d])", bracket, text)    # 51:24 -> [51:24]


Mention = tuple[int, int, frozenset[str]]   # (start, end, the player(s) it may refer to)


def _mentions(sentence: str, names: set[str], cited: set[str]) -> list[Mention]:
    """Where a sentence names players, in order.

    Full names count first. A bare surname ("Tanaka's press") refers to the cited player with
    that surname, falling back to every player with it. When two cited players share it, the
    surname is ambiguous whatever else the sentence says ("Hugo Tanaka scored after Tanaka's
    sprint" doesn't say which Tanaka sprinted), so the check asks for the full name; with one,
    it refers back to the most recent full name with that surname.
    """
    full_mentions = [(m.start(), m.end(), frozenset({n})) for n in names
                     for m in re.finditer(rf"\b{re.escape(n)}\b", sentence)]
    found = list(full_mentions)
    for surname in {n.split()[-1] for n in names}:
        group = {n for n in cited if n.split()[-1] == surname} \
            or {n for n in names if n.split()[-1] == surname}
        for m in re.finditer(rf"\b{re.escape(surname)}\b", sentence):
            if any(s <= m.start() < e for s, e, _ in found):
                continue
            earlier = [who for s, e, who in full_mentions
                       if e <= m.start() and next(iter(who)).split()[-1] == surname]
            ambiguous = len(group) > 1
            found.append((m.start(), m.end(),
                          frozenset(group) if ambiguous or not earlier else earlier[-1]))
    return sorted(found, key=lambda mention: mention[0])


def _actor(sentence: str, word: re.Match, mentions: list[Mention]) -> frozenset[str] | None:
    """The player(s) a sentence ties an action word to, or None if it ties it to nobody.

    In order: a possessive right before the word ("X's press", "X's quick interception")
    means X; a passive "by" ("the ball was won back by X", "a decoy run made by X") means X,
    unless X owns another noun ("set up by Y's press"); otherwise the nearest player named
    before the word ("X pressed"), together with anyone joined to them by "and" ("X and Y
    pressed" ties the press to both); otherwise the first named after it in the same clause
    ("the press from X"). A name after a comma is a new clause, so "After the press, X
    scored" ties the press to nobody.
    """
    after = [m for m in mentions if m[0] >= word.end()]
    before = [m for m in mentions if m[1] <= word.start()]
    if before and re.fullmatch(r"['’]s(?:\s+[a-z-]+){0,2}\s+", sentence[before[-1][1]:word.start()]):
        return before[-1][2]
    if after and re.fullmatch(r"(?:\s+(?:runs?|was|were|is|made|carried out|completed|won|back|it))*"
                              r"\s+by\s+", sentence[word.end():after[0][0]]) \
            and not re.match(r"['’]s\b", sentence[after[0][1]:]):
        return after[0][2]
    if before:
        who = set(before[-1][2])
        for previous, current in zip(reversed(before[:-1]), reversed(before)):
            if re.fullmatch(r",?\s+and\s+|,\s+", sentence[previous[1]:current[0]]):
                who |= previous[2]
            else:
                break
        return frozenset(who)
    if after and not re.search(r"[,;:\u2013\u2014]", sentence[word.end():after[0][0]]):
        return after[0][2]
    return None


def check_report(text: str, match: MatchFacts, min_citations: int = 2) -> list[str]:
    """Problems with a written report, empty if it's faithful to the facts.

    A citation [mm:ss] must match a fact's clock (to within a second); a sentence that
    cites a moment may only name players who are in the fact(s) at that time (a
    build-up fact credits everyone it names); a decoy run, press, sprint, scan or regain the
    sentence claims must be recorded at that moment, by the player the sentence ties it to
    (see _actor; same-surname players are told apart); every number outside citations must
    appear in the facts (allowing rounding); and the report must cite at least
    `min_citations` moments.
    """
    problems = []
    clocks = {f.clock for f in match.facts if f.clock}
    seconds = lambda c: int(c.split(":")[0]) * 60 + int(c.split(":")[1])
    known = {seconds(c) for c in clocks}
    cited = CITATION.findall(text)
    for c in sorted(set(cited)):
        if not any(abs(seconds(c) - k) <= 1 for k in known):
            problems.append(f"[{c}] doesn't match the time of any event in the facts.")
    if len(set(cited)) < min_citations:
        problems.append(f"Cite at least {min_citations} moments by writing their time in square "
                        f"brackets right after the sentence, exactly like [27:55].")

    # Attribution: who a sentence credits must match the moment it cites.
    names = match.player_names()
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        sentence_cites = CITATION.findall(sentence)
        if not sentence_cites:
            continue
        where = ", ".join("[" + c + "]" for c in sentence_cites)
        cited_facts = [f for f in match.facts if f.clock
                       and any(abs(seconds(f.clock) - seconds(c)) <= 1 for c in sentence_cites)]
        cited_names = set().union(set(), *(f.everyone for f in cited_facts))
        named = {n for n in names if n in sentence}
        for name in sorted(named):
            if name not in cited_names:
                about = ", ".join(sorted(cited_names)) or "nobody named"
                problems.append(f"The sentence citing {where} credits {name}, but that moment is about "
                                f"{about}. Cite {name}'s own moment.")
        # Actions: each decoy run, press, sprint, scan or regain the sentence claims must be one
        # the cited facts record, by the player the sentence ties it to (anyone, if it ties it to nobody).
        cited_people = set().union(set(), *(f.everyone for f in cited_facts))
        mentions = _mentions(sentence, names, cited_people)
        credited = set().union(set(), *(f.actions for f in cited_facts))
        for action, words in ACTIONS.items():
            doers = {n for n, a in credited if a == action}
            for word in words.finditer(sentence):
                if not doers:
                    problems.append(f"The sentence citing {where} mentions a {action}, but the facts record no "
                                    f"{action} at that moment. Only describe actions the cited fact states.")
                    break
                who = _actor(sentence, word, mentions)
                by = ", ".join(sorted(n for n in doers if n)) or "another player"
                if who and len(who) > 1 and not who <= doers:
                    problems.append(f"The sentence citing {where} gives a {action} to "
                                    f"'{word.group(0)}' by {' or '.join(sorted(who))}: write the full name "
                                    f"of the player who did it ({by}).")
                elif who and not who & doers:
                    problems.append(f"The sentence citing {where} credits a {action} to "
                                    f"{', '.join(sorted(who))}, but the {action} at that moment was by {by}. "
                                    "Put each player's name next to their own action.")

    allowed = match.allowed_numbers()
    for value in NUMBER.findall(CITATION.sub(" ", text)):
        number = _number(value)
        if not any(abs(number - a) <= max(0.051 * abs(a), 0.006) for a in allowed):
            problems.append(f"The number {value} doesn't appear in the facts.")
    return list(dict.fromkeys(problems))
