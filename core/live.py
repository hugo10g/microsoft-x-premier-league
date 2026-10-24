"""
Live match intelligence: one match replayed as its events arrive, minute by minute.

    pass_quality    every pass: distance, completed, how tight the lane was, difficulty 0-100
    shot_speeds     every shot: the ball's peak speed (km/h, from tracking)
    event_feed      the events worth showing, in order: a commentary line, why it matters,
                    and whether it's a key event (shown to every fan) or detail (analysts)
    state_at        the match up to a minute: score and both teams' running statistics
    rhythm          possession share and changes of possession per spell (control vs chaos)
    overlay_items   the feed as timed, machine-readable cues for on-screen graphics
    translate       feed lines into another language through the configured model

Everything is computed from the event file and the tracking, so the feed can be cut at
any minute: what a viewer sees at 63:00 uses only what had happened by 63:00.

Pass difficulty, 0-100, uses only what is known when the ball is struck, the same way
for every pass: 60% how tight the lane is (an opponent on the line = 1, 5 m clear = 0),
20% how pressed the passer is (an opponent on them = 1, 5 m away = 0) and 20% how
forward it goes (straight at goal = 1, square or back = 0). On the synthetic league the
failure rate rises steadily with it: under 1% of passes rated 0-30 fail, about a third
of those rated 60+. (Length isn't used: an intercepted pass's intended length is unknown.)
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from core.match_tools import MatchFacts, clock
from core.moments import KINDS
from core.tracking import TrackingData

RHYTHM_SPELL = 3.0                # minutes per bar of the rhythm chart
SHOT_WORDS = {"ON TARGET-GOAL": "scores", "ON TARGET-SAVED": "forces a save", "OFF TARGET-OUT": "fires wide",
              "BLOCKED": "has the shot blocked"}
# Why each kind of key moment matters, in a sentence a casual fan can follow.
WHY = {
    "decoy": "A run without the ball that dragged a defender away and opened space for teammates.",
    "press": "A press that cut off the carrier's passing lanes and forced a worse option.",
    "defensive_transition": "A sprint back the moment the ball was lost, before the opponent could break.",
    "attacking_transition": "An instant sprint forward after winning the ball, to attack before the defence set.",
    "scan": "Head checks before the ball arrived: the player knew the picture and kept possession.",
}
FEED_COLUMNS = ["time", "minute", "clock", "kind", "tier", "team", "player", "name", "text", "why", "data"]


def _frame(tr: TrackingData, event_frame) -> np.ndarray:
    return np.clip(np.searchsorted(tr.frames, np.asarray(event_frame)), 0, len(tr.frames) - 1)


def pass_quality(tr: TrackingData, events: pd.DataFrame) -> pd.DataFrame:
    """One row per pass: time, team, player, receiver, distance (m), completed, lane (m), difficulty, speed_kmh."""
    # An intercepted pass is logged as BALL LOST / INTERCEPTION: it was a pass, just not completed.
    passes = events[(events["Type"] == "PASS") | ((events["Type"] == "BALL LOST") & (events["Subtype"] == "INTERCEPTION"))]
    rows = []
    columns = ["Start Time [s]", "End Time [s]", "Start Frame", "Team", "From", "To", "Start X", "Start Y", "End X", "End Y"]
    for t0, t1, event_frame, team, src, dst, sx, sy, ex, ey in passes[columns].itertuples(index=False):
        start, end = np.array([sx, sy]), np.array([ex, ey])
        line = end - start
        length = float(np.linalg.norm(line))
        f = _frame(tr, event_frame)
        rivals = tr.positions[f, tr.teams != team]
        rivals = rivals[~np.isnan(rivals).any(axis=1)]
        t = np.clip(((rivals - start) @ line) / max(line @ line, 1e-6), 0.05, 0.95)
        lane = float(np.min(np.linalg.norm(rivals - (start + t[:, None] * line), axis=1))) if len(rivals) else 10.0
        pressed = float(np.min(np.linalg.norm(rivals - start, axis=1))) if len(rivals) else 10.0
        forward = max(0.0, line[0] / max(length, 1e-6) * (1 if tr.attacks_right(team)[f] else -1))
        difficulty = 100 * (0.6 * np.clip(1 - lane / 5, 0, 1) + 0.2 * np.clip(1 - pressed / 5, 0, 1) + 0.2 * forward)
        rows.append((t0, team, src, dst if isinstance(dst, str) else None, length, isinstance(dst, str), lane, round(difficulty),
                     length / max(t1 - t0, 0.04) * 3.6))
    return pd.DataFrame(rows, columns=["time", "team", "player", "receiver", "distance", "completed", "lane",
                                       "difficulty", "speed_kmh"])


def shot_speeds(tr: TrackingData, events: pd.DataFrame) -> pd.Series:
    """Peak ball speed (km/h) of every shot, indexed like `events`."""
    shots = events[events["Type"] == "SHOT"]
    dt = float(np.median(np.diff(tr.timestamps[:50])))
    speeds = {}
    for k, start, end in zip(shots.index, shots["Start Frame"], shots["End Frame"]):
        a, b = _frame(tr, start), _frame(tr, end)
        steps = np.linalg.norm(np.diff(tr.ball[a:b + 1], axis=0), axis=1)
        speeds[k] = float(np.nanmax(steps) / dt * 3.6) if len(steps) and np.isfinite(steps).any() else np.nan
    return pd.Series(speeds, dtype=float)


def event_feed(events: pd.DataFrame, tr: TrackingData, minutes: np.ndarray, moments: pd.DataFrame,
               facts: MatchFacts, names: dict[str, str], clubs: dict[str, str]) -> pd.DataFrame:
    """The live feed (columns FEED_COLUMNS), in time order.

    names: tracking id -> player name; clubs: "Home"/"Away" -> club name. Goals carry the
    assist's pass quality and the build-up from `facts`; shots carry their speed.
    """
    minute_of = lambda seconds: float(np.interp(seconds, tr.timestamps, minutes))
    passes, speeds = pass_quality(tr, events), shot_speeds(tr, events)
    build_ups = {round(f.minute, 2): f.text for f in facts.of_kind("build_up") if f.minute is not None}
    rows, score = [], {"Home": 0, "Away": 0}
    name = lambda pid: names.get(pid, pid) if isinstance(pid, str) else "a player"
    ordered = events.sort_values("Start Frame", kind="stable")
    previous = ordered.shift(1)
    for k, e in ordered.iterrows():
        kind, sub, side, t = e["Type"], str(e["Subtype"]), e["Team"], float(e["Start Time [s]"])
        row = None
        if kind == "SHOT":
            speed = speeds.get(k, np.nan)
            data = {"speed_kmh": None if np.isnan(speed) else round(speed)}
            penalty = previous.loc[k, "Type"] == "SET PIECE" and previous.loc[k, "Subtype"] == "PENALTY"
            if "GOAL" in sub:
                score[side] += 1
                assist = passes[(passes["receiver"] == e["From"]) & (passes["time"] <= t) & passes["completed"]
                                & (passes["time"] >= t - 15)].tail(1)
                text = (f"GOAL! {name(e['From'])} scores {'from the penalty spot ' if penalty else ''}for "
                        f"{clubs[side]}: {clubs['Home']} {score['Home']}-{score['Away']} {clubs['Away']}.")
                if len(assist):
                    a = assist.iloc[0]
                    text += f" Assisted by {name(a['player'])}'s {a['distance']:.0f} m pass (difficulty {a['difficulty']}/100)."
                    data.update(assist_distance=round(a["distance"]), assist_difficulty=int(a["difficulty"]))
                why = build_ups.get(round(minute_of(t), 2), "")
                row = ("goal", "key", text, why, data)
            else:
                speed_text = f" ({data['speed_kmh']} km/h)" if data["speed_kmh"] else ""
                row = ("shot", "key" if sub.startswith("ON TARGET") else "detail",
                       f"{name(e['From'])} {SHOT_WORDS.get(sub, 'shoots')}{speed_text}.", "", data)
        elif kind == "CARD":
            fouled = ordered[(ordered["Type"] == "FAULT RECEIVED") & (ordered["Start Frame"] == e["Start Frame"])]
            why = f"For a foul on {name(fouled['From'].iloc[0])}." if len(fouled) else ""
            row = ("card", "key", f"Yellow card: {name(e['From'])} ({clubs[side]}).", why, {})
        elif kind == "SET PIECE" and sub == "PENALTY":
            row = ("penalty", "key", f"Penalty to {clubs[side]}! {name(e['From'])} steps up.",
                   "A foul inside the box: the best chance in football, scored about three times in four.", {})
        elif kind == "SET PIECE" and sub == "CORNER KICK":
            row = ("corner", "detail", f"Corner to {clubs[side]}.", "", {})
        elif kind == "OFFSIDE":
            row = ("offside", "detail", f"{name(e['From'])} is caught offside.", "", {})
        if row:
            rows.append((t, side, e["From"], *row))
    for m in moments.itertuples():
        rows.append((m.time, m.player.split("_")[0], m.player, m.kind, "key",
                     f"{KINDS[m.kind]}: {name(m.player)} ({clubs[m.player.split('_')[0]]}).",
                     f"{WHY[m.kind]} {m.detail[:1].upper() + m.detail[1:]}.", {}))
    feed = pd.DataFrame(rows, columns=["time", "team", "player", "kind", "tier", "text", "why", "data"])
    feed["minute"] = [minute_of(t) for t in feed["time"]]
    feed["clock"] = feed["minute"].map(clock)
    feed["name"] = feed["player"].map(name)
    return feed.sort_values("time", kind="stable", ignore_index=True)[FEED_COLUMNS]


def state_at(events: pd.DataFrame, tr: TrackingData, minutes: np.ndarray, possession: np.ndarray,
             upto: float, passes: pd.DataFrame | None = None) -> dict:
    """{"Home"/"Away": running statistics} for everything up to match minute `upto`."""
    seconds = float(np.interp(upto, minutes, tr.timestamps))
    done = events[events["Start Time [s]"] <= seconds]
    sub = done["Subtype"].astype(str)
    held = possession[tr.timestamps <= seconds]
    in_play = sum(isinstance(o, str) for o in held) or 1
    passes = pass_quality(tr, events) if passes is None else passes
    passes = passes[passes["time"] <= seconds]
    out = {}
    for side in ("Home", "Away"):
        mine, theirs = done["Team"] == side, done["Team"] != side
        own = passes[passes["team"] == side]
        shots = mine & (done["Type"] == "SHOT")
        out[side] = {
            "goals": int((shots & sub.str.contains("GOAL")).sum()),
            "shots": int(shots.sum()), "on_target": int((shots & sub.str.startswith("ON TARGET")).sum()),
            "possession": round(100 * sum(o == side for o in held) / in_play),
            "passes": len(own), "pass_accuracy": round(100 * own["completed"].mean()) if len(own) else 0,
            "pass_difficulty": round(own.loc[own["completed"], "difficulty"].mean()) if own["completed"].any() else 0,
            "fouls": int((theirs & (done["Type"] == "FAULT RECEIVED")).sum()),
            "yellow_cards": int((mine & (done["Type"] == "CARD")).sum()),
            "corners": int((mine & sub.eq("CORNER KICK")).sum()),
            "offsides": int((mine & (done["Type"] == "OFFSIDE")).sum()),
        }
    return out


def rhythm(tr: TrackingData, minutes: np.ndarray, possession: np.ndarray, upto: float | None = None,
           spell: float = RHYTHM_SPELL) -> pd.DataFrame:
    """Per `spell`-minute spell: Home's share of the ball in play and the changes of possession.

    A high, steady share is control; many changes of possession are chaos.
    """
    upto = float(minutes[-1]) if upto is None else upto
    team = pd.Series(possession, dtype=object)
    owned = team.ffill()
    changed = (owned != owned.shift()) & owned.notna() & owned.shift().notna()
    bins = (np.asarray(minutes) // spell * spell)
    frame = pd.DataFrame({"spell": bins, "home": team.eq("Home"), "away": team.eq("Away"), "changes": changed})
    frame = frame[np.asarray(minutes) <= upto]
    grouped = frame.groupby("spell").sum()
    share = grouped["home"] / (grouped["home"] + grouped["away"]).replace(0, np.nan)
    return pd.DataFrame({"spell": grouped.index, "home_share": (100 * share).round(1).to_numpy(),
                         "changes": grouped["changes"].to_numpy()})


def overlay_items(feed: pd.DataFrame, game: str) -> list[dict]:
    """The feed as timed graphics cues: when to show what, for how long, about whom."""
    return [{"match": game, "t": round(float(r.time), 2), "clock": r.clock, "type": r.kind, "tier": r.tier,
             "team": r.team, "player": r.player if isinstance(r.player, str) else None, "name": r.name,
             "text": r.text, "why": r.why, "data": r.data, "show_for_s": 8 if r.tier == "key" else 5}
            for r in feed.itertuples()]


def translate(lines: list[str], language: str, config) -> list[str] | None:
    """`lines` in `language` through the configured model, or None if that fails.

    Names, numbers and times are kept exactly as written (the model is told to), so the
    translated feed says the same thing.
    """
    if language == "English" or not lines:
        return lines
    if config is None:
        return None
    from openai import OpenAI
    prompt = (f"Translate each line into {language}. Keep player and club names, numbers, times and units "
              "exactly as written; use a decimal point, not a comma. Reply with only a JSON list of strings, "
              "one per line, in the same order.\n\n" + json.dumps(lines, ensure_ascii=False))
    try:
        client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        reply = client.chat.completions.create(model=config.model, messages=[{"role": "user", "content": prompt}])
        text = reply.choices[0].message.content or ""
        out = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
        return [str(x) for x in out] if len(out) == len(lines) else None
    except Exception:
        return None
