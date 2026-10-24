"""
Off-ball valuation dashboard for a synthetic, football-realistic league.

    Live Match          a match replayed as its events arrive: running score and stats,
                        commentary that says why each moment matters, control vs chaos,
                        a downloadable timed overlay feed for broadcast graphics.
    Tactical Playback   a Football-Manager-style 2D pitch that plays the 25 Hz
                        tracking in the browser, with a scoreboard and highlight clips
                        of the goals and the four pillars beside it,
                        head-check cones, and pitch-control / xT overlays on paused frames.
    Match Leaderboards  every player in the selected match ranked by the Off-Ball Index,
                        with the team battle, intensity by period and a moments timeline.
    Player Profile      one player's four pillars against their position, match by
                        match, with the most similar players in the league.
    Match Report        a fact-checked report written by a team of agents.
    Off-Ball Index      the composite 0-100 indexes: equal weights over ten metrics
                        (core.composite), or the five tactical metrics weighted by their
                        impact on shots (core.impact); rankings and attribute polygons.
    Value Finder        every outfielder across the league: market value against
                        the Off-Ball Index, with median lines marking the "Moneyball" quadrant.

The headline score is the Off-Ball Index. The Pillar score (column off_ball_value) is the
plain mean of the four pillar percentiles, kept as a building block.
    AI Scout            questions about the league's off-ball work, answered from the data.
    Model Check         do the metrics recover the hidden talents the simulation gave
                        each player, and does scanning really keep the ball?

The four pillars: scanning (head checks before receiving), space (decoy runs),
pressing (lane cuts) and transition latency.

All data is synthetic (core/synthetic.py, scripts/generate_synthetic_league.py):
fictional clubs, players and values, simulated at 25 Hz in Metrica's file layout,
plus a head-orientation feed. No real match data is used anywhere.

Layers: core/ computes (tracking, pillars, pipeline, moments, analysis); ui/payload.py
turns those DataFrames into plain arrays for one playback window; ui/charts.py builds
the Plotly figures; ui/pitch_player.html only draws.

"Your view" in the sidebar personalises every page: Fan or Analyst mode, a favourite club,
a player to follow (ringed on the pitch with speed and distance tags, starred in the feeds)
and a language for the commentary and match reports.

Pages are chosen in the sidebar and only the open one is computed, so a Key Moment
or a cited report moment can switch straight to Tactical Playback.

Run:  streamlit run app.py
"""

import copy
import html
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import analysis, composite, impact, live, ml_model, moments, pipeline, roster, tracking
from core.agent import ModelConfig, OffBallAgent
from core.match_agents import LANGUAGES, PERSONAS, generate_report
from core.match_tools import Fact, MatchFacts, build_match_facts, clock   # one clock everywhere
from core.config import MOMENTS, PRESSING, PRESSURE, RECEIPTS, SCANNING, SPACE, TRANSITION
from ui import charts, payload, pitch_player

WINDOW_SECONDS = 30                # length of one playback window
GAMES = {folder.name: folder for folder in pipeline.available_games()}

TEAM_COLOUR = {"Home": "#e63946", "Away": "#3a86ff"}
# Value Finder roles: the reference palette's blue, orange, aqua, violet and pink (skipping
# its yellow, which sits too close to orange for a scatter) plus a marker shape each, so a
# role never depends on colour alone.
ROLE_STYLE = {"Space creator": ("#2a78d6", "circle"), "Lane-cutting presser": ("#eb6834", "square"),
              "Transition sprinter": ("#1baf7a", "diamond"), "Space occupier": ("#6250d6", "triangle-up"),
              "Scanner": ("#e87ba4", "star")}
OTHER_ROLE = ("#8a8a8a", "x")      # a second cluster led by the same metric ("Scanner 5")
# Value Finder y axes: label -> (column, higher is better).
Y_AXES = {"Off-Ball Index": ("index", True), "Off-Ball Impact": ("impact_index", True), "Pillar score": ("off_ball_value", True), "ML predicted value": ("Predicted_Value", True),
          "OBSO /90": ("obso_p90", True), "Pressure Intensity /90": ("pressure_intensity", True),
          "Runs in behind /90": ("runs_in_behind_p90", True),
          "Receipts in space": ("receipts_in_space_pct", True),
          "Defensive Responsibility /90": ("def_responsibility_p90", False)}


# ── Cached data ──────────────────────────────────────────────────────────────
# A full match (145k frames x 28 players, positions and velocities) is held as a
# shared resource: st.cache_data would pickle a ~130 MB copy on every rerun, which
# is exactly what makes a slider freeze. Everything derived from it is small and
# goes through st.cache_data.
@st.cache_resource(show_spinner="Loading tracking...", max_entries=2)
def match(game: str):
    tr, events = tracking.load_sample_game(GAMES[game])
    return tr, events, tr.velocities(), tracking.possession_by_frame(events, tr.frames)


@st.cache_resource(show_spinner=False, max_entries=2)
def head_yaw(game: str) -> np.ndarray:
    """The match's (frames, players) head yaw in degrees; all NaN without a head feed."""
    return tracking.load_head_orientation(GAMES[game], match(game)[0])


def games_signature() -> tuple:
    """Cache key for the scored games: the pipeline version plus (game, last-modified) pairs.

    Changing a game file, or the shape of core.pipeline's output (bump
    pipeline.VERSION), invalidates the disk cache instead of loading stale tables.
    """
    return pipeline.games_signature(list(GAMES.values()))


@st.cache_data(show_spinner="Learning xT from the league's passes, carries and shots...")
def xt_grid(signature: tuple) -> np.ndarray:
    return pipeline.learn_xt([GAMES[name] for name, _ in signature if name in GAMES])


@st.cache_data(show_spinner=False, persist="disk")
def load_global_metrics(signature: tuple) -> dict[str, pd.DataFrame]:
    """Every game through the pillars, concatenated (core.pipeline.score_all).

    Event-level tables (decoys, pressing, transitions) plus per-player profiles, all
    with game, player and uid columns. Persisted to disk: this is the slow part
    (about 5 s a game), and it only reruns when a game file changes. Tables saved by
    scripts/precompute_scores.py for this exact signature load instead, so a fresh
    container starts in seconds.
    """
    if (saved := pipeline.load_scored(signature)) is not None:
        return saved
    bar = st.progress(0.0, "Scoring every match (first run only)...")
    result = pipeline.score_all(
        xt_grid(signature), [GAMES[name] for name, _ in signature if name in GAMES],
        progress=lambda i, n: bar.progress(i / n, f"Scored match {i} of {n} (first run only)..."))
    bar.empty()
    return result


@st.cache_data(show_spinner=False)
def league_names(signature: tuple) -> pd.DataFrame:
    """uid -> fictional name, club, position and value, from every match's roster file."""
    rosters = [pipeline.load_roster(GAMES[name]).assign(uid=lambda t, n=name: n + ":" + t["player"])
               for name, _ in signature if name in GAMES]
    table = pd.concat(rosters, ignore_index=True).rename(columns={"club": "real_team"})
    return table[["uid", "name", "real_team", "position", "detailed_position", "market_value_m"]]


@st.cache_data(show_spinner=False)
def season_totals(signature: tuple) -> pd.DataFrame:
    """Every name's totals across all games (core.pipeline.player_totals on uid)."""
    data = load_global_metrics(signature)
    everyone = roster.static_timeline(data["profiles"], league_names(signature), key="uid")
    return pipeline.player_totals(data, everyone, on="uid")


INDEX_CONFIG = composite.load_config()
CATEGORIES = {name: category["label"] for name, category in INDEX_CONFIG["categories"].items()}


@st.cache_data(show_spinner=False)
def index_table(signature: tuple, weights: tuple = ()) -> pd.DataFrame:
    """The Composite Off-Ball Index of every name in the league (core.composite).

    `weights` overrides category weights as (category, weight) pairs; () keeps the config's.
    """
    config = copy.deepcopy(INDEX_CONFIG)
    for name, weight in weights:
        config["categories"][name]["weight"] = weight
    board = pipeline.off_ball_value(season_totals(signature))       # adds decoy_p90 and lane_cut_p90
    return composite.composite(board.assign(season="synthetic"), config)


@st.cache_data(show_spinner="Fitting the impact weights on every match...")
def impact_weights(signature: tuple) -> pd.DataFrame:
    """core.impact.fit_weights on every game's 5-minute team windows."""
    games = [GAMES[name] for name, _ in signature if name in GAMES]
    return impact.fit_weights(impact.team_blocks(load_global_metrics(signature), games))


@st.cache_data(show_spinner=False)
def impact_table(signature: tuple) -> pd.DataFrame:
    """The Off-Ball Impact Index of every name in the league (core.impact weights, core.composite scoring)."""
    config = impact.index_config(impact_weights(signature)["weight"])
    board = pipeline.off_ball_value(season_totals(signature))
    return composite.composite(board.assign(season="synthetic"), config)


@st.cache_data(show_spinner="Training the valuation model and clustering roles...")
def value_models(signature: tuple, pool_minutes: int, by_position: bool):
    """The Value Finder pool with Off-Ball Value, out-of-fold ML price and role clusters.

    Returns (pool, valuation diagnostics, role profile table).
    """
    totals = season_totals(signature)
    pool = totals[(totals["minutes"] >= pool_minutes) & (totals["broad_position"] != "Goalkeeper")]
    pool = pipeline.off_ball_value(pool, within="broad_position" if by_position else None)
    # The index always ranks against the whole league (its own minutes floor), whatever this pool's.
    pool = pool.merge(index_table(signature)[["name", "real_team", "index", "coverage"]],
                      on=["name", "real_team"], how="left")
    pool = pool.merge(impact_table(signature)[["name", "real_team", "index"]]
                      .rename(columns={"index": "impact_index"}), on=["name", "real_team"], how="left")
    if len(pool) < 8:
        return pool, None, None
    _, pool = ml_model.train_valuation_model(pool)
    valuation = pool.attrs["valuation"]
    pool = ml_model.cluster_player_roles(pool)
    roles = pool.attrs["clusters"]
    pool.attrs = {}        # DataFrames in attrs break later merges (pandas compares attrs with ==)
    return pool, valuation, roles


SCOUT_POOL_MINUTES = 30          # the AI Scout sees the same pool as the Value Finder's default


def match_index(board: pd.DataFrame, min_minutes: float) -> pd.DataFrame:
    """`board` (one match, after pipeline.off_ball_value) with the Off-Ball Index within that match.

    Same method as the league index, compared within position across both teams, with
    samples sized for one match: shrinkage at 30 minutes, 3 receptions, 3 transitions.
    """
    config = copy.deepcopy(INDEX_CONFIG) | {"min_minutes": min_minutes}
    for category in config["categories"].values():
        for spec in category["metrics"].values():
            spec["min_sample"] = 30 if spec["sample"] == "minutes" else 3
    scored = composite.composite(board.assign(season="match"), config)
    return board.assign(index=scored["index"])


@st.cache_data(show_spinner=False)
def match_facts(signature: tuple, game: str) -> MatchFacts:
    """The verified, numbered facts of one match that the report agents work from."""
    data = load_global_metrics(signature)
    events = {key: table[table["game"] == game] for key, table in data.items()}
    profiles = events["profiles"]
    timeline = roster.static_timeline(profiles, league_names(signature), key="uid")
    board = pipeline.player_totals(events, timeline)
    board = pipeline.off_ball_value(board[(board["minutes"] >= 20) & (board["broad_position"] != "Goalkeeper")])
    board = match_index(board, 20)
    raw_events = tracking.load_events(GAMES[game] / f"{game}_RawEventsData.csv")
    return build_match_facts(game, raw_events, pipeline.load_roster(GAMES[game]),
                             game_moments(signature, game), board, tables=events)


REPORT_KINDS = {"goal": "Goal", "build_up": "Build-up to goal"}


def moment_label(kind: str) -> str:
    return (moments.KINDS | REPORT_KINDS).get(kind, kind.replace("_", " ").title())


@st.cache_data(show_spinner="Profiling every player match by match...")
def per_match(signature: tuple) -> pd.DataFrame:
    """One row per (match, player): that match's pillars and Off-Ball Value (core.analysis)."""
    return analysis.per_match_totals(load_global_metrics(signature), league_names(signature))


@st.cache_data(show_spinner=False)
def hidden_talent(signature: tuple) -> pd.DataFrame:
    """The simulation's hidden player attributes (league_players.csv), for the Model Check only."""
    path = next(iter(GAMES.values())).parent / "league_players.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_resource(show_spinner="Preparing the AI Scout...")
def scout_agent(signature: tuple) -> OffBallAgent:
    """One agent per dataset, shared by all sessions. Each session keeps its own chat history."""
    pool, _, _ = value_models(signature, SCOUT_POOL_MINUTES, False)
    return OffBallAgent(pool)


def match_label(game: str) -> str:
    """'Matchday 1 · Harbourside FC v Northvale Athletic' from the folder name and roster."""
    clubs = pipeline.load_roster(GAMES[game]).drop_duplicates("club")["club"].tolist()
    day = game.split("_")[1] if game.startswith("Matchday_") else game
    return f"Matchday {day} · {' v '.join(clubs)}"


@st.cache_data(show_spinner=False)
def game_minutes(game: str) -> np.ndarray:
    """Match clock (minutes, second half from 45) for every tracking frame."""
    return roster.match_minutes(match(game)[0])


@st.cache_data(show_spinner=False)
def game_moments(signature: tuple, game: str) -> pd.DataFrame:
    """Key moments for one game, read from the pipeline's event tables (core.moments)."""
    data = load_global_metrics(signature)
    return moments.key_moments({key: table[table["game"] == game] for key, table in data.items()})


@st.cache_data(show_spinner="Preparing playback...")
def player_payload(signature: tuple, game: str, start: int, end: int, names: tuple, team_names: tuple,
                   highlight: tuple | None, autoplay: bool, focus: str | None = None) -> dict:
    """One playback window for the pitch player (ui.payload.build_window)."""
    tr, _, vel, possession = match(game)
    mark = dict(zip(("player", "start", "end", "label"), highlight)) if highlight else None
    return payload.build_window(tr, vel, possession, xt_grid(signature), game_minutes(game), start, end,
                                list(names), dict(team_names), TEAM_COLOUR, mark, autoplay,
                                head_yaw=head_yaw(game), focus=focus)


def match_cast(game: str) -> tuple[dict, dict]:
    """(tracking id -> player name, "Home"/"Away" -> club) for one match."""
    table = pipeline.load_roster(GAMES[game])
    return dict(zip(table["player"], table["name"])), {
        side: table.loc[table["player"].str.startswith(side), "club"].iloc[0] for side in ("Home", "Away")}


@st.cache_data(show_spinner="Building the live feed...")
def live_feed(signature: tuple, game: str) -> pd.DataFrame:
    """core.live.event_feed for one match."""
    tr, events, _, _ = match(game)
    return live.event_feed(events, tr, game_minutes(game), game_moments(signature, game),
                           match_facts(signature, game), *match_cast(game))


@st.cache_data(show_spinner=False)
def live_passes(signature: tuple, game: str) -> pd.DataFrame:
    tr, events, _, _ = match(game)
    return live.pass_quality(tr, events)


@st.cache_data(show_spinner="Translating the commentary...")
def translated_feed(signature: tuple, game: str, language: str) -> list[str] | None:
    """The live feed's lines and reasons in `language` (None if no model or the call fails)."""
    feed = live_feed(signature, game)
    return live.translate(feed["text"].tolist() + feed["why"].tolist(), language, ModelConfig.from_env())


# ── Charts ───────────────────────────────────────────────────────────────────
def value_finder_figure(pool: pd.DataFrame, log_x: bool, axis: str) -> go.Figure:
    """Market value against the metric chosen in Y_AXES.

    "ML predicted value" is drawn against a fair-price diagonal (above it = cheaper than
    their off-ball work); every other metric gets median lines marking the Moneyball
    quadrant: below-median value, better-than-median on the metric.
    """
    y_col, higher = Y_AXES[axis]
    predicted, percentile = y_col == "Predicted_Value", y_col in ("off_ball_value", "index", "impact_index")
    pool = pool.dropna(subset=[y_col])
    values = pool[["market_value_m", y_col]] if predicted else pool[["market_value_m"]]
    low = values.min().min() * 0.8 if log_x else 0
    high = values.max().max() * 1.1
    top = high if predicted else 100 if percentile else float(pool[y_col].max()) * 1.12
    fig = go.Figure()
    guide = dict(color="rgba(128,128,128,0.8)", width=1, dash="dash")
    # Regions and guides as traces (not shapes) so they sit correctly on log axes.
    if predicted:
        fig.add_trace(go.Scatter(x=[low, high, low], y=[low, high, high], fill="toself", mode="lines",
                                 line_width=0, fillcolor="rgba(128,128,128,0.12)", hoverinfo="skip",
                                 name="Priced below off-ball work"))
        fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", line=guide,
                                 hoverinfo="skip", name="Fair price (predicted = actual)"))
        finds = pool.assign(gap=pool[y_col] / pool["market_value_m"]).sort_values("gap", ascending=False).head(3)
        headline = "<b>Above the line</b>: off-ball work suggests a higher price"
    else:
        x_median, y_median = pool["market_value_m"].median(), pool[y_col].median()
        band = [y_median, y_median, top, top] if higher else [0, 0, y_median, y_median]
        fig.add_trace(go.Scatter(x=[low, x_median, x_median, low], y=band, fill="toself", mode="lines",
                                 line_width=0, fillcolor="rgba(128,128,128,0.12)", hoverinfo="skip",
                                 name="Moneyball quadrant"))
        for xs, ys, label in (([x_median] * 2, [0, top], f"Median value €{x_median:.0f}m"),
                              ([low, high], [y_median] * 2, f"Median {axis} {charts.fmt(y_median)}")):
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False, line=guide))
            fig.add_annotation(x=xs[-1], y=ys[-1], text=label, showarrow=False, font_size=11,
                               xanchor="right" if label.startswith("Median ") and xs[0] == low else "left",
                               yanchor="bottom")
        better = pool[y_col] > y_median if higher else pool[y_col] < y_median
        finds = pool[(pool["market_value_m"] < x_median) & better] \
            .sort_values(y_col, ascending=not higher).head(3)
        headline = (f"<b>Moneyball</b>: below-median value, {'above' if higher else 'below'}-median "
                    f"{axis}")

    metric_line = "" if percentile or predicted else f"{axis}: %{{y:.3~f}}<br>"
    present = list(pool["Cluster_Role"].dropna().unique())
    for role in [r for r in ROLE_STYLE if r in present] + sorted(r for r in present if r not in ROLE_STYLE):
        colour, symbol = ROLE_STYLE.get(role, OTHER_ROLE)
        group = pool[pool["Cluster_Role"] == role]
        fig.add_trace(go.Scatter(
            x=group["market_value_m"], y=group[y_col], mode="markers", name=role,
            marker=dict(size=11, color=colour, symbol=symbol, line=dict(width=1.5, color="white")),
            customdata=np.column_stack([group["name"], group["real_team"].fillna(""), group["broad_position"],
                                        group["minutes"].round(0), group["index"].round(0).astype(object)
                                        .where(group["index"].notna(), "–"),
                                        group["Predicted_Value"].round(1), group["decoy_p90"].round(3),
                                        group["lane_cut_p90"].round(1), group["pitch_control_pct"].round(1),
                                        group["scan_rate"].round(2)]),
            hovertemplate=("<b>%{customdata[0]}</b> (%{customdata[1]}, %{customdata[2]})<br>" + metric_line +
                           "Value €%{x:.1f}m · ML price €%{customdata[5]}m · Off-Ball Index %{customdata[4]}<br>"
                           "Decoy /90 %{customdata[6]} · Lane cuts /90 %{customdata[7]} s · "
                           "Scans %{customdata[9]}/s · Pitch control %{customdata[8]}%<br>"
                           "%{customdata[3]} min<extra></extra>")))
    # Label only the top three, alternating above and below so neighbours don't collide.
    for k, row in enumerate(finds.itertuples()):
        fig.add_annotation(x=row.market_value_m, y=getattr(row, y_col), text=row.name, showarrow=False,
                           yshift=14 if k % 2 == 0 else -14, font_size=11)
    fig.add_annotation(x=low, y=top, text=headline, showarrow=False, xanchor="left", yanchor="top", font_size=12)
    axis_range = lambda title: dict(title=title, type="log" if log_x else "linear", zeroline=False,
                                    range=[np.log10(low), np.log10(high)] if log_x else [0, high])
    y_title = {"Predicted_Value": "ML-predicted value from off-ball work (€m)",
               "off_ball_value": "Pillar score (mean percentile of the four pillars)",
               "index": "Off-Ball Index (0-100, within position)",
               "impact_index": "Off-Ball Impact (0-100, within position)"}.get(
        y_col, f"{axis}{'' if higher else ' (lower is better)'}")
    fig.update_layout(
        height=600, margin=dict(l=10, r=10, t=40, b=10), hovermode="closest",
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
        xaxis=axis_range("Market value (€m, synthetic)"),
        yaxis=axis_range(y_title) if predicted else dict(title=y_title, range=[0, 101 if percentile else top]))
    return fig


# ── Page ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Off-Ball Value", layout="wide")
st.title("Off-Ball Value")

if not GAMES:
    st.error("No synthetic matches found in data/synthetic/. Generate the league first: "
             "`python scripts/generate_synthetic_league.py`")
    st.stop()

signature = games_signature()
xt = xt_grid(signature)
global_data = load_global_metrics(signature)
names = league_names(signature)
# Page -> what it covers. Match pages follow the sidebar's match; the rest pool the league.
PAGES = {"Live Match": "This match · as it happens", "Tactical Playback": "This match · the 2D pitch", "Match Leaderboards": "This match · rankings",
         "Match Report": "This match · fact-checked AI report", "Off-Ball Index": "League · composite index",
         "Player Profile": "League · one player in depth", "Value Finder": "League · price v off-ball work",
         "AI Scout": "League · ask the data", "Model Check": "League · do the metrics find talent?"}
MATCH_PAGES = ["Live Match", "Tactical Playback", "Match Leaderboards", "Match Report"]
with st.sidebar:
    page = st.radio("View", list(PAGES), key="page", captions=list(PAGES.values()))
    st.divider()
    st.header("Match")
    game = st.selectbox("Playback and leaderboards", list(GAMES), format_func=match_label)
    st.caption(f"The Value Finder always pools all {len(GAMES)} matches. Every club, player and "
               "value is fictional: synthetic, football-realistic data from `core/synthetic.py`.")
    st.divider()
    st.header("Your view")
    view_mode = st.radio("Mode", ["Fan", "Analyst"], horizontal=True, key="view_mode",
                         help="Fan: the key moments in plain words. Analyst: every event, with the numbers.")
    fav_club = st.selectbox("Favourite club", ["None", *sorted(names["real_team"].unique())], key="fav_club")
    fav_player = st.selectbox("Follow a player", ["None", *sorted(names["name"].unique())], key="fav_player",
                              help="Ringed on the pitch with speed and distance tags, starred in the feeds")
    language = st.selectbox("Language", LANGUAGES, key="language",
                            help="For the live commentary and match reports (needs the AI model)")
fav_club = None if fav_club == "None" else fav_club
fav_player = None if fav_player == "None" else fav_player
game_profiles = global_data["profiles"][global_data["profiles"]["game"] == game]
cast, side_club = match_cast(game)
followed = next((pid for pid, who in cast.items() if who == fav_player), None)   # tracking id in this match
favourite = lambda team, name: name == fav_player or side_club.get(team) == fav_club
timeline = roster.static_timeline(game_profiles, names, key="uid")
clubs = pipeline.load_roster(GAMES[game]).drop_duplicates("player").set_index("player")["club"]

tr, events, velocities, possession = match(game)
minutes = game_minutes(game)
game_events = {key: table[table["game"] == game] for key, table in global_data.items()}


def names_at(minute: float) -> list[str]:
    """Display name of every tracked player at a match minute."""
    current = roster.active(timeline, minute).set_index("player")["name"]
    return [current[pid] if pid in current.index else pid for pid in tr.player_ids]


def name_of(pid: str, minute: float) -> str:
    return names_at(minute)[tr.index(pid)]


# ── Key Moments ──────────────────────────────────────────────────────────────
# The playback target lives in session state: a key-moment button or the "go to"
# control sets it, and the Tactical Playback tab builds its window around it.
def jump_to_moment(row: dict) -> None:
    st.session_state["target"] = {"game": game, "time": row["time"], "end": row["end"],
                                  "player": row["player"], "kind": row["kind"],
                                  "minute": row["minute"], "detail": row["detail"], "autoplay": True}
    st.session_state["page"] = "Tactical Playback"


def jump_to_fact(fact: Fact) -> None:
    """Load a cited moment from the match report into the Tactical Playback."""
    st.session_state["target"] = {"game": game, "time": fact.time, "end": fact.end, "player": fact.player,
                                  "kind": fact.kind, "minute": fact.minute, "detail": fact.text,
                                  "autoplay": True}
    st.session_state["page"] = "Tactical Playback"


def jump_to_minute() -> None:
    minute = st.session_state["goto_minute"]
    period = 1 if minute < 45 else 2
    rows = np.flatnonzero(tr.periods == period)
    row = rows[min(np.searchsorted(minutes[rows], minute), len(rows) - 1)]
    st.session_state["target"] = {"game": game, "time": float(tr.timestamps[row]), "autoplay": False}


key_moments = game_moments(signature, game)
target = st.session_state.get("target")
if not target or target["game"] != game:          # default: the game's first key moment, paused
    first = key_moments.iloc[0].to_dict() if len(key_moments) else None
    target = ({"game": game, "time": first["time"], "end": first["end"], "player": first["player"],
               "kind": first["kind"], "minute": first["minute"], "detail": first["detail"],
               "autoplay": False} if first else {"game": game, "time": float(tr.timestamps[0]), "autoplay": False})

# Highlight clips beside the pitch: the goals, then each pillar's key moments.
CLIP_GROUPS = {"Goals": ("goal",), "Scanning": ("scan",), "Space": ("decoy",), "Pressing": ("press",),
               "Transition": ("attacking_transition", "defensive_transition")}


def scoreboard(facts: MatchFacts, upto: float | None = None, at: str | None = None, spoilers: bool = True) -> None:
    """Both clubs, the score and the scorers. With `upto` (tracking seconds) the score at that
    moment, full time beside it, and goals still to come greyed out; spoilers=False hides the
    full-time score and goals still to come (the live view)."""
    side = lambda goal: (goal.player or "").split("_")[0]
    goals = facts.of_kind("goal")
    scored = lambda goal: upto is None or (goal.time is not None and goal.time <= upto)
    count = {s: sum(side(g) == s and scored(g) for g in goals) for s in ("Home", "Away")}
    scorers = {s: " · ".join(f'<span style="opacity:{1 if scored(g) else 0.4}">{html.escape(g.name or g.player)} '
                             f'{g.clock}</span>' for g in goals if side(g) == s and (spoilers or scored(g)))
                  or "&nbsp;" for s in ("Home", "Away")}
    status = ("Full time" if upto is None else f"Score at {at}" if not spoilers
              else f"Score at {at} · full time {facts.score[0]}–{facts.score[1]}")
    club = lambda s, name: (f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;'
                            f'background:{TEAM_COLOUR[s]};margin:0 8px"></span><b>{html.escape(name)}</b>')
    st.markdown(f"""
<div style="border:1px solid rgba(128,128,128,.3);border-radius:12px;padding:10px 16px;margin-bottom:12px">
  <div style="display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:16px;font-size:1.15rem">
    <div style="text-align:right">{html.escape(facts.home)}{club("Home", "")}</div>
    <div style="font:700 2rem/1 ui-monospace,Menlo,monospace;letter-spacing:.05em">{count["Home"]} – {count["Away"]}</div>
    <div>{club("Away", "")}{html.escape(facts.away)}</div>
    <div style="text-align:right;font-size:.85rem;opacity:.8">{scorers["Home"]}</div>
    <div style="text-align:center;font-size:.75rem;opacity:.7">{status}</div>
    <div style="font-size:.85rem;opacity:.8">{scorers["Away"]}</div>
  </div>
</div>""", unsafe_allow_html=True)


def highlights(facts: MatchFacts) -> None:
    """The clip list beside the pitch: pick goals or a pillar, click a clip to play it."""
    clips = {group: (facts.of_kind("goal") if kinds == ("goal",) else
                     key_moments[key_moments["kind"].isin(kinds)].sort_values("minute"))
             for group, kinds in CLIP_GROUPS.items()}
    group = st.radio("Highlights", list(clips), key=f"clip-group-{game}", horizontal=True,
                     index=0 if len(clips["Goals"]) else 2, format_func=lambda g: f"{g} ({len(clips[g])})")
    playing = lambda kind, time: target.get("kind") == kind and target.get("time") == time
    with st.container(height=640):
        if group == "Goals":
            for fact in clips["Goals"]:
                st.button(f"⚽ {fact.clock} · {fact.name or fact.player}", key=f"goal-{game}-{fact.id}",
                          help=fact.text, width="stretch", on_click=jump_to_fact, args=(fact,),
                          type="primary" if playing("goal", fact.time) else "secondary")
        else:
            for n, row in clips[group].iterrows():
                phase = {"attacking_transition": " · attacking", "defensive_transition": " · defensive"}
                star = "★ " if favourite(row.player.split("_")[0], name_of(row.player, row.minute)) else ""
                st.button(f"{star}{clock(row.minute)} · {name_of(row.player, row.minute)}{phase.get(row.kind, '')}",
                          key=f"moment-{game}-{n}", help=f"{moments.KINDS[row.kind]}: {row.detail}",
                          width="stretch", on_click=jump_to_moment, args=(row.to_dict(),),
                          type="primary" if playing(row.kind, row.time) else "secondary")
        if not len(clips[group]):
            st.caption(f"No {group.lower()} clips in this match.")
    st.caption("Clips are the match's standout moments for each pillar; the playing one is highlighted.")


st.header(page)
if page in MATCH_PAGES:
    st.caption(f"{match_label(game)} · change the match in the sidebar")
    if page not in ("Tactical Playback", "Live Match"):   # those show the score at the moment being watched
        scoreboard(match_facts(signature, game))
else:
    st.caption(f"All {len(GAMES)} matches of the synthetic league")

LIVE_SPEEDS = {"1 min per second": 1.0, "3 min per second": 3.0, "10 min per second": 10.0}
FEED_ICONS = {"goal": "⚽", "card": "🟨", "penalty": "❗", "shot": "🎯", "corner": "⛳", "offside": "🚩",
              "decoy": "🏃", "press": "🧲", "defensive_transition": "⚡", "attacking_transition": "⚡", "scan": "👀"}

if page == "Live Match":
    feed, state = live_feed(signature, game), st.session_state
    end_minute = float(minutes[-1])
    if state.get("live_game") != game:                     # a new match starts from kick-off, paused
        state["live_game"], state["live_minute"], state["live_playing"] = game, 0.0, False
    analyst = view_mode == "Analyst"
    texts = translated_feed(signature, game, language) if language != "English" else None
    controls = st.columns([1, 1, 3])
    controls[0].button("⏸ Pause" if state["live_playing"] else "▶ Play live", type="primary", width="stretch",
                       key="live-play", on_click=lambda: state.update(live_playing=not state["live_playing"]))
    controls[1].selectbox("Speed", list(LIVE_SPEEDS), index=1, key="live-speed", label_visibility="collapsed")
    controls[2].caption("The match arrives event by event: every number below uses only what has happened so far. "
                        + ("Analyst mode: every event with its numbers." if analyst else "Fan mode: the key moments.")
                        + ("" if language == "English" or texts else " Translation needs the AI model: in English."))

    @st.fragment(run_every=1.0 if state["live_playing"] else None)
    def live_panel() -> None:
        if state["live_playing"]:                          # the clock moves on, then everything redraws
            state["live_minute"] = min(end_minute, state["live_minute"] + LIVE_SPEEDS[state["live-speed"]])
            if state["live_minute"] >= end_minute:
                state["live_playing"] = False
        minute = st.slider("Match minute", 0.0, end_minute, key="live_minute", step=0.5, format="%.1f")
        seconds = float(np.interp(minute, minutes, tr.timestamps))
        scoreboard(match_facts(signature, game), seconds, clock(minute), spoilers=False)
        left, right = st.columns([3, 2], gap="medium")
        so_far = feed[feed["minute"] <= minute]
        so_far = so_far if analyst else so_far[so_far["tier"] == "key"]
        with left:
            st.markdown("**Live commentary**")
            with st.container(height=560):
                for k, row in so_far.iloc[::-1].head(25).iterrows():
                    text = texts[k] if texts else row.text
                    why = texts[len(feed) + k] if texts else row.why
                    star = "★ " if favourite(row["team"], row["name"]) else ""     # row.name is the index
                    st.markdown(f"`{row.clock}` {FEED_ICONS.get(row.kind, '•')} {star}{text}")
                    if why:
                        st.caption(why)
                if so_far.empty:
                    st.caption("Kick-off: the commentary appears as the events arrive.")
        with right:
            stats = live.state_at(events, tr, minutes, possession, minute, live_passes(signature, game))
            rows = {"Possession %": "possession", "Shots": "shots", "On target": "on_target",
                    "Yellow cards": "yellow_cards", "Corners": "corners"}
            if analyst:
                rows |= {"Passes": "passes", "Pass accuracy %": "pass_accuracy",
                         "Completed-pass difficulty (0-100)": "pass_difficulty", "Fouls": "fouls", "Offsides": "offsides"}
            table = pd.DataFrame({side_club[side]: [stats[side][v] for v in rows.values()] for side in ("Home", "Away")},
                                 index=list(rows))
            st.dataframe(table, width="stretch")
            st.plotly_chart(charts.live_rhythm(live.rhythm(tr, minutes, possession, minute), side_club["Home"],
                                               side_club["Away"]), key="live-rhythm")
            st.caption("Control vs chaos: who had the ball in each 3-minute spell, and how often it changed hands.")

    live_panel()
    st.download_button("⬇ Overlay feed (JSON)", json.dumps(live.overlay_items(feed, game), ensure_ascii=False, indent=1),
                       file_name=f"{game}_overlay_feed.json", mime="application/json",
                       help="Every event as a timed graphics cue (time, clock, type, player, text, why, how long "
                            "to show it): machine-readable for a broadcast overlay system.")

if page == "Tactical Playback":
    # The window: a short lead-in before the target, WINDOW_SECONDS long, inside one period.
    row = int(np.searchsorted(tr.timestamps, target["time"]))
    period_rows = np.flatnonzero(tr.periods == tr.periods[min(row, len(tr.timestamps) - 1)])
    fps = round(1 / float(np.median(np.diff(tr.timestamps[:50]))))
    start = int(max(period_rows[0], row - MOMENTS.lead_in * fps))
    end = int(min(period_rows[-1] + 1, start + WINDOW_SECONDS * fps))
    window_minute = float(minutes[start])

    facts = match_facts(signature, game)
    scoreboard(facts, float(tr.timestamps[start]), clock(window_minute))
    left, side = st.columns([3, 1], gap="medium")
    with side:
        highlights(facts)
    left_area = left.container()
    heading, goto = left_area.columns([3, 1])
    if "kind" in target:
        who = name_of(target["player"], target["minute"])
        heading.markdown(f"**{moment_label(target['kind'])}** · {who} · {clock(target['minute'])}  \n"
                         f"{target['detail']}")
    else:
        heading.markdown(f"**Free view** from {clock(window_minute)}")
    goto.number_input("Go to minute", 0.0, float(np.ceil(minutes.max())), value=round(window_minute, 1),
                      step=1.0, key="goto_minute", on_change=jump_to_minute)

    highlight = None
    if "kind" in target:
        hl_start, hl_end = np.searchsorted(tr.timestamps, [target["time"], target["end"]])
        highlight = (target["player"], int(hl_start), int(hl_end),
                     f"{moment_label(target['kind'])} · {name_of(target['player'], target['minute'])}")
    team_names = {side: clubs[clubs.index.str.startswith(side)].iloc[0] for side in ("Home", "Away")}
    current = roster.active(timeline, window_minute)
    with left_area:
        pitch_player.render(player_payload(signature, game, start, end, tuple(names_at(window_minute)),
                                           tuple(team_names.items()), highlight, target["autoplay"], followed),
                            height=780)
    left_area.caption("Space plays and pauses; arrow keys step a frame (Shift: a second). Cones point the "
               "way each player is moving (towards the ball when nearly still). With Head direction on, "
               f"a yellow cone shows a head check: a player looking more than {SCANNING.on_angle:g}° away "
               "from the ball, from the synthetic head-orientation feed. Shadows and pink lanes show "
               f"passes cut by pressers within {PRESSING.max_press_distance:g} m of the carrier. "
               "Overlays draw on paused frames.")

    st.subheader(f"Line-ups at {clock(window_minute)}")
    display = names_at(window_minute)
    for column, side in zip(st.columns(2), ("Home", "Away")):
        rows = current[current["team"] == side].sort_values("depth")
        shape = rows["formation"].dropna().iloc[0] if rows["formation"].notna().any() \
            else roster.formation(rows["position"])
        column.markdown(f"**{side}: {team_names[side]}** · {shape} · {len(rows)} on the pitch")
        column.dataframe(
            pd.DataFrame({"#": [pid.split("_")[1] for pid in rows["player"]],
                          "Player": [display[tr.index(pid)] for pid in rows["player"]],
                          "Position": rows["detailed_position"].to_numpy(),
                          "Value": rows["market_value_m"].to_numpy()}),
            hide_index=True, width="stretch",
            column_config={"Value": st.column_config.NumberColumn("Value", format="€%.1fm")})


def leaderboard_columns() -> dict:
    """Column formatting shared by the Match Leaderboard and the Value Finder table."""
    return {
        "rank": st.column_config.NumberColumn("#", width="small"),
        "name": st.column_config.TextColumn("Player"),
        "metrica_ids": st.column_config.TextColumn(
            "Tracking ID(s)", help="Tracked players who carried this name; it can pass from one "
                                  "to another at a substitution"),
        "team": "Team", "real_team": "Club",
        "market_value_m": st.column_config.NumberColumn(
            "Market value", format="€%.1fm", help="The fictional player's simulated market value"),
        "position": st.column_config.TextColumn(
            "Position", help="The player's position in the squad; the one "
                             "held longest if it changed"),
        "minutes": st.column_config.NumberColumn("Minutes", format="%.0f"),
        "index": st.column_config.ProgressColumn(
            "Off-Ball Index", min_value=0, max_value=100, format="%.0f",
            help="The headline score: ten off-ball metrics in five categories, as percentiles within "
                 "position (50 = the middle). Blank for goalkeepers and players under the minutes floor"),
        "off_ball_value": st.column_config.ProgressColumn(
            "Pillar score", min_value=0, max_value=100, format="%.0f",
            help="Average percentile across the four pillars: Decoy Run Value, Pressing Lane Cuts, "
                 "Transition score and Scan rate"),
        "decoy_runs": st.column_config.NumberColumn("Off-ball runs", format="%d"),
        "decoy_p90": st.column_config.NumberColumn(
            "Decoy Run Value /90", format="%.3f",
            help="xT zones of extra control created for teammates by dragging markers, per 90"),
        "lane_cut_p90": st.column_config.NumberColumn(
            "Pressing Lane Cuts /90", format="%.1f s",
            help="Seconds of threat-weighted passing lanes cut while within "
                 f"{PRESSING.max_press_distance:g} m of the ball, per 90"),
        "mean_lane_cut": st.column_config.NumberColumn(
            "Avg lane cut", format="%.2f", help="Mean share of lanes cut per pressing frame"),
        "transitions": st.column_config.NumberColumn(
            "Transitions", format="%d",
            help=f"Turnovers where the player was within {TRANSITION.max_ball_distance:g} m of the ball"),
        "median_latency": st.column_config.NumberColumn(
            "Transition Latency", format="%.1f s",
            help=f"Median time to a sustained {TRANSITION.sprint_kmh:g} km/h sprint, "
                 "over the transitions where the player sprinted"),
        "sprint_rate": st.column_config.NumberColumn(
            "Sprint rate", format="percent",
            help=f"Share of transitions with a sprint within {TRANSITION.horizon:g} s"),
        "transition_score": st.column_config.NumberColumn(
            "Transition score", format="%.2f", help="Mean of exp(-latency/τ) - λ·walk fraction"),
        "receptions": st.column_config.NumberColumn("Receptions", format="%d"),
        "scan_rate": st.column_config.NumberColumn(
            "Scan rate", format="%.2f /s",
            help=f"Head checks per second in the {SCANNING.window:g} s before receiving the ball"),
        "pre_reception_scans": st.column_config.NumberColumn(
            "Scans before receiving", format="%.1f",
            help="Average head checks before each reception with head data"),
        "retention_rate": st.column_config.NumberColumn(
            "Kept the ball", format="percent", help="Share of receptions where the team kept the ball"),
        "pressures": st.column_config.NumberColumn(
            "Pressures", format="%d",
            help=f"Spells closing down the ball carrier from within {PRESSURE.radius:g} m"),
        "pressure_intensity": st.column_config.NumberColumn(
            "Pressure Intensity /90", format="%.0f m",
            help="Ground made up on the carrier per 90, weighted by how close it was made up"),
        "press_closing_kmh": st.column_config.NumberColumn(
            "Closing speed", format="%.1f km/h", help="Average peak speed closing down the carrier"),
        "press_wins": st.column_config.NumberColumn(
            "Pressures won", format="%d",
            help=f"Pressures after which the team won the ball within {PRESSURE.win_window:g} s: a tackle, "
                 "an interception, or a pass forced out of play"),
        "press_wins_p90": st.column_config.NumberColumn(
            "Pressures won /90", format="%.1f", help="Pressures that won the ball back, per 90"),
        "press_win_pct": st.column_config.NumberColumn(
            "Press win rate", format="percent", help="Share of this player's pressures that won the ball back"),
        "receipt_space": st.column_config.NumberColumn(
            "Space at receipt", format="%.1f m",
            help="Median distance to the nearest opponent when receiving a pass"),
        "receipts_in_space_pct": st.column_config.NumberColumn(
            "Receipts in space", format="percent",
            help=f"Share of receptions with the nearest opponent {RECEIPTS.in_space:g}+ m away"),
        "def_responsibility_p90": st.column_config.NumberColumn(
            "Def. Responsibility /90", format="%.3f",
            help="xT of opponent passes into the zone this player's position covers, per 90. "
                 "Lower is better"),
        "obso_p90": st.column_config.NumberColumn(
            "OBSO /90", format="%.2f",
            help="Off-Ball Scoring Opportunity: threat offered by free, reachable positions, per 90"),
        "runs_in_behind_p90": st.column_config.NumberColumn("Runs in behind /90", format="%.1f"),
        "pulling_wide_p90": st.column_config.NumberColumn("Pulling wide /90", format="%.1f"),
        "dropping_deep_p90": st.column_config.NumberColumn("Dropping deep /90", format="%.1f"),
    }


INTENSITY_METRICS = {"lane_cut_seconds": "Pressing lane-cut seconds", "sprints": "Transition sprints",
                     "decoy_xT": "xT opened by decoy runs", "scans": "Head checks before receiving",
                     "pressure_intensity": "Pressure intensity (m closed down)",
                     "obso": "Off-ball scoring opportunity (xT·s)"}

METRIC_COLUMNS = ["minutes", "index", "off_ball_value", "receptions", "scan_rate", "pre_reception_scans",
                  "retention_rate", "decoy_runs", "decoy_p90", "lane_cut_p90", "mean_lane_cut",
                  "transitions", "median_latency", "sprint_rate", "transition_score"] + pipeline.TACTICAL_COLUMNS

if page == "Match Leaderboards":
    filters = st.columns([1, 2])
    team = filters[0].radio("Team", ["Both", "Home", "Away"], horizontal=True)
    min_minutes = filters[1].slider("Minimum minutes played", 0, 90, 30)

    board = pipeline.player_totals(game_events, timeline)
    board = pipeline.off_ball_value(board[board["minutes"] >= min_minutes])
    board = match_index(board, min_minutes)          # ranked across both teams, then filtered
    if team != "Both":
        board = board[board["team"] == team]
    board = board.sort_values(["index", "off_ball_value"], ascending=False, na_position="last")
    board.insert(0, "rank", range(1, len(board) + 1))

    columns = (["rank", "name", "metrica_ids"]
               + ["market_value_m"]
               + ["team", "position"] + METRIC_COLUMNS)
    st.dataframe(board[columns], hide_index=True, width="stretch",
                 height=min(38 * (len(board) + 1), 900),
                 column_config=leaderboard_columns())
    with st.expander("How the columns are computed"):
        st.markdown(f"""
- **Decoy Run Value**: for every run of at least {SPACE.run_speed_kmh:g} km/h held for
  {SPACE.min_run_duration:g} s and covering {SPACE.min_run_distance:g} m while the player's team has the
  ball (and they don't), ending level with or ahead of the ball (behind it is the defensive line
  moving up, not a decoy), and not starting within
  {SPACE.transition_window:g} s of winning it (that's a transition, scored below), the
  extra xT-weighted pitch control teammates gained because the runner's markers followed them.
  The comparison is against those markers having stayed with the defensive block. A run that
  pulls markers into more dangerous space counts as zero, not negative.
- **Pressing Lane Cuts**: on every frame the player is within {PRESSING.max_press_distance:g} m
  of the ball while the opponent has it, the threat-weighted share of the carrier's passing lanes
  inside their cover shadow, projected {PRESSING.projection:g} s ahead. Summed over time, per 90.
- **Transition Latency**: time from an open-play turnover to a {TRANSITION.min_sprint_duration:g} s
  sprint above {TRANSITION.sprint_kmh:g} km/h. Only players within
  **{TRANSITION.max_ball_distance:g} m of the ball** at the turnover are judged. The latency column
  is the median over transitions where the player did sprint; *Sprint rate* says how often that
  was, and *Transition score* penalises both slow sprints and walking.
- **Scanning**: a head check is a quick turn of the head (peak at least
  {SCANNING.min_peak_speed:g}°/s) more than {SCANNING.on_angle:g}° away from the ball, lasting
  {SCANNING.min_duration:g}-{SCANNING.max_duration:g} s. *Scan rate* counts them per second in the
  {SCANNING.window:g} s before each reception; receptions with too little head data are left out
  rather than counted as zero. *Kept the ball* is how often the team still had it after.
- **Pressure Intensity**: while the opponent has the ball under control, every spell of at least
  {PRESSURE.min_duration:g} s in which the player is within {PRESSURE.radius:g} m of the carrier and
  closing faster than {PRESSURE.min_closing_kmh:g} km/h. Intensity adds up the ground made up on the
  carrier, weighted by how close: closing from 2 m counts more than from 9 m. Per 90.
- **Receipts in Space**: on the frame a pass arrives, the distance to the nearest opponent.
  *Receipts in space* is the share with {RECEIPTS.in_space:g} m or more.
- **Defensive Responsibility**: each defender's zone is where their average position in the shape
  sits at the moment of a pass (the team's centroid plus their usual offset). Every opponent pass
  that gains xT is charged to the zones around where it ends, shared by distance. Per 90; lower is
  better, and it also reflects how much the team lets the opponent play into that zone.
- **OBSO** (Off-Ball Scoring Opportunity): five times a second while their team has the ball,
  each off-ball attacker's xT, times how free they are, times how reachable they are from the ball.
  It counts whether or not the pass ever came. Per 90.
- **Run types**: each off-ball run, measured against the team's own movement (so a whole block
  dropping isn't anyone's run), is *in behind* when it heads for goal and ends at or past the last
  defender, *pulling wide* when it heads for the touchline, *dropping deep* when it comes back
  towards the ball. Per 90.
- **Off-Ball Index**: the headline score, as on the Off-Ball Index page but within this match:
  ten metrics as percentiles among the match's players in the same position, small samples pulled
  towards 50 (one match is a small sample), five equally weighted categories. Goalkeepers have none.
- **Pillar score**: the mean percentile of the four pillar columns (Decoy Run Value, Pressing
  Lane Cuts, Transition score, Scan rate) among the players shown.
- **Names over time**: every run, pressing frame, transition and scan is credited to the name the
  tracked player carried at that minute, so a player replaced at 64' only gets credit up to 64'.
""")

    st.subheader("Match analysis")
    facts = match_facts(signature, game)
    shown_moments = key_moments.assign(
        team=key_moments["player"].str.split("_").str[0],
        name=[name_of(pid, minute) for pid, minute in zip(key_moments["player"], key_moments["minute"])])
    goals = pd.DataFrame([{"minute": f.minute, "team": f.player.split("_")[0], "name": f.name or f.player}
                          for f in facts.of_kind("goal") if f.player], columns=["minute", "team", "name"])
    st.plotly_chart(charts.match_timeline(shown_moments, goals, facts.home, facts.away), key="timeline")
    st.caption("Every key moment from the four pillars, by team, with the goals. Hover a marker for the "
               "detail; the Highlights beside the pitch in Tactical Playback play any of them.")

    battle, intensity = st.columns(2)
    with battle:
        teams = analysis.team_match_summary(game_events, pipeline.load_roster(GAMES[game]))
        # A diverging share bar reads "more is better", so lower-is-better and plain counts stay out.
        teams = teams.drop(index=["Median transition latency (s)", "Transitions (total)"], errors="ignore")
        st.plotly_chart(charts.team_comparison(teams), key="battle")
        st.caption("Each row splits the two clubs' total. Labels are the raw values.")
    with intensity:
        blocks = analysis.intensity_blocks(game_events, pipeline.load_roster(GAMES[game]))
        metric = st.selectbox("Metric", list(INTENSITY_METRICS), format_func=INTENSITY_METRICS.get,
                              key="intensity-metric")
        st.plotly_chart(charts.intensity_timeline(blocks.assign(value=blocks[metric]), metric,
                                                  INTENSITY_METRICS[metric]), key="intensity")
        st.caption("Each club's off-ball work per 15-minute block: who faded, who finished strongest.")

    effect = analysis.scanning_effect(game_events.get("scanning"))
    st.plotly_chart(charts.scanning_effect(effect), key="scan-effect-match")
    st.caption(f"Every reception in this match with head data, by head checks in the "
               f"{SCANNING.window:g} s before it. Whiskers are 95% intervals: one match is a small "
               "sample, so the Model Check tab pools the whole league.")

def index_columns(categories: dict[str, str], title: str) -> dict:
    """Column formatting for the index tables: every score is a 0-100 bar."""
    bar = lambda label, help=None: st.column_config.ProgressColumn(label, min_value=0, max_value=100,
                                                                   format="%.0f", help=help)
    return {"rank": st.column_config.NumberColumn("#", width="small"), "name": "Player", "real_team": "Club",
            "broad_position": "Position", "minutes": st.column_config.NumberColumn("Minutes", format="%.0f"),
            "index": bar(title, "Weighted mean of the subscores; 50 = the position's middle"),
            "coverage": st.column_config.NumberColumn("Coverage", format="percent",
                                                      help="Share of the index's weight with data")} \
        | {name: bar(label) for name, label in categories.items()}


METRIC_LABELS = {m: label for m, (label, _) in (analysis.PILLARS | analysis.TACTICAL).items()} \
    | {"pitch_control_pct": "Pitch control"}

if page == "Off-Ball Index":
    by_impact = st.segmented_control(
        "Weighting", ["Off-Ball Index", "Off-Ball Impact"], default="Off-Ball Index", key="index-mode",
        help="Off-Ball Index: ten metrics in five categories, equal weights. Off-Ball Impact: the five "
             "tactical metrics, each weighted by how much it moves the team's shot balance.") == "Off-Ball Impact"
    controls = st.columns([1, 1, 2])
    position = controls[0].selectbox("Position", ["All", "Defender", "Midfielder", "Forward"], key="index-position")
    if by_impact:
        fitted = impact_weights(signature)
        config, scored, title = impact.index_config(fitted["weight"]), impact_table(signature), "Off-Ball Impact"
        st.caption("The five tactical metrics, weighted by what they do for the team: every match is cut into "
                   "5-minute windows, and each metric's weight is how much more of it (per standard deviation of "
                   "team work) goes with a better shot balance, after allowing for possession. Players are then "
                   "scored as in the Off-Ball Index: percentiles within position, 50 = the middle.")
        left, right = st.columns([3, 2])
        left.plotly_chart(charts.impact_weights(fitted, METRIC_LABELS), key="impact-weights")
        shares = {c["label"]: c["weight"] for c in config["categories"].values()}
        total = sum(shares.values()) or 1
        right.markdown("**Pillar weights in the index**  \n" + "  \n".join(
            f"{label}: **{weight / total:.0%}**" for label, weight in sorted(shares.items(), key=lambda kv: -kv[1])))
        right.caption("A metric with no detectable effect on shots gets 0, never a negative weight. "
                      "Whiskers are 95% intervals from resampling whole matches: with six matches, read "
                      "the order, not the exact numbers.")
    else:
        st.caption("One 0-100 score per player from ten off-ball metrics, each ranked against the same "
                   "position (50 = the middle), grouped into five equally weighted categories.")
        with controls[1].popover("Tune category weights", width="stretch"):
            weights = tuple((name, st.slider(label, 0.0, 3.0, float(INDEX_CONFIG["categories"][name]["weight"]),
                                             0.5, key=f"index-weight-{name}")) for name, label in CATEGORIES.items())
            st.caption("Each category's share of the index. Metric weights within a category are in "
                       "`config/composite_index.json`.")
        if not any(weight for _, weight in weights):
            st.warning("Every weight is zero, so the default weights are used.")
            weights = ()
        config, scored, title = INDEX_CONFIG, index_table(signature, weights), "Off-Ball Index"
    categories = {name: c["label"] for name, c in config["categories"].items()}
    ranked = scored[scored["eligible"]].sort_values("index", ascending=False)
    if position != "All":
        ranked = ranked[ranked["broad_position"] == position]
    ranked = ranked.assign(rank=range(1, len(ranked) + 1))
    controls[2].caption(f"**{len(ranked)}** players ranked. {(~scored['eligible']).sum()} are not: under "
                        f"{config['min_minutes']} minutes, or goalkeepers (these metrics don't describe their job).")

    table, radar = st.columns([3, 2])
    table.dataframe(ranked[["rank", "name", "real_team", "broad_position", "minutes", "index", *categories,
                            "coverage"]], hide_index=True, width="stretch", height=560,
                    column_config=index_columns(categories, title))
    with radar:
        compare = st.multiselect("Compare on the polygon", ranked["name"].tolist(),
                                 default=ranked["name"].head(2).tolist(), max_selections=4,
                                 key=f"index-compare-{position}")
        records = {r["name"]: r for r in composite.polygon(scored[scored["name"].isin(compare)], config)}
        st.plotly_chart(charts.attribute_polygon([records[n] for n in compare if n in records]),
                        key="index-polygon")
    if compare:
        metrics = [(m, f"{c['label']} · {METRIC_LABELS.get(m, m)}")
                   for c in config["categories"].values() for m in c["metrics"]]
        chosen_rows = scored.set_index("name").loc[[n for n in compare if n in records]]
        breakdown = pd.DataFrame({name: [chosen_rows.loc[name, f"{m}_score"] for m, _ in metrics]
                                  for name in chosen_rows.index}, index=[label for _, label in metrics])
        st.markdown("**Metric scores behind the polygon** (0-100 within position, shrunk towards 50 on a "
                    "small sample)")
        st.dataframe(breakdown.round(0), width="stretch",
                     column_config={n: st.column_config.ProgressColumn(n, min_value=0, max_value=100, format="%.0f")
                                    for n in breakdown.columns})
    with st.expander("How the index is computed"):
        st.markdown(f"""
- **Metrics**: Spacing = pitch control and receipts in space; Movement = decoy runs and transition
  reaction; Creation = OBSO; Scanning = scan rate; Defence = lane cuts, pressure intensity, pressures won and
  defensive responsibility (flipped, lower is better). Overlapping metrics (latency, sprint rate,
  closing speed...) and run types (style, not quality) are left out.
- **Normalisation**: each metric is a mid-rank percentile among players of the same position and
  season, so 100 is always best and units don't matter.
- **Small samples**: a score is pulled towards 50 by n / (n + k), with n the player's minutes,
  receptions or transitions and k the metric's minimum (90 minutes, 10 receptions, 10 transitions).
- **Formula**: subscore = weighted mean of its metric scores; index = weighted mean of the
  subscores. A missing metric drops out and the weights renormalise; *Coverage* is the share of the
  weight observed, and below {INDEX_CONFIG['min_coverage']:.0%} there is no index.
- **Caveats**: lane cuts, pressure intensity and pressures won overlap (ρ ≈ 0.8), so pressing volume dominates
  within Defence; defensive responsibility also reflects the team's press; Spacing and Creation do not
  track any hidden talent in this league, so read them as role context. Six matches is a small sample:
  one rank moves a percentile by 7-10 points. Full method: `docs/composite_index.md`.
""" if not by_impact else """
- **Pillars**: Pressure (pressure intensity), Receipts in space, Defensive cover (defensive
  responsibility, flipped), OBSO, and Run types (runs in behind, pulling wide, dropping deep).
- **Weights**: each match is cut into 5-minute windows per team. The team's total of each metric is
  divided by the possession it applies to (pressing and cover per minute the opponent has the ball;
  OBSO and runs per minute of their own; receipts as a share). The window's shot difference is
  regressed on the standardised metrics, after removing possession share, by non-negative least
  squares. A pillar's weight is the sum of its metrics'.
- **Why shots**: defensive responsibility is built from the xT of opponents' passes, so fitting to
  xT would be circular. Shots are the nearest outcome no metric is built from.
- **Scoring players**: the same percentiles within position and small-sample shrinkage as the
  Off-Ball Index; inside a pillar its metrics count equally, so the polygon shows the player's work
  on each pillar and the weights decide how much each pillar counts.
- **Caveats**: OBSO and defensive cover sit closest to the shot (being free in dangerous space,
  letting the ball into yours), so they are bound to explain shots best; pressing and runs pay off
  indirectly and can look worthless here. It is association, not causation, from six matches.
  Full method: `docs/impact_index.md`.
""")

PILLAR_LABELS = {column: label for column, (label, _) in (analysis.PILLARS | analysis.TACTICAL).items()}
RUN_LABELS = {"runs_in_behind_p90": "runs in behind", "pulling_wide_p90": "pulling wide",
              "dropping_deep_p90": "dropping deep"}

if page == "Player Profile":
    pool, valuation, _ = value_models(signature, SCOUT_POOL_MINUTES, False)
    if pool.empty:
        st.info("No outfielders with enough minutes to profile.")
    else:
        ranked = pool.sort_values(["index", "off_ball_value"], ascending=False, na_position="last")
        picker = st.columns([2, 1])
        options = ranked["name"].tolist()
        chosen = picker[0].selectbox(
            "Player", options, key="profile-player", index=options.index(fav_player) if fav_player in options else 0,
            format_func=lambda n: f"{n} · {ranked.loc[ranked['name'] == n, 'real_team'].iloc[0]}")
        pillar = picker[1].selectbox("Metric", list(PILLAR_LABELS), format_func=PILLAR_LABELS.get,
                                     key="profile-pillar", help="The four pillars, then the five tactical metrics")
        profile = analysis.player_profile(chosen, pool, per_match(signature))
        row = profile["row"]
        tiles = st.columns(6)
        tiles[0].metric("Off-Ball Index", "–" if pd.isna(row["index"]) else f"{row['index']:.0f}",
                        help="Composite of ten metrics within position (Off-Ball Index page); blank under "
                             f"{INDEX_CONFIG['min_minutes']} minutes")
        tiles[1].metric("Pillar score", f"{row['off_ball_value']:.0f}",
                        help="Mean percentile of the four pillars across the league's outfielders")
        tiles[2].metric("Market value", f"€{row['market_value_m']:.1f}m")
        tiles[3].metric("ML price", f"€{row['Predicted_Value']:.1f}m",
                        delta=f"{row['Predicted_Value'] - row['market_value_m']:+.1f}m vs market",
                        help="What the out-of-fold model prices their off-ball work at")
        tiles[4].metric("Role (ML cluster)", str(row.get("Cluster_Role", "–")))
        tiles[5].metric("Minutes", f"{row['minutes']:.0f}", help=f"{row['broad_position']}, all matches")
        peers = pool[pool["broad_position"] == row["broad_position"]]
        tactical = analysis.percentile_table(peers, analysis.TACTICAL).loc[row.name, list(analysis.TACTICAL)]
        standout = [analysis.TACTICAL[c][0] for c in tactical.dropna().sort_values(ascending=False).index[:2]]
        runs = row[list(RUN_LABELS)].astype(float)
        style = (f"mostly **{RUN_LABELS[runs.idxmax()]}** ({runs.max():.1f} per 90)"
                 if runs.notna().any() and runs.max() > 0 else "no classified runs")
        st.markdown(f"**Strengths** for a {row['broad_position'].lower()}: {', '.join(profile['strengths']) or '–'}  \n"
                    f"**To work on**: {', '.join(profile['weaknesses']) or '–'}  \n"
                    f"**Stands out tactically**: {', '.join(standout) or '–'} · **Runs**: {style}")

        left, right = st.columns(2)
        indexed = index_table(signature)
        records = composite.polygon(indexed[(indexed["name"] == chosen)
                                            & (indexed["real_team"] == row["real_team"])], INDEX_CONFIG)
        left.plotly_chart(charts.attribute_polygon(records, f"{chosen}: attribute polygon"), key="profile-polygon")
        left.caption("The Off-Ball Index's five subscores against the same position (dashed ring = 50)."
                     if records else f"Not on the index yet: under {INDEX_CONFIG['min_minutes']} minutes.")
        right.plotly_chart(charts.pillar_profile(row, peers, analysis.PILLARS), key="profile-pillars")
        right.caption(f"Percentile among the league's {len(peers)} {row['broad_position'].lower()}s "
                      "(50 = their median). Bars are labelled with the raw value.")

        left, right = st.columns(2)
        left.plotly_chart(charts.pillar_profile(row, peers, analysis.TACTICAL, "tactical profile"),
                          key="profile-tactical")
        left.caption("The five tactical metrics against the same players. Defensive responsibility is "
                     "flipped so a long bar is always good; run types show style, not quality.")
        right.plotly_chart(charts.metric_distribution(pool, pillar, PILLAR_LABELS[pillar], [chosen]),
                           key="profile-distribution")
        right.caption("Every outfielder, by position; the bar marks each position's median.")
        matches = per_match(signature)
        st.plotly_chart(charts.match_trend(matches[matches["name"] == chosen], pillar, PILLAR_LABELS[pillar],
                                              benchmark=float(peers[pillar].median())), key="profile-trend")
        st.subheader(f"How the ML price of €{row['Predicted_Value']:.1f}m is calculated")
        if valuation is None:
            st.info("Too few players to train the valuation model.")
        else:
            X = ml_model.design_matrix(pool)
            at = pool.index.get_loc(row.name)
            steps = ml_model.price_steps(valuation["fold_models"][int(row["Price_Fold"])], X, at)
            left, right = st.columns([3, 2])
            steps = [(ml_model.SHORT_NAMES.get(label, label), price) for label, price in steps]
            left.plotly_chart(charts.price_waterfall(steps, f"{chosen}: how the ML price is built"),
                              key="profile-price")
            left.caption("The first bar prices a typical player in the same position (every off-ball number at "
                         "the league median); each next bar swaps in one of this player's own numbers. The bars "
                         "add up exactly, but how a change is shared out depends on the order, because the trees "
                         "combine features.")
            with right:
                st.markdown("**The formula.** A gradient-boosted tree model learns the *logarithm* of market "
                            "value (prices are skewed: a few players cost far more):")
                st.latex(rf"\ln(\text{{price}}) = c + \sum_{{t=1}}^{{{valuation['trees']}}} "
                         rf"{valuation['learning_rate']:g}\cdot\text{{tree}}_t(x)")
                st.latex(r"\text{price} = e^{\ln(\text{price})}")
                st.markdown(
                    f"- **x**: {', '.join(c for c in valuation['features'] if not c.startswith('Position_'))}, "
                    "and the player's position.\n"
                    f"- **tree_t**: {valuation['trees']} small trees of {valuation['leaves']} leaves, each fitted to "
                    "what the trees before it got wrong; **c** is the average log price.\n"
                    f"- **Out-of-fold**: {chosen} was priced by a model trained on the *other* four-fifths of the "
                    "league, never on their own price.\n"
                    f"- **Accuracy**: cross-validated R² {valuation['r2']:.2f} (0 = no better than the average "
                    f"price), typical error €{valuation['mae_m']:.0f}m. Off-ball work explains little of a price, "
                    "which is mostly finishing, passing and position: the gap to market value is the scouting "
                    "signal, not a precise valuation.")

        st.subheader("Most similar players")
        st.caption(f"Nearest {row['broad_position'].lower()}s on the four standardised pillars, "
                   "with their price: a like-for-like shortlist.")
        similar = profile["comparables"].merge(
            pool[["name", "real_team", "market_value_m", "Predicted_Value", "index", "off_ball_value"]],
            on=["name", "real_team"], how="left")
        st.dataframe(similar[["name", "real_team", "minutes", "index", "off_ball_value", "market_value_m",
                              "Predicted_Value", *analysis.PILLARS, "similarity"]],
                     hide_index=True, width="stretch",
                     column_config=leaderboard_columns()
                     | {"Predicted_Value": st.column_config.NumberColumn("ML price", format="€%.1fm"),
                        "similarity": st.column_config.ProgressColumn(
                            "Similarity", min_value=0, max_value=1, format="%.2f",
                            help="1 / (1 + mean z-score distance over the four pillars)")})

if page == "Value Finder":
    st.caption("Each point is a fictional player across the synthetic league, at their simulated "
               "market value, coloured by the off-ball role machine learning groups them into.")
    controls = st.columns([2, 1, 1, 1])
    pool_minutes = controls[0].slider("Minimum minutes played", 0, 90, 30, key="pool_minutes")
    y_axis = controls[1].selectbox(
        "Market value against", list(Y_AXES), key="finder-axis",
        help="Off-Ball Index: the composite of ten metrics within position (players with 90+ minutes). "
             "Pillar score: the four-pillar percentile score. ML predicted value: what a "
             "gradient-boosted model prices the player at from off-ball work. The others are the "
             "tactical metrics; for Defensive Responsibility lower is better.")
    log_x = controls[2].toggle("Log scale for value", value=False)
    by_position = controls[3].toggle(
        "Rank within position", value=False,
        help="Compare each player only with their own position. Decoy runs and pressing near "
             "the ball are mostly midfield and forward work, so one pool ranks defenders low "
             "for their role rather than their quality.")

    pool, valuation, roles = value_models(signature, pool_minutes, by_position)
    predicted = y_axis == "ML predicted value"
    y_col, higher = Y_AXES[y_axis]
    if valuation is None:
        st.info("Too few players at this minutes threshold; lower it to see the chart.")
    else:
        st.plotly_chart(value_finder_figure(pool, log_x, y_axis), key="finder")
        st.caption(f"{len(pool)} outfielders across {len(GAMES)} matches. Goalkeepers are left out: "
                   "these metrics don't describe their job.")
        st.markdown(
            f"**ML price** ({valuation['engine']}): each player is priced by a model trained on "
            f"*other* players (5-fold cross-validation), from "
            f"{', '.join(dict.fromkeys('position' if f.startswith('Position_') else f for f in valuation['features']))}. "
            f"Cross-validated R² **{valuation['r2']:.2f}** (log value; 0 = no better than the average), "
            f"typical error **€{valuation['mae_m']:.0f}m**. Off-ball work explains little of a price, "
            "which is driven mostly by finishing, passing, speed and position: the gap between "
            "price and contribution is the scouting signal, not a precise valuation. The Player Profile "
            "breaks any player's price down step by step, with the formula.")

        if predicted:
            finds = pool.assign(gap=pool["Predicted_Value"] / pool["market_value_m"])
            finds = finds[finds["gap"] > 1].sort_values("gap", ascending=False)
            st.subheader(f"Priced below their off-ball work · {len(finds)} players")
            st.caption("A weak model pulls its predictions towards the average price, so cheap "
                       "players tend to land above the line. Read the ranking as relative, and "
                       "trust the biggest gaps most.")
        else:
            x_median, y_median = pool["market_value_m"].median(), pool[y_col].median()
            better = pool[y_col] > y_median if higher else pool[y_col] < y_median
            finds = pool[(pool["market_value_m"] < x_median) & better] \
                .sort_values(y_col, ascending=not higher)
            st.subheader(f"Moneyball quadrant on {y_axis} · {len(finds)} players")
        finds = finds.assign(rank=range(1, len(finds) + 1))
        st.dataframe(finds[["rank", "name", "real_team", "broad_position", "Cluster_Role", "market_value_m",
                            "Predicted_Value", "pitch_control_pct"] + METRIC_COLUMNS],
                     hide_index=True, width="stretch",
                     column_config=leaderboard_columns()
                     | {"broad_position": "Position", "Cluster_Role": "Role (ML cluster)",
                        "Predicted_Value": st.column_config.NumberColumn(
                            "ML price", format="€%.1fm", help="Out-of-fold model prediction"),
                        "pitch_control_pct": st.column_config.NumberColumn(
                            "Pitch Control %", format="%.1f%%",
                            help="Share of the pitch the player controls while their team has the ball")})
        if predicted:
            st.plotly_chart(charts.value_ladder(pool), key="ladder")
            st.caption("The biggest gaps between price and off-ball work, as a multiple of market value.")
        with st.expander("What the roles mean"):
            st.markdown("K-means (k = 4) on the standardised off-ball metrics. Each cluster is named "
                        "after the metric it leads the others on (for Transition Latency, by being "
                        "fastest). Average values per role:")
            st.dataframe(roles, width="stretch")


SCOUT_SUGGESTIONS = [
    "Which forward creates the most space with decoy runs?",
    "Who is the fastest midfielder in transition?",
    "Which defender cuts the most passing lanes when pressing?",
    "Who is the biggest bargain for their off-ball work?",
]

if page == "AI Scout":
    agent = scout_agent(signature)
    st.caption(f"Ask about off-ball work across the league's {len(agent.df)} outfielders with "
               f"{SCOUT_POOL_MINUTES}+ minutes. Answers come from **{agent.mode}**, are grounded in "
               "the same metrics as the other tabs, and quote the numbers behind them. Every club "
               "and player is fictional.")
    # The conversation lives in this browser session, never in the shared agent.
    conversation: list[dict[str, str]] = st.session_state.setdefault("scout_messages", [])

    suggested = None
    if not conversation:
        columns = st.columns(2)
        for k, suggestion in enumerate(SCOUT_SUGGESTIONS):
            if columns[k % 2].button(suggestion, key=f"scout-suggestion-{k}", width="stretch"):
                suggested = suggestion
    for message in conversation:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask the AI Scout about off-ball work...") or suggested
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            answer = st.write_stream(agent.stream(question, conversation))
        conversation += [{"role": "user", "content": question},
                         {"role": "assistant", "content": answer if isinstance(answer, str) else str(answer)}]

    if conversation and st.button("Clear conversation", key="scout-clear"):
        conversation.clear()
        st.rerun()


if page == "Match Report":
    config = ModelConfig.from_env()
    engine = f"{config.provider} · {config.model}" if config else "offline (code-written, no model configured)"
    st.caption("A team of four agents writes the match report, orchestrated with **Microsoft Agent Framework** "
               f"on **{engine}**: the **Analyst** reads the verified match facts with tools and picks the "
               "story, the **Narrator** writes it citing every moment, the **Fact-checker** (code, not a "
               "model) rejects any time, number or player the facts don't support and sends it back, and "
               "the **Persona editor** retells it for your audience, then gets checked again.")
    facts = match_facts(signature, game)
    choose, generate = st.columns([3, 1])
    persona = choose.radio("Audience", list(PERSONAS), horizontal=True, key="report-persona",
                           index=list(PERSONAS).index("Analyst" if view_mode == "Analyst" else "Fan"))
    focus = fav_player or fav_club or ""
    choose.caption(f"In {language}" + (f", following {focus}" if focus else "") + " (set in Your view)."
                   + ("" if language == "English" or config else " Other languages need the AI model; "
                      "offline reports are in English."))
    reports: dict = st.session_state.setdefault("match_reports", {})
    key = (game, persona, language, focus)
    if generate.button("Generate report" if key not in reports else "Regenerate", type="primary",
                 key="report-generate", width="stretch"):
        with st.status("The agents are writing the report...", expanded=False) as status:
            report = generate_report(facts, persona, language=language, focus=focus)
            status.update(label=f"Report ready in {report.trace[-1].seconds:.0f} s", state="complete")
        reports[key] = report

    report = reports.get(key)
    if report is None:
        st.info(f"Choose an audience and generate a report on {facts.home} {facts.score[0]}-"
                f"{facts.score[1]} {facts.away}.")
    else:
        if report.verified:
            st.success("Fact-checked: every cited moment, number and player matches the match data.", icon="✅")
        else:
            st.warning("This report could not be fully verified against the match data.", icon="⚠️")
        st.markdown(report.text)
        st.caption(f"Written by {report.mode}.")

        moments_cited = [f for f in report.cited if f.time is not None]
        if moments_cited:
            st.markdown("**Watch the cited moments**")
            columns = st.columns(min(3, len(moments_cited)))
            for k, fact in enumerate(moments_cited):
                columns[k % len(columns)].button(
                    f"▶ {fact.clock} · {moment_label(fact.kind)}", key=f"report-jump-{game}-{fact.id}",
                    help=fact.text, width="stretch", on_click=jump_to_fact, args=(fact,))

        with st.expander("Agent trace"):
            st.dataframe(pd.DataFrame([{"Agent": t.agent, "Action": t.action, "Detail": t.detail,
                                        "Seconds": t.seconds or None} for t in report.trace]),
                         hide_index=True, width="stretch")
        with st.expander(f"The {len(facts.facts)} verified facts the agents could use"):
            st.code(facts.lines(), language=None)



ROLE_ORDER = ["Centre-Forward", "Left Winger", "Right Winger", "Left Midfield", "Right Midfield",
              "Central Midfield", "Defensive Midfield", "Left-Back", "Right-Back", "Centre-Back"]
VALIDATION_TITLES = {"decoy_p90": "Decoy runs vs work rate", "lane_cut_p90": "Lane cuts vs pressing",
                     "transition_score": "Transition score vs work rate",
                     "median_latency": "Transition latency vs work rate (should fall)",
                     "scan_rate": "Scan rate vs awareness"}

if page == "Model Check":
    st.caption("The simulation gives every fictional player hidden talents (work rate, pressing, "
               "awareness). The metrics never see them. If a metric is real, it should recover the "
               "talent it is meant to measure from the tracking alone.")
    pool, _, _ = value_models(signature, SCOUT_POOL_MINUTES, False)
    truth = hidden_talent(signature)
    if truth.empty or pool.empty:
        st.info("No hidden attributes found (data/synthetic/league_players.csv).")
    else:
        checked = {column for pair in analysis.VALIDATION_PAIRS for column in pair[:2]}
        joined = analysis.rank_within(analysis.attach_truth(pool, truth), checked)
        table = analysis.validation(joined)
        st.dataframe(table.assign(metric=table["metric"].map(lambda m: VALIDATION_TITLES.get(m, m))),
                     hide_index=True, width="stretch",
                     column_config={"attribute": "Hidden talent", "metric": "Check",
                                    "rho": st.column_config.NumberColumn("Spearman ρ", format="%.2f"),
                                    "low": st.column_config.NumberColumn("95% low", format="%.2f"),
                                    "high": st.column_config.NumberColumn("95% high", format="%.2f"),
                                    "n": "Players", "verdict": "Verdict"})
        st.caption("ρ is the rank correlation between talent and metric (1 = same order), comparing "
                   "each player only with their own position: decoy runs and pressing are mostly "
                   "midfield and forward work, so a league-wide comparison would mostly measure "
                   "position. The 95% interval comes from 1,000 bootstrap resamples of the players. "
                   "Scatter axes are percentiles within position. Decoy runs only count when they end level "
                   "with or ahead of the ball, so the check rests on forwards: centre-backs make no decoy "
                   "runs, and midfielders' decoy value doesn't yet track their work rate.")
        pairs = [(a, m, VALIDATION_TITLES.get(m, m)) for a, m, _ in analysis.VALIDATION_PAIRS]
        st.plotly_chart(charts.validation_scatter(joined, pairs), key="validation")

    st.subheader("Do the tactical metrics find the roles?")
    order = [r for r in ROLE_ORDER if r in set(pool["position"])]
    medians = pool.groupby("position")[list(analysis.TACTICAL)].median().reindex(order)
    medians.columns = [analysis.TACTICAL[c][0] for c in medians.columns]
    st.plotly_chart(charts.role_heatmap(medians, {analysis.TACTICAL["def_responsibility_p90"][0]}),
                    key="role-heatmap")
    st.caption("The simulator gives each role a style: strikers mostly run in behind, wide players and "
               "full-backs pull wide, midfielders drop deep, and defensive and central midfielders press "
               "hardest. The metrics never see those settings; they only see the tracking. Each column is "
               "shaded from its lowest to its best value (for defensive responsibility, lowest is best). "
               f"{len(pool)} outfielders with {SCOUT_POOL_MINUTES}+ minutes; some positions have only a few players.")

    st.subheader("Does scanning keep the ball?")
    league_effect = analysis.scanning_effect(global_data.get("scanning"))
    st.plotly_chart(charts.scanning_effect(league_effect), key="scan-effect-league")
    if len(league_effect) and league_effect["n"].sum():
        none, most = league_effect.iloc[0], league_effect.iloc[-1]
        st.caption(f"All {int(league_effect['n'].sum()):,} receptions with head data across {len(GAMES)} "
                   f"matches. Players with no head check kept the ball {none['retention']:.0%} of the time; "
                   f"with {most['bucket']} it was {most['retention']:.0%}. In this synthetic league that link "
                   "is built into the simulation, so this checks the pipeline finds it, not that it holds "
                   "in real football.")
