"""
Charts for the off-ball statistics: pure functions from DataFrames to plotly figures.

Every function takes plain pandas input, returns a plotly.graph_objects.Figure and
never raises on empty or all-NaN input (it returns a figure carrying a "No data"
annotation instead), so the app can call them before any data exists.

Conventions shared by every chart, so they read as one system:

    colour      follows the entity, never its rank: Home/Away always wear the team
                colours, other categories take the fixed palette in order (never cycled)
    identity    is never colour alone: teams also differ by marker symbol, and charts
                with up to four series carry direct labels next to the legend
    axes        one y-axis only; recessive hairline gridlines; text stays in the
                theme's neutral ink (no series-coloured text)
    hover       every mark has a tooltip with the exact value and its unit
    theme       paper/plot backgrounds and font colours are left to Streamlit's theme
                (st.plotly_chart), so the charts follow light and dark mode
"""

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

# Team identity and the validated categorical order (blue, orange, aqua first: the three
# that stay distinguishable all-pairs, so scatters use only those).
TEAM_COLOUR = {"Home": "#e63946", "Away": "#3a86ff"}
TEAM_SYMBOL = {"Home": "circle", "Away": "diamond"}
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# Neutral greys as translucent mid-grey so they recede on both light and dark surfaces.
GRID = "rgba(128,128,128,0.18)"
GUIDE = "rgba(128,128,128,0.75)"
NO_DATA = "rgba(128,128,128,0.30)"
RING = dict(width=1, color="rgba(128,128,128,0.45)")  # neutral ring separating overlapping markers
# Goal-tag width estimate: ~6.5 px per character at 11 px plus padding, on a ~700 px plot area.
TAG_CHAR_PX, TAG_PAD_PX, PLOT_PX = 6.5, 12.0, 700.0

# Moment lanes, top to bottom, with the labels core.moments.KINDS uses.
MOMENT_LANES = {
    "decoy": "High-xT decoy run",
    "press": "Lane-cutting press",
    "defensive_transition": "Elite defensive transition",
    "attacking_transition": "Explosive attacking transition",
    "scan": "Elite scan before receiving",
}
POSITION_ORDER = ["Goalkeeper", "Defender", "Midfielder", "Forward"]

# Units for the hover of known metric columns (anything else is shown unitless).
UNITS = {
    "decoy_p90": "xT /90",
    "lane_cut_p90": "s /90",
    "transition_score": "(0-1)",
    "scan_rate": "scans/s",
    "pre_reception_scans": "scans",
    "retention_rate": "",
    "pitch_control_pct": "%",
    "off_ball_value": "(percentile score)",
    "median_latency": "s",
    "sprint_rate": "",
    "market_value_m": "€m",
    "pressure_intensity": "m /90",
    "receipts_in_space_pct": "",
    "def_responsibility_p90": "xT /90",
    "obso_p90": "xT·s /90",
    "runs_in_behind_p90": "runs /90",
    "pulling_wide_p90": "runs /90",
    "dropping_deep_p90": "runs /90",
    "Predicted_Value": "€m",
}


# ── Shared helpers ───────────────────────────────────────────────────────────
def fmt(value: float) -> str:
    """A raw value to a compact string: more decimals for small numbers, '–' for NaN."""
    if value is None or not np.isfinite(value):
        return "–"
    size = abs(value)
    return f"{value:.0f}" if size >= 100 else f"{value:.1f}" if size >= 10 \
        else f"{value:.2f}" if size >= 1 else f"{value:.3f}"


def with_unit(value: float, column: str) -> str:
    unit = UNITS.get(column, "")
    return f"{fmt(value)} {unit}".strip()


def clock(minute: float) -> str:
    """Match minute -> 'mm:ss' (the same clock as core.match_tools, truncated to whole seconds)."""
    seconds = int(round(minute * 60, 6))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def style(fig: go.Figure, title: str, height: int | None = None, legend_below: bool = False) -> go.Figure:
    """The common frame: a title saying what the chart shows, a horizontal legend (top right,
    or below the plot where labels already crowd the top), recessive grids."""
    legend = (dict(orientation="h", y=-0.12, yanchor="top", x=0, xanchor="left") if legend_below
              else dict(orientation="h", y=1.0, yanchor="bottom", x=1, xanchor="right"))
    fig.update_layout(title=dict(text=title, x=0, xanchor="left"), height=height,
                      margin=dict(l=10, r=10, t=60, b=10), hovermode="closest", legend=legend)
    fig.update_xaxes(gridcolor=GRID, gridwidth=1, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, zeroline=False)
    return fig


def empty_figure(title: str, message: str = "No data") -> go.Figure:
    """A blank, axis-free figure with a centred message, for empty input."""
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font_size=14)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, title, height=300)


def is_empty(df: pd.DataFrame | None, *columns: str) -> bool:
    """True when df is missing, has no rows, lacks a column, or a column is all NaN."""
    if df is None or len(df) == 0:
        return True
    return any(c not in df.columns or df[c].isna().all() for c in columns)


def side_of(team: str) -> str:
    return team if team in TEAM_COLOUR else "Home"


# ── 1. Pillar profile ────────────────────────────────────────────────────────
def percentile(value: float, values: pd.Series, higher_is_better: bool, own_index=None) -> float:
    """Percentile (0-100) of value within values via rank(pct=True); inverted for lower-is-better.

    When own_index is a row of values holding this same value, that row is ranked in
    place; otherwise the value joins the pool as one extra row.
    """
    if not np.isfinite(value):
        return np.nan
    pool = values.dropna()
    if pool.empty:
        return np.nan      # nothing to rank against: show "no data", not a lone 100th percentile
    if (own_index is None or not pool.index.is_unique or own_index not in pool.index
            or pool[own_index] != value):
        own_index = "__player__"
        pool = pd.concat([pool, pd.Series([value], index=[own_index])])
    ranks = pool.rank(pct=True, ascending=higher_is_better)
    return float(ranks[own_index] * 100)


def pillar_profile(player: pd.Series, pool: pd.DataFrame, pillars: dict[str, tuple[str, bool]],
                   subject: str = "off-ball profile") -> go.Figure:
    """One player's percentile on each pillar within a comparison pool.

    player:  a row of player totals (index = metric columns, optional 'name'); its
             Series name is used to recognise the player's own row in pool.
    pool:    the comparison players, one row each, holding the pillar columns.
    pillars: column -> (label, higher_is_better), drawn top to bottom in this order.
    subject: what the chart shows, after the player's name in the title.

    Bars run 0-100 with a reference line at 50 (the pool median); each bar is labelled
    with the raw value and the hover adds its unit and percentile. A pillar with no
    value for the player, or none in the pool, is drawn as a full-length grey "no data" bar.
    """
    name = player.get("name", player.name) if player is not None else None
    title = f"{name}: {subject}" if name is not None else subject[:1].upper() + subject[1:]
    columns = [c for c in pillars if c in pool.columns] if pool is not None else []
    if player is None or is_empty(pool) or not columns:
        return empty_figure(title)

    rows = []
    for column in columns:
        label, higher = pillars[column]
        raw = pd.to_numeric(player.get(column, np.nan), errors="coerce")
        raw = float(raw) if pd.notna(raw) else np.nan
        pct = percentile(raw, pd.to_numeric(pool[column], errors="coerce"), higher, player.name)
        rows.append(dict(label=label, raw=raw, pct=pct, unit_text=with_unit(raw, column),
                         direction="higher is better" if higher else "lower is better"))
    table = pd.DataFrame(rows)
    scored, missing = table[table["pct"].notna()], table[table["pct"].isna()]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=scored["label"], x=scored["pct"], orientation="h", name="Percentile",
        marker=dict(color=PALETTE[0], cornerradius=4), text=scored["raw"].map(fmt),
        textposition="outside", cliponaxis=False, showlegend=False,
        customdata=np.column_stack([scored["unit_text"], scored["direction"]]) if len(scored) else None,
        hovertemplate="<b>%{y}</b><br>Value: %{customdata[0]}<br>Percentile: %{x:.0f} "
                      "(%{customdata[1]})<extra></extra>"))
    if len(missing):
        fig.add_trace(go.Bar(
            y=missing["label"], x=[100] * len(missing), orientation="h", name="No data",
            marker=dict(color=NO_DATA, cornerradius=4), text=["no data"] * len(missing),
            textposition="inside", insidetextanchor="middle", showlegend=False,
            hovertemplate="<b>%{y}</b><br>No data to rank on this pillar<extra></extra>"))
    fig.add_vline(x=50, line=dict(color=GUIDE, width=1, dash="dot"),
                  annotation=dict(text="Pool median", font_size=11), annotation_position="top")
    style(fig, f"{title} (percentile vs {len(pool)} players)", height=90 + 48 * len(columns))
    fig.update_xaxes(range=[0, 112], tickvals=[0, 25, 50, 75, 100], title="Percentile within pool")
    fig.update_yaxes(categoryorder="array", categoryarray=list(table["label"]), autorange="reversed",
                     showgrid=False)
    fig.update_layout(barmode="overlay", bargap=0.35)
    return fig


# ── 2. Metric distribution ───────────────────────────────────────────────────
def jitter(n: int, spread: float = 0.6) -> np.ndarray:
    """Deterministic vertical offsets in (-spread/2, spread/2): golden-ratio steps, so
    consecutive points land far apart and the same input always draws the same way."""
    return ((np.arange(n) * 0.6180339887) % 1 - 0.5) * spread


def group_order(groups: pd.Series) -> list:
    present = list(pd.unique(groups.dropna()))
    known = [g for g in POSITION_ORDER if g in present]
    return known + sorted((g for g in present if g not in known), key=str)


def metric_distribution(pool: pd.DataFrame, column: str, label: str, highlight: list[str],
                        by: str = "broad_position") -> go.Figure:
    """Every player's value of one metric as a strip plot, one row per group.

    pool:      one row per player with column, by, and optionally name / real_team.
    column:    the metric to plot along x; label is its axis title (include the unit).
    highlight: players to call out (matched against name, uid or the index), drawn as
               orange diamonds with direct name labels.
    by:        the grouping column (rows); missing -> one row "All players".

    Points are jittered deterministically within each row; a short vertical bar marks
    each group's median. Hover shows name, club and the exact value.
    """
    title = f"{label}: where every player sits"
    if is_empty(pool, column):
        return empty_figure(title)
    data = pool.assign(_value=pd.to_numeric(pool[column], errors="coerce")).dropna(subset=["_value"])
    data["_group"] = data[by].fillna("Unknown") if by in data.columns else "All players"
    data["_name"] = data["name"] if "name" in data.columns else data.index.astype(str)
    data["_club"] = data["real_team"].fillna("") if "real_team" in data.columns else ""
    groups = group_order(data["_group"])
    position = {g: k for k, g in enumerate(groups)}
    data = data.sort_values("_value")
    data["_y"] = data["_group"].map(position).astype(float)
    data["_y"] += data.groupby("_group")["_value"].transform(lambda s: jitter(len(s)))

    wanted = set(map(str, highlight or []))
    keys = [data["_name"].astype(str), data.index.astype(str).to_series(index=data.index)]
    if "uid" in data.columns:
        keys.append(data["uid"].astype(str))
    is_hot = np.logical_or.reduce([k.isin(wanted).to_numpy() for k in keys])
    hover = (f"<b>%{{customdata[0]}}</b> (%{{customdata[1]}})<br>{label}: "
             f"%{{x:.3~f}} {UNITS.get(column, '')}<extra></extra>")

    fig = go.Figure()
    rest, hot = data[~is_hot], data[is_hot]
    fig.add_trace(go.Scatter(
        x=rest["_value"], y=rest["_y"], mode="markers", name="All players",
        marker=dict(size=8, color=PALETTE[0], opacity=0.55, line=RING),
        customdata=np.column_stack([rest["_name"], rest["_club"]]) if len(rest) else None,
        hovertemplate=hover))
    medians = data.groupby("_group")["_value"].median()
    xs, ys = [], []
    for group, median in medians.items():
        xs += [median, median, None]
        ys += [position[group] - 0.38, position[group] + 0.38, None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Group median",
                             line=dict(color=GUIDE, width=2), hoverinfo="skip"))
    # Invisible points at the medians carry their hover (lines with gaps don't hover well).
    fig.add_trace(go.Scatter(
        x=medians.to_numpy(), y=[position[g] for g in medians.index], mode="markers",
        marker=dict(size=14, opacity=0), showlegend=False, customdata=list(medians.index),
        hovertemplate=f"Median %{{customdata}}: %{{x:.3~f}} {UNITS.get(column, '')}<extra></extra>"))
    if len(hot):
        fig.add_trace(go.Scatter(
            x=hot["_value"], y=hot["_y"], mode="markers+text", name="Highlighted",
            marker=dict(size=13, color=PALETTE[1], symbol="diamond", line=RING),
            text=hot["_name"], textposition="top center", textfont_size=11,
            customdata=np.column_stack([hot["_name"], hot["_club"]]), hovertemplate=hover))
    style(fig, f"{title} ({len(data)} players)", height=140 + 110 * len(groups))
    fig.update_xaxes(title=label)
    fig.update_yaxes(tickvals=list(range(len(groups))), ticktext=groups, range=[len(groups) - 0.5, -0.5],
                     showgrid=False)
    return fig


# ── 3. Match timeline ────────────────────────────────────────────────────────
def goal_tag_layout(minutes: list[float], texts: list[str], end: float) -> list[tuple[int, str]]:
    """(stack level, xanchor) for each goal tag so no two tags overlap on the minute axis.

    Each tag's width in minutes is estimated from its character count; tags are placed
    left to right on the lowest level whose last tag ends before this one starts. Tags
    that would run off either end of the axis are anchored inward instead of centred.
    """
    per_minute = PLOT_PX / max(end, 1.0)
    right_edges: list[float] = []         # right edge (minutes) of the last tag on each level
    out = []
    for at, text in zip(minutes, texts):
        width = (len(text) * TAG_CHAR_PX + TAG_PAD_PX) / per_minute
        if at - width / 2 < 0:
            anchor, left, right = "left", at, at + width
        elif at + width / 2 > end:
            anchor, left, right = "right", at - width, at
        else:
            anchor, left, right = "center", at - width / 2, at + width / 2
        level = next((k for k, edge in enumerate(right_edges) if edge < left), len(right_edges))
        if level == len(right_edges):
            right_edges.append(right)
        else:
            right_edges[level] = right
        out.append((level, anchor))
    return out


def match_timeline(moments: pd.DataFrame, goals: pd.DataFrame, home: str, away: str) -> go.Figure:
    """Key off-ball moments along the match clock, one lane per kind, with the goals.

    moments: rows with kind, minute, team ('Home'/'Away'), and optionally label (lane
             name, defaults to the core.moments label of kind), name, detail.
    goals:   rows with minute (or time in s), team ('Home'/'Away') and name (scorer).
    home, away: club names for the legend.

    Markers wear the team colour and symbol; goals are vertical lines in the scorer's
    team colour with a "GOAL name" tag (team-coloured border, neutral text); a dotted
    divider marks half-time at 45'.
    """
    title = f"Off-ball moments: {home} v {away}"
    has_moments = not is_empty(moments, "minute")
    has_goals = not is_empty(goals) and ("minute" in goals.columns or "time" in goals.columns)
    if not has_moments and not has_goals:
        return empty_figure(title)

    fig = go.Figure()
    lanes: list[str] = []
    if has_moments:
        data = moments.dropna(subset=["minute"]).copy()
        kind = data["kind"] if "kind" in data.columns else pd.Series("", index=data.index)
        fallback = kind.map(MOMENT_LANES).fillna(kind)
        data["_lane"] = data["label"].fillna(fallback) if "label" in data.columns else fallback
        rank = {k: n for n, k in enumerate(MOMENT_LANES)}
        data["_rank"] = kind.map(rank).fillna(len(rank))
        lanes = list(data.sort_values("_rank")["_lane"].drop_duplicates())
        data["_team"] = data["team"].map(side_of) if "team" in data.columns else "Home"
        data["_name"] = data["name"].fillna("") if "name" in data.columns else data.get("player", "")
        data["_detail"] = data["detail"].fillna("") if "detail" in data.columns else ""
        data["_clock"] = data["minute"].map(clock)
        for side, club in (("Home", home), ("Away", away)):
            rows = data[data["_team"] == side]
            fig.add_trace(go.Scatter(
                x=rows["minute"], y=rows["_lane"], mode="markers", name=club,
                marker=dict(size=12, color=TEAM_COLOUR[side], symbol=TEAM_SYMBOL[side], line=RING),
                customdata=np.column_stack([rows["_clock"], rows["_name"], rows["_detail"],
                                            [club] * len(rows)]) if len(rows) else None,
                hovertemplate="<b>%{customdata[1]}</b> (%{customdata[3]}) · %{customdata[0]}<br>"
                              "%{y}<br>%{customdata[2]}<extra></extra>"))

    end, top_level = 95.0, 0
    if has_moments:
        end = max(end, float(data["minute"].max()) + 1)
    if has_goals:
        minutes = goals["minute"] if "minute" in goals.columns else goals["time"] / 60
        tagged = goals.assign(at=pd.to_numeric(minutes, errors="coerce")).dropna(subset=["at"]).sort_values("at")
        if len(tagged):
            end = max(end, float(tagged["at"].max()) + 1)
        tags = []
        for row in tagged.itertuples():
            name = getattr(row, "name", "")
            name = "" if pd.isna(name) else str(name).strip()
            text = f"GOAL {name} {clock(row.at)}" if name else f"GOAL {clock(row.at)}"
            tags.append((row, side_of(getattr(row, "team", "Home")), text))
        layout = goal_tag_layout([row.at for row, _, _ in tags], [t for _, _, t in tags], end)
        for (row, side, text), (level, anchor) in zip(tags, layout):
            top_level = max(top_level, level)
            fig.add_vline(x=row.at, line=dict(color=TEAM_COLOUR[side], width=2))
            fig.add_annotation(
                x=row.at, y=1.0, xref="x", yref="paper", yanchor="bottom", xanchor=anchor,
                yshift=2 + 20 * level, text=text,
                showarrow=False, font_size=11, bordercolor=TEAM_COLOUR[side], borderwidth=1.5,
                borderpad=2, hovertext=f"{home if side == 'Home' else away} goal at {clock(row.at)}")

    fig.add_vline(x=45, line=dict(color=GUIDE, width=1, dash="dot"),
                  annotation=dict(text="Half-time", font_size=10), annotation_position="bottom right")
    style(fig, title, height=220 + 52 * max(len(lanes), 1), legend_below=True)
    # Room above the plot for the title plus however many rows of goal tags were stacked.
    fig.update_layout(margin=dict(t=90 + 20 * (top_level + 1)), legend=dict(y=-0.2))
    fig.update_xaxes(range=[0, end], tickvals=list(range(0, int(end) + 1, 15)), ticksuffix="'",
                     title="Match minute")
    if lanes:
        fig.update_yaxes(categoryorder="array", categoryarray=lanes, autorange="reversed")
    else:
        fig.update_yaxes(visible=False)
    return fig


# ── 4. Team comparison ───────────────────────────────────────────────────────
def team_comparison(teams: pd.DataFrame) -> go.Figure:
    """Each off-ball metric split between the two clubs as a diverging 100% bar.

    teams: index = metric labels (top to bottom), exactly two columns = club names,
           home first. Each row is normalised to the pair's total, so the home share
           runs left of centre and the away share right; bars are labelled with the
           raw values and the hover gives raw value and share.
    """
    title = "Who won the off-ball battle"
    if teams is None or teams.shape[1] < 2 or len(teams) == 0:
        return empty_figure(title)
    home, away = teams.columns[0], teams.columns[1]
    raw = teams[[home, away]].apply(pd.to_numeric, errors="coerce")
    total = raw[home].abs() + raw[away].abs()
    home_share = (raw[home].abs() / total.where(total > 0) * 100)
    if home_share.isna().all():
        return empty_figure(title)
    away_share = 100 - home_share
    metrics = [str(m) for m in teams.index]

    fig = go.Figure()
    for side, club, share, sign in (("Home", home, home_share, -1), ("Away", away, away_share, 1)):
        fig.add_trace(go.Bar(
            y=metrics, x=sign * share, orientation="h", name=str(club),
            marker=dict(color=TEAM_COLOUR[side], cornerradius=4),
            text=raw[club].map(fmt), textposition="outside", cliponaxis=False,
            customdata=np.column_stack([raw[club].map(fmt), share.round(1).map(fmt)]),
            hovertemplate=f"<b>%{{y}}</b><br>{club}: %{{customdata[0]}} "
                          f"(%{{customdata[1]}}% of the pair)<extra></extra>"))
    # Direct labels for the two sides above the bars.
    for club, x, anchor in ((home, -100, "left"), (away, 100, "right")):
        fig.add_annotation(x=x, y=1.0, xref="x", yref="paper", yanchor="bottom", xanchor=anchor,
                           text=f"<b>{club}</b>", showarrow=False, font_size=12)
    fig.add_vline(x=0, line=dict(color=GUIDE, width=1))
    style(fig, title, height=160 + 44 * len(metrics), legend_below=True)
    fig.update_layout(barmode="relative", bargap=0.35, margin=dict(t=80),
                      legend=dict(y=-0.2, x=1, xanchor="right"))     # clear of the centred axis title
    fig.update_xaxes(range=[-118, 118], tickvals=[-100, -50, 0, 50, 100],
                     ticktext=["100%", "50%", "0", "50%", "100%"], title="Share of the pair's total")
    fig.update_yaxes(categoryorder="array", categoryarray=metrics, autorange="reversed", showgrid=False)
    return fig


# ── 5. Intensity timeline ────────────────────────────────────────────────────
def intensity_timeline(per_block: pd.DataFrame, metric: str, label: str) -> go.Figure:
    """One metric per club through the match in time blocks (e.g. 0-15', 15-30', ...).

    per_block: rows with block_start (minute), club, value, and optionally side
               ('Home'/'Away', sets the team colour; otherwise the first club seen is
               Home) and metric (rows are filtered to metric when that column exists).
    metric:    the metric name (also the hover unit lookup); label is the y-axis title.
    """
    title = f"{label} through the match"
    data = per_block
    if data is not None and "metric" in data.columns:
        data = data[data["metric"] == metric]
    if is_empty(data, "block_start", "value") or "club" not in data.columns:
        return empty_figure(title)
    data = data.dropna(subset=["club"])
    if data.empty:
        return empty_figure(title)
    data = data.assign(value=pd.to_numeric(data["value"], errors="coerce")).sort_values("block_start")
    starts = np.sort(data["block_start"].dropna().unique())
    width = float(np.min(np.diff(starts))) if len(starts) > 1 else 15.0
    ticks = {s: f"{s:g}-{s + width:g}'" for s in starts}
    clubs = list(pd.unique(data["club"]))
    if "side" in data.columns:
        sides = data.groupby("club")["side"].first().map(side_of).to_dict()
        clubs.sort(key=lambda c: sides.get(c, "Home") != "Home")
    else:
        sides = {club: ("Home", "Away")[min(k, 1)] for k, club in enumerate(clubs)}

    fig = go.Figure()
    unit = UNITS.get(metric, "")
    for k, club in enumerate(clubs[:2]):
        rows = data[data["club"] == club]
        side = sides.get(club, "Home")
        fig.add_trace(go.Scatter(
            x=rows["block_start"] + width / 2, y=rows["value"], mode="lines+markers", name=str(club),
            line=dict(color=TEAM_COLOUR[side], width=2),
            marker=dict(size=9, color=TEAM_COLOUR[side], symbol=TEAM_SYMBOL[side], line=RING),
            customdata=rows["block_start"].map(ticks),
            hovertemplate=f"<b>{club}</b> · %{{customdata}}<br>{label}: %{{y:.3~f}} {unit}<extra></extra>"))
        tail = rows.dropna(subset=["value"]).tail(1)
        if len(tail):
            fig.add_annotation(x=float(tail["block_start"].iloc[0] + width / 2), y=float(tail["value"].iloc[0]),
                               text=str(club), showarrow=False, xanchor="left", xshift=10, font_size=11)
    if starts.min() < 45 < starts.max() + width:
        fig.add_vline(x=45, line=dict(color=GUIDE, width=1, dash="dot"))
    style(fig, title, height=380)
    fig.update_layout(hovermode="x unified", margin=dict(r=90))
    fig.update_xaxes(tickvals=list(starts + width / 2), ticktext=list(ticks.values()),
                     range=[starts.min(), starts.max() + width], title="Match period", showgrid=False)
    fig.update_yaxes(title=label, rangemode="tozero")
    return fig


# ── 6. Scanning effect ───────────────────────────────────────────────────────
def scanning_effect(effect: pd.DataFrame) -> go.Figure:
    """Ball retention by how many head checks the receiver made before receiving.

    effect: one row per bucket in display order with bucket (e.g. "0", "1", "2", "3+"),
            retention (0-1), low and high (95% CI bounds, 0-1) and n (receptions).
    Bars show retention %, whiskers the confidence interval, the label n; a dotted
    line marks the n-weighted retention over all receptions.
    """
    title = "Players who scan keep the ball"
    if is_empty(effect, "retention"):
        return empty_figure(title)
    data = effect.dropna(subset=["retention"]).copy()
    data["bucket"] = data["bucket"].astype(str)
    pct = data["retention"] * 100
    low = (data["low"] * 100) if "low" in data.columns else pct
    high = (data["high"] * 100) if "high" in data.columns else pct
    n = data["n"] if "n" in data.columns else pd.Series(np.nan, index=data.index)

    fig = go.Figure(go.Bar(
        x=data["bucket"], y=pct, name="Retention", marker=dict(color=PALETTE[0], cornerradius=4),
        error_y=dict(type="data", array=(high - pct).clip(lower=0), arrayminus=(pct - low).clip(lower=0),
                     color=GUIDE, thickness=1.5, width=6),
        text=[f"n={int(v)}" if pd.notna(v) else "" for v in n], textposition="inside",
        insidetextanchor="start", textfont=dict(color="white", size=11),
        customdata=np.column_stack([low.round(1), high.round(1), n.fillna(0).astype(int)]),
        hovertemplate="<b>%{x} head checks</b> in the 5 s before receiving<br>"
                      "Kept the ball: %{y:.1f}%<br>95% CI: %{customdata[0]:.1f}-%{customdata[1]:.1f}%<br>"
                      "%{customdata[2]} receptions<extra></extra>"))
    if n.notna().any() and n.sum() > 0:
        overall = float((pct * n).sum() / n.sum())
        fig.add_hline(y=overall, line=dict(color=GUIDE, width=1, dash="dot"),
                      annotation=dict(text=f"All receptions {overall:.0f}%", font_size=11),
                      annotation_position="top left")
    style(fig, title, height=380)
    fig.update_layout(showlegend=False, bargap=0.35)
    fig.update_xaxes(title="Head checks in the 5 s before receiving", type="category")
    fig.update_yaxes(title="Ball retained (%)", range=[0, 100], ticksuffix="%")
    return fig


# ── 7. Validation scatter ────────────────────────────────────────────────────
def spearman(x: pd.Series, y: pd.Series) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = spearmanr(x, y).statistic
    return float(rho) if np.isfinite(rho) else np.nan


def validation_scatter(df: pd.DataFrame, pairs: list[tuple[str, str, str]]) -> go.Figure:
    """Small multiples: each measured metric against the hidden attribute it should track.

    df:    one row per player with the hidden ground-truth columns and the metric
           columns (optionally name for the hover).
    pairs: (hidden_col, metric_col, title) per panel; each panel gets a least-squares
           trend line and a subtitle with Spearman's rho and n.
    """
    title = "Do the metrics find the hidden talent?"
    pairs = [p for p in (pairs or []) if df is not None and p[0] in df.columns and p[1] in df.columns]
    if is_empty(df) or not pairs:
        return empty_figure(title)
    cols = min(len(pairs), 3)
    rows = int(np.ceil(len(pairs) / cols))
    stats, subsets = [], []
    for hidden, metric, _ in pairs:
        sub = df.assign(_x=pd.to_numeric(df[hidden], errors="coerce"),
                        _y=pd.to_numeric(df[metric], errors="coerce")).dropna(subset=["_x", "_y"])
        subsets.append(sub)
        stats.append((spearman(sub["_x"], sub["_y"]) if len(sub) >= 3 else np.nan, len(sub)))
    titles = [f"{t}<br><sup>Spearman ρ = {fmt(rho)} · n = {n}</sup>" for (_, _, t), (rho, n) in zip(pairs, stats)]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, horizontal_spacing=0.08,
                        vertical_spacing=0.2 if rows > 1 else 0.1)
    for k, ((hidden, metric, _), sub) in enumerate(zip(pairs, subsets)):
        r, c = k // cols + 1, k % cols + 1
        names = sub["name"] if "name" in sub.columns else sub.index.astype(str)
        fig.add_trace(go.Scatter(
            x=sub["_x"], y=sub["_y"], mode="markers", name="Player", legendgroup="player", showlegend=k == 0,
            marker=dict(size=7, color=PALETTE[0], opacity=0.6, line=RING),
            customdata=list(names),
            hovertemplate=(f"<b>%{{customdata}}</b><br>{hidden}: %{{x:.3~f}}<br>{metric}: %{{y:.3~f}} "
                           f"{UNITS.get(metric, '')}<extra></extra>")), row=r, col=c)
        if len(sub) >= 2 and sub["_x"].nunique() > 1:
            slope, intercept = np.polyfit(sub["_x"], sub["_y"], 1)
            xs = np.array([sub["_x"].min(), sub["_x"].max()])
            fig.add_trace(go.Scatter(
                x=xs, y=slope * xs + intercept, mode="lines", name="Least-squares fit", legendgroup="fit",
                showlegend=k == 0, line=dict(color=GUIDE, width=2),
                hovertemplate=f"Fit: {metric} = {slope:.3g} × {hidden} + {intercept:.3g}<extra></extra>"),
                row=r, col=c)
        fig.update_xaxes(title_text=f"Hidden {hidden.replace('_', ' ')}", row=r, col=c)
        fig.update_yaxes(title_text=metric.replace("_", " "), row=r, col=c)
    style(fig, title, height=160 + 330 * rows, legend_below=True)
    fig.update_layout(margin=dict(t=110), legend=dict(y=-0.15 / rows))
    return fig


# ── 8. Value ladder ──────────────────────────────────────────────────────────
def value_ladder(pool: pd.DataFrame, top: int = 15) -> go.Figure:
    """Dumbbells from market value to ML price for the most underpriced players.

    pool: one row per player with market_value_m, Predicted_Value (both €m) and
          optionally name / real_team. The top players by Predicted_Value /
          market_value_m are drawn, the biggest gap first; each line joins the two
          prices and the ML marker is labelled with the multiple.
    """
    title = "Underpriced for their off-ball work: market value → ML price"
    if is_empty(pool, "market_value_m", "Predicted_Value"):
        return empty_figure(title)
    data = pool.assign(market=pd.to_numeric(pool["market_value_m"], errors="coerce"),
                       ml=pd.to_numeric(pool["Predicted_Value"], errors="coerce"))
    data = data[(data["market"] > 0) & data["ml"].notna()]
    if data.empty:
        return empty_figure(title)
    data = data.assign(ratio=data["ml"] / data["market"]).sort_values("ratio", ascending=False).head(top)
    names = (data["name"] if "name" in data.columns else data.index.to_series()).astype(str)
    clubs = data["real_team"].fillna("").astype(str) if "real_team" in data.columns else pd.Series("", index=data.index)
    labels = names.where(~names.duplicated(keep=False), names + " (" + clubs + ")")
    data = data.assign(_label=labels.to_numpy(), _club=clubs.to_numpy())
    custom = np.column_stack([data["_club"], data["market"].round(1), data["ml"].round(1), data["ratio"].round(2)])
    hover = ("<b>%{y}</b> (%{customdata[0]})<br>Market value €%{customdata[1]}m · "
             "ML price €%{customdata[2]}m<br>ML price is %{customdata[3]}× market value<extra></extra>")

    fig = go.Figure()
    xs, ys = [], []
    for market, ml, label in zip(data["market"], data["ml"], data["_label"]):
        xs += [market, ml, None]
        ys += [label, label, None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=GUIDE, width=2),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=data["market"], y=data["_label"], mode="markers", name="Market value",
        marker=dict(size=11, color=PALETTE[0], symbol="circle", line=RING), customdata=custom,
        hovertemplate=hover))
    fig.add_trace(go.Scatter(
        x=data["ml"], y=data["_label"], mode="markers+text", name="ML price (off-ball work)",
        marker=dict(size=12, color=PALETTE[1], symbol="diamond", line=RING),
        text=[f"×{r:.1f}" for r in data["ratio"]], textposition="middle right", textfont_size=11,
        cliponaxis=False,
        customdata=custom, hovertemplate=hover))
    style(fig, title, height=120 + 30 * len(data))
    fig.update_xaxes(title="€m (synthetic)", range=[0, data[["market", "ml"]].max().max() * 1.15])
    fig.update_yaxes(categoryorder="array", categoryarray=list(data["_label"]), autorange="reversed",
                     showgrid=False)
    return fig


# ── 9. Match-by-match trend ──────────────────────────────────────────────────
def match_trend(matches: pd.DataFrame, column: str, label: str, benchmark: float | None = None) -> go.Figure:
    """One player's value of a metric in each match they played.

    matches:   rows with matchday, column and optionally opponent / minutes (the
               output of core.analysis.player_profile's "matches").
    benchmark: an optional reference value (e.g. the position median), drawn dotted.
    """
    title = f"{label}: match by match"
    if is_empty(matches, "matchday", column):
        return empty_figure(title)
    data = matches.assign(_value=pd.to_numeric(matches[column], errors="coerce")).sort_values("matchday")
    opponent = data["opponent"].fillna("") if "opponent" in data.columns else pd.Series("", index=data.index)
    played = data["minutes"].round(0) if "minutes" in data.columns else pd.Series(np.nan, index=data.index)
    unit = UNITS.get(column, "")
    ticks = [f"MD{int(d)}<br>v {o}" if o else f"MD{int(d)}" for d, o in zip(data["matchday"], opponent)]
    fig = go.Figure(go.Scatter(
        x=ticks, y=data["_value"], mode="lines+markers", name=label, connectgaps=False,
        line=dict(color=PALETTE[0], width=2), marker=dict(size=9, color=PALETTE[0], line=RING),
        customdata=np.column_stack([opponent, played.fillna(0).astype(int)]),
        hovertemplate=(f"<b>%{{x}}</b><br>{label}: %{{y:.3~f}} {unit}<br>"
                       "%{customdata[1]} min<extra></extra>")))
    if benchmark is not None and np.isfinite(benchmark):
        fig.add_hline(y=benchmark, line=dict(color=GUIDE, width=1, dash="dot"),
                      annotation=dict(text=f"Position median {fmt(benchmark)}", font_size=11),
                      annotation_position="top left")
    style(fig, title, height=320)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="Match", type="category", showgrid=False)
    fig.update_yaxes(title=label, rangemode="tozero")
    return fig


# ── 10. Profiles by role ─────────────────────────────────────────────────────
def wrap(label: str, width: int = 12) -> str:
    """A label broken onto lines of about `width` characters with <br>, at spaces."""
    lines = []
    for word in str(label).split():
        if lines and len(lines[-1]) + 1 + len(word) <= width:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return "<br>".join(lines)


def role_heatmap(medians: pd.DataFrame, lower_is_better: set[str] | None = None) -> go.Figure:
    """Median of each metric per position, shaded within each metric's column.

    medians: index = positions (top to bottom), columns = metric labels, values = medians.
    Shade is the cell's share of its column's best value (one blue ramp, light -> dark), so
    every column reads "who leads this metric"; for columns in `lower_is_better` the
    lowest value is darkest. Cells carry the raw median; hover adds the position.
    """
    title = "Do the metrics find the roles? Median by position"
    if medians is None or medians.empty or medians.isna().all().all():
        return empty_figure(title)
    lower = lower_is_better or set()
    shade = medians.copy().astype(float)
    for column in shade.columns:
        values = shade[column]
        span = values.max() - values.min()
        scaled = (values - values.min()) / span if span > 0 else values * 0 + 0.5
        shade[column] = 1 - scaled if column in lower else scaled
    fig = go.Figure(go.Heatmap(
        z=shade.to_numpy(), x=[wrap(c) for c in medians.columns], y=list(medians.index),
        colorscale=[[0, "#eaf2fb"], [1, "#1c5aa6"]], zmin=0, zmax=1, showscale=False, xgap=2, ygap=2,
        text=medians.map(fmt).to_numpy(), texttemplate="%{text}", textfont_size=11,
        customdata=medians.to_numpy(),
        hovertemplate="<b>%{y}</b> · %{x}<br>Median %{customdata:.3~f}<extra></extra>"))
    style(fig, title, height=140 + 34 * len(medians))
    fig.update_xaxes(side="top", showgrid=False, tickangle=0)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(margin=dict(t=135))
    return fig


def attribute_polygon(records: list[dict], title: str = "Attribute polygon") -> go.Figure:
    """Radar of core.composite.polygon records: one closed polygon per player.

    Axes are the categories in record order, 0-100 with a dashed ring at 50 (the
    position's middle). Up to four players take the palette in order; a category a
    player has no data for is drawn at 50 (no information) and says so in the hover.
    """
    records = [r for r in records if r.get("axes")][:4]
    if not records:
        return empty_figure(title)
    labels = [axis["label"] for axis in records[0]["axes"]]
    fig = go.Figure(go.Scatterpolar(r=[50] * (len(labels) + 1), theta=labels + labels[:1], mode="lines",
                                    line=dict(color=GUIDE, width=1, dash="dash"), name="Position middle (50)",
                                    hoverinfo="skip"))
    for colour, record in zip(PALETTE, records):
        scores = [axis["score"] for axis in record["axes"]]
        shown = [50 if s is None else s for s in scores]
        notes = ["no data" if s is None else f"{s:.0f}" for s in scores]
        index = "–" if record.get("index") is None else f"{record['index']:.0f}"
        fig.add_trace(go.Scatterpolar(
            r=shown + shown[:1], theta=labels + labels[:1], mode="lines+markers", fill="toself",
            name=f"{record['name']} · {index}", line=dict(color=colour, width=2),
            marker=dict(size=8, line=RING), customdata=notes + notes[:1],
            fillcolor="rgba({}, {}, {}, 0.15)".format(*(int(colour[k:k + 2], 16) for k in (1, 3, 5))),
            hovertemplate=f"<b>{record['name']}</b><br>%{{theta}}: %{{customdata}}<extra></extra>"))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75, 100], gridcolor=GRID,
                                                 linecolor=GRID, tickfont_size=10),
                                 angularaxis=dict(gridcolor=GRID, linecolor=GRID)))
    return style(fig, title, height=420, legend_below=True)


def impact_weights(weights: pd.DataFrame, labels: dict[str, str]) -> go.Figure:
    """Each metric's fitted weight (shots per SD of team work) with its bootstrap 95% interval.

    weights: core.impact.fit_weights output (index = metric); labels: metric -> label.
    """
    title = "What each metric is worth in shots"
    if is_empty(weights, "weight"):
        return empty_figure(title)
    table = weights.iloc[::-1]                      # first metric on top
    names = [labels.get(m, m) for m in table.index]
    fig = go.Figure(go.Bar(
        y=names, x=table["weight"], orientation="h", marker=dict(color=PALETTE[0], cornerradius=4),
        error_x=dict(type="data", symmetric=False, array=table["high"] - table["weight"],
                     arrayminus=table["weight"] - table["low"], color=GUIDE, thickness=1.5),
        text=table["weight"].map(lambda v: f"{v:.2f}"), textposition="outside", cliponaxis=False,
        customdata=np.column_stack([table["low"], table["high"], table["positive_share"] * 100]),
        hovertemplate="<b>%{y}</b><br>Weight: %{x:.3f} shots per SD<br>95% interval: %{customdata[0]:.3f}"
                      " to %{customdata[1]:.3f}<br>Above 0 in %{customdata[2]:.0f}% of resamples<extra></extra>"))
    fig.update_xaxes(title="Shot difference per standard deviation of team work (0 = no detectable effect)",
                     rangemode="tozero")
    return style(fig, title, height=320)
