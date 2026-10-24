"""
AI Scout: a language-model agent that answers scouting questions from the off-ball metrics.

    OffBallAgent(global_df).query("Which forward creates the most space with decoy runs?")

How it stays grounded:
    1. The whole player table (Off-Ball Index, Pillar score, the four pillar metrics: decoy runs,
       pressing lane cuts, transition reaction and scanning, plus ML role clusters
       and prices) goes into the context window as CSV, with a plain
       definition of every column, including which direction is "better".
    2. Exact rankings (top five per metric, overall and per position) are computed here
       in pandas and included too, so the model reads answers off a sorted list instead
       of sorting fifty rows itself, which language models do unreliably.
    3. The system prompt tells it to answer only from that data and to quote numbers.

It never breaks a demo: with no credentials configured, or if the API call fails, it
answers offline from the same table with a rule-based ranking, and says so.

Credentials come from environment variables, falling back to the project's `.env`:
    Microsoft Foundry / Azure OpenAI   AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
                                       AZURE_OPENAI_DEPLOYMENT (e.g. gpt-4.1-mini)
    GitHub Models                      GITHUB_TOKEN (model: GITHUB_MODEL, default openai/gpt-4.1-mini)
    OpenAI                             OPENAI_API_KEY (model: OPENAI_MODEL, default gpt-4.1-mini)
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
GITHUB_MODELS_URL = "https://models.github.ai/inference"

# Table column -> (label shown to the model, definition). Order is the order in the context.
COLUMNS: dict[str, tuple[str, str]] = {
    "name": ("player", "fictional player name"),
    "real_team": ("club", "the player's club in the synthetic league"),
    "broad_position": ("position", "Defender, Midfielder or Forward (goalkeepers are excluded)"),
    "minutes": ("minutes", "minutes played across all matches"),
    "off_ball_value": ("pillar_score", "0-100 percentile score averaging the four pillars "
                                         "(decoy runs, pressing, transitions, scanning); higher is "
                                         "better"),
    "index": ("off_ball_index", "0-100 Composite Off-Ball Index: weighted mean of spacing, movement, "
                                "creation, scanning and defence subscores, each a percentile within "
                                "position (50 = middle); blank = under 90 minutes"),
    "impact_index": ("off_ball_impact", "0-100 Off-Ball Impact Index: the five tactical metrics weighted by "
                                        "how much each moves the team's shot balance, percentiles within "
                                        "position (50 = middle); blank = under 90 minutes"),
    "decoy_p90": ("decoy_run_value_p90", "expected threat (xT) of space opened for teammates by "
                                         "dragging markers with off-ball runs, per 90 minutes; higher = "
                                         "creates more space"),
    "decoy_runs": ("off_ball_runs", "number of sustained off-ball runs made"),
    "lane_cut_p90": ("pressing_lane_cuts_p90", "seconds of threat-weighted passing lanes cut while "
                                               "pressing near the ball, per 90; higher = better presser"),
    "mean_lane_cut": ("avg_lane_cut", "average share of passing lanes cut per pressing moment (0-1)"),
    "median_latency": ("transition_latency_s", "median seconds from a turnover to a 20+ km/h sprint, over "
                                               "turnovers where they sprinted; LOWER is better; blank = "
                                               "never sprinted"),
    "sprint_rate": ("transition_sprint_rate", "share of nearby turnovers where they sprinted (0-1); higher "
                                              "= reacts more often"),
    "transition_score": ("transition_score", "transition reaction score; higher is better"),
    "pitch_control_pct": ("pitch_control_pct", "share of the pitch the player controls while their team "
                                               "has the ball, in percent"),
    "scan_rate": ("scan_rate", "scans per second in the 5 s before receiving; higher = scans more "
                               "(checks surroundings before the ball arrives)"),
    "pre_reception_scans": ("scans_before_receiving", "average number of head checks in the 5 s before "
                                                      "receiving a pass"),
    "receptions": ("receptions", "completed passes received (the sample behind the scanning numbers)"),
    "retention_rate": ("retention_rate", "share of receptions where the player's next action kept the "
                                         "ball (completed pass or shot), 0-1; higher is better"),
    "pressure_intensity": ("pressure_intensity_p90", "metres closed down on the ball carrier per 90, "
                                                     "weighted by how close (within 10 m); higher = "
                                                     "presses more aggressively"),
    "receipts_in_space_pct": ("receipts_in_space", "share of receptions with the nearest opponent 5+ m "
                                                   "away (0-1); higher = finds space before the ball arrives"),
    "def_responsibility_p90": ("xt_conceded_in_zone_p90", "expected threat of opponent passes into the zone "
                                                          "the player's position in the shape covers, per 90; "
                                                          "LOWER is better"),
    "obso_p90": ("obso_p90", "Off-Ball Scoring Opportunity: xT-seconds of threat offered by free, "
                             "reachable positions per 90, whether or not the ball came; higher is better"),
    "runs_in_behind_p90": ("runs_in_behind_p90", "off-ball runs past the opponent's last defender per 90"),
    "pulling_wide_p90": ("pulling_wide_p90", "off-ball runs towards the touchline per 90"),
    "dropping_deep_p90": ("dropping_deep_p90", "off-ball runs back towards the ball per 90"),
    "Cluster_Role": ("role", "machine-learned off-ball role (K-means): Space creator, Lane-cutting "
                             "presser, Transition sprinter, Space occupier or Scanner"),
    "market_value_m": ("market_value_eur_m", "market value in € millions"),
    "Predicted_Value": ("ml_price_eur_m", "what a gradient-boosted model prices the player at from "
                                          "off-ball work and position (out-of-fold, € millions); "
                                          "ml_price above market value = cheap for their off-ball work"),
}

# (column, best-first sort, label) for the precomputed rankings and the offline answers.
RANKINGS: list[tuple[str, bool, str]] = [
    ("index", False, "Off-Ball Index"),
    ("off_ball_value", False, "Pillar score"),
    ("decoy_p90", False, "Decoy Run Value per 90 (space created)"),
    ("lane_cut_p90", False, "Pressing Lane Cuts per 90"),
    ("median_latency", True, "Transition Latency in seconds (fastest first)"),
    ("pitch_control_pct", False, "Pitch Control %"),
    ("scan_rate", False, "Scanning rate (scans per second before receiving)"),
    ("retention_rate", False, "Ball retention after receiving"),
    ("pressure_intensity", False, "Pressure Intensity per 90"),
    ("receipts_in_space_pct", False, "Share of receptions in space"),
    ("def_responsibility_p90", True, "Defensive Responsibility: xT conceded in zone per 90 (lowest first)"),
    ("obso_p90", False, "Off-Ball Scoring Opportunity (OBSO) per 90"),
    ("runs_in_behind_p90", False, "Runs in behind per 90"),
    ("pulling_wide_p90", False, "Runs pulling wide per 90"),
    ("dropping_deep_p90", False, "Runs dropping deep per 90"),
    ("value_gap", False, "ML price relative to market value (cheapest for their off-ball work)"),
]

SYSTEM_PROMPT = """You are AI Scout, an analyst for a football club's recruitment team.
You answer questions about players' OFF-BALL contribution using only the data provided below.
The headline score is the Off-Ball Index (nine metrics, percentiles within position). The Pillar
score averages the four pillars: decoy runs (space created for teammates), pressing lane cuts,
transition reaction (sprinting after a turnover) and scanning (head checks before receiving).
Five tactical metrics add detail: pressure intensity, receipts in space, defensive responsibility,
off-ball scoring opportunity (OBSO) and run types (in behind, pulling wide, dropping deep).

Rules:
- Use only the PLAYER TABLE and RANKINGS. If the data can't answer a question, say so plainly.
- Quote the exact numbers you rely on (with units) and name the metric, e.g. "Decoy Run Value 0.21 per 90".
- Prefer the precomputed RANKINGS for "who is best/most/fastest" questions; they are exact.
- Transition latency: lower is better. All other metrics: higher is better unless stated.
- Keep answers short: a direct answer first, then up to three supporting bullet points.
- Every club and player is fictional (a synthetic, football-realistic league). Don't present them as real.
- Metrics measure off-ball work only; say so if asked about finishing, passing or overall quality."""


# ── Configuration ────────────────────────────────────────────────────────────
def _read_env_file(path: Path) -> dict[str, str]:
    """KEY=VALUE lines from a .env file (no quotes or exports needed); missing file -> {}."""
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        match = re.match(r"^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$", line)
        if match and match.group(2) and "<your-resource>" not in match.group(2):
            values[match.group(1)] = match.group(2).strip("'\"")
    return values


@dataclass(frozen=True)
class ModelConfig:
    """Which API the agent calls. Build with `from_env()`; None means "answer offline"."""

    provider: str          # "Microsoft Foundry", "GitHub Models" or "OpenAI"
    model: str             # deployment or model name
    base_url: str | None
    api_key: str

    @classmethod
    def from_env(cls, env_file: Path = ENV_FILE) -> ModelConfig | None:
        file_values = _read_env_file(env_file)
        get = lambda key, default="": os.getenv(key) or file_values.get(key) or default

        if get("AZURE_OPENAI_ENDPOINT") and get("AZURE_OPENAI_API_KEY"):
            # A Foundry "v1" endpoint may be copied with its operation on the end
            # (.../openai/v1/responses); the client wants the base, ending in /openai/v1/.
            endpoint = re.sub(r"/(responses|chat/completions)/?$", "", get("AZURE_OPENAI_ENDPOINT").strip())
            endpoint = endpoint.rstrip("/")
            if not re.search(r"/openai/v1$", endpoint):
                endpoint = endpoint + "/openai/v1"
            return cls("Microsoft Foundry", get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini"), endpoint + "/",
                       get("AZURE_OPENAI_API_KEY"))
        if get("GITHUB_TOKEN"):
            return cls("GitHub Models", get("GITHUB_MODEL", "openai/gpt-4.1-mini"), GITHUB_MODELS_URL,
                       get("GITHUB_TOKEN"))
        if get("OPENAI_API_KEY"):
            return cls("OpenAI", get("OPENAI_MODEL", "gpt-4.1-mini"), None, get("OPENAI_API_KEY"))
        return None

    def __repr__(self) -> str:      # never show the key in logs or tracebacks
        return f"ModelConfig(provider={self.provider!r}, model={self.model!r}, base_url={self.base_url!r})"


# ── The agent ────────────────────────────────────────────────────────────────
class OffBallAgent:
    """Answers scouting questions about the players in `global_df`.

    global_df: one row per player, e.g. the Value Finder pool (core.pipeline.player_totals
    -> off_ball_value -> core.ml_model). Columns not in COLUMNS are ignored; missing
    ones are simply left out of the context.
    config: which API to call; by default read from the environment / .env.
    client: an OpenAI-compatible client to use instead of building one (handy in tests).
    offline: answer from the rule-based ranking only, without calling any model.
    """

    def __init__(self, global_df: pd.DataFrame, config: ModelConfig | None = None,
                 client: Any | None = None, offline: bool = False, max_history: int = 6) -> None:
        self.df = global_df.copy()
        if {"Predicted_Value", "market_value_m"} <= set(self.df.columns):
            self.df["value_gap"] = self.df["Predicted_Value"] / self.df["market_value_m"]
        if offline:
            self.config, client = None, None
        elif config is None and client is None:
            config = ModelConfig.from_env()
        self.config = config
        self.max_history = max_history
        self._client = client
        self.context = self.build_context()

    # ── Context ──────────────────────────────────────────────────────────────
    def build_context(self) -> str:
        """The data block the model sees: column definitions, the table, exact rankings."""
        present = [c for c in COLUMNS if c in self.df.columns]
        table = self.df[present].rename(columns={c: COLUMNS[c][0] for c in present})
        definitions = "\n".join(f"- {COLUMNS[c][0]}: {COLUMNS[c][1]}" for c in present)
        return (f"COLUMN DEFINITIONS\n{definitions}\n\n"
                f"PLAYER TABLE ({len(table)} players, CSV)\n"
                f"{table.round(3).to_csv(index=False)}\n"
                f"RANKINGS (exact, computed from the table)\n{self.rankings_text()}")

    def ranking(self, column: str, best_first_ascending: bool, position: str | None = None,
                top: int = 5) -> pd.DataFrame:
        """Top players on one metric, optionally within a position. Blank values are skipped."""
        pool = self.df if position is None else self.df[self.df["broad_position"] == position]
        pool = pool.dropna(subset=[column])
        return pool.sort_values(column, ascending=best_first_ascending).head(top)

    def rankings_text(self, top: int = 5) -> str:
        lines = []
        positions = [None] + (sorted(self.df["broad_position"].dropna().unique())
                              if "broad_position" in self.df else [])
        for column, ascending, label in RANKINGS:
            if column not in self.df:
                continue
            if self.df[column].notna().sum() == 0:
                lines.append(f"{label}: (no data yet)")
                continue
            for position in positions:
                rows = self.ranking(column, ascending, position, top)
                if rows.empty:
                    lines.append(f"{label} - {position or 'all positions'}: (no data)")
                    continue
                entries = "; ".join(f"{k}. {r['name']} ({r.get('real_team', '')}) {r[column]:.3g}"
                                    for k, (_, r) in enumerate(rows.iterrows(), 1))
                lines.append(f"{label} - {position or 'all positions'}: {entries}")
        return "\n".join(lines)

    # ── Querying ─────────────────────────────────────────────────────────────
    @property
    def mode(self) -> str:
        """Human-readable description of where answers come from."""
        if self.config is None and self._client is None:
            return "offline (no model configured)"
        return f"{self.config.provider} · {self.config.model}" if self.config else "custom client"

    def messages(self, user_prompt: str, history: Sequence[dict[str, str]] = ()) -> list[dict[str, str]]:
        """System prompt + data, the recent conversation, then the new question."""
        recent = [m for m in history if m.get("role") in ("user", "assistant")][-2 * self.max_history:]
        return ([{"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{self.context}"}]
                + [{"role": m["role"], "content": m["content"]} for m in recent]
                + [{"role": "user", "content": user_prompt}])

    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI          # imported lazily: offline mode needs no SDK
            self._client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
        return self._client

    def stream(self, user_prompt: str, history: Sequence[dict[str, str]] = ()) -> Iterator[str]:
        """Yield the answer in pieces as the model writes it (for st.write_stream)."""
        if self.config is None and self._client is None:
            yield self.offline_answer(user_prompt)
            return
        try:
            response = self.client().chat.completions.create(
                model=self.config.model if self.config else "model",
                messages=self.messages(user_prompt, history),
                temperature=0.2, max_tokens=700, stream=True)
            produced = False
            for chunk in response:
                # Azure can send chunks with no choices (e.g. content-filter results).
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    produced = True
                    yield chunk.choices[0].delta.content
            if not produced:
                yield self.offline_answer(user_prompt, note="The model returned an empty answer.")
        except Exception as error:             # network, auth, quota, content filter...
            yield self.offline_answer(user_prompt, note=f"The model call failed ({type(error).__name__}).")

    def query(self, user_prompt: str, history: Sequence[dict[str, str]] = ()) -> str:
        """Answer one scouting question. `history` holds earlier {"role", "content"} turns."""
        return "".join(self.stream(user_prompt, history))

    # ── Offline fallback ─────────────────────────────────────────────────────
    POSITION_WORDS = {"forward": "Forward", "striker": "Forward", "winger": "Forward", "attacker": "Forward",
                      "midfielder": "Midfielder", "midfield": "Midfielder",
                      "defender": "Defender", "centre-back": "Defender", "center back": "Defender",
                      "full-back": "Defender", "fullback": "Defender"}
    # First match wins, matched at word starts ("scan" doesn't match "inscan"; "head"
    # alone is avoided so "header" stays an aerial question). Price
    # comes first so "looking for a cheap scanner" is a bargain question.
    METRIC_WORDS = [
        (("bargain", "cheap", "undervalued", "value for money", "price", "moneyball"), "value_gap"),
        (("retention", "retain", "keep the ball", "keeps the ball", "lose the ball"), "retention_rate"),
        (("scan", "awareness", "aware", "head check", "head turn", "heads up", "look around", "looks around", "looking around",
          "shoulder"), "scan_rate"),
        (("decoy", "space", "runs", "dragging"), "decoy_p90"),
        (("press", "lane", "cut"), "lane_cut_p90"),
        (("transition", "sprint", "latency", "react", "counter"), "median_latency"),
        (("control", "occup"), "pitch_control_pct"),
    ]

    def offline_answer(self, user_prompt: str, note: str = "") -> str:
        """A rule-based answer from the same table: detect position and metric, rank, quote numbers."""
        text = user_prompt.lower()
        said = lambda word: re.search(rf"\b{re.escape(word)}", text) is not None
        position = next((p for word, p in self.POSITION_WORDS.items() if said(word)), None)
        column = next((c for words, c in self.METRIC_WORDS if any(said(w) for w in words)), "index")
        spec = next(((c, asc, label) for c, asc, label in RANKINGS if c == column and c in self.df), None)
        if spec is None:
            asked = next((label for c, _, label in RANKINGS if c == column), None)
            if asked and column != "index":
                note = f"{note} There's no {asked} data yet.".strip()
            spec = next(((c, asc, label) for c, asc, label in RANKINGS if c in self.df), None)
        header = "_Offline answer (no language model used)._"
        if note:
            header = f"_{note} Offline answer from the same data._"
        if spec is None:
            return f"{header}\n\nThere's no player data to rank yet."
        column, ascending, label = spec
        rows = self.ranking(column, ascending, position, top=3)
        if rows.empty:
            return f"{header}\n\nNo {position.lower() + 's' if position else 'players'} have data for {label}."
        lines = [f"{k}. **{r['name']}** ({r.get('real_team', '')}, {r.get('broad_position', '')}): "
                 f"{label} {r[column]:.3g}"
                 + next((f", {name} {r[c]:.0f}" for c, name in (("index", "Off-Ball Index"), ("off_ball_value", "Pillar score"))
                         if c in r and pd.notna(r[c]) and column != c), "")
                 for k, (_, r) in enumerate(rows.iterrows(), 1)]
        who = f"{position.lower()}s" if position else "players"
        return f"{header}\n\nTop {who} by {label}:\n\n" + "\n".join(lines)
