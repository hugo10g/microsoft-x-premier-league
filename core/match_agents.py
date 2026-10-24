"""
Off-ball match reports by a team of agents, orchestrated with Microsoft Agent Framework.

The report explains how the work players do without the ball shaped the match:
decoy runs, pressing lane cuts, transitions and scanning. The result is context.

    Analyst ──► Narrator ──► Fact-checker ──► Persona editor ──► Fact-checker ──► report
                   ▲             │ problems         ▲                  │ problems
                   └─ feedback ──┘                  └──── feedback ────┘

    Analyst         LLM agent with tools over the verified match facts (core.match_tools);
                    picks the off-ball facts that tell the story and an angle for the audience.
    Narrator        LLM agent; writes a neutral off-ball report from those facts, with a
                    headline about off-ball work, citing each moment as [mm:ss].
    Fact-checker    plain code, not a model: every [mm:ss] must be a real event and
                    every number must appear in the facts. Problems go back to the
                    author with the exact list, for up to MAX_REVISIONS rewrites.
    Persona editor  LLM agent; retells the verified recap for a fan, an analyst or a
                    broadcaster without changing citations or numbers.

Shared state: the facts, the analyst's brief and the verified draft live in the
workflow's state; every step is written to a trace the app shows. Recovery: a model
error falls back to a code-written step; a persona rewrite that keeps failing the
check falls back to the verified neutral recap; a narrator that keeps failing falls
back to a code-written recap. With no model configured the whole report is
code-written, and says so.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Annotated, Any

from agent_framework import Agent, Executor, WorkflowBuilder, WorkflowContext, handler, tool

from core.agent import ModelConfig
from core.match_tools import CITATION, Fact, MatchFacts, check_report, normalise_citations

MAX_REVISIONS = 2

PERSONAS: dict[str, str] = {
    "Fan": "a supporter: lively and warm, short paragraphs, celebrate the unseen off-ball graft behind "
           "the drama, at most two emojis, about 150 words",
    "Analyst": "a performance analyst: precise and tactical, lead with the off-ball metrics, use 3-5 "
               "bullet points after one summary sentence, about 170 words",
    "Broadcaster": "a TV presenter's script: spoken, punchy sentences to read aloud, open with a hook, "
                   "end with a line handing over to the pundit, about 150 words",
}

ANALYST_INSTRUCTIONS = """You are the Analyst on an off-ball match-report team.
The report explains how the work players did WITHOUT the ball shaped this match: decoy runs that
opened space (xT), presses that cut passing lanes, sprints into transitions, and scanning (head
checks before receiving). The result is context, not the story.
Use your tools to read the match facts. Start with build_up_facts (the off-ball work behind each
goal), then team_offball and pillar_leaders, then key_moments. Choose the 5 to 8 facts that best
show the off-ball story for the given audience: the result and the goals are always included for
you, so spend your picks on build-up facts, pillar leaders, team off-ball totals and moments.
Reply in exactly this format and nothing else:
Angle: <one sentence on how off-ball work shaped the match, e.g. which pillar or player made the difference>
Facts: <comma-separated fact IDs, e.g. F1, F4, F9>"""

NARRATOR_INSTRUCTIONS = """You are the Narrator on an off-ball match-report team.
Write a neutral report of about 180 words on how off-ball work shaped this match, from the facts
you are given.
Structure:
- First line: a headline about the off-ball work (a decoy run, a press, a transition, scanning or
  a pillar leader), not the scoreline.
- Then the story: lead with the off-ball actions behind the goals and the players who led each
  pillar; give the result and goals as context.
Rules:
- Use only the facts given. Never invent events, players, scores or numbers.
- After every sentence about something that happened at a moment, cite its time exactly as
  written in the facts, in square brackets. Example: "Bram Ulloa's decoy run opened 0.031 xT of
  space before Hugo Tanaka's goal [27:55]."
- Cite at least three different moments.
- Copy numbers exactly as they appear in the facts.
- One moment per sentence: never name players from different moments in the same sentence. A
  build-up fact is one moment: the players it names may share a sentence citing that goal's time.
- Credit each player only with the action the fact gives them: if a build-up fact says one player
  pressed and another made a decoy run, never swap them, and never add a decoy run, press, sprint,
  head check or ball regain that the fact doesn't state. Keep each action next to the full name of
  the player who did it ("Leo Brand's press", "a decoy run by Bram Ulloa"); use full names when
  two players share a surname. The fact-checker verifies who did what.
- Facts without a time (team totals, pillar leaders, performers) get no citation: write each in
  its own sentence, without naming players from timed moments.
- Don't claim one moment caused or led to another (e.g. "scored after a sprint") unless a build-up
  fact states that link; retell a build-up fact's links as written. Moments at different times are
  separate events.
- Never write fact IDs such as F4 in the text.
- Every club and player is fictional; write about them naturally."""

PERSONA_INSTRUCTIONS = """You are the Persona editor on an off-ball match-report team.
Rewrite the verified report for the audience described. Keep the focus on off-ball work (decoy
runs, pressing, transitions, scanning), with a headline about it; the result stays context.
Change tone, order and length freely, but:
- keep every [mm:ss] citation attached to the sentence it supports, exactly as written;
- copy every number exactly; never add new facts, players, numbers or times;
- keep one moment per sentence, keep sentences without a citation free of players from timed
  moments, and don't add cause-and-effect links beyond those the recap already states;
- keep each player's action as the recap states it (who made the decoy run, who pressed, who
  sprinted, who made head checks, who won the ball back), each next to that player's full name;
  never add one;
- never invent names of anyone or anything not in the recap: no pundit, presenter, stadium or
  manager names (say "the pundit", "the ground")."""


# ── Messages passed between the steps ────────────────────────────────────────
@dataclass
class ReportRequest:
    persona: str


@dataclass
class Brief:
    angle: str
    fact_ids: list[str]


@dataclass
class Draft:
    stage: str              # "narrator" or "persona"
    text: str
    attempt: int = 1
    author: str = "model"   # "model" or "code" (a fallback)


@dataclass
class Feedback:
    stage: str
    draft: Draft
    problems: list[str]


@dataclass
class Verified:
    draft: Draft


@dataclass
class TraceStep:
    agent: str
    action: str
    detail: str = ""
    seconds: float = 0.0


@dataclass
class MatchReport:
    persona: str
    text: str
    cited: list[Fact]
    trace: list[TraceStep]
    mode: str
    verified: bool


# ── Code-written fallbacks (also the whole report when no model is configured) ──
# Kinds that aren't key moments: the result, match totals and the goal-related facts.
MATCH_KINDS = ("score", "team", "team_offball", "goal", "build_up", "pillar_leader", "performer")


def default_brief(match: MatchFacts) -> Brief:
    """The result, each goal with its build-up, the pillar leaders, and the strongest moment of each kind."""
    picks = (match.of_kind("score", "goal", "build_up") + match.of_kind("pillar_leader")
             + match.of_kind("performer")[:1])
    seen = set()
    for fact in match.facts:
        if fact.kind not in seen and fact.kind not in MATCH_KINDS:
            picks.append(fact)
            seen.add(fact.kind)
    return Brief(f"{match.home} {match.score[0]}-{match.score[1]} {match.away}: the off-ball work behind "
                 "the goals and the players who led each pillar.", [f.id for f in picks])


def code_recap(match: MatchFacts, brief: Brief) -> str:
    """A plain, faithful report built straight from the facts, under an off-ball headline."""
    facts = match.by_id()
    lines = [f"Off the ball: how the work without it shaped {match.home} {match.score[0]}-"
             f"{match.score[1]} {match.away}.", facts["F1"].text if "F1" in facts else ""]
    for fact_id in brief.fact_ids:
        fact = facts.get(fact_id)
        if fact and fact.id != "F1":
            lines.append(f"{fact.text} [{fact.clock}]" if fact.clock else fact.text)
    return "\n\n".join(line for line in lines if line)


# ── Workflow steps ───────────────────────────────────────────────────────────
class Trace:
    """The run's step log; shared by every executor."""

    def __init__(self) -> None:
        self.steps: list[TraceStep] = []

    def add(self, agent: str, action: str, detail: str = "", started: float | None = None) -> None:
        self.steps.append(TraceStep(agent, action, detail, round(time.time() - started, 1) if started else 0.0))


class AnalystStep(Executor):
    def __init__(self, match: MatchFacts, agent: Agent | None, trace: Trace) -> None:
        super().__init__(id="analyst")
        self.match, self.agent, self.trace = match, agent, trace

    @handler
    async def start(self, request: ReportRequest, ctx: WorkflowContext[Brief]) -> None:
        ctx.set_state("persona", request.persona)
        started = time.time()
        brief = None
        if self.agent is not None:
            try:
                reply = (await self.agent.run(f"Audience: {PERSONAS[request.persona]}. Build the brief.")).text
                ids = [i for i in re.findall(r"\bF\d+\b", reply.split("Facts:")[-1]) if i in self.match.by_id()]
                angle = re.search(r"Angle:\s*(.+)", reply)
                if ids:
                    brief = Brief(angle.group(1).strip() if angle else "", list(dict.fromkeys(ids)))
                    self.trace.add("Analyst", "Chose facts with tools", f"{len(brief.fact_ids)} facts: "
                                   f"{', '.join(brief.fact_ids)}. Angle: {brief.angle}", started)
                else:
                    self.trace.add("Analyst", "Reply had no fact IDs, using the default brief")
            except Exception as error:      # network, quota, content filter...
                self.trace.add("Analyst", "Model call failed, using the default brief", type(error).__name__)
        if brief is None:
            brief = default_brief(self.match)
            self.trace.add("Analyst", "Default brief (code)", f"{len(brief.fact_ids)} facts")
        # Always keep the result, the goals and their build-up, whatever the analyst chose.
        must = [f.id for f in self.match.of_kind("score", "goal", "build_up")]
        brief.fact_ids = list(dict.fromkeys(must + brief.fact_ids))
        ctx.set_state("brief", brief)
        await ctx.send_message(brief)


class NarratorStep(Executor):
    def __init__(self, match: MatchFacts, agent: Agent | None, trace: Trace) -> None:
        super().__init__(id="narrator")
        self.match, self.agent, self.trace = match, agent, trace

    def prompt(self, brief: Brief) -> str:
        facts = self.match.by_id()
        chosen = [facts[i] for i in brief.fact_ids if i in facts] + self.match.of_kind("team", "team_offball")
        return f"Story angle: {brief.angle}\n\nFacts:\n{self.match.lines(list(dict.fromkeys(chosen)))}"

    async def write(self, prompt: str, attempt: int, ctx: WorkflowContext[Draft]) -> None:
        started = time.time()
        if self.agent is not None:
            try:
                text = (await self.agent.run(prompt)).text
                self.trace.add("Narrator", f"Wrote recap (attempt {attempt})", f"{len(text.split())} words", started)
                await ctx.send_message(Draft("narrator", text, attempt))
                return
            except Exception as error:
                self.trace.add("Narrator", "Model call failed, using the code recap", type(error).__name__)
        brief = ctx.get_state("brief")
        await ctx.send_message(Draft("narrator", code_recap(self.match, brief), attempt, author="code"))

    @handler
    async def from_brief(self, brief: Brief, ctx: WorkflowContext[Draft]) -> None:
        await self.write(self.prompt(brief), 1, ctx)

    @handler
    async def revise(self, feedback: Feedback, ctx: WorkflowContext[Draft]) -> None:
        if feedback.draft.attempt > MAX_REVISIONS:      # give up on the model: faithful by construction
            self.trace.add("Narrator", "Too many failed checks, using the code recap")
            await ctx.send_message(Draft("narrator", code_recap(self.match, ctx.get_state("brief")),
                                         feedback.draft.attempt + 1, author="code"))
            return
        prompt = (f"{self.prompt(ctx.get_state('brief'))}\n\nYour previous recap:\n{feedback.draft.text}\n\n"
                  "The fact-checker found these problems. Fix all of them:\n- " + "\n- ".join(feedback.problems))
        await self.write(prompt, feedback.draft.attempt + 1, ctx)


class FactCheckStep(Executor):
    def __init__(self, match: MatchFacts, trace: Trace) -> None:
        super().__init__(id="fact_checker")
        self.match, self.trace = match, trace

    @handler
    async def check(self, draft: Draft, ctx: WorkflowContext[Feedback | Verified, MatchReport]) -> None:
        draft.text = normalise_citations(draft.text, self.match)      # formatting only
        problems = check_report(draft.text, self.match)
        if not problems:
            self.trace.add("Fact-checker", f"Approved the {draft.stage} draft", f"attempt {draft.attempt}")
            if draft.stage == "narrator":
                ctx.set_state("verified_recap", draft)
                await ctx.send_message(Verified(draft))
            else:
                await ctx.yield_output(self.report(draft, ctx, verified=True))
            return
        self.trace.add("Fact-checker", f"Rejected the {draft.stage} draft", "; ".join(problems[:4]))
        if draft.stage == "persona" and draft.attempt > MAX_REVISIONS:
            # Recovery: the neutral recap already passed, so publish that instead.
            fallback = ctx.get_state("verified_recap")
            self.trace.add("Fact-checker", "Falling back to the verified neutral recap")
            await ctx.yield_output(self.report(fallback, ctx, verified=True))
            return
        if draft.author == "code":                      # can't happen by construction, but never loop
            await ctx.yield_output(self.report(draft, ctx, verified=False))
            return
        await ctx.send_message(Feedback(draft.stage, draft, problems))

    def report(self, draft: Draft, ctx: WorkflowContext[Any, MatchReport], verified: bool) -> MatchReport:
        by_clock = {f.clock: f for f in self.match.facts if f.clock}
        cited = [by_clock[c] for c in dict.fromkeys(CITATION.findall(draft.text)) if c in by_clock]
        return MatchReport(ctx.get_state("persona"), draft.text, cited, [], "", verified)


class PersonaStep(Executor):
    def __init__(self, agent: Agent | None, trace: Trace) -> None:
        super().__init__(id="persona_editor")
        self.agent, self.trace = agent, trace

    async def write(self, prompt: str, source: Draft, attempt: int, ctx: WorkflowContext[Draft]) -> None:
        started = time.time()
        persona = ctx.get_state("persona")
        if self.agent is not None:
            try:
                text = (await self.agent.run(prompt)).text
                self.trace.add("Persona editor", f"Rewrote for {persona} (attempt {attempt})",
                               f"{len(text.split())} words", started)
                await ctx.send_message(Draft("persona", text, attempt))
                return
            except Exception as error:
                self.trace.add("Persona editor", "Model call failed, keeping the verified recap", type(error).__name__)
        await ctx.send_message(Draft("persona", source.text, attempt, author="code"))

    @handler
    async def from_verified(self, verified: Verified, ctx: WorkflowContext[Draft]) -> None:
        persona = ctx.get_state("persona")
        prompt = f"Audience: {PERSONAS[persona]}\n\nVerified recap:\n{verified.draft.text}"
        await self.write(prompt, verified.draft, 1, ctx)

    @handler
    async def revise(self, feedback: Feedback, ctx: WorkflowContext[Draft]) -> None:
        persona = ctx.get_state("persona")
        source = ctx.get_state("verified_recap")
        prompt = (f"Audience: {PERSONAS[persona]}\n\nVerified recap:\n{source.text}\n\nYour previous rewrite:\n"
                  f"{feedback.draft.text}\n\nThe fact-checker found these problems. Fix all of them:\n- "
                  + "\n- ".join(feedback.problems))
        await self.write(prompt, source, feedback.draft.attempt + 1, ctx)


# ── Tools the Analyst can call ───────────────────────────────────────────────
def analyst_tools(match: MatchFacts) -> list:
    """Read-only tools over one match's facts, for the Analyst agent."""

    @tool
    def match_summary() -> str:
        """The final score and each team's shots, passing and share of the passes."""
        return match.lines(match.of_kind("score", "team"))

    @tool
    def goals() -> str:
        """Every goal with its match time and the scorer."""
        return match.lines(match.of_kind("goal")) or "No goals."

    @tool
    def build_up_facts() -> str:
        """The off-ball work behind each goal: decoy runs and the xT they opened, the press that
        won the ball back, sprints into the attack and the scorer's head checks before receiving."""
        return match.lines(match.of_kind("build_up")) or "No goals, so no build-ups."

    @tool
    def team_offball() -> str:
        """Each team's off-ball totals: decoy runs and xT opened, pressing lane-cut seconds,
        transition sprint rate, head checks per reception and receptions kept."""
        return match.lines(match.of_kind("team_offball")) or "No team off-ball totals."

    @tool
    def pillar_leaders() -> str:
        """The match's top player in each off-ball pillar: decoy runs, pressing, transitions, scanning."""
        return match.lines(match.of_kind("pillar_leader")) or "No pillar leaders."

    @tool
    def key_moments(kind: Annotated[str, "One of: decoy, press, defensive_transition, "
                                         "attacking_transition, scan, or all"] = "all") -> str:
        """Standout off-ball moments: decoy runs, presses, transitions and scans, with times and players."""
        kinds = ("decoy", "press", "defensive_transition", "attacking_transition", "scan")
        chosen = kinds if kind == "all" else (kind,)
        return match.lines(match.of_kind(*chosen)) or "No moments of that kind."

    @tool
    def top_performers() -> str:
        """The players with the highest Off-Ball Value in this match."""
        return match.lines(match.of_kind("performer")) or "No performer data."

    return [match_summary, goals, build_up_facts, team_offball, pillar_leaders, key_moments, top_performers]


# ── Running a report ─────────────────────────────────────────────────────────
def build_workflow(match: MatchFacts, config: ModelConfig | None, trace: Trace,
                   agents: dict[str, Any] | None = None, http_client: Any = None):
    """The four-step workflow graph; agents are None (code fallbacks) without a model.

    `agents` ({"analyst": ..., "narrator": ..., "persona": ...}) replaces the model-backed
    agents with any objects whose async `run(prompt)` returns something with `.text`
    (used by the tests to script failures and revisions).
    """
    analyst = narrator = persona = None
    if agents is not None:
        analyst, narrator, persona = agents.get("analyst"), agents.get("narrator"), agents.get("persona")
    elif config is not None:
        from agent_framework.openai import OpenAIChatCompletionClient
        client = OpenAIChatCompletionClient(model=config.model, async_client=http_client)
        analyst = Agent(client=client, name="Analyst", instructions=ANALYST_INSTRUCTIONS,
                        tools=analyst_tools(match))
        narrator = Agent(client=client, name="Narrator", instructions=NARRATOR_INSTRUCTIONS)
        persona = Agent(client=client, name="PersonaEditor", instructions=PERSONA_INSTRUCTIONS)

    analyst_step = AnalystStep(match, analyst, trace)
    narrator_step = NarratorStep(match, narrator, trace)
    checker = FactCheckStep(match, trace)
    persona_step = PersonaStep(persona, trace)
    return (WorkflowBuilder(start_executor=analyst_step, max_iterations=30)
            .add_edge(analyst_step, narrator_step)
            .add_edge(narrator_step, checker)
            .add_edge(checker, narrator_step,
                      condition=lambda m: isinstance(m, Feedback) and m.stage == "narrator")
            .add_edge(checker, persona_step,
                      condition=lambda m: isinstance(m, Verified) or (isinstance(m, Feedback) and m.stage == "persona"))
            .add_edge(persona_step, checker)
            .build())


async def generate_report_async(match: MatchFacts, persona: str = "Fan", config: ModelConfig | None = None,
                                offline: bool = False, agents: dict[str, Any] | None = None) -> MatchReport:
    """Run the agent team on one match.

    config defaults to the environment / .env; offline=True skips the model entirely.
    """
    if persona not in PERSONAS:
        raise ValueError(f"persona must be one of {list(PERSONAS)}")
    if offline:
        config = None
    elif config is None and agents is None:
        config = ModelConfig.from_env()
    trace = Trace()
    started = time.time()
    http_client = None
    if config is not None and agents is None:
        from openai import AsyncOpenAI
        http_client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
    try:
        result = await build_workflow(match, config, trace, agents, http_client).run(ReportRequest(persona))
    finally:
        if http_client is not None:          # close inside this event loop, not after it has ended
            await http_client.close()
    outputs = result.get_outputs()
    if outputs:
        report = outputs[-1]
    else:                                   # the graph ended without a verdict: never leave the user empty
        brief = default_brief(match)
        report = MatchReport(persona, code_recap(match, brief), [], [], "", verified=True)
        trace.add("Orchestrator", "Workflow ended without a report, using the code recap")
    report.trace = trace.steps + [TraceStep("Orchestrator", "Finished", "", round(time.time() - started, 1))]
    if agents is not None:
        report.mode = "scripted agents (test)"
    elif config is not None:
        report.mode = f"{config.provider} · {config.model} · Microsoft Agent Framework"
    else:
        report.mode = "offline: code-written from the verified facts (no model used)"
    if not report.cited:
        by_clock = {f.clock: f for f in match.facts if f.clock}
        report.cited = [by_clock[c] for c in dict.fromkeys(CITATION.findall(report.text)) if c in by_clock]
    return report


def generate_report(match: MatchFacts, persona: str = "Fan", config: ModelConfig | None = None,
                    offline: bool = False) -> MatchReport:
    """Synchronous wrapper for scripts and Streamlit."""
    return asyncio.run(generate_report_async(match, persona, config, offline))
