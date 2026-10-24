import asyncio
from types import SimpleNamespace

from core.match_agents import generate_report_async
from core.match_tools import Fact, MatchFacts, check_report


def match() -> MatchFacts:
    return MatchFacts("Test_Match", "Harbourside FC", "Northvale Athletic", (1, 0), [
        Fact("F1", "score", "Final score: Harbourside FC 1-0 Northvale Athletic."),
        Fact("F2", "team", "Harbourside FC: 9 shots (4 on target), 472 of 526 passes completed (90%)."),
        Fact("F3", "goal", "Goal: Hugo Tanaka (Harbourside FC) scores; Harbourside FC 1-0 Northvale Athletic.",
             minute=27 + 55 / 60, time=1675.0, end=1680.0, player="Home_9"),
        Fact("F4", "press", "Lane-cutting press: Hugo Oyelaran (Harbourside FC): 79% of threat-weighted "
                            "lanes cut for 3.5 s.", minute=63.5, time=3900.0, end=3904.0, player="Home_4"),
        Fact("F5", "performer", "Top off-ball performer: Bram Ulloa (Harbourside FC), Off-Ball Value 88."),
    ])


GOOD = ("Harbourside FC beat Northvale Athletic 1-0. Hugo Tanaka scored [27:55]. "
        "Hugo Oyelaran cut 79% of the lanes for 3.5 s [63:30].")
INVENTED = "Harbourside FC won 1-0 after Hugo Tanaka ran 37.7 km [27:55] and scored again [88:00]."


class Scripted:
    """A stand-in agent: returns (or raises) the scripted replies in order, recording prompts."""

    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    async def run(self, prompt):
        self.prompts.append(prompt)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(text=reply)


def run(**kwargs):
    return asyncio.run(generate_report_async(match(), **kwargs))


def actions(report):
    return [f"{s.agent}: {s.action}" for s in report.trace]


def test_offline_report_is_code_written_verified_and_cited():
    report = run(persona="Analyst", offline=True)
    assert report.verified and report.mode.startswith("offline")
    assert check_report(report.text, match()) == []
    assert {f.id for f in report.cited} >= {"F3", "F4"}
    assert "Fact-checker: Approved the narrator draft" in actions(report)


def test_narrator_draft_with_invented_facts_is_sent_back_and_fixed():
    narrator = Scripted(INVENTED, GOOD)
    report = run(agents={"analyst": Scripted("Angle: a narrow win\nFacts: F3, F4"),
                         "narrator": narrator, "persona": Scripted(GOOD)})
    steps = actions(report)
    assert "Fact-checker: Rejected the narrator draft" in steps
    assert steps.index("Fact-checker: Rejected the narrator draft") < steps.index("Fact-checker: Approved the narrator draft")
    # The rewrite request carried the checker's exact problems.
    assert "37.7" in narrator.prompts[1] and "[88:00]" in narrator.prompts[1]
    assert report.verified and report.text == GOOD


def test_persona_that_keeps_failing_falls_back_to_the_verified_recap():
    report = run(agents={"analyst": Scripted("Facts: F3"), "narrator": Scripted(GOOD),
                         "persona": Scripted(INVENTED)})
    assert "Fact-checker: Falling back to the verified neutral recap" in actions(report)
    assert report.text == GOOD and report.verified


def test_analyst_choice_always_keeps_the_result_and_goals():
    narrator = Scripted(GOOD)
    run(agents={"analyst": Scripted("Angle: pressing told the story\nFacts: F4"), "narrator": narrator,
                "persona": Scripted(GOOD)})
    assert "F1 " in narrator.prompts[0] and "F3 [27:55]" in narrator.prompts[0] and "F4 [63:30]" in narrator.prompts[0]


def test_model_errors_fall_back_to_code_without_breaking_the_workflow():
    report = run(agents={"analyst": Scripted(RuntimeError("down")), "narrator": Scripted(RuntimeError("down")),
                         "persona": Scripted(RuntimeError("down"))})
    steps = actions(report)
    assert "Analyst: Model call failed, using the default brief" in steps
    assert "Narrator: Model call failed, using the code recap" in steps
    assert report.verified and check_report(report.text, match()) == []


def facts_with_names() -> MatchFacts:
    m = match()
    named = {"F3": "Hugo Tanaka", "F4": "Hugo Oyelaran", "F5": "Bram Ulloa"}
    m.facts = [Fact(f.id, f.kind, f.text, f.minute, f.time, f.end, f.player, named.get(f.id)) for f in m.facts]
    return m


def test_checker_rejects_crediting_the_wrong_player_for_a_moment():
    m = facts_with_names()
    wrong = "Harbourside FC won 1-0. Hugo Oyelaran scored [27:55]. Hugo Tanaka cut 79% of the lanes for 3.5 s [63:30]."
    problems = check_report(wrong, m)
    assert any("credits Hugo Oyelaran" in p and "about Hugo Tanaka" in p for p in problems)
    assert check_report(GOOD, m) == []


def test_checker_allows_minutes_of_real_moments():
    text = "Hugo Tanaka struck in the 28th minute [27:55], and Hugo Oyelaran pressed 63 minutes in [63:30]."
    assert check_report(text, facts_with_names()) == []


def test_bare_times_of_real_events_become_citations_but_wrong_times_do_not():
    from core.match_tools import normalise_citations
    m = facts_with_names()
    text = normalise_citations("Hugo Tanaka scored at 27:55, Hugo Oyelaran pressed (63:30), and 12:00 was quiet.", m)
    assert "[27:55]" in text and "[63:30]" in text and "[12:00]" not in text


def test_narrator_that_forgets_brackets_still_passes_after_normalising():
    bare = "Harbourside FC beat Northvale Athletic 1-0. Hugo Tanaka scored at 27:55. Hugo Oyelaran pressed at 63:30."
    report = run(agents={"analyst": Scripted("Facts: F3, F4"), "narrator": Scripted(bare), "persona": Scripted(bare)})
    assert "Fact-checker: Approved the narrator draft" in actions(report)
    assert "Fact-checker: Rejected the narrator draft" not in actions(report)
    assert "[27:55]" in report.text


def test_fact_ids_are_stripped_from_reports():
    from core.match_tools import normalise_citations
    text = normalise_citations("Harbourside FC won 1-0 [F1]. Hugo Tanaka scored [27:55] [F3, F4].", facts_with_names())
    assert "[F" not in text and "[27:55]" in text


def test_clock_absorbs_float_error_and_truncates_seconds():
    from core.match_tools import clock
    assert clock(27 + 55 / 60) == "27:55"
    assert clock(27.9086) == "27:54"            # 27:54.5 -> whole seconds, like the dashboard
    assert clock(93 + 14 / 60) == "93:14"


def test_language_and_followed_player_reach_the_persona_editor_and_the_rewrite_is_still_checked():
    spanish = ("Harbourside FC ganó 1-0 a Northvale Athletic. Hugo Tanaka marcó [27:55]. "
               "Hugo Oyelaran cortó el 79% de las líneas durante 3.5 s [63:30].")
    persona = Scripted(spanish.replace("79%", "97%"), spanish)          # first try changes a number
    report = run(agents={"analyst": Scripted("Angle: a narrow win\nFacts: F3, F4"), "narrator": Scripted(GOOD),
                         "persona": persona}, language="Español", focus="Hugo Tanaka")
    assert "write it in Español" in persona.prompts[0] and "the reader follows Hugo Tanaka" in persona.prompts[0]
    assert "Fact-checker: Rejected the persona draft" in actions(report)          # 97% isn't in the facts
    assert report.verified and report.text == spanish
