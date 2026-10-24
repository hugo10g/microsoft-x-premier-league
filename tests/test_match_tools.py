import numpy as np
import pandas as pd

from core.match_tools import build_match_facts, check_report, pillar_leaders, team_offball_text

ROSTER = pd.DataFrame({
    "player": ["Home_9", "Home_7", "Home_4", "Home_10", "Home_11", "Home_6", "Away_8", "Away_5"],
    "name": ["Hugo Tanaka", "Bram Ulloa", "Hugo Oyelaran", "Leo Brand", "Kai Ferro", "Tom Reyes",
             "Nico Vance", "Ravi Holt"],
    "club": ["Harbourside FC"] * 6 + ["Northvale Athletic"] * 2})

COLUMNS = ["Team", "Type", "Subtype", "Period", "Start Frame", "Start Time [s]", "End Frame",
           "End Time [s]", "From", "To"]


def events() -> pd.DataFrame:
    rows = [
        ("Home", "SET PIECE", "KICK OFF", 0.0, 0.0, "Home_9", None),
        ("Home", "PASS", None, 0.5, 1.5, "Home_9", "Home_7"),
        ("Home", "BALL LOST", "INTERCEPTION", 90.0, 90.0, "Home_7", None),
        ("Away", "RECOVERY", "INTERCEPTION", 90.2, 90.2, "Away_5", None),
        ("Away", "PASS", None, 95.0, 96.0, "Away_5", "Away_8"),
        ("Away", "BALL LOST", "INTERCEPTION", 100.0, 100.0, "Away_8", None),
        # Home's goal: Oyelaran's regain 12 s before Tanaka's shot.
        ("Home", "RECOVERY", "INTERCEPTION", 100.2, 100.2, "Home_4", None),
        ("Home", "PASS", None, 102.0, 103.0, "Home_4", "Home_7"),
        ("Home", "PASS", None, 110.0, 111.0, "Home_7", "Home_9"),
        ("Home", "SHOT", "ON TARGET-GOAL", 112.0, 113.0, "Home_9", None),
        ("Away", "SET PIECE", "KICK OFF", 140.0, 140.0, "Away_8", None),
        ("Away", "BALL OUT", None, 380.0, 380.0, "Away_5", None),
        # Away's equaliser: from a throw-in 10 s before Vance's shot.
        ("Away", "SET PIECE", "THROW IN", 390.0, 390.0, "Away_5", None),
        ("Away", "PASS", None, 393.0, 394.0, "Away_5", "Away_8"),
        ("Away", "SHOT", "ON TARGET-GOAL", 400.0, 401.0, "Away_8", None),
    ]
    table = pd.DataFrame([(team, kind, sub, 1, int(start * 25), start, int(end * 25), end, src, dst)
                          for team, kind, sub, start, end, src, dst in rows], columns=COLUMNS)
    return table.astype({"Subtype": object})


def press_frames(player: str, start: float, n: int, cut: float) -> pd.DataFrame:
    times = start + 0.5 * np.arange(n)
    return pd.DataFrame({"player": player, "time": times, "minute": times / 60, "lane_cut": cut, "seconds": 0.5})


def tables() -> dict[str, pd.DataFrame]:
    decoys = pd.DataFrame({"player": ["Home_7", "Home_7", "Away_5", "Home_10"],
                           "time": [105.0, 60.0, 106.0, 108.0], "duration": 2.0,
                           "decoy_value": [0.031, 0.5, 0.2, 0.0001]})
    pressing = pd.concat([press_frames("Home_10", 97.0, 7, 0.8),      # 3.5 s at 80% before the regain
                          press_frames("Home_4", 99.0, 2, 0.3),
                          press_frames("Away_5", 150.0, 5, 0.6)], ignore_index=True)
    transitions = pd.DataFrame({
        "player": ["Home_11", "Home_11", "Home_11", "Home_9", "Home_6", "Away_5", "Away_5", "Away_5"],
        "phase": ["attacking"] * 5 + ["defensive"] * 3,
        "latency": [0.4, 0.5, 0.6, 0.1, 3.0, 0.6, 0.7, 0.8],
        "reached_sprint": [True, True, True, True, False, True, True, True],
        "score": 0.5, "time": [100.2, 300.0, 500.0, 100.2, 100.2, 100.2, 300.0, 500.0]})
    scanning = pd.DataFrame({
        "player": ["Home_9", "Home_9"] + ["Home_7"] * 6 + ["Away_8"] * 5,
        "time": [50.0, 111.0] + [20.0 + k for k in range(6)] + [200.0, 210.0, 220.0, 230.0, 394.0],
        "n_scans": [5, 3] + [2] * 6 + [3, 3, 3, 2, 4],
        "retained": [True, True] + [True] * 3 + [False] * 3 + [True] * 5})
    scanning = scanning.assign(minute=scanning["time"] / 60, scan_rate=scanning["n_scans"] / 5,
                               ball_fixation=0.5)
    for table in (decoys, transitions):
        table["minute"] = table["time"] / 60
    return {"decoys": decoys, "pressing": pressing, "transitions": transitions, "scanning": scanning}


def moments() -> pd.DataFrame:
    return pd.DataFrame({"kind": ["press"], "player": ["Away_5"], "time": [150.0], "end": [152.0],
                         "minute": [2.5], "score": [1.0], "detail": ["60% of threat-weighted lanes cut for 2.5 s"]})


def facts(with_tables: bool = True):
    return build_match_facts("Test", events(), ROSTER, moments(), pd.DataFrame(),
                             tables=tables() if with_tables else None)


def test_build_up_credits_the_press_regain_decoy_sprint_and_scans_in_the_goal_possession():
    home, away = facts().of_kind("build_up")
    assert home.clock == "01:52" and home.name == "Hugo Tanaka"
    assert home.text == (
        "Build-up to Hugo Tanaka's goal for Harbourside FC (Harbourside FC 1-0 Northvale Athletic): "
        "Leo Brand's press cut 80% of the threat-weighted passing lanes for 3.5 s before Hugo Oyelaran won "
        "the ball back with an interception 12 s before the shot; Bram Ulloa's decoy run 7 s before the shot "
        "opened 0.031 xT of space for teammates; Kai Ferro sprinted (20+ km/h) into the attack 0.4 s after "
        "the turnover; Hugo Tanaka made 3 head checks in the 5 s before receiving.")
    # The window is the goal's possession: Ulloa's earlier run, the other team's run, a negligible
    # run, a sprint under the reaction floor and Tanaka's earlier reception are all left out.
    assert "0.500" not in home.text and "Ravi Holt" not in home.text and "0.1 s" not in home.text
    assert set(home.names) == {"Leo Brand", "Hugo Oyelaran", "Bram Ulloa", "Kai Ferro"}
    # Playback starts at Brand's first pressing frame, so Watch shows the whole press.
    assert home.time == 97.0 and home.end == 116.0
    assert away.text == ("Build-up to Nico Vance's goal for Northvale Athletic (Harbourside FC 1-1 Northvale "
                         "Athletic): The move began from a throw-in 10 s before the shot; Nico Vance made "
                         "4 head checks in the 5 s before receiving.")
    assert away.names == ()


def test_team_offball_totals():
    home, away = facts().of_kind("team_offball")
    assert home.text == ("Harbourside FC off the ball: 3 decoy runs opening 0.531 xT for teammates, 3 s of "
                         "threat-weighted passing lanes cut while pressing, a sprint (20+ km/h) in 80% of their "
                         "players' transition reactions, 2.5 head checks per reception and 62% of receptions kept.")
    assert away.text.startswith("Northvale Athletic off the ball: 1 decoy run opening 0.200 xT")
    assert "3.0 head checks per reception and 100% of receptions kept" in away.text


def test_pillar_leaders_need_enough_sprints_and_receptions():
    leaders = {f.text.split(":")[0]: f for f in facts().of_kind("pillar_leader")}
    assert leaders["Decoy-run leader"].name == "Bram Ulloa"
    assert "2 decoy runs opening 0.531 xT" in leaders["Decoy-run leader"].text
    assert leaders["Pressing leader"].name == "Leo Brand" and "2.8 s" in leaders["Pressing leader"].text
    # Tanaka's 0.1 s is below the reaction floor and he has one sprint; Ferro's median is 0.5 s.
    assert leaders["Transition leader"].name == "Kai Ferro"
    assert "a median 0.5 s to sprint over 3 sprints" in leaders["Transition leader"].text
    # Tanaka averages 4 checks but over only 2 receptions.
    assert leaders["Scanning leader"].name == "Nico Vance"
    assert "3.0 head checks per reception over 5 receptions, 100% of them kept" in leaders["Scanning leader"].text


def test_without_tables_the_facts_still_build_from_events_alone():
    match = facts(with_tables=False)
    assert not match.of_kind("team_offball", "pillar_leader")
    home = match.of_kind("build_up")[0]
    assert home.text.endswith("Hugo Oyelaran won the ball back with an interception 12 s before the shot.")
    assert [f.kind for f in match.facts[:3]] == ["score", "team", "team"]


def test_checker_accepts_build_up_names_but_rejects_unrelated_ones():
    match = facts()
    good = ("Bram Ulloa's decoy run opened 0.031 xT of space before Hugo Tanaka's goal [01:52]. "
            "Leo Brand's press cut 80% of the lanes for 3.5 s in that move [01:52]. "
            "Nico Vance made 4 head checks before equalising [06:40].")
    assert check_report(good, match) == []
    wrong = good.replace("Leo Brand's press", "Ravi Holt's press")
    problems = check_report(wrong, match)
    assert any("credits Ravi Holt" in p and "that moment is about" in p for p in problems)
    assert any("0.047" in p for p in check_report(good.replace("0.031", "0.047"), match))


def test_checker_rejects_actions_given_to_the_wrong_player_or_invented():
    match = facts()
    swapped = ("Kai Ferro's decoy run opened 0.031 xT of space before Hugo Tanaka's goal [01:52]. "
               "Nico Vance scored [06:40].")
    assert any("credits a decoy to Kai Ferro, but the decoy at that moment was by Bram Ulloa" in p
               for p in check_report(swapped, match))
    invented = "Bram Ulloa's decoy run came before Hugo Tanaka's goal [01:52]. Nico Vance's decoy run set it up [06:40]."
    problems = check_report(invented, match)
    assert problems == ["The sentence citing [06:40] mentions a decoy, but the facts record no decoy at that "
                        "moment. Only describe actions the cited fact states."]
    # A key moment credits its own action; a sentence naming nobody may still describe it.
    assert check_report("Ravi Holt pressed [02:30]. A decoy run opened space before the opener [01:52].", match) == []


def test_checker_rejects_two_players_swapping_actions_in_one_sentence():
    match = facts()
    right = "Bram Ulloa's decoy run and Leo Brand's press set up Hugo Tanaka's goal [01:52]. Nico Vance scored [06:40]."
    assert check_report(right, match) == []
    assert check_report("A decoy run by Bram Ulloa set up Hugo Tanaka's goal [01:52]. Nico Vance scored [06:40].",
                        match) == []
    swapped = "Leo Brand's decoy run and Bram Ulloa's press set up Hugo Tanaka's goal [01:52]. Nico Vance scored [06:40]."
    problems = check_report(swapped, match)
    assert any("credits a decoy to Leo Brand" in p for p in problems)
    assert any("credits a press to Bram Ulloa" in p for p in problems)


def test_checker_checks_who_won_the_ball_back_and_shoulder_checks():
    match = facts()
    ok = "Hugo Oyelaran won the ball back with an interception before Hugo Tanaka scored [01:52]. Nico Vance scored [06:40]."
    assert check_report(ok, match) == []
    wrong = ok.replace("Hugo Oyelaran won", "Bram Ulloa won")
    assert any("credits a regain to Bram Ulloa" in p and "Hugo Oyelaran" in p for p in check_report(wrong, match))
    shoulder = "Leo Brand checked his shoulder 3 times before Hugo Tanaka scored [01:52]. Nico Vance scored [06:40]."
    assert any("credits a scan to Leo Brand" in p for p in check_report(shoulder, match))
    assert check_report(shoulder.replace("Leo Brand", "Hugo Tanaka", 1), match) == []


def test_checker_tells_same_surname_teammates_apart():
    roster = ROSTER.assign(name=ROSTER["name"].replace("Kai Ferro", "Elias Tanaka"))
    match = build_match_facts("Test", events(), roster, moments(), pd.DataFrame(), tables=tables())
    tail = " Nico Vance scored [06:40]."
    # Hugo Tanaka scored; his teammate Elias Tanaka made the build-up sprint.
    assert check_report("Elias Tanaka sprinted into the attack before Hugo Tanaka scored [01:52]." + tail, match) == []
    stolen = "Hugo Tanaka sprinted into the attack 0.4 s after the turnover and scored [01:52]." + tail
    assert any("credits a sprint to Hugo Tanaka" in p and "Elias Tanaka" in p for p in check_report(stolen, match))
    # With two Tanakas cited, a bare "Tanaka" is ambiguous, even after one full name: ask for it.
    for sentence in ["Hugo Tanaka scored after Tanaka's sprint [01:52].", "Tanaka's sprint set up the goal [01:52]."]:
        assert any("write the full name" in p for p in check_report(sentence + tail, match)), sentence
    assert check_report("Hugo Tanaka scored after Elias Tanaka's sprint [01:52]." + tail, match) == []


def test_numbers_with_thousands_separators_are_read_whole():
    match = facts()
    text = "Hugo Tanaka scored [01:52]. Nico Vance replied [06:40]. Harbourside made 1,000 decoy runs."
    assert check_report(text, match) == ["The number 1,000 doesn't appear in the facts."]


def test_offline_report_leads_with_off_ball_work_and_passes_the_check():
    import asyncio

    from core.match_agents import default_brief, generate_report_async
    match = facts()
    brief = default_brief(match)
    assert {f.id for f in match.of_kind("build_up", "pillar_leader")} <= set(brief.fact_ids)
    report = asyncio.run(generate_report_async(match, "Fan", offline=True))
    assert report.verified and check_report(report.text, match) == []
    assert report.text.startswith("Off the ball:")
    assert {f.kind for f in report.cited} >= {"build_up", "press"}


def test_passive_and_chain_phrasing_keeps_each_action_with_its_owner():
    m = facts()
    for sentence in ["Hugo Oyelaran's interception was set up by Leo Brand's press [01:52].",
                     "Bram Ulloa's decoy run followed by Kai Ferro's sprint helped Hugo Tanaka score [01:52].",
                     "Leo Brand's press was followed by Hugo Oyelaran's interception [01:52]."]:
        assert check_report(sentence + " Nico Vance scored [06:40].", m) == [], sentence


def test_passive_regain_and_recovery_wording():
    m = facts()
    wrong = "The ball was won back by Bram Ulloa before Hugo Tanaka scored [01:52]. Nico Vance scored [06:40]."
    assert any("regain to Bram Ulloa" in p for p in check_report(wrong, m))
    fine = "Northvale never recovered after Hugo Tanaka scored [01:52]. Nico Vance scored [06:40]."
    assert check_report(fine, m) == []


def test_every_player_in_a_joint_subject_must_have_done_the_action():
    m = facts()
    joint = "Bram Ulloa and Leo Brand pressed before Hugo Tanaka scored [01:52]. Nico Vance scored [06:40]."
    assert any("press" in p and "Bram Ulloa" in p for p in check_report(joint, m))


def test_tactical_leaders_and_team_totals_come_from_the_new_tables():
    tables = {
        "pressures": pd.DataFrame({"player": ["Home_4", "Home_4", "Away_8"], "time": [10.0, 50.0, 70.0],
                                   "intensity": [3.0, 4.0, 2.0]}),
        "runs": pd.DataFrame({"player": ["Away_9", "Away_9", "Home_7"], "time": [5.0, 9.0, 30.0],
                              "run_type": ["Run in Behind", "Run in Behind", "Pulling Wide"]}),
        "obso": pd.DataFrame({"player": ["Away_9", "Home_7"], "time": [0.0, 10.0], "obso": [2.5, 1.0]}),
    }
    leaders = dict((text.split(":")[0], player) for player, text in pillar_leaders(tables))
    assert leaders == {"Pressure-intensity leader": "Home_4", "Runs-in-behind leader": "Away_9",
                       "Off-ball scoring opportunity (OBSO) leader": "Away_9"}
    text = team_offball_text("Away", "Away FC", tables)
    assert "2 runs in behind" in text and "2 m of ground closed down" in text
    assert "0 runs in behind" in team_offball_text("Home", "Home FC", tables)
