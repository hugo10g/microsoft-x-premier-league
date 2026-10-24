from types import SimpleNamespace

import pandas as pd

from core.agent import ModelConfig, OffBallAgent


def league() -> pd.DataFrame:
    return pd.DataFrame({
        "name": ["Ada Runner", "Ben Presser", "Cal Quick", "Dan Deep"],
        "real_team": ["Harbourside FC", "Northvale Athletic", "Redmoor Rovers", "Ashcombe City"],
        "broad_position": ["Forward", "Forward", "Midfielder", "Defender"],
        "minutes": [180.0, 170, 160, 190],
        "off_ball_value": [80.0, 70, 60, 40],
        "decoy_p90": [0.30, 0.10, 0.05, 0.01],
        "lane_cut_p90": [40.0, 150, 90, 20],
        "median_latency": [2.5, 3.0, 1.1, float("nan")],
        "pitch_control_pct": [3.0, 2.5, 4.0, 7.5],
        "scan_rate": [0.2, 0.3, 0.9, 0.5],
        "pre_reception_scans": [1.0, 1.5, 4.5, 2.5],
        "receptions": [40, 35, 80, 60],
        "retention_rate": [0.7, 0.75, 0.9, 0.95],
        "Cluster_Role": ["Space creator", "Lane-cutting presser", "Transition sprinter", "Scanner"],
        "market_value_m": [20.0, 60, 10, 30],
        "Predicted_Value": [40.0, 30, 25, 15],
    })


class FakeClient:
    """Records the request and streams back a canned answer, like the OpenAI SDK."""

    def __init__(self, reply: str = "Ada Runner.", fail: bool = False):
        self.requests: list[dict] = []
        self.reply, self.fail = reply, fail
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("network down")
        return iter([SimpleNamespace(choices=[]),                      # Azure-style empty chunk
                     *[SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=word))])
                       for word in self.reply.split(" ")]])


def test_context_carries_the_table_definitions_and_exact_rankings():
    agent = OffBallAgent(league(), client=FakeClient())
    assert "Ada Runner" in agent.context and "decoy_run_value_p90" in agent.context
    assert "LOWER is better" in agent.context
    # The exact ranking the example question needs is precomputed: Ada first among forwards.
    line = next(l for l in agent.context.splitlines() if l.startswith("Decoy Run Value per 90 (space created) - Forward"))
    assert line.split(": ", 1)[1].startswith("1. Ada Runner")
    # Fastest transition first; a player who never sprinted (NaN) is left out, not ranked first.
    fastest = next(l for l in agent.context.splitlines() if l.startswith("Transition Latency in seconds (fastest first) - all"))
    assert fastest.split(": ", 1)[1].startswith("1. Cal Quick") and "Dan Deep" not in fastest


def test_query_sends_data_history_and_question_to_the_model():
    client = FakeClient("Ada Runner creates the most space.")
    agent = OffBallAgent(league(), client=client)
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    answer = agent.query("Which forward creates the most space with decoy runs?", history)
    assert answer.replace(" ", "") == "AdaRunnercreatesthemostspace."
    messages = client.requests[0]["messages"]
    assert messages[0]["role"] == "system" and "PLAYER TABLE" in messages[0]["content"]
    assert [m["content"] for m in messages[1:]] == ["Hi", "Hello", "Which forward creates the most space with decoy runs?"]
    assert client.requests[0]["stream"] is True


def test_offline_answers_the_example_question_from_the_data():
    agent = OffBallAgent(league(), offline=True)
    assert agent.mode.startswith("offline")
    answer = agent.query("Which forward creates the most space with decoy runs?")
    assert "Offline answer" in answer
    assert answer.index("Ada Runner") < answer.index("Ben Presser")         # forwards only, best first
    assert "Cal Quick" not in answer


def test_failed_model_call_falls_back_to_offline_ranking():
    agent = OffBallAgent(league(), client=FakeClient(fail=True))
    answer = agent.query("Who is the fastest in transition?")
    assert "model call failed" in answer and answer.index("Cal Quick") < answer.index("Ada Runner")


def test_bargain_question_uses_ml_price_against_market_value():
    answer = OffBallAgent(league(), offline=True).query("Who is the biggest bargain?")
    assert answer.index("Cal Quick") < answer.index("Ada Runner")          # 25/10 beats 40/20


def test_config_reads_foundry_endpoint_and_hides_the_key(tmp_path, monkeypatch):
    for key in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT",
                "GITHUB_TOKEN", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / ".env"
    env.write_text("AZURE_OPENAI_ENDPOINT=https://demo.services.ai.azure.com/api/projects/p1/openai/v1/responses\n"
                   "AZURE_OPENAI_API_KEY=secret-key-123\nAZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini\n")
    config = ModelConfig.from_env(env)
    assert config.provider == "Microsoft Foundry" and config.model == "gpt-4.1-mini"
    assert config.base_url == "https://demo.services.ai.azure.com/api/projects/p1/openai/v1/"
    assert "secret-key-123" not in repr(config)
    assert ModelConfig.from_env(tmp_path / "missing.env") is None


def test_context_and_prompt_cover_the_scanning_pillar():
    agent = OffBallAgent(league(), client=FakeClient())
    assert "scans per second in the 5 s before receiving" in agent.context
    assert "retention_rate" in agent.context and "scans_before_receiving" in agent.context
    line = next(l for l in agent.context.splitlines() if l.startswith("Scanning rate") and "all positions" in l)
    assert line.split(": ", 1)[1].startswith("1. Cal Quick")
    system = agent.messages("hi")[0]["content"]
    assert "four pillars" in system and "scanning" in system


def test_offline_scanning_questions():
    agent = OffBallAgent(league(), offline=True)
    answer = agent.query("Which forward has the best awareness before receiving?")
    assert "Scanning rate" in answer and answer.index("Ben Presser") < answer.index("Ada Runner")
    assert "Scanning rate" in agent.query("Who checks their shoulder and scans the most?")
    assert "Ball retention" in agent.query("Which player keeps the ball best after receiving?")
    # "ahead" is not "head"; "looking for a cheap" is a price question, not scanning.
    assert "Scanning" not in agent.query("Who gets ahead fastest in transition?")
    assert "ML price" in agent.query("I'm looking for a cheap midfielder")


def test_offline_says_when_scanning_data_is_missing():
    agent = OffBallAgent(league().drop(columns=["scan_rate"]), offline=True)
    answer = agent.query("Who scans the most?")
    assert "no Scanning rate" in answer and "Off-Ball Value" in answer


def test_header_questions_are_not_scanning_and_empty_rankings_say_so():
    agent = OffBallAgent(league(), offline=True)
    assert "Scanning" not in agent.query("Best header of the ball?")
    assert "Scanning rate" in agent.query("Who makes the most head checks?")
    blank = OffBallAgent(league().assign(scan_rate=float("nan")), offline=True)
    lines = [l for l in blank.rankings_text().splitlines() if l.startswith("Scanning rate")]
    assert lines == ["Scanning rate (scans per second before receiving): (no data yet)"]
