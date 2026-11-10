import json
from pathlib import Path

import pytest

from mtva_bench.config import ConfigError
from mtva_bench.llm.fake import FakeLLM
from mtva_bench.llm.types import Request, Timing, Usage
from mtva_bench.run import RunStore, simulate_stage
from mtva_bench.run.latency import check_scored, latency_dir, latency_stage, scripts
from mtva_bench.run.score import score_stage
from mtva_bench.scoring.latency import speed
from mtva_bench.simulate.types import Conversation, ModelCall
from tests.test_run import make_bundle, make_config

LINES = ("I need to move my visit.", "Friday works.", "Thanks, bye.")


def recorded_caller(request: Request) -> str:
    """Each line in turn, from the caller's own history, and done on the last."""
    said = sum(message.role == "assistant" for message in request.messages)
    line = min(said, len(LINES) - 1)
    return json.dumps({"utterance": LINES[line], "beats": [], "done": line == len(LINES) - 1})


async def test_the_same_recorded_calls_are_replayed_for_any_model(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    reference = make_config(tmp_path, name="reference")
    agent = FakeLLM(lambda _: "Sure, one moment.")
    await simulate_stage(reference, bundle, agent, FakeLLM(recorded_caller))
    picked = scripts(bundle, RunStore(reference.run_dir))
    assert len(picked) == len(bundle.agents)
    assert picked == scripts(bundle, RunStore(reference.run_dir))

    measured = make_config(tmp_path)
    score_stage(reference, bundle)
    leaderboard = tmp_path / "leaderboard"
    leaderboard.mkdir()
    summary = json.loads((reference.run_dir / "summary.json").read_text())
    summary["run"] = measured.model.name
    (leaderboard / f"{measured.model.name}.json").write_text(json.dumps(summary))
    check_scored(measured.model, leaderboard)
    changed = measured.model.model_copy(update={"params": {"temperature": 0.1}})
    with pytest.raises(ConfigError, match="differs from the spec"):
        check_scored(changed, leaderboard)

    model = FakeLLM(lambda _: "Okay. Let me check that for you.")
    result = await latency_stage(
        measured.model,
        measured.benchmark,
        bundle,
        model,
        reference.run_dir,
        "test",
        tmp_path / "runs",
    )
    assert (result.calls, result.failed) == (len(picked), 0)
    [key] = picked
    store = RunStore(latency_dir(tmp_path / "runs", "test", measured.model.name))
    replayed = store.conversation(key).conversation
    heard = [m.content for m in replayed.messages if m.role == "user"]
    recorded = [f for turn in picked[key][1].conversation.caller_turns for f in turn.heard]
    assert heard[-len(recorded) :] == recorded
    assert len(replayed.agent_turns) == len(recorded) + 1
    assert (store.root / "result.json").is_file()

    again = FakeLLM(lambda _: "unused")
    await latency_stage(
        measured.model,
        measured.benchmark,
        bundle,
        again,
        reference.run_dir,
        "test",
        tmp_path / "runs",
    )
    assert again.requests == []


def test_speed_counts_tool_rounds_before_the_first_sentence() -> None:
    def call(turn: int, round_: int, first: float, sentence: float | None, total: float):
        return ModelCall(
            turn=turn,
            round=round_,
            timing=Timing(first_token_ms=first, first_sentence_ms=sentence, total_ms=total),
            usage=Usage(output_tokens=11),
            spoke=sentence is not None,
            tool_calls=0 if sentence is not None else 1,
            finish_reason="stop",
        )

    conversation = Conversation(
        scenario_id="s",
        agent_id="a",
        model="m",
        rollout=0,
        seed=0,
        messages=[],
        model_calls=[
            call(0, 0, 900, 1000, 1100),
            call(1, 0, 200, None, 300),
            call(1, 1, 150, 400, 600),
            call(2, 0, 300, 500, 800),
        ],
        end_reason="caller_done",
        duration_ms=1,
    )
    measured = speed([conversation])
    assert measured is not None
    assert (measured.ttft_cold_p50, measured.first_sentence_cold_p50) == (900, 1000)
    # Turn 1: a 300 ms tool round, then a sentence 400 ms into the next request.
    assert (measured.ttft_warm_p50, measured.first_sentence_warm_p50) == (250, 600)
    # 10 tokens after the first, over 200, 450 and 500 ms of generation.
    assert measured.tokens_per_second_p50 == 22.2
