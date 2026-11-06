import json
from pathlib import Path

import pytest

from mtva_bench import arena
from mtva_bench.llm.fake import FakeLLM
from mtva_bench.llm.types import Message, Output, Request, ToolCall
from mtva_bench.run import RunStore, StaleRunError
from mtva_bench.run.arena import (
    JUDGE,
    StoredArenaVerdict,
    arena_stage,
    pair_dir,
    summarize_arena,
)
from mtva_bench.scoring.arena import Outcome, arena_scores
from mtva_bench.simulate.call import OPENING_TURN
from mtva_bench.simulate.types import Conversation
from tests.test_run import hang_up, make_bundle, make_config, run


def test_the_judge_reads_only_what_was_said() -> None:
    conversation = Conversation(
        scenario_id="s",
        agent_id="a",
        model="m",
        rollout=0,
        seed=0,
        messages=[
            Message(role="system", content="secret prompt"),
            Message(role="user", content=OPENING_TURN),
            Message(role="assistant", content="Hello, City Clinic."),
            Message(role="user", content="I need to move my visit."),
            Message(
                role="assistant",
                content="One moment.",
                tool_calls=[ToolCall(id="1", name="find_visit", arguments="{}")],
            ),
            Message(role="tool", tool_call_id="1", content='{"visit": "Friday"}'),
            Message(role="assistant", content="It is on Friday."),
            Message(role="user", content="Hello?"),
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="2", name="end_call", arguments="{}")],
            ),
        ],
        end_reason="terminal_tool",
        duration_ms=1,
    )
    assert arena.spoken(conversation) == (
        "Agent: Hello, City Clinic.\n"
        "Caller: I need to move my visit.\n"
        "Agent: One moment. It is on Friday.\n"
        "Caller: Hello?\n"
        "Agent: (silence)"
    )


def test_each_run_is_shown_first_in_half_the_scenarios() -> None:
    stems = [f"s{i}__r0" for i in range(600)]
    first = arena.first_in_half(stems, seed=0)
    assert len(first) == 300
    assert first == arena.first_in_half(list(reversed(stems)), seed=0)
    assert first != arena.first_in_half(stems, seed=1)


def test_the_winner_follows_the_position() -> None:
    def stored(first: str, verdict: str) -> StoredArenaVerdict:
        return StoredArenaVerdict.model_validate(
            {
                "inputs": {},
                "first": first,
                "verdict": {"reasons": "", "verdict": verdict},
                "usage": [],
            }
        )

    assert stored("a", "A").winner == "a"
    assert stored("b", "A").winner == "b"
    assert stored("b", "B").winner == "a"
    assert stored("a", "tie").winner == "tie"


def test_rates_and_position_lean() -> None:
    outcomes = [
        Outcome("s1", "x", "Hindi", "a", "a"),
        Outcome("s2", "x", "Hindi", "a", "b"),
        Outcome("s3", "y", "Tamil", "tie", "a"),
        Outcome("s4", "y", "Tamil", "b", "b"),
    ]
    scores = arena_scores(outcomes)
    assert (scores.wins.value, scores.ties.value, scores.losses.value) == (50.0, 25.0, 25.0)
    assert scores.by_language["Tamil"].ties == 50.0
    assert scores.first_shown_won == round(200 / 3, 1)


def always(answer: str) -> FakeLLM:
    def respond(_: Request) -> Output:
        return Output(
            text=json.dumps({"reasons": "sounds natural", "verdict": answer}),
            finish_reason="stop",
        )

    return FakeLLM(respond)


async def test_arena_stage_judges_resumes_and_refuses_stale(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    a, b = make_config(tmp_path, name="a"), make_config(tmp_path, name="b")
    await run(a, bundle, hang_up())
    await run(b, bundle, hang_up())

    judge = always("A")
    result = await arena_stage(a, b, bundle, judge)
    assert (result.planned, result.judged, result.failed) == (3, 3, 0)
    request = judge.requests[0].messages[0].content or ""
    assert "Agent: Goodbye." in request
    assert "Clinic" not in request and "end_call" not in request

    summary = summarize_arena(a, b, bundle)
    assert summary.coverage.judged == 3
    assert summary.scores is not None
    # The judge always picks the call shown first, so each run wins where it was shown first.
    first_a = arena.first_in_half([f"demo_dental_0{i}__r0" for i in (1, 2, 3)], 0)
    assert summary.scores.wins.value == round(100 * len(first_a) / 3, 1)
    assert summary.scores.first_shown_won == 100.0

    again = always("A")
    assert (await arena_stage(a, b, bundle, again)).judged == 0
    assert again.requests == []

    store = RunStore(a.run_dir)
    stored = store.conversation("demo_dental_01__r0")
    stored.conversation.messages[-1].content = "Bye now."
    store.save_conversation(stored)
    with pytest.raises(StaleRunError, match=r"a \(1\)"):
        await arena_stage(a, b, bundle, always("A"))
    redone = await arena_stage(a, b, bundle, always("B"), discard_stale=True)
    assert redone.judged == 1
    pair = RunStore(pair_dir(tmp_path / "runs", "a", "b"))
    assert pair.verdict_path(JUDGE, "demo_dental_01__r0").is_file()
