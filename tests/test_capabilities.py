import asyncio
import json
from pathlib import Path

import pytest

from mtva_bench.capabilities import conduct
from mtva_bench.capabilities.conduct import VerdictError, validate
from mtva_bench.capabilities.conduct.checks import check_actions
from mtva_bench.capabilities.conduct.transcript import transcript
from mtva_bench.config import ModelSpec
from mtva_bench.data import load_bundle
from mtva_bench.data.schema import KeyAction
from mtva_bench.llm.fake import FakeLLM, scripted
from mtva_bench.llm.http import LLMError
from mtva_bench.llm.protocols import messages as anthropic
from mtva_bench.llm.types import (
    BatchStatus,
    Message,
    Output,
    Request,
    Response,
    Timing,
    ToolCall,
    Usage,
)
from mtva_bench.run import RunStore, StaleRunError, evaluate_stage
from mtva_bench.run.evaluate import JUDGE, call_inputs, judge_inputs
from mtva_bench.run.stages import plan
from mtva_bench.run.store import BatchState
from mtva_bench.simulate import simulate
from mtva_bench.simulate.call import OPENING_TURN
from mtva_bench.simulate.types import Conversation, ToolEvent
from tests.test_run import hang_up, make_bundle, make_config, run

BUNDLE = load_bundle(Path(__file__).parent / "fixtures" / "bundle")
SCENARIO = BUNDLE.scenarios["demo_dental_01"]
AGENT = BUNDLE.agent(SCENARIO)
SPEC = ModelSpec(
    name="fake", provider="custom", model="fake", base_url="http://x", serving={"kind": "api"}
)


def event(name: str, arguments: dict, turn: int = 1) -> ToolEvent:
    return ToolEvent(
        turn=turn,
        call_id=f"{name}-{turn}",
        name=name,
        arguments=json.dumps(arguments),
        rule=None,
        reply={},
        terminal=name == "end_call",
    )


def answer(**changes: object) -> str:
    verdict = {
        "focus_rules": [
            {
                "id": "confirm_first",
                "evidence": "read back",
                "turns": [2],
                "verdict": "followed",
                "severity": None,
            }
        ],
        "key_actions": [{"id": "book", "verdict": "met", "evidence": "booked"}],
        "findings": [
            {
                "capability": "instruction_following",
                "turn": 2,
                "severity": "minor",
                "quote": "Always confirm the date and time back to the caller before booking.",
                "what_happened": "confirmed the date but not the time",
            },
            {
                "capability": "faithfulness",
                "turn": 2,
                "severity": "major",
                "quote": "booked for Saturday",
                "what_happened": "said a day it did not book",
            },
        ],
        "task_completion": {
            "caller_heard": [{"item": 1, "heard": "yes", "turn": 2}],
            "evidence": "booked",
            "outcome": "reached",
        },
        "simulation_problem": None,
        "summary": "Booked the visit.",
        **changes,
    }
    return json.dumps(verdict)


async def booked_call() -> Conversation:
    agent_llm = FakeLLM(
        scripted(
            [
                "Namaste, MetroMolar se Priya bol rahi hoon.",
                Output(
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            name="book_visit",
                            arguments='{"phone": "98765 43210", "date": "2026-08-21"}',
                        )
                    ]
                ),
                "Friday, 21 August, 11:30 AM booked hai.",
                Output(
                    text="Dhanyavaad!",
                    tool_calls=[ToolCall(id="c2", name="end_call", arguments="{}")],
                ),
            ]
        )
    )
    caller_llm = FakeLLM(
        scripted(
            [
                json.dumps({"utterance": "Friday ko cleaning?", "beats": ["b1"], "done": False}),
                json.dumps({"utterance": "Thank you.", "beats": ["b3"], "done": True}),
            ]
        )
    )
    return await simulate(
        agent=AGENT,
        scenario=SCENARIO,
        model=agent_llm,
        model_spec=SPEC.model_copy(update={"needs_opening_user_turn": True}),
        caller=caller_llm,
        voice_guide=BUNDLE.voice_guide,
        max_turns=5,
        seed=0,
    )


def test_key_action_checks() -> None:
    def status(events: list[ToolEvent], scenario=SCENARIO) -> list[tuple[str, int | None]]:
        return [(r.status, r.turn) for r in check_actions(AGENT, scenario, events)]

    booked = event("book_visit", {"phone": "+91 98765 43210", "date": "21 August 2026"}, turn=2)
    assert status([booked]) == [("met", 3)]
    assert status([event("book_visit", {"phone": "9876543210", "date": "2026-08-22"})]) == [
        ("wrong_arguments", 2)
    ]
    rejected = check_actions(AGENT, SCENARIO, [event("book_visit", {"date": "2026-08-21"})])
    assert (rejected[0].status, rejected[0].rejected) == ("not_called", 1)

    never = KeyAction(
        id="no_hangup", tool="end_call", expect="never", importance=2, focus="f", note="n"
    )
    after = SCENARIO.evaluation.key_actions[0].model_copy(update={"id": "late", "after": ["book"]})
    evaluation = SCENARIO.evaluation.model_copy(
        update={"key_actions": [*SCENARIO.evaluation.key_actions, after, never]}
    )
    scenario = SCENARIO.model_copy(update={"evaluation": evaluation})
    assert status([booked], scenario) == [("met", 3), ("out_of_order", 3), ("met", None)]
    assert status([booked, booked, event("end_call", {}, turn=4)], scenario) == [
        ("met", 3),
        ("met", 3),
        ("called", 5),
    ]


async def test_transcript_numbers_agent_turns() -> None:
    conversation = await booked_call()
    assert conversation.messages[1].content == OPENING_TURN
    text, turns = transcript(conversation)
    assert turns == 3
    assert OPENING_TURN not in text
    assert text.splitlines()[:6] == [
        "Agent [1]:",
        "  says: Namaste, MetroMolar se Priya bol rahi hoon.",
        "Caller: Friday ko cleaning?",
        "Agent [2]:",
        '  → book_visit {"phone": "98765 43210", "date": "2026-08-21"}',
        '  ← book_visit: {"status": "booked", "booking_id": "BK-482913", "time": "11:30 AM"}',
    ]
    silent = conversation.model_copy(
        update={"messages": [*conversation.messages, Message(role="user", content="Hello?")]}
    )
    silent.messages.append(Message(role="assistant"))
    assert transcript(silent)[0].endswith("Caller: Hello?\nAgent [4]: (said nothing)")


async def test_judge_request_layout() -> None:
    conversation = await booked_call()
    checks = check_actions(AGENT, SCENARIO, conversation.tool_events)
    messages = conduct.request_messages(AGENT, SCENARIO, conversation, checks)
    body = anthropic.encode(Request(messages=messages, json_schema=conduct.SCHEMA), "m", True)
    blocks = body["messages"][0]["content"]
    assert "system" not in body and len(body["messages"]) == 1
    assert ["cache_control" in block for block in blocks] == [True, False]
    assert blocks[0]["text"].startswith("You are evaluating one simulated phone call")
    assert "\n# Your task" in blocks[1]["text"]
    shared = conversation.messages[0].cache_prefix
    prompt = conversation.messages[0].content
    assert blocks[0]["text"].endswith(prompt[:shared])
    assert blocks[1]["text"].startswith(prompt[shared:])
    assert "Today is" not in blocks[0]["text"]
    assert "[The call ended when the agent called end_call.]" in blocks[1]["text"]
    assert body["output_config"]["format"]["schema"] == conduct.SCHEMA
    assert conduct.SILENT_TOOLS not in blocks[1]["text"]
    silent = conduct.request_messages(AGENT, SCENARIO, conversation, checks, False)
    assert silent[0] == messages[0]
    assert conduct.SILENT_TOOLS in (silent[1].content or "")


async def test_verdicts_are_checked() -> None:
    conversation = await booked_call()
    verdict = validate(answer(), AGENT, SCENARIO, conversation, 3)
    assert [f.quote_found for f in verdict.findings] == [True, False]
    assert conduct.scores(verdict, SCENARIO) == {
        "instruction_following": 4,
        "tool_use": 5,
        "faithfulness": 3,
        "task_completion": 5,
    }
    with pytest.raises(VerdictError, match="severity goes with partly"):
        focus = {"id": "confirm_first", "evidence": "", "turns": [], "verdict": "broken"}
        validate(
            answer(focus_rules=[{**focus, "severity": None}]), AGENT, SCENARIO, conversation, 3
        )
    with pytest.raises(VerdictError, match="focus rules"):
        validate(answer(focus_rules=[]), AGENT, SCENARIO, conversation, 3)
    with pytest.raises(VerdictError, match=r"turns \[7\]"):
        heard = {"outcome": "reached", "caller_heard": [{"item": 1, "heard": "yes", "turn": 7}]}
        validate(
            answer(task_completion={**heard, "evidence": ""}), AGENT, SCENARIO, conversation, 3
        )
    with pytest.raises(VerdictError, match="schema"):
        validate("{}", AGENT, SCENARIO, conversation, 3)


async def test_a_step_missed_because_the_call_ended_sits_after_the_last_turn() -> None:
    conversation = await booked_call()
    focus = {"id": "confirm_first", "evidence": "", "verdict": "broken", "severity": "major"}
    after = validate(
        answer(focus_rules=[{**focus, "turns": [4]}]), AGENT, SCENARIO, conversation, 3
    )
    assert after.focus_rules[0].turns == [4]
    with pytest.raises(VerdictError, match=r"turns \[5\]"):
        validate(answer(focus_rules=[{**focus, "turns": [5]}]), AGENT, SCENARIO, conversation, 3)
    with pytest.raises(VerdictError, match=r"turns \[4\]"):
        heard = {"outcome": "reached", "caller_heard": [{"item": 1, "heard": "yes", "turn": 4}]}
        validate(
            answer(task_completion={**heard, "evidence": ""}), AGENT, SCENARIO, conversation, 3
        )


async def test_judge_retries_a_bad_answer_once() -> None:
    conversation = await booked_call()
    llm = FakeLLM(scripted([Output(text=answer(key_actions=[]), finish_reason="stop")] * 2))
    result = await conduct.judge(llm, AGENT, SCENARIO, conversation)
    assert result.verdict is None and "key actions" in result.error
    assert len(llm.requests) == 2

    good = Output(text=answer(), finish_reason="stop")
    llm = FakeLLM(scripted([Output(text="{", finish_reason="length"), good]))
    result = await conduct.judge(llm, AGENT, SCENARIO, conversation)
    assert result.error is None and result.verdict is not None
    assert "a matching call ran at Agent [2]" in llm.requests[0].messages[-1].content


def fake_judge(order: list[str]) -> FakeLLM:
    def respond(request: Request) -> Output:
        order.append(request.messages[1].content.split("Clinic ")[-1][:1])
        heard = {
            "outcome": "not_reached",
            "caller_heard": [{"item": 1, "heard": "no", "turn": None}],
        }
        return Output(
            text=answer(
                focus_rules=[
                    {
                        "id": "confirm_first",
                        "evidence": "",
                        "turns": [1],
                        "verdict": "broken",
                        "severity": "major",
                    }
                ],
                findings=[],
                task_completion={**heard, "evidence": ""},
            ),
            finish_reason="stop",
        )

    return FakeLLM(respond)


async def test_evaluate_stage_resumes_and_refuses_stale_verdicts(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    config = make_config(tmp_path)
    await run(config, bundle, hang_up())
    order: list[str] = []
    result = await evaluate_stage(config, bundle, fake_judge(order))
    assert (result.planned, result.judged, result.failed) == (3, 3, 0)
    store = RunStore(config.run_dir)
    assert (
        store.verdict(JUDGE, "demo_dental_02__r0").verdict.task_completion.outcome == "not_reached"
    )
    judging = store.manifest().judging
    assert (judging.planned, judging.judged, judging.failed) == (3, 3, 0)
    await run(config, bundle, hang_up())
    assert store.manifest().judging == judging

    again = await evaluate_stage(config, bundle, fake_judge(order))
    assert (again.skipped, again.judged) == (3, 0)

    stored = store.verdict(JUDGE, "demo_dental_01__r0")
    stored.inputs["prompt"] = "old"
    store.save_verdict("demo_dental_01__r0", stored)
    with pytest.raises(StaleRunError, match=r"judge prompt \(1\)"):
        await evaluate_stage(config, bundle, fake_judge(order))
    redone = await evaluate_stage(
        make_config(tmp_path, discard_stale=True), bundle, fake_judge(order)
    )
    assert redone.judged == 1

    # A field the judge never reads does not make a verdict stale; a changed call does.
    stored = store.conversation("demo_dental_03__r0")
    stored.conversation.duration_ms += 1
    store.save_conversation(stored)
    assert (await evaluate_stage(config, bundle, fake_judge(order))).judged == 0
    stored.conversation.messages[-1].content = "Goodbye now."
    store.save_conversation(stored)
    assert (await evaluate_stage(config, bundle, fake_judge(order))).judged == 1


async def test_a_grading_edit_rejudges_without_resimulating(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    config = make_config(tmp_path)
    await run(config, bundle, hang_up())
    await evaluate_stage(config, bundle, fake_judge([]))

    path = bundle.root / "scenarios" / "demo_dental" / "demo_dental_02.json"
    scenario = json.loads(path.read_text())
    scenario["evaluation"]["key_actions"][0]["note"] = "the booking is what the caller wants"
    path.write_text(json.dumps(scenario))
    edited = load_bundle(bundle.root)

    simulated = await run(config, edited, hang_up())
    assert simulated.simulated == 0
    with pytest.raises(StaleRunError, match=r"scenario grading \(1\)"):
        await evaluate_stage(config, edited, fake_judge([]))
    redone = await evaluate_stage(make_config(tmp_path, discard_stale=True), edited, fake_judge([]))
    assert (redone.skipped, redone.judged) == (2, 1)


async def test_verdicts_from_an_older_schema_are_stale(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    config = make_config(tmp_path)
    await run(config, bundle, hang_up())
    await evaluate_stage(config, bundle, fake_judge([]))
    store = RunStore(config.run_dir)
    path = store.verdict_path(JUDGE, "demo_dental_01__r0")
    old = json.loads(path.read_text())
    old["verdict"]["scores"] = {"tool_use": 5}
    old["inputs"]["schema"] = "old"
    path.write_text(json.dumps(old))
    with pytest.raises(StaleRunError, match=r"verdict schema \(1\)"):
        await evaluate_stage(config, bundle, fake_judge([]))
    redone = await evaluate_stage(make_config(tmp_path, discard_stale=True), bundle, fake_judge([]))
    assert redone.judged == 1


async def test_an_agents_first_verdict_comes_before_the_rest(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    config = make_config(tmp_path)
    await run(config, bundle, hang_up())
    events: list[str] = []
    judge = fake_judge([])

    class SlowFirst:
        async def stream(self, request: Request) -> Response:
            clinic = request.messages[1].content.split("Clinic ")[-1][:1]
            events.append(f"start {clinic}")
            if clinic == "A":
                await asyncio.sleep(0.05)
            events.append(f"end {clinic}")
            return await judge.stream(request)

    await evaluate_stage(config, bundle, SlowFirst())
    assert events[:2] == ["start A", "end A"]


def test_scores_from_verdicts() -> None:
    assert [
        conduct.severity_score(found)
        for found in (
            [],
            ["minor", "minor"],
            ["minor"] * 3,
            ["major"],
            ["major", "major"],
            ["critical", "minor"],
            ["critical", "critical"],
        )
    ] == [5, 4, 3, 3, 2, 2, 1]
    verdict = conduct.ConductVerdict.model_validate_json(
        answer(
            key_actions=[{"id": "book", "evidence": "", "verdict": "partly"}],
            findings=[],
            task_completion={
                "caller_heard": [{"item": 1, "heard": "partly", "turn": 2}],
                "evidence": "",
                "outcome": "alternative",
            },
        )
    )
    # `book` has importance 3: partly met counts one level lower, as major.
    assert conduct.scores(verdict, SCENARIO) == {
        "task_completion": 4,
        "instruction_following": 5,
        "tool_use": 3,
        "faithfulness": 5,
    }


class FakeBatches:
    """A batch API that answers each request with `answers[custom id]`, after one poll."""

    def __init__(self, answers: dict[str, Response | LLMError]):
        self.answers = answers
        self.submitted: list[dict[str, Request]] = []
        self.polls = 0

    async def submit(self, requests: dict[str, Request]) -> str:
        self.submitted.append(requests)
        return f"batch_{len(self.submitted)}"

    async def status(self, batch_id: str) -> BatchStatus:
        self.polls += 1
        ended = self.polls % 2 == 0
        return BatchStatus(id=batch_id, ended=ended, counts={"processing": 0 if ended else 3})

    async def results(self, batch_id: str) -> dict[str, Response | LLMError]:
        return dict(self.answers)


def batch_response(text: str) -> Response:
    return Response(
        text=text,
        finish_reason="stop",
        model="judge-1",
        usage=Usage(input_tokens=100, output_tokens=10, cache_read_tokens=80),
        timing=Timing(total_ms=0),
    )


def good_answer() -> str:
    """A valid answer for the one-turn fixture calls."""
    return answer(
        focus_rules=[
            {
                "id": "confirm_first",
                "evidence": "",
                "turns": [1],
                "verdict": "not_reached",
                "severity": None,
            }
        ],
        findings=[],
        task_completion={
            "caller_heard": [{"item": 1, "heard": "no", "turn": None}],
            "evidence": "",
            "outcome": "not_reached",
        },
    )


async def test_batch_judging(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    await run(make_config(tmp_path), bundle, hang_up())
    config = make_config(tmp_path, batch=True)
    batches = FakeBatches(
        {
            "demo_dental_01__r0": batch_response(good_answer()),
            "demo_dental_02__r0": LLMError("batch request errored", retryable=True),
            "demo_dental_03__r0": batch_response(answer(key_actions=[])),
        }
    )
    realtime = FakeLLM(lambda _: Output(text=good_answer(), finish_reason="stop"))
    result = await evaluate_stage(config, bundle, realtime, batch_api=batches, poll_seconds=0)

    assert (result.judged, result.failed, result.retried) == (3, 0, 2)
    [requests] = batches.submitted
    assert list(requests) == ["demo_dental_01__r0", "demo_dental_02__r0", "demo_dental_03__r0"]
    assert all(request.cache_ttl == "1h" for request in requests.values())
    assert len(realtime.requests) == 2
    store = RunStore(config.run_dir)
    assert store.batch_state(JUDGE) is None
    first = store.verdict(JUDGE, "demo_dental_01__r0")
    assert first.batch_id == "batch_1" and len(first.usage) == 1
    assert first.judge_model == "judge-1"
    invalid = store.verdict(JUDGE, "demo_dental_03__r0")
    assert invalid.batch_id == "batch_1" and len(invalid.usage) == 2
    assert store.verdict(JUDGE, "demo_dental_02__r0").batch_id is None
    assert result.batch_usage.input_tokens == 200


async def test_a_saved_batch_is_collected_not_paid_again(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    await run(make_config(tmp_path), bundle, hang_up())
    config = make_config(tmp_path, batch=True)
    batches = FakeBatches(
        {
            stem: batch_response(good_answer())
            for stem in ("demo_dental_01__r0", "demo_dental_02__r0", "demo_dental_03__r0")
        }
    )
    # A run killed after submitting batch_9 with two rollouts; the second has changed since.
    store = RunStore(config.run_dir)
    rollouts = {rollout.stem: rollout for rollout in plan(bundle, config)}
    inputs = {
        stem: call_inputs(
            judge_inputs(config.benchmark), rollouts[stem], store.conversation(stem), bundle
        )
        for stem in ("demo_dental_01__r0", "demo_dental_02__r0")
    }
    inputs["demo_dental_02__r0"]["conversation"] = "older"
    store.save_batch_state(JUDGE, BatchState(batch_id="batch_9", inputs=inputs))

    result = await evaluate_stage(config, bundle, FakeLLM(str), batch_api=batches, poll_seconds=0)
    assert result.judged == 3
    assert store.verdict(JUDGE, "demo_dental_01__r0").batch_id == "batch_9"
    [resubmitted] = batches.submitted
    assert sorted(resubmitted) == ["demo_dental_02__r0", "demo_dental_03__r0"]
    assert store.batch_state(JUDGE) is None


def refusing() -> FakeLLM:
    return FakeLLM(lambda _: Output(finish_reason="refusal", model="primary"))


def fallback_judge() -> FakeLLM:
    return FakeLLM(lambda _: Output(text=good_answer(), finish_reason="stop", model="fallback-1"))


async def test_a_refused_call_goes_to_the_fallback_judge(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    config = make_config(tmp_path)
    await run(config, bundle, hang_up())
    primary, fallback = refusing(), fallback_judge()
    result = await evaluate_stage(config, bundle, primary, fallback)
    assert (result.judged, result.failed, result.fallback) == (3, 0, 3)
    # A refusal is not retried on the judge that refused.
    assert len(primary.requests) == 3
    store = RunStore(config.run_dir)
    verdict = store.verdict(JUDGE, "demo_dental_01__r0")
    assert verdict.judge_model == "fallback-1" and "fallback" in verdict.inputs
    judging = store.manifest().judging
    assert (judging.fallback_model, judging.fallback_judged) == ("claude-opus-4-8", 3)

    again = await evaluate_stage(config, bundle, primary, fallback)
    assert (again.skipped, again.judged) == (3, 0)

    without = config.benchmark.model_copy(
        update={"judge": config.benchmark.judge.model_copy(update={"fallback": None})}
    )
    with pytest.raises(StaleRunError, match=r"fallback judge \(3\)"):
        await evaluate_stage(config.model_copy(update={"benchmark": without}), bundle, primary)

    unrefused = await evaluate_stage(config, bundle, refusing())
    assert unrefused.judged == 0


async def test_a_call_refused_in_the_batch_skips_the_judge_in_realtime(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    await run(make_config(tmp_path), bundle, hang_up())
    config = make_config(tmp_path, batch=True)
    refused = Response(finish_reason="refusal", timing=Timing(total_ms=0))
    batches = FakeBatches(
        {
            "demo_dental_01__r0": batch_response(good_answer()),
            "demo_dental_02__r0": refused,
            "demo_dental_03__r0": batch_response(good_answer()),
        }
    )
    primary, fallback = refusing(), fallback_judge()
    result = await evaluate_stage(
        config, bundle, primary, fallback, batch_api=batches, poll_seconds=0
    )
    assert (result.judged, result.failed, result.retried, result.fallback) == (3, 0, 1, 1)
    assert primary.requests == [] and len(fallback.requests) == 1
