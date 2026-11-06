import json
from pathlib import Path

from mtva_bench.config import ModelSpec
from mtva_bench.data import load_bundle
from mtva_bench.data.schema import Channel, Scenario
from mtva_bench.llm.fake import FakeLLM, scripted
from mtva_bench.llm.http import LLMError
from mtva_bench.llm.types import Output, Request, ToolCall
from mtva_bench.simulate import derive_seed, simulate
from mtva_bench.simulate.agent import MAX_TOOL_ROUNDS
from mtva_bench.simulate.channel import deliver
from mtva_bench.simulate.tools import MockTools, argument_errors

BUNDLE = load_bundle(Path(__file__).parent / "fixtures" / "bundle")
SCENARIO = BUNDLE.scenarios["demo_dental_01"]
AGENT = BUNDLE.agent(SCENARIO)
SPEC = ModelSpec(name="demo", provider="openai", model="demo", serving={"kind": "api"})


def call(name: str, arguments: dict, call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))


def caller(utterance: str, beats: list[str], done: bool = False) -> str:
    return json.dumps({"utterance": utterance, "beats": beats, "done": done})


def with_tools(scenario: Scenario, tools: dict) -> Scenario:
    return scenario.model_validate({**scenario.model_dump(), "tools": tools})


def test_mock_tools_answer_by_rule() -> None:
    mock = MockTools(AGENT, SCENARIO)
    assert mock.call("book_visit", '{"phone": "98765", "date": "2026-08-21"}') == (
        {"status": "unavailable", "note": "no slot for that date"},
        None,
    )
    reply, rule = mock.call("book_visit", '{"phone": "+91 98765-43210", "date": "21/08/2026"}')
    assert (rule, reply["status"]) == ("book_friday", "booked")
    assert mock.call("book_visit", "{not json")[1] is None
    assert mock.call("book_visit", '{"phone": "9876543210"}') == (
        {"error": "Invalid arguments: Missing required parameter: 'date'"},
        None,
    )
    assert "no tool named" in mock.call("cancel_visit", "{}")[0]["error"]
    assert mock.is_terminal("end_call") and not mock.is_terminal("book_visit")


def test_first_replies_and_requires() -> None:
    tools = SCENARIO.model_dump()["tools"]
    tools["book_visit"]["rules"][0]["first_replies"] = [{"status": "system busy"}]
    tools["end_call"]["rules"] = [
        {"id": "after_booking", "when": {}, "requires": ["book_friday"], "reply": {"ok": True}}
    ]
    mock = MockTools(AGENT, with_tools(SCENARIO, tools))
    assert mock.call("end_call", "{}") == ({"status": "ended"}, None)
    booking = '{"phone": "9876543210", "date": "2026-08-21"}'
    assert mock.call("book_visit", booking)[0] == {"status": "system busy"}
    assert mock.call("book_visit", booking)[0]["status"] == "booked"
    assert mock.call("book_visit", booking)[0]["status"] == "booked"
    assert mock.call("end_call", "{}") == ({"ok": True}, "after_booking")


def test_channel() -> None:
    assert deliver("Is Friday free?", Channel(), 0, 1, []) == (["Is Friday free?"], [])
    light, edits = deliver("Is Friday free for cleaning?", Channel(asr="light"), 0, 2, ["Friday"])
    assert edits[0].type == "drop" and edits[0].source != "Friday" and "Friday" in light[0]
    authored = Channel.model_validate(
        {
            "asr": "authored",
            "vad": "natural_splits",
            "events": [{"type": "substitute", "turn": 1, "source": "21", "target": "31"}],
        }
    )
    heard, edits = deliver("It's the 21st <pause> in the morning", authored, 1, 0, [])
    assert heard == ["It's the 31st", "in the morning"]
    assert edits[0].applied


async def test_fake_call_end_to_end() -> None:
    agent_llm = FakeLLM(
        scripted(
            [
                "Namaste, MetroMolar se Priya bol rahi hoon.",
                Output(
                    tool_calls=[call("book_visit", {"phone": "98765 43210", "date": "2026-08-21"})]
                ),
                "Friday, 21 August, 11:30 AM booked hai.",
                Output(text="Dhanyavaad!", tool_calls=[call("end_call", {}, "c2")]),
            ]
        )
    )
    caller_llm = FakeLLM(
        scripted(
            [
                caller("  ", ["b1"]),
                caller("Friday ko cleaning ho sakti hai?", ["b1", "b9"]),
                caller("Theek hai, thank you.", ["b1", "b3"], done=True),
            ]
        )
    )
    conversation = await simulate(
        agent=AGENT,
        scenario=SCENARIO,
        model=agent_llm,
        model_spec=SPEC,
        caller=caller_llm,
        voice_guide=BUNDLE.voice_guide,
        max_turns=5,
        seed=0,
    )
    assert conversation.end_reason == "terminal_tool", conversation.error
    assert [turn.text for turn in conversation.agent_turns] == [
        "Namaste, MetroMolar se Priya bol rahi hoon.",
        "Friday, 21 August, 11:30 AM booked hai.",
        "Dhanyavaad!",
    ]
    assert [(event.name, event.rule) for event in conversation.tool_events] == [
        ("book_visit", "book_friday"),
        ("end_call", None),
    ]
    assert [turn.beats for turn in conversation.caller_turns] == [["b1"], ["b3"]]
    assert [call.turn for call in conversation.caller_calls] == [0, 0, 1]
    roles = [message.role for message in conversation.messages]
    turn_1 = ["user", "assistant", "tool", "assistant"]
    turn_2 = ["user", "assistant", "tool"]
    assert roles == ["system", "assistant", *turn_1, *turn_2]
    assert all(turn.latency_ms is not None for turn in conversation.agent_turns)
    breakpoints = [[m.cache_breakpoint for m in r.messages] for r in agent_llm.requests]
    assert all(marks == [False] * (len(marks) - 1) + [True] for marks in breakpoints)
    assert not any(m.cache_breakpoint for m in conversation.messages)
    caller_view = caller_llm.requests[-1].messages
    assert "book_visit" not in " ".join(message.content or "" for message in caller_view)
    assert "Today is Wednesday, August 19, 2026" in conversation.messages[0].content


async def test_caller_done_and_opening_turn() -> None:
    spec = SPEC.model_copy(update={"needs_opening_user_turn": True})
    conversation = await simulate(
        agent=AGENT,
        scenario=SCENARIO,
        model=FakeLLM(scripted(["Hello?", "Okay, goodbye."])),
        model_spec=spec,
        caller=FakeLLM(scripted([caller("Never mind, bye.", [], done=True)])),
        voice_guide=None,
        max_turns=5,
        seed=0,
    )
    assert conversation.end_reason == "caller_done"
    assert conversation.messages[1].role == "user"
    assert conversation.messages[1].content == "[system: call started]"


async def test_concluding_message_for_models_without_text_and_tools() -> None:
    def respond(request: Request) -> Output:
        end_call = next(tool for tool in request.tools if tool.name == "end_call")
        assert "concluding_message" in end_call.parameters["required"]
        return Output(tool_calls=[call("end_call", {"concluding_message": "Sorry, goodbye."})])

    conversation = await simulate(
        agent=AGENT,
        scenario=SCENARIO,
        model=FakeLLM(respond),
        model_spec=SPEC.model_copy(update={"text_with_tool_calls": False}),
        caller=FakeLLM(scripted([])),
        voice_guide=None,
        max_turns=5,
        seed=0,
    )
    assert conversation.end_reason == "terminal_tool"
    assert conversation.agent_turns[0].text == "Sorry, goodbye."
    assert conversation.tool_events[0].arguments == "{}"


async def test_tool_loop_is_capped_and_failures_are_recorded() -> None:
    def respond(request: Request) -> Output:
        if request.tools:
            return Output(tool_calls=[call("book_visit", {})])
        return Output(text="Let me help you another way.")

    capped = await simulate(
        agent=AGENT,
        scenario=SCENARIO,
        model=FakeLLM(respond),
        model_spec=SPEC,
        caller=FakeLLM(scripted([caller("Hello?", [], done=True)])),
        voice_guide=None,
        max_turns=1,
        seed=0,
    )
    opening = [c for c in capped.model_calls if c.turn == 0]
    assert len(opening) == MAX_TOOL_ROUNDS + 1
    assert capped.agent_turns[0].text == "Let me help you another way."

    def fail(_: Request) -> Output:
        raise LLMError("upstream 500", retryable=True)

    failed = await simulate(
        agent=AGENT,
        scenario=SCENARIO,
        model=FakeLLM(fail),
        model_spec=SPEC,
        caller=FakeLLM(scripted([])),
        voice_guide=None,
        max_turns=1,
        seed=0,
    )
    assert (failed.end_reason, failed.error) == ("error", "LLMError: upstream 500")


def test_seeds_are_stable_and_differ_by_rollout() -> None:
    assert derive_seed(0, "demo_dental_01", 0) == derive_seed(0, "demo_dental_01", 0)
    assert derive_seed(0, "demo_dental_01", 0) != derive_seed(0, "demo_dental_01", 1)


def test_argument_errors_match_electron_core() -> None:
    parameters = {
        "type": "object",
        "properties": {
            "role": {"type": "string", "enum": ["member", "other"]},
            "count": {"type": "integer"},
        },
        "required": ["role"],
    }
    assert argument_errors(parameters, {"role": "member", "count": 2}) is None
    assert argument_errors(parameters, {"role": "son"}) == (
        "Parameter 'role' value 'son' not in allowed values: ['member', 'other']"
    )
    assert argument_errors(parameters, {"count": "2"}) == (
        "Missing required parameter: 'role'; Parameter 'count' expected type 'integer', got 'str'"
    )


async def test_a_call_that_outgrows_the_context_window_ends_and_is_kept() -> None:
    limit = LLMError(
        "HTTP 400 from x: This model's maximum context length is 32768 tokens.", status=400
    )
    replies = [Output(text="Namaste!", model="demo-2026-10-01")]

    def respond(request: Request) -> Output:
        assert request.max_output_tokens == 4096
        if not replies:
            raise limit
        return replies.pop(0)

    conversation = await simulate(
        agent=AGENT,
        scenario=SCENARIO,
        model=FakeLLM(respond),
        model_spec=SPEC,
        caller=FakeLLM(scripted([caller("Hello?", ["b1"])])),
        voice_guide=BUNDLE.voice_guide,
        max_turns=5,
        seed=0,
        max_reply_tokens=4096,
    )
    assert conversation.end_reason == "context_limit" and conversation.error is None
    assert [turn.text for turn in conversation.agent_turns] == ["Namaste!", ""]
    assert conversation.model_calls[0].model == "demo-2026-10-01"


def test_calls_saved_before_the_served_model_was_recorded_still_load() -> None:
    from mtva_bench.simulate.types import ModelCall

    old = {
        "turn": 1,
        "round": 0,
        "timing": {"total_ms": 5.0},
        "usage": {},
        "spoke": True,
        "tool_calls": 0,
        "finish_reason": "stop",
    }
    assert ModelCall.model_validate(old).model is None
