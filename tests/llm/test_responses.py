import pytest

from mtva_bench.llm.http import LLMError, SSEEvent
from mtva_bench.llm.protocols import responses
from mtva_bench.llm.types import Message, Request, Tool, ToolCall

FINAL = {
    "model": "gpt-5.6-sol",
    "status": "completed",
    "output": [
        {"type": "message", "content": [{"type": "output_text", "text": "Let me check."}]},
        {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": '{"id": 7}'},
    ],
    "usage": {
        "input_tokens": 100,
        "output_tokens": 12,
        "input_tokens_details": {"cached_tokens": 64},
    },
}


def test_encode_flattens_tools_and_tool_turns() -> None:
    request = Request(
        messages=[
            Message(role="system", content="Agent prompt"),
            Message(role="user", content="Hi"),
            Message(
                role="assistant", tool_calls=[ToolCall(id="c1", name="lookup", arguments="{}")]
            ),
            Message(role="tool", tool_call_id="c1", content="found"),
            Message(role="assistant", provider_data={"responses": FINAL["output"]}),
        ],
        tools=[Tool(name="lookup", description="Find")],
        json_schema={"type": "object"},
        seed=4,
    )
    body = responses.encode(request, "gpt-5.6-sol", stream=True)
    assert body["input"][:4] == [
        {"role": "system", "content": "Agent prompt"},
        {"role": "user", "content": "Hi"},
        {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "found"},
    ]
    assert body["input"][4:] == FINAL["output"]
    assert body["tools"][0] == {
        "type": "function",
        "name": "lookup",
        "description": "Find",
        "parameters": {"type": "object", "properties": {}},
    }
    assert body["text"]["format"]["schema"] == {"type": "object"}
    assert "seed" not in body


def test_decode_completed_and_incomplete() -> None:
    output = responses.decode(FINAL)
    assert output.text == "Let me check."
    assert output.tool_calls[0].id == "c1"
    assert output.finish_reason == "tool_calls"
    assert output.usage.cache_read_tokens == 64
    truncated = responses.decode(
        {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [],
        }
    )
    assert truncated.finish_reason == "length"
    with pytest.raises(LLMError):
        responses.decode({"status": "failed", "error": {"code": "server_error"}})


def test_stream_decoder_uses_final_response() -> None:
    decoder = responses.StreamDecoder()
    stream = [
        {"type": "response.created", "response": {}},
        {"type": "response.output_text.delta", "delta": "Let me "},
        {"type": "response.output_text.delta", "delta": "check."},
        {"type": "response.output_item.added", "item": {"type": "function_call", "call_id": "c1"}},
        {"type": "response.function_call_arguments.delta", "delta": "{}"},
        {"type": "response.completed", "response": FINAL},
    ]
    deltas = [decoder.feed(SSEEvent(item["type"], item)) for item in stream]
    assert [d for d in deltas if d] == [("Let me ", False), ("check.", False), ("", True)]
    assert decoder.output().tool_calls[0].name == "lookup"


def test_stream_without_final_response_is_retryable() -> None:
    with pytest.raises(LLMError) as raised:
        responses.StreamDecoder().output()
    assert raised.value.retryable


def test_cache_prefix_becomes_an_explicit_breakpoint() -> None:
    request = Request(
        messages=[
            Message(role="system", content="Shared rules.\nToday is Monday.", cache_prefix=14),
            Message(role="user", content="Hi"),
        ]
    )
    body = responses.encode(request, "gpt-5.6-sol", stream=False)
    assert body["input"][0] == {
        "role": "system",
        "content": [
            {
                "type": "input_text",
                "text": "Shared rules.\n",
                "prompt_cache_breakpoint": {"mode": "explicit"},
            },
            {"type": "input_text", "text": "Today is Monday."},
        ],
    }
    assert body["input"][1] == {"role": "user", "content": "Hi"}
    usage = responses.decode_usage(
        {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 6}}
    )
    assert (usage.cache_read_tokens, usage.cache_write_tokens) == (4, 6)
