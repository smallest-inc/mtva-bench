import json

import pytest

from mtva_bench.llm.http import LLMError, SSEEvent
from mtva_bench.llm.protocols import messages
from mtva_bench.llm.types import Message, Request, Tool, ToolCall


def test_encode_hoists_system_merges_turns_and_marks_cache() -> None:
    request = Request(
        messages=[
            Message(role="system", content="Judge preamble"),
            Message(role="system", content="Agent prompt", cache_breakpoint=True),
            Message(role="user", content="Transcript", cache_breakpoint=True),
            Message(role="user", content="Instructions"),
        ],
        tools=[Tool(name="lookup", parameters={"type": "object"})],
        json_schema={"type": "object"},
    )
    body = messages.encode(request, "claude-opus-5", stream=True)
    assert body["system"] == [
        {"type": "text", "text": "Judge preamble"},
        {"type": "text", "text": "Agent prompt", "cache_control": {"type": "ephemeral"}},
    ]
    assert body["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcript", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "Instructions"},
            ],
        }
    ]
    assert body["tools"] == [
        {"name": "lookup", "description": "", "input_schema": {"type": "object"}}
    ]
    assert body["output_config"] == {
        "format": {"type": "json_schema", "schema": {"type": "object"}}
    }
    assert body["stream"] is True
    assert "seed" not in body


def test_encode_tool_round_trip_and_replayed_blocks() -> None:
    replayed = [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "tool_use", "id": "t1", "name": "lookup", "input": {"id": 7}},
    ]
    request = Request(
        messages=[
            Message(role="user", content="Hi"),
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="t1", name="lookup", arguments='{"id": 7}')],
                provider_data={"messages": replayed},
            ),
            Message(role="tool", tool_call_id="t1", content="found"),
            Message(
                role="assistant",
                content="Done.",
                tool_calls=[ToolCall(id="t2", name="end_call", arguments="")],
            ),
        ]
    )
    turns = messages.encode(request, "m", stream=False)["messages"]
    assert turns[1]["content"] == replayed
    assert turns[1]["content"] is not replayed
    assert turns[2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "found"}],
    }
    assert turns[3]["content"] == [
        {"type": "text", "text": "Done."},
        {"type": "tool_use", "id": "t2", "name": "end_call", "input": {}},
    ]


def test_encode_rejects_late_system_and_too_many_breakpoints() -> None:
    with pytest.raises(ValueError, match="before the conversation"):
        messages.encode(
            Request(
                messages=[Message(role="user", content="a"), Message(role="system", content="b")]
            ),
            "m",
            stream=False,
        )
    many = [Message(role="user", content=str(i), cache_breakpoint=True) for i in range(5)]
    with pytest.raises(ValueError, match="5 cache breakpoints"):
        messages.encode(Request(messages=many), "m", stream=False)


def test_decode_body() -> None:
    output = messages.decode(
        {
            "model": "claude-opus-5",
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "Checking."},
                {"type": "tool_use", "id": "t1", "name": "lookup", "input": {"id": "७"}},
            ],
            "usage": {
                "input_tokens": 20,
                "output_tokens": 10,
                "cache_read_input_tokens": 9000,
                "cache_creation_input_tokens": 0,
            },
        }
    )
    assert output.text == "Checking."
    assert output.tool_calls[0].arguments == '{"id": "७"}'
    assert output.finish_reason == "tool_calls"
    assert output.usage.cache_read_tokens == 9000
    assert output.usage.input_tokens == 9020
    assert output.provider_data["messages"][1]["name"] == "lookup"


def events(*items: dict) -> list[SSEEvent]:
    return [SSEEvent(item["type"], item) for item in items]


def test_stream_decoder() -> None:
    decoder = messages.StreamDecoder()
    deltas = [
        decoder.feed(event)
        for event in events(
            {
                "type": "message_start",
                "message": {
                    "model": "claude-opus-5",
                    "usage": {"input_tokens": 30, "cache_creation_input_tokens": 9000},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Hmm"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "s"},
            },
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "One moment."},
            },
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {"type": "tool_use", "id": "t1", "name": "lookup", "input": {}},
            },
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "input_json_delta", "partial_json": '{"id":'},
            },
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "input_json_delta", "partial_json": " 7}"},
            },
            {"type": "content_block_stop", "index": 2},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 42},
            },
            {"type": "message_stop"},
        )
    ]
    assert [d for d in deltas if d] == [
        ("One moment.", False),
        ("", True),
        ("", True),
        ("", True),
    ]
    output = decoder.output()
    assert output.text == "One moment."
    assert json.loads(output.tool_calls[0].arguments) == {"id": 7}
    assert output.usage.output_tokens == 42
    assert output.usage.cache_write_tokens == 9000
    assert output.provider_data["messages"][0] == {
        "type": "thinking",
        "thinking": "Hmm",
        "signature": "s",
    }


def test_stream_error_event_is_retryable() -> None:
    with pytest.raises(LLMError) as raised:
        messages.StreamDecoder().feed(
            SSEEvent("error", {"type": "error", "error": {"type": "overloaded_error"}})
        )
    assert raised.value.retryable


def test_cache_prefix_splits_the_message() -> None:
    request = Request(
        messages=[
            Message(role="system", content="Shared rules.\nToday is Monday.", cache_prefix=14),
            Message(role="user", content="Hi", cache_breakpoint=True),
        ]
    )
    body = messages.encode(request, "claude-opus-5", stream=False)
    assert body["system"] == [
        {"type": "text", "text": "Shared rules.\n", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "Today is Monday."},
    ]
    assert body["messages"][0]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_one_hour_cache_marks() -> None:
    request = Request(
        messages=[
            Message(role="system", content="Shared rules.\nToday is Monday.", cache_prefix=14),
            Message(role="user", content="Call", cache_breakpoint=True),
        ],
        cache_ttl="1h",
    )
    body = messages.encode(request, "claude", stream=False)
    marks = [
        block["cache_control"]
        for block in body["system"] + body["messages"][0]["content"]
        if "cache_control" in block
    ]
    assert marks == [{"type": "ephemeral", "ttl": "1h"}] * 2
