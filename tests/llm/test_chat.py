import pytest

from mtva_bench.llm.http import LLMError, SSEEvent
from mtva_bench.llm.protocols import chat
from mtva_bench.llm.types import Delta, Message, Request, Tool, ToolCall


def test_encode_round_trips_tool_calls_and_provider_data() -> None:
    request = Request(
        messages=[
            Message(role="system", content="You are an agent."),
            Message(role="user", content="Hi"),
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="lookup",
                        arguments='{"id": 7}',
                        provider_data={"extra_content": {"google": {"thought_signature": "s"}}},
                    )
                ],
            ),
            Message(role="tool", tool_call_id="c1", content='{"ok": true}'),
        ],
        tools=[Tool(name="lookup", description="Find a record")],
        json_schema={"type": "object"},
        seed=3,
    )
    body = chat.encode(request, "gpt-5.2", stream=True)
    assert body["model"] == "gpt-5.2"
    assert body["seed"] == 3
    assert body["stream_options"] == {"include_usage": True}
    assert body["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert body["tools"][0]["function"]["name"] == "lookup"
    assert body["messages"][2] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"id": 7}'},
                "extra_content": {"google": {"thought_signature": "s"}},
            }
        ],
    }
    assert body["messages"][3] == {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true}'}


def test_decode_body() -> None:
    output = chat.decode(
        {
            "model": "gpt-5.2-2026-01-01",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "end_call", "arguments": "{}"},
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 8,
                "prompt_tokens_details": {"cached_tokens": 100},
            },
        }
    )
    assert output.text is None
    assert output.tool_calls[0].name == "end_call"
    assert output.tool_calls[0].provider_data == {}
    assert output.usage.cache_read_tokens == 100
    assert output.model == "gpt-5.2-2026-01-01"


def feed(decoder: chat.StreamDecoder, chunks: list[dict]) -> list:
    return [decoder.feed(SSEEvent(None, chunk)) for chunk in chunks]


def test_stream_decoder_openai_chunks() -> None:
    decoder = chat.StreamDecoder()
    deltas = feed(
        decoder,
        [
            {"model": "m", "choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "One "}}]},
            {"choices": [{"delta": {"content": "moment."}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": '{"id"'},
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ": 7}"}}]}}
                ]
            },
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 9}},
        ],
    )
    assert deltas == [
        None,
        Delta("One "),
        Delta("moment."),
        Delta(tool_call=True),
        Delta(tool_call=True),
        None,
        None,
    ]
    output = decoder.output()
    assert output.text == "One moment."
    assert [(c.id, c.name, c.arguments) for c in output.tool_calls] == [
        ("c1", "lookup", '{"id": 7}')
    ]
    assert output.finish_reason == "tool_calls"
    assert output.usage.output_tokens == 9


def test_stream_decoder_gemini_calls_without_index() -> None:
    decoder = chat.StreamDecoder()
    feed(
        decoder,
        [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "id": "a",
                                    "function": {"name": "first", "arguments": "{}"},
                                    "extra_content": {"google": {"thought_signature": "x"}},
                                },
                                {"id": "b", "function": {"name": "second", "arguments": "{}"}},
                            ]
                        }
                    }
                ]
            }
        ],
    )
    calls = decoder.output().tool_calls
    assert [call.name for call in calls] == ["first", "second"]
    assert calls[0].provider_data == {"extra_content": {"google": {"thought_signature": "x"}}}


def test_stream_error_chunk_is_retryable() -> None:
    with pytest.raises(LLMError) as raised:
        chat.StreamDecoder().feed(SSEEvent(None, {"error": {"message": "upstream overloaded"}}))
    assert raised.value.retryable


def test_reply_cap_in_every_protocol() -> None:
    from mtva_bench.llm.protocols import messages, responses

    request = Request(messages=[Message(role="user", content="Hi")], max_output_tokens=4096)
    assert chat.encode(request, "m", stream=False)["max_tokens"] == 4096
    assert responses.encode(request, "m", stream=False)["max_output_tokens"] == 4096
    assert messages.encode(request, "m", stream=False)["max_tokens"] == 4096
    assert "max_tokens" not in chat.encode(Request(messages=request.messages), "m", stream=False)


def test_context_limit_errors() -> None:
    from mtva_bench.llm.http import context_limit

    vllm = "This model's maximum context length is 32768 tokens. However, you requested ..."
    assert context_limit(LLMError(f"HTTP 400 from x: {vllm}", status=400))
    assert context_limit(LLMError("HTTP 400: prompt is too long: 210000 tokens", status=400))
    assert not context_limit(LLMError("HTTP 400: invalid tool schema", status=400))
    assert not context_limit(LLMError("HTTP 503: maximum context length", status=503))
