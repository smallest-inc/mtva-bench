import json

import httpx
import pytest

from mtva_bench.llm import http
from mtva_bench.llm.client import LLMClient
from mtva_bench.llm.endpoints import Endpoint
from mtva_bench.llm.fake import FakeLLM, scripted
from mtva_bench.llm.http import LLMError
from mtva_bench.llm.types import Message, Output, Request

REQUEST = Request(messages=[Message(role="user", content="Hello")], seed=11)

BODY = {
    "model": "gpt-5.2",
    "choices": [{"finish_reason": "stop", "message": {"content": "Hi there"}}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
}


def sse(*chunks: dict) -> bytes:
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks).encode() + (
        b"data: [DONE]\n\n"
    )


def client_for(handler, monkeypatch, **endpoint_fields) -> LLMClient:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    endpoint = Endpoint(provider="openai", model="gpt-5.2", **endpoint_fields)
    return LLMClient(endpoint, transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch) -> None:
    async def sleep(_: float) -> None:
        return None

    monkeypatch.setattr(http.asyncio, "sleep", sleep)


async def test_complete_merges_params_and_drops_unsupported(monkeypatch) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=BODY)

    client = client_for(
        handler,
        monkeypatch,
        params={"temperature": 0.7, "reasoning_effort": "none"},
        unsupported=["seed"],
    )
    response = await client.complete(REQUEST)
    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer sk-test"
    assert body["temperature"] == 0.7
    assert body["reasoning_effort"] == "none"
    assert "seed" not in body
    assert response.text == "Hi there"
    assert response.timing.first_token_ms is None
    assert response.timing.total_ms >= 0


async def test_stream_records_token_speech_and_sentence_times(monkeypatch) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse(
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1"}]}}]},
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "x"}}]}}]},
                {"choices": [{"delta": {"content": "Sure. "}}]},
                {"choices": [{"delta": {"content": "One moment"}, "finish_reason": "stop"}]},
            ),
            headers={"content-type": "text/event-stream"},
        )

    response = await client_for(handler, monkeypatch).stream(REQUEST)
    timing = response.timing
    assert response.text == "Sure. One moment"
    assert response.tool_calls[0].name == "x"
    assert timing.first_token_ms <= timing.first_speech_token_ms <= timing.first_sentence_ms
    assert timing.first_sentence_ms <= timing.total_ms


async def test_a_stream_cut_off_mid_event_is_retried(monkeypatch) -> None:
    replies = [
        b'data: {"choices": [{"delta": {"content": "Sure, one mom',
        sse({"choices": [{"delta": {"content": "Sure."}, "finish_reason": "stop"}]}),
    ]

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=replies.pop(0), headers={"content-type": "text/event-stream"}
        )

    response = await client_for(handler, monkeypatch).stream(REQUEST)
    assert response.text == "Sure."
    assert replies == []


async def test_retries_rate_limits(monkeypatch) -> None:
    replies = [httpx.Response(429, headers={"retry-after": "1"}), httpx.Response(200, json=BODY)]

    def handler(_: httpx.Request) -> httpx.Response:
        return replies.pop(0)

    response = await client_for(handler, monkeypatch).complete(REQUEST)
    assert response.text == "Hi there"
    assert replies == []


async def test_client_errors_are_not_retried(monkeypatch) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": {"message": "bad model"}})

    with pytest.raises(LLMError, match="HTTP 400") as raised:
        await client_for(handler, monkeypatch).complete(REQUEST)
    assert raised.value.status == 400
    assert calls == 1


async def test_fake_llm_records_requests() -> None:
    fake = FakeLLM(scripted(["Hello", Output(text=None, finish_reason="stop")]))
    first = await fake.stream(REQUEST)
    second = await fake.complete(REQUEST)
    assert first.text == "Hello"
    assert first.timing.first_speech_token_ms == first.timing.first_sentence_ms == 0.0
    assert second.timing.first_speech_token_ms is None
    assert fake.requests == [REQUEST, REQUEST]


async def test_messages_api_merges_nested_params(monkeypatch) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "model": "claude-opus-5",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "{}"}],
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    endpoint = Endpoint(
        provider="anthropic",
        model="claude-opus-5",
        params={"max_tokens": 8000, "output_config": {"effort": "low"}},
    )
    client = LLMClient(endpoint, transport=httpx.MockTransport(handler))
    request = Request(messages=[Message(role="user", content="x")], json_schema={"type": "object"})
    response = await client.complete(request)
    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/v1/messages"
    assert seen[0].headers["x-api-key"] == "sk-ant"
    assert body["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": {"type": "object"}},
    }
    assert response.finish_reason == "stop"
