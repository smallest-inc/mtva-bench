import json

import httpx
import pytest

from mtva_bench.llm.batch import Batcher
from mtva_bench.llm.client import LLMClient
from mtva_bench.llm.endpoints import Endpoint
from mtva_bench.llm.http import LLMError
from mtva_bench.llm.types import Message, Request

REQUESTS = {
    "faithfulness-0": Request(messages=[Message(role="user", content="Judge call A")]),
    "faithfulness-1": Request(messages=[Message(role="user", content="Judge call B")]),
}


def anthropic_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if request.method == "POST" and path == "/v1/messages/batches":
        sent = json.loads(request.content)["requests"]
        assert [item["custom_id"] for item in sent] == list(REQUESTS)
        assert sent[0]["params"]["max_tokens"] == 4000
        assert "stream" not in sent[0]["params"]
        return httpx.Response(200, json={"id": "msgbatch_1"})
    if path == "/v1/messages/batches/msgbatch_1":
        return httpx.Response(
            200,
            json={
                "processing_status": "ended",
                "request_counts": {"succeeded": 1, "errored": 1},
                "results_url": "https://api.anthropic.com/v1/messages/batches/msgbatch_1/results",
            },
        )
    if path == "/v1/messages/batches/msgbatch_1/results":
        lines = [
            {
                "custom_id": "faithfulness-0",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "model": "claude-opus-5",
                        "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": "{}"}],
                        "usage": {"input_tokens": 9, "output_tokens": 1},
                    },
                },
            },
            {
                "custom_id": "faithfulness-1",
                "result": {"type": "errored", "error": {"type": "overloaded_error"}},
            },
        ]
        return httpx.Response(200, text="\n".join(json.dumps(line) for line in lines))
    raise AssertionError(f"unexpected {request.method} {path}")


async def test_anthropic_batch(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    endpoint = Endpoint(provider="anthropic", model="claude-opus-5", params={"max_tokens": 4000})
    batcher = Batcher(LLMClient(endpoint, transport=httpx.MockTransport(anthropic_handler)))
    batch_id = await batcher.submit(REQUESTS)
    assert (await batcher.status(batch_id)).ended
    results = await batcher.results(batch_id)
    assert results["faithfulness-0"].text == "{}"
    assert isinstance(results["faithfulness-1"], LLMError)


def openai_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/v1/files" and request.method == "POST":
        lines = [
            json.loads(line.strip())
            for line in request.content.split(b"\n")
            if line.startswith(b'{"custom_id"')
        ]
        assert [line["url"] for line in lines] == ["/v1/chat/completions"] * 2
        return httpx.Response(200, json={"id": "file_in"})
    if path == "/v1/batches" and request.method == "POST":
        assert json.loads(request.content)["endpoint"] == "/v1/chat/completions"
        return httpx.Response(200, json={"id": "batch_1", "status": "validating"})
    if path == "/v1/batches/batch_1":
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output_file_id": "file_out",
                "request_counts": {"total": 2},
            },
        )
    if path == "/v1/files/file_out/content":
        body = {"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}
        lines = [
            {"custom_id": "faithfulness-0", "response": {"status_code": 200, "body": body}},
            {"custom_id": "faithfulness-1", "response": {"status_code": 500, "body": {}}},
        ]
        return httpx.Response(200, text="\n".join(json.dumps(line) for line in lines))
    raise AssertionError(f"unexpected {request.method} {path}")


async def test_openai_batch(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    endpoint = Endpoint(provider="openai", model="gpt-5.2")
    batcher = Batcher(LLMClient(endpoint, transport=httpx.MockTransport(openai_handler)))
    batch_id = await batcher.submit(REQUESTS)
    assert (await batcher.status(batch_id)).ended
    results = await batcher.results(batch_id)
    assert results["faithfulness-0"].text == "ok"
    assert results["faithfulness-1"].status == 500


async def test_batch_rejects_unsupported_providers_and_bad_ids(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    with pytest.raises(ValueError, match="no batch api"):
        Batcher(LLMClient(Endpoint(provider="gemini", model="gemini-3.6-flash")))
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    batcher = Batcher(LLMClient(Endpoint(provider="openai", model="gpt-5.2")))
    with pytest.raises(ValueError, match="batch request ids"):
        await batcher.submit({"faithfulness/scenario.r0": REQUESTS["faithfulness-0"]})
