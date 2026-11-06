import copy
from typing import Any

from mtva_bench.llm.http import LLMError, SSEEvent
from mtva_bench.llm.types import Delta, FinishReason, Message, Output, Request, ToolCall, Usage

PATH = "/responses"
EXPLICIT = {"mode": "explicit"}

_INCOMPLETE_REASONS: dict[str, FinishReason] = {
    "max_output_tokens": "length",
    "content_filter": "content_filter",
}


def headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def encode(request: Request, model: str, stream: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "input": [item for message in request.messages for item in input_items(message)],
    }
    if request.tools:
        body["tools"] = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in request.tools
        ]
    if request.json_schema is not None:
        body["text"] = {
            "format": {"type": "json_schema", "name": "response", "schema": request.json_schema}
        }
    if request.max_output_tokens is not None:
        body["max_output_tokens"] = request.max_output_tokens
    if stream:
        body["stream"] = True
    return body


def input_items(message: Message) -> list[dict[str, Any]]:
    if message.role == "tool":
        return [
            {
                "type": "function_call_output",
                "call_id": message.tool_call_id,
                "output": message.content or "",
            }
        ]
    if message.role == "assistant" and "responses" in message.provider_data:
        return copy.deepcopy(message.provider_data["responses"])
    items: list[dict[str, Any]] = []
    parts = message.text_parts()
    if any(marked for _, marked in parts):
        kind = "output_text" if message.role == "assistant" else "input_text"
        content: str | list[dict[str, Any]] = [
            {
                "type": kind,
                "text": text,
                **({"prompt_cache_breakpoint": EXPLICIT} if marked else {}),
            }
            for text, marked in parts
        ]
        items.append({"role": message.role, "content": content})
    elif message.content or message.role != "assistant":
        items.append({"role": message.role, "content": message.content or ""})
    items += [
        {
            "type": "function_call",
            "call_id": call.id,
            "name": call.name,
            "arguments": call.arguments,
        }
        for call in message.tool_calls
    ]
    return items


def decode(body: dict[str, Any]) -> Output:
    if body.get("status") == "failed":
        raise LLMError(f"response failed: {body.get('error')}", retryable=True)
    items = body.get("output") or []
    calls = [
        ToolCall(id=item["call_id"], name=item["name"], arguments=item.get("arguments") or "")
        for item in items
        if item["type"] == "function_call"
    ]
    text = "".join(
        part.get("text", "")
        for item in items
        if item["type"] == "message"
        for part in item.get("content") or []
        if part.get("type") == "output_text"
    )
    return Output(
        text=text or None,
        tool_calls=calls,
        finish_reason=finish_reason(body, bool(calls)),
        model=body.get("model"),
        usage=decode_usage(body.get("usage")),
        provider_data={"responses": items},
    )


def finish_reason(body: dict[str, Any], called_tools: bool) -> FinishReason:
    if body.get("status") == "incomplete":
        reason = (body.get("incomplete_details") or {}).get("reason") or ""
        return _INCOMPLETE_REASONS.get(reason, "other")
    for item in body.get("output") or []:
        for part in item.get("content") or []:
            if part.get("type") == "refusal":
                return "refusal"
    return "tool_calls" if called_tools else "stop"


def decode_usage(usage: dict[str, Any] | None) -> Usage:
    if not usage:
        return Usage()
    details = usage.get("input_tokens_details") or {}
    return Usage(
        input_tokens=usage.get("input_tokens") or 0,
        output_tokens=usage.get("output_tokens") or 0,
        cache_read_tokens=details.get("cached_tokens") or 0,
        cache_write_tokens=details.get("cache_write_tokens") or 0,
    )


class StreamDecoder:
    def __init__(self) -> None:
        self.final: dict[str, Any] | None = None

    def feed(self, event: SSEEvent) -> Delta | None:
        data = event.data
        kind = data.get("type")
        if kind == "error":
            raise LLMError(f"error in stream: {data}", retryable=True)
        if kind == "response.output_text.delta":
            return Delta(data["delta"])
        if kind == "response.output_item.added" and data["item"]["type"] == "function_call":
            return Delta(tool_call=True)
        if kind in ("response.completed", "response.incomplete", "response.failed"):
            self.final = data["response"]
        return None

    def output(self) -> Output:
        if self.final is None:
            raise LLMError("stream ended without a final response", retryable=True)
        return decode(self.final)
