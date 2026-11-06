from typing import Any

from mtva_bench.llm.http import LLMError, SSEEvent
from mtva_bench.llm.types import (
    Delta,
    FinishReason,
    Message,
    Output,
    Request,
    ToolCall,
    Usage,
)

PATH = "/chat/completions"

_TOOL_CALL_KEYS = {"index", "id", "type", "function"}
_FINISH_REASONS: dict[str, FinishReason] = {
    "stop": "stop",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "length": "length",
    "content_filter": "content_filter",
}


def headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def encode(request: Request, model: str, stream: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [encode_message(message) for message in request.messages],
    }
    if request.tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in request.tools
        ]
    if request.json_schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": request.json_schema},
        }
    if request.seed is not None:
        body["seed"] = request.seed
    if request.max_output_tokens is not None:
        body["max_tokens"] = request.max_output_tokens
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def encode_message(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content}
    encoded: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        encoded["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
                **call.provider_data,
            }
            for call in message.tool_calls
        ]
    return encoded


def decode(body: dict[str, Any]) -> Output:
    choice = body["choices"][0]
    message = choice["message"]
    return Output(
        text=message.get("content") or None,
        tool_calls=[
            ToolCall(
                id=call["id"],
                name=call["function"]["name"],
                arguments=call["function"].get("arguments") or "",
                provider_data={k: v for k, v in call.items() if k not in _TOOL_CALL_KEYS},
            )
            for call in message.get("tool_calls") or []
        ],
        finish_reason=finish_reason(choice.get("finish_reason")),
        model=body.get("model"),
        usage=decode_usage(body.get("usage")),
    )


def finish_reason(value: str | None) -> FinishReason | None:
    return None if value is None else _FINISH_REASONS.get(value, "other")


def decode_usage(usage: dict[str, Any] | None) -> Usage:
    if not usage:
        return Usage()
    details = usage.get("prompt_tokens_details") or {}
    return Usage(
        input_tokens=usage.get("prompt_tokens") or 0,
        output_tokens=usage.get("completion_tokens") or 0,
        cache_read_tokens=details.get("cached_tokens") or 0,
    )


class StreamDecoder:
    def __init__(self) -> None:
        self.text: list[str] = []
        self.calls: dict[int, dict[str, Any]] = {}
        self.call_index_by_id: dict[str, int] = {}
        self.finish_reason: FinishReason | None = None
        self.model: str | None = None
        self.usage = Usage()

    def feed(self, event: SSEEvent) -> Delta | None:
        chunk = event.data
        if "error" in chunk:
            raise LLMError(f"error in stream: {chunk['error']}", retryable=True)
        self.model = chunk.get("model") or self.model
        if chunk.get("usage"):
            self.usage = decode_usage(chunk["usage"])
        if not chunk.get("choices"):
            return None
        choice = chunk["choices"][0]
        self.finish_reason = finish_reason(choice.get("finish_reason")) or self.finish_reason
        delta = choice.get("delta") or {}
        calls = delta.get("tool_calls") or []
        for call in calls:
            self.add_call_delta(call)
        text = delta.get("content") or ""
        self.text.append(text)
        return Delta(text, bool(calls)) if text or calls else None

    def add_call_delta(self, delta: dict[str, Any]) -> None:
        index = delta.get("index")
        if index is None:
            index = self.call_index_by_id.setdefault(delta.get("id") or "", len(self.calls))
        call = self.calls.setdefault(
            index, {"id": "", "name": "", "arguments": "", "provider_data": {}}
        )
        if delta.get("id"):
            call["id"] = delta["id"]
            self.call_index_by_id[delta["id"]] = index
        function = delta.get("function") or {}
        call["name"] += function.get("name") or ""
        call["arguments"] += function.get("arguments") or ""
        call["provider_data"].update({k: v for k, v in delta.items() if k not in _TOOL_CALL_KEYS})

    def output(self) -> Output:
        return Output(
            text="".join(self.text) or None,
            tool_calls=[ToolCall(**self.calls[index]) for index in sorted(self.calls)],
            finish_reason=self.finish_reason,
            model=self.model,
            usage=self.usage,
        )
