import copy
import json
from typing import Any

from mtva_bench.llm.http import LLMError, SSEEvent
from mtva_bench.llm.types import Delta, FinishReason, Message, Output, Request, ToolCall, Usage

PATH = "/messages"
EPHEMERAL = {"type": "ephemeral"}
ONE_HOUR = {"type": "ephemeral", "ttl": "1h"}
VERSION = "2023-06-01"
MAX_CACHE_BREAKPOINTS = 4

_FINISH_REASONS: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "refusal": "refusal",
}


def headers(api_key: str | None) -> dict[str, str]:
    return {"x-api-key": api_key or "", "anthropic-version": VERSION}


def encode(request: Request, model: str, stream: bool) -> dict[str, Any]:
    leading = 0
    while leading < len(request.messages) and request.messages[leading].role == "system":
        leading += 1
    if any(message.role == "system" for message in request.messages[leading:]):
        raise ValueError("system messages must come before the conversation")
    marker = ONE_HOUR if request.cache_ttl == "1h" else EPHEMERAL
    body: dict[str, Any] = {
        "model": model,
        "messages": encode_turns(request.messages[leading:], marker),
    }
    system = [
        block for message in request.messages[:leading] for block in text_blocks(message, marker)
    ]
    if system:
        body["system"] = system
    if request.tools:
        body["tools"] = [
            {"name": tool.name, "description": tool.description, "input_schema": tool.parameters}
            for tool in request.tools
        ]
    if request.json_schema is not None:
        body["output_config"] = {"format": {"type": "json_schema", "schema": request.json_schema}}
    if request.max_output_tokens is not None:
        body["max_tokens"] = request.max_output_tokens
    if stream:
        body["stream"] = True
    blocks = system + [block for turn in body["messages"] for block in turn["content"]]
    breakpoints = sum("cache_control" in block for block in blocks)
    if breakpoints > MAX_CACHE_BREAKPOINTS:
        raise ValueError(
            f"{breakpoints} cache breakpoints; Anthropic allows {MAX_CACHE_BREAKPOINTS}"
        )
    return body


def encode_turns(
    messages: list[Message], marker: dict[str, str] = EPHEMERAL
) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for message in messages:
        role = "assistant" if message.role == "assistant" else "user"
        blocks = content_blocks(message, marker)
        if not blocks:
            continue
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"].extend(blocks)
        else:
            turns.append({"role": role, "content": blocks})
    return turns


def content_blocks(message: Message, marker: dict[str, str] = EPHEMERAL) -> list[dict[str, Any]]:
    if message.role == "tool":
        blocks: list[dict[str, Any]] = [
            {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": message.content or "",
            }
        ]
    elif message.role == "assistant" and "messages" in message.provider_data:
        blocks = copy.deepcopy(message.provider_data["messages"])
    else:
        blocks = [
            {"type": "text", "text": text, **({"cache_control": dict(marker)} if marked else {})}
            for text, marked in message.text_parts()
        ]
        blocks += [
            {
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": json.loads(call.arguments or "{}"),
            }
            for call in message.tool_calls
        ]
    if message.cache_breakpoint and blocks:
        blocks[-1]["cache_control"] = dict(marker)
    return blocks


def text_blocks(message: Message, marker: dict[str, str] = EPHEMERAL) -> list[dict[str, Any]]:
    return content_blocks(message.model_copy(update={"role": "user"}), marker)


def decode(body: dict[str, Any]) -> Output:
    return output_from(body["content"], body.get("stop_reason"), body.get("model"), body["usage"])


def output_from(
    blocks: list[dict[str, Any]], stop_reason: str | None, model: str | None, usage: dict
) -> Output:
    return Output(
        text="".join(block["text"] for block in blocks if block["type"] == "text") or None,
        tool_calls=[
            ToolCall(
                id=block["id"],
                name=block["name"],
                arguments=json.dumps(block["input"], ensure_ascii=False),
            )
            for block in blocks
            if block["type"] == "tool_use"
        ],
        finish_reason=None if stop_reason is None else _FINISH_REASONS.get(stop_reason, "other"),
        model=model,
        usage=decode_usage(usage),
        provider_data={"messages": blocks},
    )


def decode_usage(usage: dict[str, Any]) -> Usage:
    """Anthropic's input_tokens is only the uncached remainder; Usage counts all input."""
    cache_read = usage.get("cache_read_input_tokens") or 0
    cache_write = usage.get("cache_creation_input_tokens") or 0
    return Usage(
        input_tokens=(usage.get("input_tokens") or 0) + cache_read + cache_write,
        output_tokens=usage.get("output_tokens") or 0,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )


class StreamDecoder:
    def __init__(self) -> None:
        self.blocks: dict[int, dict[str, Any]] = {}
        self.partial_json: dict[int, str] = {}
        self.stop_reason: str | None = None
        self.model: str | None = None
        self.usage: dict[str, Any] = {}

    def feed(self, event: SSEEvent) -> Delta | None:
        data = event.data
        kind = data.get("type")
        if kind == "error":
            error = data.get("error") or {}
            raise LLMError(f"error in stream: {error}", retryable=True)
        if kind == "message_start":
            self.model = data["message"].get("model")
            self.usage.update(data["message"].get("usage") or {})
        elif kind == "message_delta":
            self.stop_reason = (data.get("delta") or {}).get("stop_reason") or self.stop_reason
            self.usage.update({k: v for k, v in (data.get("usage") or {}).items() if v is not None})
        elif kind == "content_block_start":
            block = dict(data["content_block"])
            self.blocks[data["index"]] = block
            if block["type"] == "tool_use":
                self.partial_json[data["index"]] = ""
                return Delta(tool_call=True)
        elif kind == "content_block_delta":
            return self.add_delta(data["index"], data["delta"])
        elif kind == "content_block_stop" and data["index"] in self.partial_json:
            raw = self.partial_json[data["index"]]
            self.blocks[data["index"]]["input"] = json.loads(raw) if raw else {}
        return None

    def add_delta(self, index: int, delta: dict[str, Any]) -> Delta | None:
        block = self.blocks[index]
        kind = delta["type"]
        if kind == "text_delta":
            block["text"] = block.get("text", "") + delta["text"]
            return Delta(delta["text"])
        if kind == "input_json_delta":
            self.partial_json[index] += delta["partial_json"]
            return Delta(tool_call=True)
        if kind == "thinking_delta":
            block["thinking"] = block.get("thinking", "") + delta["thinking"]
        elif kind == "signature_delta":
            block["signature"] = delta["signature"]
        return None

    def output(self) -> Output:
        blocks = [self.blocks[index] for index in sorted(self.blocks)]
        return output_from(blocks, self.stop_reason, self.model, self.usage)
