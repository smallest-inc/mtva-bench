from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

FinishReason = Literal["stop", "tool_calls", "length", "refusal", "content_filter", "other"]


class LLMModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolCall(LLMModel):
    id: str
    name: str
    arguments: str
    provider_data: dict[str, Any] = Field(default_factory=dict)


class Message(LLMModel):
    """`cache_breakpoint` marks the end of this message for caching; `cache_prefix` marks the
    end of its first that many characters, a prefix shared with other requests. Protocols
    with explicit cache breakpoints honour them; the others cache prefixes on their own."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    cache_breakpoint: bool = False
    cache_prefix: int | None = None
    provider_data: dict[str, Any] = Field(default_factory=dict)

    def text_parts(self) -> list[tuple[str, bool]]:
        """The content as (text, marked for caching) parts, split at `cache_prefix`."""
        if not self.content:
            return []
        if self.cache_prefix and 0 < self.cache_prefix < len(self.content):
            return [
                (self.content[: self.cache_prefix], True),
                (self.content[self.cache_prefix :], False),
            ]
        return [(self.content, False)]


class Tool(LLMModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class Request(LLMModel):
    """`cache_ttl` is how long a cached prefix lives where the protocol lets it be chosen: one
    hour suits a batch, whose requests may run far apart."""

    messages: list[Message]
    tools: list[Tool] = Field(default_factory=list)
    json_schema: dict[str, Any] | None = None
    seed: int | None = None
    max_output_tokens: int | None = None
    cache_ttl: Literal["5m", "1h"] = "5m"


class Usage(LLMModel):
    """`input_tokens` counts all input, cached or not; the cache fields are parts of it."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class Delta(NamedTuple):
    text: str = ""
    tool_call: bool = False


class Timing(LLMModel):
    first_token_ms: float | None = None
    first_speech_token_ms: float | None = None
    first_sentence_ms: float | None = None
    total_ms: float


class Output(LLMModel):
    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: FinishReason | None = None
    model: str | None = None
    usage: Usage = Field(default_factory=Usage)
    provider_data: dict[str, Any] = Field(default_factory=dict)


class Response(Output):
    timing: Timing


class BatchStatus(LLMModel):
    id: str
    ended: bool
    counts: dict[str, int] = Field(default_factory=dict)
