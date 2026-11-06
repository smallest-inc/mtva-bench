from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mtva_bench.llm.types import FinishReason, Message, Timing, Usage

EndReason = Literal["terminal_tool", "caller_done", "max_turns", "context_limit", "error"]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChannelEdit(Record):
    type: Literal["drop", "double", "substitute"]
    source: str
    target: str
    applied: bool


class CallerTurn(Record):
    turn: int
    intended: str
    heard: list[str]
    beats: list[str] = Field(default_factory=list)
    done: bool = False
    edits: list[ChannelEdit] = Field(default_factory=list)


class ToolEvent(Record):
    turn: int
    call_id: str
    name: str
    arguments: str
    rule: str | None
    reply: dict[str, Any]
    terminal: bool


class CallerCall(Record):
    """One request to the caller model, retries of a rejected reply included."""

    turn: int
    usage: Usage


class ModelCall(Record):
    """One request to the model under test. `round` counts tool rounds within the turn;
    `model` is the model the provider says answered, often a dated version."""

    turn: int
    round: int
    model: str | None = None
    timing: Timing
    usage: Usage
    spoke: bool
    tool_calls: int
    finish_reason: FinishReason | None


class AgentTurn(Record):
    """What the caller heard from the agent in one turn.

    `latency_ms` is the response latency: model time from the caller's turn to the first
    spoken token, summed over tool rounds before speech, mock-tool time excluded. None when
    the turn had no speech.
    """

    turn: int
    text: str
    latency_ms: float | None


class Conversation(Record):
    scenario_id: str
    agent_id: str
    model: str
    rollout: int
    seed: int
    messages: list[Message]
    agent_turns: list[AgentTurn] = Field(default_factory=list)
    caller_turns: list[CallerTurn] = Field(default_factory=list)
    tool_events: list[ToolEvent] = Field(default_factory=list)
    model_calls: list[ModelCall] = Field(default_factory=list)
    caller_calls: list[CallerCall] = Field(default_factory=list)
    end_reason: EndReason
    error: str | None = None
    duration_ms: float
