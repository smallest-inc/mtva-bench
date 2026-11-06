"""One agent turn: the model under test speaks, calling mock tools as often as it needs."""

import json
from dataclasses import dataclass, field

from mtva_bench.data.schema import Agent
from mtva_bench.llm.client import LLM
from mtva_bench.llm.http import LLMError, context_limit
from mtva_bench.llm.types import Message, Request, Response, Tool, ToolCall
from mtva_bench.simulate.tools import MockTools
from mtva_bench.simulate.types import ModelCall, ToolEvent

MAX_TOOL_ROUNDS = 10
CONCLUDING = "concluding_message"


@dataclass
class TurnResult:
    """`context_limit` is set when the provider refused the next request because the call no
    longer fits the model's context window."""

    text: str = ""
    ended: bool = False
    context_limit: bool = False
    latency_ms: float | None = None
    calls: list[ModelCall] = field(default_factory=list)
    events: list[ToolEvent] = field(default_factory=list)


def cached(messages: list[Message]) -> list[Message]:
    """The request's messages with a cache breakpoint on the latest one, so the next turn of
    the call reads everything before it from cache. Only the Anthropic protocol needs it;
    other providers cache matching prefixes on their own."""
    return [*messages[:-1], messages[-1].model_copy(update={"cache_breakpoint": True})]


def agent_tools(agent: Agent, text_with_tool_calls: bool) -> list[Tool]:
    """The agent's tools as the model sees them. A model that cannot speak and call a tool in
    one message gets a required `concluding_message` on terminal tools for its farewell."""
    tools = []
    for tool in agent.tools:
        parameters = tool.parameters
        if not text_with_tool_calls and tool.name in agent.terminal_tools:
            properties = {
                **parameters.get("properties", {}),
                CONCLUDING: {"type": "string", "description": "What you say before hanging up."},
            }
            required = [*parameters.get("required", []), CONCLUDING]
            parameters = {**parameters, "properties": properties, "required": required}
        tools.append(Tool(name=tool.name, description=tool.description, parameters=parameters))
    return tools


def concluding(response: Response, mock: MockTools) -> tuple[str, list[ToolCall]]:
    """Move a terminal tool's `concluding_message` into the spoken text."""
    text, calls = response.text or "", []
    for call in response.tool_calls:
        try:
            arguments = json.loads(call.arguments or "{}")
        except json.JSONDecodeError:
            arguments = None
        if mock.is_terminal(call.name) and isinstance(arguments, dict) and CONCLUDING in arguments:
            text = text or str(arguments.pop(CONCLUDING))
            call = call.model_copy(update={"arguments": json.dumps(arguments, ensure_ascii=False)})
        calls.append(call)
    return text, calls


async def agent_turn(
    llm: LLM,
    messages: list[Message],
    tools: list[Tool],
    mock: MockTools,
    turn: int,
    seed: int,
    text_with_tool_calls: bool,
    max_reply_tokens: int | None = None,
) -> TurnResult:
    """Run the model until it speaks without calling a tool, ends the call, or uses up its
    tool rounds; a last round without tools then forces it to speak. Appends to `messages`."""
    result = TurnResult()
    spoken: list[str] = []
    model_ms = 0.0
    for round_ in range(MAX_TOOL_ROUNDS + 1):
        offered = tools if round_ < MAX_TOOL_ROUNDS else []
        request = Request(
            messages=cached(messages),
            tools=offered,
            seed=seed + round_,
            max_output_tokens=max_reply_tokens,
        )
        try:
            response = await llm.stream(request)
        except LLMError as error:
            if not context_limit(error):
                raise
            result.context_limit = True
            break
        text, calls = (
            concluding(response, mock)
            if not text_with_tool_calls
            else (response.text or "", response.tool_calls)
        )
        timing = response.timing
        result.calls.append(
            ModelCall(
                turn=turn,
                round=round_,
                model=response.model,
                timing=timing,
                usage=response.usage,
                spoke=bool(text),
                tool_calls=len(calls),
                finish_reason=response.finish_reason,
            )
        )
        if text and result.latency_ms is None:
            result.latency_ms = round(
                model_ms + (timing.first_speech_token_ms or timing.total_ms), 2
            )
        model_ms += timing.total_ms
        messages.append(
            Message(
                role="assistant",
                content=text or None,
                tool_calls=calls,
                provider_data=response.provider_data,
            )
        )
        if text:
            spoken.append(text.strip())
        if not calls:
            break
        for call in calls:
            reply, rule = mock.call(call.name, call.arguments)
            terminal = mock.is_terminal(call.name)
            result.events.append(
                ToolEvent(
                    turn=turn,
                    call_id=call.id,
                    name=call.name,
                    arguments=call.arguments,
                    rule=rule,
                    reply=reply,
                    terminal=terminal,
                )
            )
            messages.append(
                Message(
                    role="tool", tool_call_id=call.id, content=json.dumps(reply, ensure_ascii=False)
                )
            )
            result.ended = result.ended or terminal
        if result.ended:
            break
    result.text = " ".join(spoken)
    return result
