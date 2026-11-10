"""A recorded call played back against another model, to time it.

The caller's lines are played back verbatim and in order, so every model hears the same caller;
the model's own replies and tool calls build the history, as they would on a live call, so a
provider's cache holds what it would hold on a real line. The recorded caller answered another
agent, so the call may drift from what this model says: it is timed, never judged.
"""

from collections.abc import Sequence
from time import perf_counter

from mtva_bench.config import ModelSpec
from mtva_bench.data.schema import Agent, Scenario
from mtva_bench.llm.client import LLM
from mtva_bench.llm.timing import elapsed_ms
from mtva_bench.llm.types import Message
from mtva_bench.simulate.agent import agent_tools
from mtva_bench.simulate.call import open_call, speak
from mtva_bench.simulate.tools import MockTools
from mtva_bench.simulate.types import CallerTurn, Conversation


async def replay(
    *,
    agent: Agent,
    scenario: Scenario,
    model: LLM,
    model_spec: ModelSpec,
    script: Sequence[CallerTurn],
    voice_guide: str | None,
    seed: int,
    max_reply_tokens: int | None = None,
) -> Conversation:
    """The model's call with the recorded caller turns in `script`. It ends when the model ends
    the call, after the model's reply to the last recorded turn, or on a failure retries could
    not fix, which is recorded as end_reason "error" instead of raised."""
    start = perf_counter()
    conversation = open_call(agent, scenario, model_spec, voice_guide, seed, rollout=0)
    mock = MockTools(agent, scenario)
    tools = agent_tools(agent, model_spec.text_with_tool_calls)
    try:
        result = await speak(conversation, model, tools, mock, model_spec, 0, max_reply_tokens)
        for turn, recorded in enumerate(script):
            if result.ended or result.context_limit:
                break
            conversation.caller_turns.append(recorded.model_copy(update={"turn": turn}))
            conversation.messages += [Message(role="user", content=f) for f in recorded.heard]
            result = await speak(
                conversation, model, tools, mock, model_spec, turn + 1, max_reply_tokens
            )
        conversation.end_reason = (
            "context_limit"
            if result.context_limit
            else "terminal_tool"
            if result.ended
            else "caller_done"
        )
    except Exception as error:  # one failed call must not stop the others
        conversation.end_reason = "error"
        conversation.error = f"{type(error).__name__}: {error}"
    conversation.duration_ms = elapsed_ms(start)
    return conversation
