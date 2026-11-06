"""One rollout: an agent, a scenario and a model become a Conversation.

The agent opens the call from its system prompt, as it does on a real line, then caller and
agent alternate until a terminal tool, the caller is done, max_turns, or the call no longer
fits the model's context window, which is the model's own outcome and is judged like any other
ending. A model or caller failure that retries could not fix ends the rollout with end_reason
"error" instead of raising, so a run can count it and retry it.
"""

import hashlib
from collections.abc import Callable
from time import perf_counter

from mtva_bench.config import ModelSpec
from mtva_bench.data.render import render
from mtva_bench.data.schema import Agent, Scenario
from mtva_bench.llm.client import LLM
from mtva_bench.llm.timing import elapsed_ms
from mtva_bench.llm.types import Message
from mtva_bench.simulate.agent import TurnResult, agent_tools, agent_turn
from mtva_bench.simulate.caller import Caller
from mtva_bench.simulate.channel import deliver
from mtva_bench.simulate.tools import MockTools
from mtva_bench.simulate.types import AgentTurn, CallerCall, CallerTurn, Conversation

# Bump when the call loop behaves differently: saved conversations then no longer match.
SIMULATOR_VERSION = "2"

# What a model that needs a user message before its greeting receives, as some providers' APIs
# require one. A model that can open the call without it gets none.
OPENING_TURN = "[system: call started]"


def derive_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()
    return int(digest[:8], 16)


async def simulate(
    *,
    agent: Agent,
    scenario: Scenario,
    model: LLM,
    model_spec: ModelSpec,
    caller: LLM,
    voice_guide: str | None,
    max_turns: int,
    seed: int,
    max_reply_tokens: int | None = None,
    rollout: int = 0,
    on_open: Callable[[], None] = lambda: None,
) -> Conversation:
    """`on_open` is called once the agent's opening turn is done, which writes the system
    prompt to the provider's cache."""
    start = perf_counter()
    base = derive_seed(seed, scenario.scenario_id, rollout)
    prompt, shared = render(agent, scenario, voice_guide)
    messages = [Message(role="system", content=prompt, cache_prefix=shared)]
    if model_spec.needs_opening_user_turn:
        messages.append(Message(role="user", content=OPENING_TURN))
    conversation = Conversation(
        scenario_id=scenario.scenario_id,
        agent_id=agent.agent_id,
        model=model_spec.name,
        rollout=rollout,
        seed=base,
        messages=messages,
        end_reason="max_turns",
        duration_ms=0,
    )
    try:
        await converse(
            conversation,
            agent,
            scenario,
            model,
            model_spec,
            caller,
            max_turns,
            max_reply_tokens,
            on_open,
        )
    except Exception as error:  # one failed call must not stop the others
        conversation.end_reason = "error"
        conversation.error = f"{type(error).__name__}: {error}"
    conversation.duration_ms = elapsed_ms(start)
    return conversation


async def converse(
    conversation: Conversation,
    agent: Agent,
    scenario: Scenario,
    model: LLM,
    model_spec: ModelSpec,
    caller_llm: LLM,
    max_turns: int,
    max_reply_tokens: int | None,
    on_open: Callable[[], None],
) -> None:
    mock = MockTools(agent, scenario)
    tools = agent_tools(agent, model_spec.text_with_tool_calls)
    caller = Caller(caller_llm, agent, scenario)
    protected = [str(value) for value in scenario.caller.facts.values()]
    base = conversation.seed

    async def speak(turn: int) -> TurnResult:
        result = await agent_turn(
            model,
            conversation.messages,
            tools,
            mock,
            turn,
            derive_seed(base, "agent", turn),
            model_spec.text_with_tool_calls,
            max_reply_tokens,
        )
        conversation.agent_turns.append(
            AgentTurn(turn=turn, text=result.text, latency_ms=result.latency_ms)
        )
        conversation.model_calls += result.calls
        conversation.tool_events += result.events
        return result

    result = await speak(0)
    on_open()
    for turn in range(max_turns):
        if result.context_limit:
            conversation.end_reason = "context_limit"
            return
        if result.ended:
            conversation.end_reason = "terminal_tool"
            return
        caller.hear(result.text)
        try:
            reply = await caller.reply(derive_seed(base, "caller", turn))
        finally:
            conversation.caller_calls += [CallerCall(turn=turn, usage=u) for u in caller.usage]
            caller.usage.clear()
        heard, edits = deliver(
            reply.utterance, scenario.channel, turn, derive_seed(base, "channel", turn), protected
        )
        conversation.caller_turns.append(
            CallerTurn(
                turn=turn,
                intended=reply.utterance,
                heard=heard,
                beats=reply.beats,
                done=reply.done,
                edits=edits,
            )
        )
        conversation.messages += [Message(role="user", content=fragment) for fragment in heard]
        result = await speak(turn + 1)
        if result.context_limit:
            conversation.end_reason = "context_limit"
            return
        if reply.done and not result.ended:
            conversation.end_reason = "caller_done"
            return
    if result.ended:
        conversation.end_reason = "terminal_tool"
