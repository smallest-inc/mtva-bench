"""The simulate stage: plan the rollouts, refuse stale ones, simulate what is missing.

An agent's scenarios share most of their system prompt, so its first call opens before the
others start and they read that prompt from the provider's cache.

A run never mixes work made from different inputs. Each saved conversation records a hash of
every input it was simulated from; when any of them differs from now, the run stops and names
what changed, unless `discard_stale` asks for those conversations to be redone. A changed limit
counts only for the conversations it would have changed: one that never reached the old limit
and fits the new one would have run the same. Conversations that ended in an error are
simulated again.
"""

import asyncio
import random
from collections.abc import Callable
from dataclasses import dataclass

from mtva_bench.config import RunConfig, Simulation, endpoint_hash, stable_hash
from mtva_bench.data import Bundle, DataError, Scenario, system_prompt
from mtva_bench.llm.client import LLM
from mtva_bench.prompts import template_hash
from mtva_bench.run.manifest import CALLER_PROMPT, build_manifest
from mtva_bench.run.store import RunStore, StoredConversation, stem
from mtva_bench.simulate import simulate
from mtva_bench.simulate.call import SIMULATOR_VERSION

INPUT_LABELS = {
    "model": "model spec",
    "caller": "caller model",
    "caller_prompt": "caller prompt",
    "agent": "agent tools",
    "scenario": "scenario (caller, world or tool rules)",
    "system_prompt": "system prompt (voice guide or variables)",
    "max_turns": "max_turns",
    "max_reply_tokens": "max_reply_tokens",
    "seed": "seed",
    "simulator": "simulator version",
}


# What of an agent and a scenario a call is simulated from. The agent prompt reaches the call
# through the rendered system prompt, hashed on its own; a scenario's grading only the judge reads.
SIMULATED_AGENT = {"tools", "terminal_tools", "direction"}
UNSIMULATED_SCENARIO = {"schema_version", "summary", "difficulty", "evaluation"}


LIMITS = ("max_turns", "max_reply_tokens")


class StaleRunError(RuntimeError):
    pass


@dataclass(frozen=True)
class Rollout:
    scenario: Scenario
    rollout: int
    inputs: dict[str, str]

    @property
    def stem(self) -> str:
        return stem(self.scenario.scenario_id, self.rollout)


@dataclass
class SimulateResult:
    planned: int
    skipped: int
    simulated: int
    failed: int


def select_scenarios(bundle: Bundle, config: RunConfig) -> list[Scenario]:
    """The scenarios the run's filters pick. `limit` takes a seeded sample that spreads evenly
    over languages."""
    options = config.options
    for kind, wanted, known in (
        ("agents", options.agents, bundle.agents),
        ("scenarios", options.scenarios, bundle.scenarios),
        ("languages", options.languages, {s.language for s in bundle.scenarios.values()}),
    ):
        unknown = sorted(set(wanted) - set(known))
        if unknown:
            raise DataError(f"unknown {kind}: {unknown}")
    scenarios = sorted(
        bundle.select(
            agents=options.agents, languages=options.languages, scenarios=options.scenarios
        ),
        key=lambda scenario: scenario.scenario_id,
    )
    if options.limit is None or options.limit >= len(scenarios):
        return scenarios
    rng = random.Random(config.benchmark.simulation.seed)
    by_language: dict[str, list[Scenario]] = {}
    for scenario in scenarios:
        by_language.setdefault(scenario.language, []).append(scenario)
    queues = [rng.sample(group, len(group)) for _, group in sorted(by_language.items())]
    picked: list[Scenario] = []
    while len(picked) < options.limit:
        for queue in queues:
            if queue and len(picked) < options.limit:
                picked.append(queue.pop())
    return sorted(picked, key=lambda scenario: scenario.scenario_id)


def plan(bundle: Bundle, config: RunConfig) -> list[Rollout]:
    simulation = config.benchmark.simulation
    guide = bundle.voice_guide if config.benchmark.data.voice_guide else None
    shared = {
        "model": config.model.hash(),
        "caller": endpoint_hash(simulation.caller),
        "caller_prompt": template_hash(CALLER_PROMPT),
        "max_turns": str(simulation.max_turns),
        "max_reply_tokens": str(simulation.max_reply_tokens),
        "seed": str(simulation.seed),
        "simulator": SIMULATOR_VERSION,
    }
    rollouts = []
    for scenario in select_scenarios(bundle, config):
        agent = bundle.agent(scenario)
        inputs = {
            **shared,
            "agent": stable_hash(agent.model_dump(mode="json", include=SIMULATED_AGENT)),
            "scenario": stable_hash(scenario.model_dump(mode="json", exclude=UNSIMULATED_SCENARIO)),
            "system_prompt": stable_hash(system_prompt(agent, scenario, guide)),
        }
        rollouts += [Rollout(scenario, k, inputs) for k in range(simulation.rollouts)]
    return rollouts


def interleave(rollouts: list[Rollout]) -> list[Rollout]:
    """Each agent's first rollout, then each agent's second, and so on, so concurrent calls
    spread over agents instead of starting an agent's scenarios together."""
    by_agent: dict[str, list[Rollout]] = {}
    for rollout in rollouts:
        by_agent.setdefault(rollout.scenario.agent_id, []).append(rollout)
    depth = max((len(group) for group in by_agent.values()), default=0)
    return [
        by_agent[agent_id][index]
        for index in range(depth)
        for agent_id in sorted(by_agent)
        if index < len(by_agent[agent_id])
    ]


def within_limit(name: str, stored: StoredConversation, simulation: Simulation) -> bool:
    """Whether a saved conversation would have gone the same under the current `name` limit:
    it never reached the limit it was simulated under, and it fits inside the new one."""
    conversation = stored.conversation
    if name == "max_turns":
        return (
            conversation.end_reason != "max_turns"
            and len(conversation.caller_turns) <= simulation.max_turns
        )
    calls = conversation.model_calls
    if any(call.finish_reason == "length" for call in calls):
        return False
    old = stored.inputs.get(name)
    if old is not None and int(old) <= simulation.max_reply_tokens:
        return True
    return all(0 < call.usage.output_tokens <= simulation.max_reply_tokens for call in calls)


def changed_inputs(
    stored: StoredConversation, rollout: Rollout, simulation: Simulation
) -> list[str]:
    return [
        name
        for name in sorted(set(stored.inputs) | set(rollout.inputs))
        if stored.inputs.get(name) != rollout.inputs.get(name)
        and not (name in LIMITS and within_limit(name, stored, simulation))
    ]


def stale_inputs(
    store: RunStore, rollouts: list[Rollout], simulation: Simulation
) -> dict[str, int]:
    """For each input that changed, how many saved conversations were made with the old one."""
    changed: dict[str, int] = {}
    for rollout in rollouts:
        stored = store.conversation(rollout.stem)
        if stored is None:
            continue
        for name in changed_inputs(stored, rollout, simulation):
            changed[name] = changed.get(name, 0) + 1
    return changed


def needs_simulation(store: RunStore, rollout: Rollout, simulation: Simulation) -> bool:
    stored = store.conversation(rollout.stem)
    return (
        stored is None
        or bool(changed_inputs(stored, rollout, simulation))
        or stored.conversation.end_reason == "error"
    )


async def simulate_stage(
    config: RunConfig,
    bundle: Bundle,
    model: LLM,
    caller: LLM,
    on_start: Callable[[int], None] = lambda _: None,
    on_done: Callable[[StoredConversation], None] = lambda _: None,
) -> SimulateResult:
    store = RunStore(config.run_dir)
    rollouts = plan(bundle, config)
    simulation = config.benchmark.simulation
    changed = stale_inputs(store, rollouts, simulation)
    if changed and not config.options.discard_stale:
        details = ", ".join(f"{INPUT_LABELS.get(name, name)} ({n})" for name, n in changed.items())
        raise StaleRunError(
            f"{config.run_dir} holds conversations made from different inputs: {details}. "
            "Use a new --name, or --discard-stale to simulate them again."
        )
    todo = interleave(
        [rollout for rollout in rollouts if needs_simulation(store, rollout, simulation)]
    )
    result = SimulateResult(len(rollouts), len(rollouts) - len(todo), 0, 0)
    if config.options.dry_run:
        return result
    on_start(len(todo))

    guide = bundle.voice_guide if config.benchmark.data.voice_guide else None
    slots = asyncio.Semaphore(config.concurrency)
    opened = {rollout.scenario.agent_id: asyncio.Event() for rollout in todo}
    leaders = {}
    for rollout in todo:
        leaders.setdefault(rollout.scenario.agent_id, rollout.stem)

    async def one(rollout: Rollout) -> None:
        gate = opened[rollout.scenario.agent_id]
        if leaders[rollout.scenario.agent_id] != rollout.stem:
            await gate.wait()
        try:
            async with slots:
                conversation = await simulate(
                    agent=bundle.agent(rollout.scenario),
                    scenario=rollout.scenario,
                    model=model,
                    model_spec=config.model,
                    caller=caller,
                    voice_guide=guide,
                    max_turns=simulation.max_turns,
                    max_reply_tokens=simulation.max_reply_tokens,
                    seed=simulation.seed,
                    rollout=rollout.rollout,
                    on_open=gate.set,
                )
        finally:
            gate.set()
        stored = StoredConversation(inputs=rollout.inputs, conversation=conversation)
        store.save_conversation(stored)
        result.simulated += 1
        result.failed += conversation.end_reason == "error"
        on_done(stored)

    with store.lock():
        store.save_run_yaml(
            {
                "benchmark": config.benchmark.model_dump(mode="json"),
                "model": config.model.model_dump(mode="json"),
                "options": config.options.model_dump(mode="json"),
            }
        )
        await asyncio.gather(*(one(rollout) for rollout in todo))
        failed = sum(
            stored.conversation.end_reason == "error"
            for rollout in rollouts
            if (stored := store.conversation(rollout.stem)) is not None
        )
        previous = store.manifest()
        store.save_manifest(
            build_manifest(
                config,
                bundle,
                len(rollouts),
                len(rollouts) - failed,
                failed,
                judging=previous.judging if previous else None,
            )
        )
    return result
