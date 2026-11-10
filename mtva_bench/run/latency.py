"""The latency test: the same recorded calls replayed against each model, one request at a time.

Latency measured during a benchmark run depends on where the run happened and how busy it was.
This test measures it on its own terms: one recorded call per agent, taken from a reference run,
played back with the caller's lines verbatim, every request after the last has finished. It
runs from the cloud region nearest the model's provider, so the network adds as little as it can.

    runs/latency/<region>/<model>/
      conversations/<scenario_id>__r0.json   one replayed call, with the inputs it was made from
      result.json                            the model's speed in that region

The model spec must be the one the model was scored with: its hash is checked against the
model's published summary.
"""

import hashlib
from collections.abc import Callable
from datetime import date
from pathlib import Path

from mtva_bench.config import Benchmark, ConfigError, ModelSpec, stable_hash
from mtva_bench.data import Bundle, Scenario
from mtva_bench.llm.client import LLM
from mtva_bench.run.evaluate import conversation_hash
from mtva_bench.run.score import Summary
from mtva_bench.run.store import RunStore, StoredConversation, stem, write_atomic
from mtva_bench.scoring.latency import Speed, speed
from mtva_bench.simulate.replay import replay
from mtva_bench.simulate.types import Record

# Bump when the replay behaves differently: saved replays then no longer match.
REPLAY_VERSION = "1"


class LatencyResult(Record):
    model: str
    model_hash: str
    region: str
    reference: str
    measured: str
    calls: int
    failed: int
    speed: Speed | None


def scripts(bundle: Bundle, reference: RunStore) -> dict[str, tuple[Scenario, StoredConversation]]:
    """One recorded call per agent, by stem: a seeded pick among the agent's scenarios the
    reference run completed, so every model replays the same calls."""
    picked = {}
    for agent_id in sorted(bundle.agents):
        done = []
        for scenario in sorted(bundle.select(agents=[agent_id]), key=lambda s: s.scenario_id):
            stored = reference.conversation(stem(scenario.scenario_id, 0))
            if stored is not None and stored.conversation.end_reason != "error":
                done.append((scenario, stored))
        if done:
            index = int(hashlib.sha256(agent_id.encode()).hexdigest(), 16) % len(done)
            scenario, stored = done[index]
            picked[stem(scenario.scenario_id, 0)] = (scenario, stored)
    return picked


def check_scored(spec: ModelSpec, leaderboard_runs: Path) -> None:
    """Refuse a model spec other than the one its published summary was scored with."""
    path = leaderboard_runs / f"{spec.name}.json"
    if not path.is_file():
        raise ConfigError(f"{spec.name} has no published summary in {leaderboard_runs}")
    summary = Summary.model_validate_json(path.read_text(encoding="utf-8"))
    if summary.provenance.model_hash != spec.hash():
        raise ConfigError(
            f"models/{spec.name}.yaml differs from the spec {spec.name} was scored with"
        )


def latency_dir(runs_dir: Path, region: str, model: str) -> Path:
    return runs_dir / "latency" / region / model


async def latency_stage(
    spec: ModelSpec,
    benchmark: Benchmark,
    bundle: Bundle,
    model: LLM,
    reference_dir: Path,
    region: str,
    runs_dir: Path,
    limit: int | None = None,
    on_start: Callable[[int], None] = lambda _: None,
    on_done: Callable[[StoredConversation], None] = lambda _: None,
) -> LatencyResult:
    """Replay every picked call the model has no current replay of, one request at a time,
    then save the model's speed in this region."""
    picked = scripts(bundle, RunStore(reference_dir))
    if limit is not None:
        picked = dict(list(picked.items())[:limit])
    if not picked:
        raise ConfigError(f"{reference_dir} holds no completed calls to replay")
    store = RunStore(latency_dir(runs_dir, region, spec.name))
    guide = bundle.voice_guide if benchmark.data.voice_guide else None
    simulation = benchmark.simulation

    def inputs(reference: StoredConversation) -> dict[str, str]:
        return {
            "model": spec.hash(),
            "reference": conversation_hash(reference),
            "max_reply_tokens": str(simulation.max_reply_tokens),
            "voice_guide": stable_hash(guide),
            "replay": REPLAY_VERSION,
        }

    def current(key: str, reference: StoredConversation) -> StoredConversation | None:
        saved = store.conversation(key)
        if saved is None or saved.inputs != inputs(reference):
            return None
        return saved if saved.conversation.end_reason != "error" else None

    todo = [key for key, (_, reference) in picked.items() if current(key, reference) is None]
    on_start(len(todo))
    with store.lock():
        for key in todo:
            scenario, reference = picked[key]
            conversation = await replay(
                agent=bundle.agent(scenario),
                scenario=scenario,
                model=model,
                model_spec=spec,
                script=reference.conversation.caller_turns,
                voice_guide=guide,
                seed=simulation.seed,
                max_reply_tokens=simulation.max_reply_tokens,
            )
            saved = StoredConversation(inputs=inputs(reference), conversation=conversation)
            store.save_conversation(saved)
            on_done(saved)
    replays = [current(key, reference) for key, (_, reference) in picked.items()]
    done = [saved.conversation for saved in replays if saved is not None]
    result = LatencyResult(
        model=spec.name,
        model_hash=spec.hash(),
        region=region,
        reference=reference_dir.name,
        measured=date.today().isoformat(),
        calls=len(done),
        failed=len(picked) - len(done),
        speed=speed(done),
    )
    write_atomic(store.root / "result.json", result.model_dump_json(indent=1) + "\n")
    return result
