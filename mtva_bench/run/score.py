"""The score stage: summarize a run's current verdicts in summary.json. It calls no model.

A call counts only when its conversation and verdict are both current under the protocol it is
scored against; every other planned call is counted as missing, with the reason. The scores are
those of the calls that count, so a run is comparable with others only at full coverage.

summary.json holds aggregates only, nothing from a scenario or a conversation, so it can be
published as it is while the run folder stays private.
"""

from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from mtva_bench.config import (
    Benchmark,
    ConfigError,
    ModelSpec,
    RunConfig,
    RunOptions,
    endpoint_hash,
    stable_hash,
)
from mtva_bench.data import Bundle
from mtva_bench.prompts import template_hash
from mtva_bench.run.evaluate import JUDGE, call_inputs, expected_inputs, fallback_hash, judge_inputs
from mtva_bench.run.manifest import CALLER_PROMPT, Code
from mtva_bench.run.stages import changed_inputs, plan
from mtva_bench.run.store import RunStore
from mtva_bench.scoring.aggregate import CallScore, Scores, aggregate, score_call
from mtva_bench.scoring.latency import Latency, ttft
from mtva_bench.simulate.call import SIMULATOR_VERSION
from mtva_bench.simulate.types import Conversation

# How a model is reached, not what it is: an endpoint address can be private.
UNPUBLISHED_MODEL_FIELDS = {"base_url", "api_key_env", "timeout_s", "concurrency"}
PUBLISHABLE_COVERAGE = 0.99


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Coverage(Model):
    planned: int
    scored: int
    missing: dict[str, int]

    @property
    def share(self) -> float:
        return self.scored / self.planned if self.planned else 0.0


class Provenance(Model):
    """The protocol the run was scored under, and the code that last ran it."""

    code: Code
    bundle_hash: str
    model_hash: str
    caller_hash: str
    caller_prompt_hash: str
    simulator_version: str
    judge_model: str
    judge_inputs_hash: str


class Summary(Model):
    run: str
    model: dict[str, Any]
    benchmark_version: str
    bundle_version: str
    weights: dict[str, float]
    coverage: Coverage
    scores: Scores | None
    latency_ms: Latency | None
    provenance: Provenance


def protocol(benchmark: Benchmark) -> dict[str, str]:
    """The hashes of what made and judged a run's calls, besides the data: runs compare only
    when these match."""
    return {
        "caller_hash": endpoint_hash(benchmark.simulation.caller),
        "caller_prompt_hash": template_hash(CALLER_PROMPT),
        "simulator_version": SIMULATOR_VERSION,
        "judge_model": benchmark.judge.model,
        "judge_inputs_hash": stable_hash(judge_inputs(benchmark)),
    }


def run_config(run_dir: Path, benchmark: Benchmark) -> RunConfig:
    """The run's model spec, from its run.yaml, under `benchmark`, over every scenario."""
    path = run_dir / "run.yaml"
    if not path.is_file():
        raise ConfigError(f"{run_dir} is not a run folder: it has no run.yaml")
    try:
        model = ModelSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))["model"])
    except (ValidationError, KeyError, TypeError) as error:
        raise ConfigError(f"{path}: {error}") from error
    return RunConfig(
        benchmark=benchmark,
        model=model,
        options=RunOptions(name=run_dir.name, runs_dir=run_dir.parent),
    )


def current_calls(
    config: RunConfig, bundle: Bundle
) -> tuple[list[CallScore], list[Conversation], Coverage]:
    store = RunStore(config.run_dir)
    simulation = config.benchmark.simulation
    shared = judge_inputs(config.benchmark)
    fallback = fallback_hash(config)
    rollouts = plan(bundle, config)
    missing: Counter[str] = Counter()
    calls: list[CallScore] = []
    conversations: list[Conversation] = []
    for rollout in rollouts:
        stored = store.conversation(rollout.stem)
        if stored is None:
            missing["not simulated"] += 1
            continue
        if changed_inputs(stored, rollout, simulation):
            missing["stale conversation"] += 1
            continue
        if stored.conversation.end_reason == "error":
            missing["call failed"] += 1
            continue
        header = store.verdict_header(JUDGE, rollout.stem)
        if header is None:
            missing["not judged"] += 1
            continue
        inputs = call_inputs(shared, rollout, stored, bundle, config.model.text_with_tool_calls)
        if header.inputs != expected_inputs(header.inputs, inputs, fallback):
            missing["stale verdict"] += 1
            continue
        verdict = store.verdict(JUDGE, rollout.stem)
        if verdict is None or verdict.verdict is None:
            missing["judge failed"] += 1
            continue
        scenario = rollout.scenario
        calls.append(
            score_call(
                rollout.stem, verdict.verdict, scenario, bundle.agent(scenario), stored.conversation
            )
        )
        conversations.append(stored.conversation)
    coverage = Coverage(
        planned=len(rollouts), scored=len(calls), missing=dict(sorted(missing.items()))
    )
    return calls, conversations, coverage


def summarize(config: RunConfig, bundle: Bundle) -> tuple[Summary, list[CallScore]]:
    """The run's summary and the scores of the calls it is made from."""
    store = RunStore(config.run_dir)
    manifest = store.manifest()
    if manifest is None:
        raise ConfigError(f"{config.run_dir} has no manifest.json; run the model first")
    calls, conversations, coverage = current_calls(config, bundle)
    weights = config.benchmark.capabilities
    summary = Summary(
        run=config.name,
        model=config.model.model_dump(mode="json", exclude=UNPUBLISHED_MODEL_FIELDS),
        benchmark_version=config.benchmark.version,
        bundle_version=bundle.version,
        weights=weights,
        coverage=coverage,
        scores=aggregate(calls, weights) if calls else None,
        latency_ms=ttft(conversations),
        provenance=Provenance(
            code=manifest.code,
            bundle_hash=bundle.hash,
            model_hash=config.model.hash(),
            **protocol(config.benchmark),
        ),
    )
    return summary, calls


def score_stage(config: RunConfig, bundle: Bundle) -> tuple[Summary, list[CallScore]]:
    """`summarize`, saving the summary as summary.json."""
    summary, calls = summarize(config, bundle)
    RunStore(config.run_dir).save_summary(summary.model_dump_json(indent=1))
    return summary, calls


def publish_problems(summary: Summary) -> list[str]:
    """Why a summary cannot go on the leaderboard; empty when it can."""
    problems = []
    if summary.coverage.share < PUBLISHABLE_COVERAGE:
        problems.append(
            f"only {summary.coverage.scored} of {summary.coverage.planned} calls are scored; "
            f"publishing needs {PUBLISHABLE_COVERAGE:.0%}"
        )
    if summary.latency_ms is None:
        problems.append("no request recorded a first token")
    return problems
