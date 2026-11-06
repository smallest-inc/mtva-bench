"""The arena stage: judge two runs head to head on every scenario both completed.

    runs/arena/<a>__vs__<b>/
      verdicts/arena/<scenario_id>__r<k>.json   a verdict and the inputs it was made from
      verdicts/arena/_batch.json                the batch being judged, until its results are in
      result.json                               win, tie and loss rates

A verdict depends on both calls, the judge, its prompt and which call was shown first. When any
of them changes, as when a run is redone with a new checkpoint, the pair stops and names what
changed, unless `discard_stale` asks for those verdicts to be redone.
"""

import asyncio
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from mtva_bench import arena
from mtva_bench.config import Benchmark, Endpoint, RunConfig, endpoint_hash, stable_hash
from mtva_bench.data import Bundle
from mtva_bench.llm.client import LLM
from mtva_bench.llm.http import LLMError
from mtva_bench.llm.types import BatchStatus, Request, Response, Usage
from mtva_bench.prompts import template_hash
from mtva_bench.run.evaluate import BatchApi, add, conversation_hash
from mtva_bench.run.stages import StaleRunError, changed_inputs, plan
from mtva_bench.run.store import BatchState, RunStore, StoredConversation, write_atomic
from mtva_bench.scoring.arena import ArenaScores, Outcome, arena_scores
from mtva_bench.simulate.types import Record

JUDGE = "arena"
# Bump when the code that shows the calls or reads the answer changes.
ARENA_VERSION = "1"
POLL_SECONDS = 60.0


class StoredArenaVerdict(Record):
    """`first` is the run whose call the judge read as Agent A."""

    inputs: dict[str, str]
    first: Literal["a", "b"]
    verdict: arena.ArenaVerdict | None
    error: str | None = None
    usage: list[Usage]
    batch_id: str | None = None
    judge_model: str | None = None

    @property
    def winner(self) -> Literal["a", "b", "tie"] | None:
        if self.verdict is None:
            return None
        if self.verdict.verdict == "tie":
            return "tie"
        second = "b" if self.first == "a" else "a"
        return self.first if self.verdict.verdict == "A" else second


def pair_dir(runs_dir: Path, a: str, b: str) -> Path:
    return runs_dir / "arena" / f"{a}__vs__{b}"


def completed(config: RunConfig, bundle: Bundle) -> dict[str, StoredConversation]:
    """The run's calls that are current under its protocol and did not end in an error."""
    store = RunStore(config.run_dir)
    simulation = config.benchmark.simulation
    found = {}
    for rollout in plan(bundle, config):
        stored = store.conversation(rollout.stem)
        if (
            stored is not None
            and stored.conversation.end_reason != "error"
            and not changed_inputs(stored, rollout, simulation)
        ):
            found[rollout.stem] = stored
    return found


def judge_spec(config: RunConfig) -> Endpoint:
    judge = config.benchmark.arena.judge
    if judge is None:
        raise ValueError("benchmark.yaml names no arena judge")
    return judge


def shared_inputs(config: RunConfig) -> dict[str, str]:
    return shared_inputs_for(config.benchmark)


def shared_inputs_for(benchmark: Benchmark) -> dict[str, str]:
    judge = benchmark.arena.judge
    if judge is None:
        raise ValueError("benchmark.yaml names no arena judge")
    return {
        "judge": endpoint_hash(judge),
        "prompt": template_hash(arena.PROMPT),
        "schema": stable_hash(arena.SCHEMA),
        "version": ARENA_VERSION,
    }


@dataclass
class ArenaResult:
    planned: int
    skipped: int
    judged: int = 0
    failed: int = 0
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True)
class Matchup:
    stem: str
    first: Literal["a", "b"]
    call_a: StoredConversation
    call_b: StoredConversation
    inputs: dict[str, str]

    def request(self, cache_ttl: Literal["5m", "1h"] = "5m") -> Request:
        shown = (self.call_a, self.call_b) if self.first == "a" else (self.call_b, self.call_a)
        return arena.build_request(shown[0].conversation, shown[1].conversation, cache_ttl)


def matchups(a: RunConfig, b: RunConfig, bundle: Bundle) -> list[Matchup]:
    """Every scenario both runs completed. Which call is shown first is fixed per scenario over
    the whole plan, so it does not move when another call completes."""
    calls_a, calls_b = completed(a, bundle), completed(b, bundle)
    first_a = arena.first_in_half([r.stem for r in plan(bundle, a)], a.benchmark.arena.seed)
    shared = shared_inputs(a)
    found = []
    for stem in sorted(calls_a.keys() & calls_b.keys()):
        first: Literal["a", "b"] = "a" if stem in first_a else "b"
        inputs = {
            **shared,
            "a": conversation_hash(calls_a[stem]),
            "b": conversation_hash(calls_b[stem]),
            "first": first,
        }
        found.append(Matchup(stem, first, calls_a[stem], calls_b[stem], inputs))
    return found


async def arena_stage(
    a: RunConfig,
    b: RunConfig,
    bundle: Bundle,
    judge: LLM,
    batch_api: BatchApi | None = None,
    discard_stale: bool = False,
    dry_run: bool = False,
    poll_seconds: float = POLL_SECONDS,
    on_start: Callable[[int], None] = lambda _: None,
    on_batch: Callable[[BatchStatus], None] = lambda _: None,
    on_done: Callable[[str, StoredArenaVerdict], None] = lambda *_: None,
) -> ArenaResult:
    store = RunStore(pair_dir(a.options.runs_dir, a.name, b.name))
    found = matchups(a, b, bundle)
    changed: dict[str, int] = {}
    todo: list[Matchup] = []
    for matchup in found:
        saved = verdict(store, matchup.stem)
        if saved is not None and saved.inputs == matchup.inputs and saved.error is None:
            continue
        if saved is not None and saved.inputs != matchup.inputs:
            for name in sorted(set(saved.inputs) | set(matchup.inputs)):
                if saved.inputs.get(name) != matchup.inputs.get(name):
                    changed[name] = changed.get(name, 0) + 1
        todo.append(matchup)
    if changed and not discard_stale:
        details = ", ".join(f"{name} ({n})" for name, n in changed.items())
        raise StaleRunError(
            f"{store.root} holds verdicts made from different inputs: {details}. "
            "Use --discard-stale to judge them again."
        )
    result = ArenaResult(planned=len(found), skipped=len(found) - len(todo))
    if dry_run or not todo:
        return result
    on_start(len(todo))

    def save(matchup: Matchup, stored: StoredArenaVerdict) -> None:
        write_atomic(store.verdict_path(JUDGE, matchup.stem), stored.model_dump_json(indent=1))
        result.judged += 1
        result.failed += stored.verdict is None
        for usage in stored.usage:
            result.usage = add(result.usage, usage)
        on_done(matchup.stem, stored)

    async def realtime(matchup: Matchup, usage: list[Usage] | None = None) -> StoredArenaVerdict:
        stored = StoredArenaVerdict(
            inputs=matchup.inputs, first=matchup.first, verdict=None, usage=list(usage or [])
        )
        for _ in range(arena.ATTEMPTS):
            try:
                response = await judge.stream(matchup.request())
            except LLMError as error:
                stored.error = f"{type(error).__name__}: {error}"
                continue
            stored.usage.append(response.usage)
            try:
                stored.verdict = arena.read_answer(response)
            except arena.ArenaError as error:
                stored.error = str(error)
                continue
            stored.error = None
            stored.judge_model = response.model
            break
        return stored

    async def retried(matchup: Matchup, usage: list[Usage]) -> None:
        save(matchup, await realtime(matchup, usage))

    async def collect(api: BatchApi, state: BatchState) -> list[tuple[Matchup, list[Usage]]]:
        """Save the batch's verdicts; return what it could not judge, to retry in realtime."""
        while True:
            status = await api.status(state.batch_id)
            on_batch(status)
            if status.ended:
                break
            await asyncio.sleep(poll_seconds)
        results = await api.results(state.batch_id)
        by_stem = {matchup.stem: matchup for matchup in todo}
        retry = []
        for stem, inputs in state.inputs.items():
            matchup = by_stem.get(stem)
            # A matchup judged since, or changed since it was submitted, is not this batch's.
            if matchup is None or matchup.inputs != inputs:
                continue
            answer = results.get(stem)
            usage = [answer.usage] if isinstance(answer, Response) else []
            try:
                if not isinstance(answer, Response):
                    raise arena.ArenaError(str(answer) if answer else "missing from the batch")
                stored = StoredArenaVerdict(
                    inputs=inputs,
                    first=matchup.first,
                    verdict=arena.read_answer(answer),
                    usage=usage,
                    batch_id=state.batch_id,
                    judge_model=answer.model,
                )
            except arena.ArenaError:
                retry.append((matchup, usage))
                continue
            save(matchup, stored)
        store.clear_batch_state(JUDGE)
        return retry

    with store.lock():
        if batch_api is None:

            async def judged(matchup: Matchup) -> None:
                save(matchup, await realtime(matchup))

            await asyncio.gather(*(judged(matchup) for matchup in todo))
        else:
            state = store.batch_state(JUDGE)
            retry = await collect(batch_api, state) if state is not None else []
            retrying = {matchup.stem for matchup, _ in retry}
            left = [m for m in todo if m.stem not in retrying and not current(store, m)]
            if left:
                state = BatchState(
                    batch_id=await batch_api.submit({m.stem: m.request("1h") for m in left}),
                    inputs={m.stem: m.inputs for m in left},
                )
                store.save_batch_state(JUDGE, state)
                retry += await collect(batch_api, state)
            await asyncio.gather(
                *(asyncio.create_task(retried(matchup, usage)) for matchup, usage in retry)
            )
    return result


def verdict(store: RunStore, stem: str) -> StoredArenaVerdict | None:
    path = store.verdict_path(JUDGE, stem)
    if not path.is_file():
        return None
    try:
        return StoredArenaVerdict.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def current(store: RunStore, matchup: Matchup) -> bool:
    saved = verdict(store, matchup.stem)
    return saved is not None and saved.inputs == matchup.inputs and saved.error is None


class Contestant(Record):
    run: str
    label: str
    organization: str


class ArenaCoverage(Record):
    planned: int
    judged: int
    missing: dict[str, int]


class ArenaProvenance(Record):
    judge_model: str
    judge_hash: str
    prompt_hash: str
    version: str
    seed: int


class ArenaSummary(Record):
    """A pair's result: aggregates only, so it can be published as it is."""

    a: Contestant
    b: Contestant
    benchmark_version: str
    coverage: ArenaCoverage
    scores: ArenaScores | None
    provenance: ArenaProvenance


def contestant(config: RunConfig) -> Contestant:
    display = config.model.display
    return Contestant(
        run=config.name,
        label=display.label if display else config.name,
        organization=display.organization if display else "",
    )


def summarize_arena(a: RunConfig, b: RunConfig, bundle: Bundle) -> ArenaSummary:
    """The pair's current verdicts as win, tie and loss rates for `a`; every planned scenario
    without one is counted as missing, with the reason."""
    store = RunStore(pair_dir(a.options.runs_dir, a.name, b.name))
    found = {matchup.stem: matchup for matchup in matchups(a, b, bundle)}
    missing: Counter[str] = Counter()
    outcomes = []
    for rollout in plan(bundle, a):
        matchup = found.get(rollout.stem)
        if matchup is None:
            missing["a call is missing or failed"] += 1
            continue
        saved = verdict(store, rollout.stem)
        if saved is None or saved.inputs != matchup.inputs:
            missing["not judged" if saved is None else "stale verdict"] += 1
            continue
        winner = saved.winner
        if winner is None:
            missing["judge failed"] += 1
            continue
        scenario = rollout.scenario
        outcomes.append(
            Outcome(rollout.stem, scenario.agent_id, scenario.language, winner, saved.first)
        )
    shared = shared_inputs(a)
    judge = judge_spec(a)
    return ArenaSummary(
        a=contestant(a),
        b=contestant(b),
        benchmark_version=a.benchmark.version,
        coverage=ArenaCoverage(
            planned=len(plan(bundle, a)),
            judged=len(outcomes),
            missing=dict(sorted(missing.items())),
        ),
        scores=arena_scores(outcomes) if outcomes else None,
        provenance=ArenaProvenance(
            judge_model=judge.model,
            judge_hash=shared["judge"],
            prompt_hash=shared["prompt"],
            version=ARENA_VERSION,
            seed=a.benchmark.arena.seed,
        ),
    )


def save_arena_summary(a: RunConfig, b: RunConfig, summary: ArenaSummary) -> Path:
    path = pair_dir(a.options.runs_dir, a.name, b.name) / "result.json"
    write_atomic(path, summary.model_dump_json(indent=1) + "\n")
    return path
