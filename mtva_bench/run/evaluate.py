"""The evaluate stage: judge every saved conversation that has no current verdict.

A verdict is redone when its conversation changed, since it described another call. A verdict
made with a different judge, judge prompt or judge code stops the run, as stale conversations
do, unless `discard_stale` asks for those verdicts to be redone. Verdicts that ended in an
error are judged again.

An agent's calls share the cached start of every judge request. In realtime, each agent's first
request finishes before the rest of its calls are judged, so they read that start from the
provider's cache. With `batch`, every call goes into one batch at half price, ordered by agent
and cached for an hour, since a batch runs its requests in no fixed order and may take longer
than the default five minutes to get through one agent. The batch id is saved before polling, so
a killed run collects the same batch. Calls the batch could not judge are retried in realtime.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from mtva_bench.capabilities import conduct
from mtva_bench.config import Benchmark, RunConfig, endpoint_hash, stable_hash
from mtva_bench.data import Bundle
from mtva_bench.llm.client import LLM
from mtva_bench.llm.http import LLMError
from mtva_bench.llm.types import BatchStatus, Request, Response, Usage
from mtva_bench.prompts import template_hash
from mtva_bench.run.manifest import Judging, now
from mtva_bench.run.stages import Rollout, StaleRunError, interleave, plan
from mtva_bench.run.store import BatchState, RunStore, StoredConversation, StoredVerdict

# Bump when the code that reads the judge's answer changes; a change to what the judge reads
# shows in each verdict's `request` hash on its own.
JUDGE_VERSION = "1"
JUDGED_CONVERSATION = {"messages", "tool_events", "end_reason"}
JUDGE = "conduct"
INPUT_LABELS = {
    "judge": "judge model",
    "prompt": "judge prompt",
    "evaluation": "scenario grading",
    "request": "what the judge reads",
    "schema": "verdict schema",
    "version": "judge code version",
    "fallback": "fallback judge",
}


POLL_SECONDS = 60.0
# A verdict depends on the judge that made it, not on the fallback that would have stood in had
# it refused, nor on how many repeats are asked for, which only adds verdicts beside it.
UNHASHED_JUDGE_FIELDS = frozenset({"fallback", "repeats"})


class BatchApi(Protocol):
    async def submit(self, requests: dict[str, Request]) -> str: ...

    async def status(self, batch_id: str) -> BatchStatus: ...

    async def results(self, batch_id: str) -> dict[str, Response | LLMError]: ...


@dataclass
class EvaluateResult:
    """`usage` counts every judge request; `batch_usage` is the part billed at batch price."""

    planned: int
    skipped: int
    judged: int = 0
    failed: int = 0
    retried: int = 0
    fallback: int = 0
    usage: Usage = field(default_factory=Usage)
    batch_usage: Usage = field(default_factory=Usage)


def judge_inputs(benchmark: Benchmark) -> dict[str, str]:
    return {
        "judge": endpoint_hash(benchmark.judge, UNHASHED_JUDGE_FIELDS),
        "prompt": template_hash(conduct.PROMPT),
        "schema": stable_hash(conduct.SCHEMA),
        "version": JUDGE_VERSION,
    }


def fallback_hash(config: RunConfig) -> str:
    fallback = config.benchmark.judge.fallback
    return endpoint_hash(fallback) if fallback else stable_hash(None)


def expected_inputs(saved: dict[str, str], inputs: dict[str, str], fallback: str) -> dict[str, str]:
    """What a saved verdict's inputs must be to be current: a verdict the fallback judge made
    also depends on the fallback judge."""
    return {**inputs, "fallback": fallback} if "fallback" in saved else inputs


def conversation_hash(stored: StoredConversation) -> str:
    """The parts of a saved call the judge's request is built from, so a field added to saved
    calls later does not make every verdict stale."""
    return stable_hash(stored.conversation.model_dump(mode="json", include=JUDGED_CONVERSATION))


def call_inputs(
    shared: dict[str, str],
    rollout: Rollout,
    stored: StoredConversation,
    bundle: Bundle,
    text_with_tool_calls: bool = True,
) -> dict[str, str]:
    """What one verdict depends on. `request` is the exact text the judge reads, so any change
    to it (the prompt, the scenario's grading, how the call or the checks are shown) makes only
    the verdicts it touches stale."""
    request = conduct.build_request(
        bundle.agent(rollout.scenario),
        rollout.scenario,
        stored.conversation,
        text_with_tool_calls=text_with_tool_calls,
    )
    return {
        **shared,
        "conversation": conversation_hash(stored),
        "evaluation": stable_hash(rollout.scenario.evaluation.model_dump(mode="json")),
        "request": stable_hash([[message.role, message.content] for message in request.messages]),
    }


def add(total: Usage, usage: Usage) -> Usage:
    return Usage(
        **{name: getattr(total, name) + getattr(usage, name) for name in Usage.model_fields}
    )


async def evaluate_stage(
    config: RunConfig,
    bundle: Bundle,
    judge: LLM,
    fallback: LLM | None = None,
    batch_api: BatchApi | None = None,
    poll_seconds: float = POLL_SECONDS,
    on_start: Callable[[int], None] = lambda _: None,
    on_batch: Callable[[BatchStatus], None] = lambda _: None,
    on_done: Callable[[str, StoredVerdict], None] = lambda *_: None,
) -> EvaluateResult:
    if config.benchmark.judge.repeats != 1:
        raise ValueError("judge repeats above 1 are not supported yet")
    if config.options.batch and batch_api is None:
        raise ValueError("batch judging needs a batch api")
    store = RunStore(config.run_dir)
    shared = judge_inputs(config.benchmark)
    fallback_inputs = fallback_hash(config)
    speaks_with_tools = config.model.text_with_tool_calls
    ready: list[tuple[Rollout, StoredConversation, dict[str, str]]] = []
    for rollout in plan(bundle, config):
        stored = store.conversation(rollout.stem)
        if stored is not None and stored.conversation.end_reason != "error":
            inputs = call_inputs(shared, rollout, stored, bundle, speaks_with_tools)
            ready.append((rollout, stored, inputs))

    changed: dict[str, int] = {}
    todo: list[Rollout] = []
    work: dict[str, tuple[StoredConversation, dict[str, str]]] = {}
    for rollout, stored, inputs in ready:
        verdict = store.verdict_header(JUDGE, rollout.stem)
        if verdict is not None and verdict.inputs.get("conversation") == inputs["conversation"]:
            expected = expected_inputs(verdict.inputs, inputs, fallback_inputs)
            for name in sorted(set(expected) | set(verdict.inputs)):
                if verdict.inputs.get(name) != expected.get(name):
                    changed[name] = changed.get(name, 0) + 1
            if verdict.inputs == expected and verdict.error is None:
                continue
        todo.append(rollout)
        work[rollout.stem] = (stored, inputs)
    if changed and not config.options.discard_stale:
        details = ", ".join(f"{INPUT_LABELS.get(name, name)} ({n})" for name, n in changed.items())
        raise StaleRunError(
            f"{config.run_dir} holds verdicts made from different inputs: {details}. "
            "Use --discard-stale to judge them again."
        )
    result = EvaluateResult(planned=len(ready), skipped=len(ready) - len(todo))
    if config.options.dry_run:
        return result
    on_start(len(todo))

    def save(
        rollout: Rollout,
        judged: conduct.JudgeResult,
        batch_id: str | None = None,
    ) -> None:
        inputs = work[rollout.stem][1]
        verdict = StoredVerdict(
            inputs={**inputs, "fallback": fallback_inputs} if judged.fallback else inputs,
            judge=JUDGE,
            verdict=judged.verdict,
            error=judged.error,
            usage=judged.usage,
            batch_id=batch_id,
            judge_model=judged.model,
        )
        store.save_verdict(rollout.stem, verdict)
        result.judged += 1
        result.failed += judged.verdict is None
        result.fallback += judged.fallback and judged.verdict is not None
        for index, usage in enumerate(judged.usage):
            result.usage = add(result.usage, usage)
            if batch_id is not None and index == 0:
                result.batch_usage = add(result.batch_usage, usage)
        on_done(rollout.stem, verdict)

    async def realtime(
        rollout: Rollout, attempts: int = conduct.ATTEMPTS, refused: bool = False
    ) -> conduct.JudgeResult:
        stored = work[rollout.stem][0]
        return await conduct.judge(
            judge,
            bundle.agent(rollout.scenario),
            rollout.scenario,
            stored.conversation,
            attempts,
            fallback,
            refused,
            speaks_with_tools,
        )

    async def realtime_stage() -> None:
        ordered = interleave(todo)
        opened = {rollout.scenario.agent_id: asyncio.Event() for rollout in ordered}
        leaders: dict[str, str] = {}
        for rollout in ordered:
            leaders.setdefault(rollout.scenario.agent_id, rollout.stem)

        async def one(rollout: Rollout) -> None:
            gate = opened[rollout.scenario.agent_id]
            if leaders[rollout.scenario.agent_id] != rollout.stem:
                await gate.wait()
            try:
                judged = await realtime(rollout)
            finally:
                gate.set()
            save(rollout, judged)

        await asyncio.gather(*(one(rollout) for rollout in ordered))

    async def batch_stage(api: BatchApi) -> None:
        by_stem = {rollout.stem: rollout for rollout in todo}
        pending = set(by_stem)
        retry: dict[str, conduct.JudgeResult] = {}
        batch_ids: dict[str, str] = {}
        state = store.batch_state(JUDGE)
        while state is not None or pending:
            if state is None:
                ordered = sorted(
                    (by_stem[stem] for stem in pending),
                    key=lambda rollout: (rollout.scenario.agent_id, rollout.stem),
                )
                requests = {
                    rollout.stem: conduct.build_request(
                        bundle.agent(rollout.scenario),
                        rollout.scenario,
                        work[rollout.stem][0].conversation,
                        cache_ttl="1h",
                        text_with_tool_calls=speaks_with_tools,
                    )
                    for rollout in ordered
                }
                state = BatchState(
                    batch_id=await api.submit(requests),
                    inputs={stem: work[stem][1] for stem in requests},
                )
                store.save_batch_state(JUDGE, state)
            while True:
                status = await api.status(state.batch_id)
                on_batch(status)
                if status.ended:
                    break
                await asyncio.sleep(poll_seconds)
            results = await api.results(state.batch_id)
            for stem, inputs in state.inputs.items():
                # A rollout judged since, or changed since it was submitted, is not this batch's.
                if stem not in pending or work[stem][1] != inputs:
                    continue
                pending.discard(stem)
                rollout = by_stem[stem]
                conversation = work[stem][0].conversation
                answer = results.get(stem)
                judged = conduct.JudgeResult()
                if isinstance(answer, Response):
                    judged.usage.append(answer.usage)
                    judged.model = answer.model
                    try:
                        judged.verdict = conduct.read_answer(
                            answer, bundle.agent(rollout.scenario), rollout.scenario, conversation
                        )
                    except conduct.VerdictError as error:
                        judged.error = str(error)
                        judged.refused = answer.finish_reason == "refusal"
                else:
                    judged.error = str(answer) if answer else "missing from the batch results"
                if judged.verdict is not None:
                    save(rollout, judged, state.batch_id)
                else:
                    retry[stem] = judged
                    batch_ids[stem] = state.batch_id
            store.clear_batch_state(JUDGE)
            state = None

        async def again(stem: str) -> None:
            first = retry[stem]
            second = await realtime(
                by_stem[stem], attempts=conduct.ATTEMPTS - 1, refused=first.refused
            )
            second.usage = first.usage + second.usage
            save(by_stem[stem], second, batch_ids[stem] if first.usage else None)

        result.retried = len(retry)
        await asyncio.gather(*(again(stem) for stem in sorted(retry)))

    with store.lock():
        if config.options.batch:
            assert batch_api is not None
            await batch_stage(batch_api)
        else:
            await realtime_stage()
        manifest = store.manifest()
        if manifest is not None:
            current = [
                header
                for rollout, _, inputs in ready
                if (header := store.verdict_header(JUDGE, rollout.stem)) is not None
                and header.inputs == expected_inputs(header.inputs, inputs, fallback_inputs)
            ]
            manifest.judging = Judging(
                judge=JUDGE,
                model=config.benchmark.judge.model,
                inputs_hash=stable_hash(shared),
                planned=len(ready),
                judged=sum(header.error is None for header in current),
                failed=sum(header.error is not None for header in current),
                fallback_model=config.benchmark.judge.fallback.model
                if config.benchmark.judge.fallback
                else None,
                fallback_judged=sum(
                    "fallback" in header.inputs and header.error is None for header in current
                ),
                updated_at=now(),
            )
            store.save_manifest(manifest)
    return result
