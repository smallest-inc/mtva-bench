import asyncio
import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from mtva_bench.cli import cli
from mtva_bench.config import RunConfig, RunOptions, load_run
from mtva_bench.data import Bundle, hash_files, load_bundle
from mtva_bench.llm.fake import FakeLLM
from mtva_bench.llm.http import LLMError
from mtva_bench.llm.types import Output, Request, Response, ToolCall
from mtva_bench.run import RunLockedError, RunStore, StaleRunError, simulate_stage
from mtva_bench.run.evaluate import judge_inputs
from mtva_bench.run.stages import Rollout, interleave, plan, select_scenarios

ROOT = Path(__file__).parent.parent
FIXTURE = Path(__file__).parent / "fixtures" / "bundle"
MODEL = Path(__file__).parent / "fixtures" / "model.yaml"
CLINICS = {"demo_dental_01": "Clinic A", "demo_dental_02": "Clinic B", "demo_dental_03": "Clinic C"}


def make_bundle(tmp_path: Path, languages: dict[str, str] | None = None) -> Bundle:
    """The fixture bundle with three scenarios, each naming its own clinic."""
    root = Path(shutil.copytree(FIXTURE, tmp_path / "bundle"))
    source = root / "scenarios" / "demo_dental" / "demo_dental_01.json"
    base = json.loads(source.read_text())
    for scenario_id, clinic in CLINICS.items():
        scenario = {**base, "scenario_id": scenario_id}
        scenario["variables"] = {**base["variables"], "clinic_name": clinic}
        source.with_name(f"{scenario_id}.json").write_text(json.dumps(scenario))
    manifest = json.loads((root / "bundle.json").read_text())
    manifest["files"] = hash_files(root)
    (root / "bundle.json").write_text(json.dumps(manifest))
    return load_bundle(root)


def make_config(tmp_path: Path, **options: object) -> RunConfig:
    return load_run(
        MODEL,
        ROOT / "benchmark.yaml",
        RunOptions(runs_dir=tmp_path / "runs", **options),  # type: ignore[arg-type]
    )


def hang_up(failing: frozenset[str] = frozenset()) -> FakeLLM:
    """An agent that ends every call at once, failing for the clinics in `failing`."""

    def respond(request: Request) -> Output:
        prompt = request.messages[0].content or ""
        if any(clinic in prompt for clinic in failing):
            raise LLMError("upstream 503", retryable=True)
        return Output(
            text="Goodbye.", tool_calls=[ToolCall(id="c", name="end_call", arguments="{}")]
        )

    return FakeLLM(respond)


async def run(config: RunConfig, bundle: Bundle, model: FakeLLM):
    return await simulate_stage(config, bundle, model, FakeLLM(lambda _: "unused"))


async def test_resume_after_failure_and_kill(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    config = make_config(tmp_path)
    first = await run(config, bundle, hang_up(frozenset({"Clinic B"})))
    assert (first.planned, first.simulated, first.failed) == (3, 3, 1)
    store = RunStore(config.run_dir)
    assert store.conversation("demo_dental_02__r0").conversation.end_reason == "error"

    store.conversation_path("demo_dental_03__r0").unlink()
    (store.conversations_dir / ".demo_dental_03__r0.json.tmp").write_text('{"half": ')
    model = hang_up()
    second = await run(config, bundle, model)
    assert (second.skipped, second.simulated, second.failed) == (1, 2, 0)
    assert len(model.requests) == 2
    manifest = json.loads((config.run_dir / "manifest.json").read_text())
    assert (manifest["planned"], manifest["conversations"], manifest["failed"]) == (3, 3, 0)
    assert manifest["bundle_hash"] == bundle.hash

    third = await run(config, bundle, hang_up())
    assert (third.skipped, third.simulated) == (3, 0)


async def test_changed_inputs_are_refused(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    config = make_config(tmp_path)
    await run(config, bundle, hang_up())

    changed = config.model_copy(
        update={"model": config.model.model_copy(update={"params": {"temperature": 0.2}})}
    )
    with pytest.raises(StaleRunError, match=r"model spec \(3\)"):
        await run(changed, bundle, hang_up())

    path = bundle.root / "scenarios" / "demo_dental" / "demo_dental_01.json"
    scenario = json.loads(path.read_text())
    scenario["summary"] = "An edited summary."
    path.write_text(json.dumps(scenario))
    unchanged = await run(config, load_bundle(bundle.root), hang_up())
    assert unchanged.simulated == 0
    scenario["caller"]["profile"] += " She is in a hurry."
    path.write_text(json.dumps(scenario))
    edited = load_bundle(bundle.root)
    with pytest.raises(StaleRunError, match=r"scenario \(caller, world or tool rules\) \(1\)"):
        await run(config, edited, hang_up())

    slower = config.model_copy(
        update={"model": config.model.model_copy(update={"timeout_s": 30.0})}
    )
    assert (await run(slower, bundle, hang_up())).simulated == 0

    redo = changed.model_copy(
        update={"options": changed.options.model_copy(update={"discard_stale": True})}
    )
    result = await run(redo, edited, hang_up())
    assert (result.skipped, result.simulated) == (0, 3)
    run_yaml = (config.run_dir / "run.yaml").read_text()
    assert "temperature: 0.2" in run_yaml and "SMALLEST_API_KEY" not in run_yaml


def with_simulation(config: RunConfig, **changes: object) -> RunConfig:
    simulation = config.benchmark.simulation.model_copy(update=changes)
    benchmark = config.benchmark.model_copy(update={"simulation": simulation})
    return config.model_copy(update={"benchmark": benchmark})


def edit_call(store: RunStore, stem: str, **changes: object) -> None:
    """Rewrite a saved conversation as if it had gone differently: `end_reason`, or the
    `finish_reason` and `output_tokens` of its model calls."""
    stored = store.conversation(stem)
    conversation = stored.conversation
    if "end_reason" in changes:
        conversation = conversation.model_copy(update={"end_reason": changes["end_reason"]})
    calls = [
        call.model_copy(
            update={
                "finish_reason": changes.get("finish_reason", call.finish_reason),
                "usage": call.usage.model_copy(
                    update={"output_tokens": changes.get("output_tokens", 50)}
                ),
            }
        )
        for call in conversation.model_calls
    ]
    conversation = conversation.model_copy(update={"model_calls": calls})
    store.save_conversation(stored.model_copy(update={"conversation": conversation}))


async def test_a_changed_limit_stales_only_the_calls_it_changes(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    config = make_config(tmp_path)
    await run(config, bundle, hang_up())
    store = RunStore(config.run_dir)
    for stem in ("demo_dental_01__r0", "demo_dental_02__r0", "demo_dental_03__r0"):
        edit_call(store, stem)

    looser = with_simulation(config, max_turns=40, max_reply_tokens=8192)
    assert (await run(looser, bundle, hang_up())).simulated == 0
    tighter = with_simulation(config, max_turns=1, max_reply_tokens=100)
    assert (await run(tighter, bundle, hang_up())).simulated == 0

    edit_call(store, "demo_dental_01__r0", end_reason="max_turns")
    edit_call(store, "demo_dental_02__r0", finish_reason="length")
    edit_call(store, "demo_dental_03__r0", output_tokens=500)
    with pytest.raises(StaleRunError, match=r"max_turns \(1\), max_reply_tokens \(1\)"):
        await run(looser, bundle, hang_up())
    with pytest.raises(StaleRunError, match=r"max_reply_tokens \(2\)"):
        await run(tighter, bundle, hang_up())
    assert (await run(config, bundle, hang_up())).simulated == 0


def test_weights_arena_and_repeats_stale_nothing(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    config = make_config(tmp_path)
    benchmark = config.benchmark.model_copy(
        update={
            "capabilities": {"task_completion": 2.0, "tool_use": 0.5},
            "arena": config.benchmark.arena.model_copy(update={"judge": None}),
            "judge": config.benchmark.judge.model_copy(update={"repeats": 3}),
        }
    )
    reweighted = config.model_copy(update={"benchmark": benchmark})
    assert [r.inputs for r in plan(bundle, reweighted)] == [r.inputs for r in plan(bundle, config)]
    assert judge_inputs(reweighted.benchmark) == judge_inputs(config.benchmark)


async def test_dry_run_calls_nothing(tmp_path: Path) -> None:
    model = hang_up()
    result = await run(make_config(tmp_path, dry_run=True), make_bundle(tmp_path), model)
    assert (result.planned, result.simulated) == (3, 0)
    assert model.requests == []
    assert not (tmp_path / "runs").exists()


def test_limit_spreads_over_languages(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    scenarios = {**bundle.scenarios}
    scenarios["demo_dental_03"] = scenarios["demo_dental_03"].model_copy(
        update={"language": "English"}
    )
    mixed = Bundle(bundle.root, bundle.manifest, bundle.voice_guide, bundle.agents, scenarios)
    picked = select_scenarios(mixed, make_config(tmp_path, limit=2))
    assert sorted(s.language for s in picked) == ["English", "Hindi"]
    assert picked == select_scenarios(mixed, make_config(tmp_path, limit=2))
    with pytest.raises(ValueError, match="unknown agents"):
        select_scenarios(mixed, make_config(tmp_path, agents=["nobody"]))


def unpinned_benchmark(tmp_path: Path) -> Path:
    """benchmark.yaml without its data hash pin, for runs on the fixture bundle."""
    path = tmp_path / "benchmark.yaml"
    lines = (ROOT / "benchmark.yaml").read_text().splitlines()
    path.write_text("\n".join(line for line in lines if "sha256:" not in line) + "\n")
    return path


def test_cli_dry_run(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        [
            "run",
            str(MODEL),
            "--benchmark",
            str(unpinned_benchmark(tmp_path)),
            "--data",
            str(FIXTURE),
            "--runs-dir",
            str(tmp_path),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "1 rollouts planned · 0 already done · would simulate 1" in result.output


async def test_one_run_per_folder(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    config = make_config(tmp_path)
    store = RunStore(config.run_dir)
    with store.lock(), pytest.raises(RunLockedError, match="in use by another run"):
        await run(config, bundle, hang_up())
    (config.run_dir / ".lock").write_text("999999")
    result = await run(config, bundle, hang_up())
    assert result.simulated == 3
    assert not (config.run_dir / ".lock").exists()


async def test_an_agents_first_call_opens_before_the_rest(tmp_path: Path) -> None:
    events: list[str] = []

    class SlowFirst:
        async def stream(self, request: Request) -> Response:
            clinic = next(c for c in CLINICS.values() if c in (request.messages[0].content or ""))
            events.append(f"start {clinic}")
            if clinic == "Clinic A":
                await asyncio.sleep(0.05)
            events.append(f"end {clinic}")
            return await hang_up().stream(request)

        async def complete(self, request: Request) -> Response:
            return await self.stream(request)

    await simulate_stage(make_config(tmp_path), make_bundle(tmp_path), SlowFirst(), FakeLLM(str))
    assert events[:2] == ["start Clinic A", "end Clinic A"]


def test_interleave_spreads_agents() -> None:
    scenario = load_bundle(FIXTURE).scenarios["demo_dental_01"]
    rollouts = [
        Rollout(
            scenario.model_copy(update={"agent_id": agent, "scenario_id": f"{agent}_{i}"}), 0, {}
        )
        for agent in ("a", "b")
        for i in range(3)
    ]
    order = [rollout.scenario.scenario_id for rollout in interleave(rollouts)]
    assert order == ["a_0", "b_0", "a_1", "b_1", "a_2", "b_2"]
