import json
from itertools import pairwise
from pathlib import Path

import pytest
from click.testing import CliRunner

from mtva_bench.cli import cli
from mtva_bench.llm.types import Timing, Usage
from mtva_bench.run import RunStore, evaluate_stage
from mtva_bench.run.evaluate import JUDGE
from mtva_bench.run.score import publish_problems, score_stage
from mtva_bench.scoring.aggregate import CallScore, aggregate, head_to_head
from mtva_bench.scoring.latency import ttft
from mtva_bench.scoring.stats import clustered_bootstrap, percentile
from mtva_bench.simulate.types import Conversation, ModelCall
from tests.test_capabilities import fake_judge
from tests.test_run import hang_up, make_bundle, make_config, run, unpinned_benchmark

WEIGHTS = {
    "task_completion": 1.0,
    "tool_use": 1.0,
    "instruction_following": 1.0,
    "faithfulness": 1.0,
}


def call(
    stem: str, agent: str, raw: tuple[int, int, int, int], outcome: str = "reached", **slices: str
) -> CallScore:
    return CallScore(
        stem=stem,
        agent_id=agent,
        slices={
            "language": "Hindi",
            "industry": "Health",
            "difficulty": "hard",
            "direction": "inbound",
            "agent": agent,
            **slices,
        },
        raw=dict(zip(WEIGHTS, raw, strict=True)),
        outcome=outcome,
        ending="terminal_tool",
    )


def test_percentile() -> None:
    assert percentile([3, 1, 2], 50) == 2
    assert percentile([0, 10], 90) == 9
    assert percentile([7], 10) == 7


def test_the_bootstrap_resamples_whole_agents() -> None:
    clusters = ["a"] * 50 + ["b"] * 50
    columns = {"split": [0.0] * 50 + [100.0] * 50, "flat": [40.0] * 100}
    intervals = clustered_bootstrap(clusters, columns, resamples=500)
    assert intervals["flat"] == (40.0, 40.0)
    # Two agents that disagree completely: the mean is anywhere from 0 to 100, not 50 ± 10 as
    # it would be if the 100 calls were independent.
    assert intervals["split"] == (0.0, 100.0)
    assert clustered_bootstrap(clusters, columns, resamples=500) == intervals


def test_scores_goal_and_clean_calls() -> None:
    calls = [
        call("s1", "a", (5, 5, 5, 5), language="Hindi"),
        call("s2", "a", (3, 5, 1, 5), outcome="partial", language="Tamil", difficulty="easy"),
        call("s3", "b", (5, 4, 5, 5), outcome="alternative", language="Tamil"),
    ]
    scores = aggregate(calls, WEIGHTS)
    assert scores.mtva_score.value == round((100 + 62.5 + 93.75) / 3, 1)
    low, high = scores.mtva_score.ci
    assert low <= scores.mtva_score.value <= high
    assert scores.capabilities["instruction_following"].value == round(200 / 3, 1)
    assert scores.goal_reached.value == 66.7
    assert scores.clean_calls.value == 33.3
    assert scores.outcomes == {"alternative": 1, "partial": 1, "reached": 1}
    assert list(scores.slices["difficulty"]) == ["easy", "hard"]
    tamil = scores.slices["language"]["Tamil"]
    assert (tamil.calls, tamil.mtva_score) == (2, 78.1)
    assert "agent" not in scores.slices

    weighted = aggregate(calls, {**WEIGHTS, "instruction_following": 2.0})
    assert weighted.mtva_score.value < scores.mtva_score.value


def test_head_to_head_pairs_the_same_calls() -> None:
    a = [
        call("s1", "a", (5, 5, 5, 5)),
        call("s2", "a", (3, 3, 3, 3)),
        call("s3", "b", (4, 4, 4, 4)),
    ]
    b = [
        call("s1", "a", (4, 4, 4, 4)),
        call("s2", "a", (5, 5, 5, 5)),
        call("s4", "b", (5, 5, 5, 5)),
    ]
    result = head_to_head(a, b, WEIGHTS)
    assert result.calls == 2
    assert result.mtva_score.value == round((-25 + 50) / 2, 1)
    assert (result.wins, result.ties, result.losses) == (1, 0, 1)


def test_ttft_reports_the_opening_request_apart() -> None:
    def request(turn: int, round_: int, ms: float | None) -> ModelCall:
        return ModelCall(
            turn=turn,
            round=round_,
            timing=Timing(first_token_ms=ms, total_ms=1000),
            usage=Usage(),
            spoke=True,
            tool_calls=0,
            finish_reason="stop",
        )

    conversation = Conversation(
        scenario_id="s",
        agent_id="a",
        model="m",
        rollout=0,
        seed=0,
        messages=[],
        model_calls=[
            request(0, 0, 900),
            request(0, 1, 100),
            request(1, 0, 300),
            request(2, 0, None),
        ],
        end_reason="caller_done",
        duration_ms=1,
    )
    latency = ttft([conversation])
    assert latency is not None
    assert (latency.ttft_cold_p50, latency.ttft_warm_p50) == (900, 200)
    assert (latency.cold_requests, latency.warm_requests) == (1, 2)


async def test_score_stage_counts_only_current_calls(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    config = make_config(tmp_path)
    await run(config, bundle, hang_up(frozenset({"Clinic B"})))
    await evaluate_stage(config, bundle, fake_judge([]))

    summary, calls = score_stage(config, bundle)
    assert (summary.coverage.planned, summary.coverage.scored) == (3, 2)
    assert summary.coverage.missing == {"call failed": 1}
    assert [c.stem for c in calls] == ["demo_dental_01__r0", "demo_dental_03__r0"]
    # Task not reached (2), key action met (5), one major broken rule (3), nothing unfaithful (5).
    assert summary.scores is not None
    assert summary.scores.mtva_score.value == 68.8
    assert publish_problems(summary)
    saved = (config.run_dir / "summary.json").read_text()
    assert "Clinic" not in saved and "base_url" not in saved

    store = RunStore(config.run_dir)
    stored = store.verdict(JUDGE, "demo_dental_01__r0")
    stored.inputs["prompt"] = "old"
    store.save_verdict("demo_dental_01__r0", stored)
    summary, _ = score_stage(config, bundle)
    assert summary.coverage.missing == {"call failed": 1, "stale verdict": 1}


async def test_publish_writes_the_leaderboard_file(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    config = make_config(tmp_path)
    await run(config, bundle, hang_up())
    await evaluate_stage(config, bundle, fake_judge([]))
    store = RunStore(config.run_dir)
    for path in sorted(store.conversations_dir.glob("*.json")):
        stored = store.conversation(path.stem)
        opening = stored.conversation.model_calls[0]
        stored.conversation.model_calls.append(opening.model_copy(update={"turn": 1}))
        store.save_conversation(stored)
    options = ["--benchmark", str(unpinned_benchmark(tmp_path)), "--data", str(bundle.root)]
    leaderboard = tmp_path / "leaderboard"

    result = CliRunner().invoke(
        cli, ["publish", str(config.run_dir), *options, "--to", str(leaderboard)]
    )
    assert result.exit_code == 0, result.output
    published = json.loads((leaderboard / f"{config.name}.json").read_text())
    assert published["coverage"] == {"planned": 3, "scored": 3, "missing": {}}
    assert published["scores"]["mtva_score"]["value"] == 68.8
    assert published["model"]["serving"]["kind"] == "self-hosted"

    result = CliRunner().invoke(
        cli, ["report", str(config.run_dir), str(config.run_dir), *options, "--by", "language"]
    )
    assert result.exit_code == 0, result.output
    assert "won / tied / lost" in result.output


def test_the_leaderboard_refuses_runs_from_another_protocol(tmp_path: Path) -> None:
    from mtva_bench.config import load_benchmark
    from mtva_bench.leaderboard.build import LeaderboardError, load_entries
    from mtva_bench.leaderboard.charts import charts, frontier

    root = Path(__file__).parent.parent
    benchmark = load_benchmark(root / "benchmark.yaml")
    runs = tmp_path / "runs"
    runs.mkdir()
    for path in (root / "leaderboard" / "runs").glob("*.json"):
        (runs / path.name).write_text(path.read_text())
    entries = load_entries(runs, benchmark)
    assert [e.scores.mtva_score.value for e in entries] == sorted(
        (e.scores.mtva_score.value for e in entries), reverse=True
    )
    fastest = frontier(entries)
    assert fastest[0].latency.ttft_warm_p50 == min(e.latency.ttft_warm_p50 for e in entries)
    assert all(a.scores.mtva_score.value < b.scores.mtva_score.value for a, b in pairwise(fastest))
    drawn = charts(entries, [])
    assert all(svg.startswith("<svg") for svg in drawn.values())

    other = json.loads(next(runs.glob("*.json")).read_text())
    other["provenance"]["judge_inputs_hash"] = "another judge"
    next(runs.glob("*.json")).write_text(json.dumps(other))
    with pytest.raises(LeaderboardError, match="another protocol: judge_inputs_hash"):
        load_entries(runs, benchmark)


def test_the_readme_is_rewritten_between_its_markers() -> None:
    from mtva_bench.leaderboard.readme import ReadmeError, fill

    readme = "intro\n<!-- mtva:table -->\n<!-- /mtva:table -->\nend\n"
    once = fill(readme, {"table": "| a |\n| b |"})
    assert once == "intro\n<!-- mtva:table -->\n| a |\n| b |\n<!-- /mtva:table -->\nend\n"
    assert fill(once, {"table": "| a |\n| b |"}) == once
    assert fill(once, {"table": "| c |"}).count("| a |") == 0
    with pytest.raises(ReadmeError, match="mtva:arena"):
        fill(readme, {"arena": "x"})
