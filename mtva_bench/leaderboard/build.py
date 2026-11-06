"""The leaderboard: every published summary in leaderboard/runs/, checked to be comparable.

A summary is refused when it was made under another protocol than benchmark.yaml (another
caller, judge, judge prompt or data) or scored with other capability weights, since its numbers
would not mean the same thing. Republishing the run under the current protocol fixes it.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from mtva_bench.config import Benchmark
from mtva_bench.run.arena import ArenaSummary, shared_inputs_for
from mtva_bench.run.score import Summary, protocol, publish_problems
from mtva_bench.scoring.aggregate import Scores
from mtva_bench.scoring.latency import Latency


class LeaderboardError(ValueError):
    pass


@dataclass(frozen=True)
class Entry:
    """A published run with the scores and latency the leaderboard needs."""

    summary: Summary
    scores: Scores
    latency: Latency

    @property
    def label(self) -> str:
        display = self.summary.model.get("display")
        return display["label"] if display else self.summary.run

    @property
    def organization(self) -> str:
        display = self.summary.model.get("display")
        return display["organization"] if display else ""

    @property
    def reasoning(self) -> str:
        display = self.summary.model.get("display")
        return display["reasoning"] if display else ""


def problems(summary: Summary, benchmark: Benchmark) -> list[str]:
    found = publish_problems(summary)
    if summary.benchmark_version != benchmark.version:
        found.append(f"benchmark {summary.benchmark_version}, not {benchmark.version}")
    if benchmark.data.sha256 and summary.provenance.bundle_hash != benchmark.data.sha256:
        found.append("scored on other data than benchmark.yaml pins")
    expected = protocol(benchmark)
    provenance = summary.provenance.model_dump()
    changed = [name for name, value in expected.items() if provenance[name] != value]
    if changed:
        found.append(f"made under another protocol: {', '.join(changed)}")
    if summary.weights != benchmark.capabilities:
        found.append("scored with other capability weights")
    return found


def load_entries(runs_dir: Path, benchmark: Benchmark) -> list[Entry]:
    """Every published run, best MTVA Score first."""
    entries = []
    refused = []
    for path in sorted(runs_dir.glob("*.json")):
        try:
            summary = Summary.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as error:
            refused.append(f"{path.name}: not a published summary ({error.error_count()} errors)")
            continue
        found = problems(summary, benchmark)
        if found or summary.scores is None or summary.latency_ms is None:
            refused.append(f"{path.name}: {'; '.join(found)}")
            continue
        entries.append(Entry(summary, summary.scores, summary.latency_ms))
    if refused:
        raise LeaderboardError("refused:\n  " + "\n  ".join(refused))
    if not entries:
        raise LeaderboardError(f"no published runs in {runs_dir}")
    return sorted(entries, key=lambda entry: -entry.scores.mtva_score.value)


def load_arena(arena_dir: Path, benchmark: Benchmark) -> list[ArenaSummary]:
    """Every published arena pair, refused when another judge, judge prompt or arena code made
    it, or when under 99% of its scenarios were judged."""
    expected = shared_inputs_for(benchmark)
    pairs = []
    refused = []
    for path in sorted(arena_dir.glob("*.json")):
        try:
            pair = ArenaSummary.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as error:
            refused.append(f"{path.name}: not an arena result ({error.error_count()} errors)")
            continue
        made = pair.provenance
        changed = [
            name
            for name, value in (
                ("judge", made.judge_hash),
                ("prompt", made.prompt_hash),
                ("version", made.version),
            )
            if value != expected[name]
        ]
        coverage = pair.coverage
        if changed or pair.scores is None or coverage.judged < 0.99 * coverage.planned:
            reason = f"made under another {', '.join(changed)}" if changed else "incomplete"
            refused.append(f"{path.name}: {reason}")
            continue
        pairs.append(pair)
    if refused:
        raise LeaderboardError("refused:\n  " + "\n  ".join(refused))
    return pairs


def leaderboard_json(entries: list[Entry], pairs: list[ArenaSummary], benchmark: Benchmark) -> str:
    return json.dumps(
        {
            "benchmark_version": benchmark.version,
            "bundle_version": benchmark.data.version,
            "runs": [entry.summary.model_dump(mode="json") for entry in entries],
            "arena": [pair.model_dump(mode="json") for pair in pairs],
        },
        indent=1,
        ensure_ascii=False,
    )
