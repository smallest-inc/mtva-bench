"""Terminal tables: runs side by side, a cut by slice, a run's weakest calls, and two runs head
to head."""

from collections.abc import Mapping, Sequence

from rich.table import Table

from mtva_bench.run.arena import ArenaSummary
from mtva_bench.run.score import Summary
from mtva_bench.scoring.aggregate import CallScore, HeadToHead, Interval, slices

LABELS = {
    "task_completion": "Task",
    "tool_use": "Tools",
    "instruction_following": "Instructions",
    "faithfulness": "Faithfulness",
}


def label(capability: str) -> str:
    return LABELS.get(capability, capability)


def with_ci(interval: Interval) -> str:
    low, high = interval.ci
    return f"{interval.value:.1f} [dim]\\[{low:.1f}, {high:.1f}][/dim]"


def signed(interval: Interval) -> str:
    low, high = interval.ci
    return f"{interval.value:+.1f} [dim]\\[{low:+.1f}, {high:+.1f}][/dim]"


def runs_table(summaries: Sequence[Summary]) -> Table:
    """One row per run, best MTVA Score first. Every score is 0 to 100 with its 95% interval;
    latency is time to first token in milliseconds."""
    scored = sorted(
        ((summary, summary.scores) for summary in summaries if summary.scores is not None),
        key=lambda pair: -pair[1].mtva_score.value,
    )
    capabilities = list(scored[0][0].weights) if scored else []
    table = Table(
        "run",
        "MTVA Score",
        *map(label, capabilities),
        "goal",
        "clean",
        "TTFT p50",
        "p90",
        "cold",
        "calls",
    )
    for column in table.columns[1:]:
        column.justify = "right"
    for summary, scores in scored:
        latency = summary.latency_ms
        coverage = summary.coverage
        table.add_row(
            summary.run,
            f"[bold]{with_ci(scores.mtva_score)}[/bold]",
            *(f"{scores.capabilities[name].value:.1f}" for name in capabilities),
            f"{scores.goal_reached.value:.0f}%",
            f"{scores.clean_calls.value:.0f}%",
            f"{latency.ttft_warm_p50:,}" if latency else "-",
            f"{latency.ttft_warm_p90:,}" if latency else "-",
            f"{latency.ttft_cold_p50:,}" if latency else "-",
            f"{coverage.scored}/{coverage.planned}"
            if coverage.scored < coverage.planned
            else str(coverage.scored),
        )
    return table


def slice_table(
    runs: Mapping[str, Sequence[CallScore]], weights: Mapping[str, float], field: str
) -> Table:
    """One run: its MTVA Score and capabilities in each slice. Several: each run's MTVA Score."""
    cuts = {name: slices(calls, weights, field) for name, calls in runs.items()}
    values = sorted({value for cut in cuts.values() for value in cut})
    if len(cuts) == 1:
        [cut] = cuts.values()
        table = Table(field, "calls", "MTVA Score", *map(label, weights))
        for value in values:
            part = cut[value]
            table.add_row(
                value,
                str(part.calls),
                f"[bold]{part.mtva_score:.1f}[/bold]",
                *(f"{part.capabilities[name]:.1f}" for name in weights),
            )
    else:
        table = Table(field, *cuts)
        for value in values:
            table.add_row(
                value,
                *(f"{cut[value].mtva_score:.1f}" if value in cut else "-" for cut in cuts.values()),
            )
    for column in table.columns[1:]:
        column.justify = "right"
    return table


def worst_table(calls: Sequence[CallScore], weights: Mapping[str, float], count: int) -> Table:
    """The run's lowest-scoring calls, with each capability's 1 to 5 score."""
    table = Table(
        "call", "language", "difficulty", "MTVA Score", *map(label, weights), "outcome", "ending"
    )
    for column in table.columns[3:-2]:
        column.justify = "right"
    for call in sorted(calls, key=lambda c: (c.mtva(weights), c.stem))[:count]:
        table.add_row(
            call.stem,
            call.slices["language"],
            call.slices["difficulty"],
            f"{call.mtva(weights):.1f}",
            *(str(call.raw[name]) for name in weights),
            call.outcome,
            call.ending,
        )
    return table


def head_to_head_table(a: str, b: str, result: HeadToHead) -> Table:
    """B minus A, with the 95% interval of the paired difference."""
    table = Table("", f"{b} - {a}", title=f"{b} against {a} on {result.calls} shared calls")
    table.columns[1].justify = "right"
    table.add_row("[bold]MTVA Score[/bold]", f"[bold]{signed(result.mtva_score)}[/bold]")
    for name, delta in result.capabilities.items():
        table.add_row(label(name), signed(delta))
    table.add_row("calls won / tied / lost", f"{result.wins} / {result.ties} / {result.losses}")
    return table


def arena_table(summary: ArenaSummary) -> Table:
    """A's wins, ties and losses against B, in percent of the judged scenarios, overall with
    95% intervals and by language."""
    scores = summary.scores
    a, b = summary.a.label, summary.b.label
    table = Table(
        "",
        "calls",
        f"{a} wins",
        "ties",
        f"{b} wins",
        title=f"{a} against {b}: which sounds more human",
    )
    for column in table.columns[1:]:
        column.justify = "right"
    if scores is None:
        return table
    table.add_row(
        "[bold]all[/bold]",
        str(scores.calls),
        f"[bold]{with_ci(scores.wins)}[/bold]",
        with_ci(scores.ties),
        with_ci(scores.losses),
        end_section=True,
    )
    for language, rates in scores.by_language.items():
        table.add_row(
            language,
            str(rates.calls),
            f"{rates.wins:.1f}",
            f"{rates.ties:.1f}",
            f"{rates.losses:.1f}",
        )
    return table
