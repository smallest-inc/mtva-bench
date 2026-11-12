"""The README's generated parts: everything between a `<!-- mtva:NAME -->` marker and its
`<!-- /mtva:NAME -->` is rewritten from the leaderboard, so no number in it is typed by hand."""

import re
from collections.abc import Sequence

from mtva_bench.leaderboard.build import Entry
from mtva_bench.leaderboard.charts import highlighted
from mtva_bench.scoring.index import WEIGHTS
from mtva_bench.scoring.latency import Speed

CAPABILITY_COLUMNS = (
    ("task_completion", "Task completion"),
    ("tool_use", "Tool calling"),
    ("instruction_following", "Instruction following"),
    ("faithfulness", "Faithfulness"),
)


class ReadmeError(ValueError):
    pass


def markdown_table(header: list[str], text_columns: int, rows: list[list[str]]) -> str:
    """A table whose first column is centered, the next `text_columns` left-aligned and the
    rest right-aligned."""
    numbers = len(header) - 1 - text_columns
    lines = [header, [":-:"] + [":--"] * text_columns + ["--:"] * numbers, *rows]
    return "\n".join("| " + " | ".join(line) + " |" for line in lines)


def bold(entry: Entry, text: str) -> str:
    return f"**{text}**" if highlighted(entry) else text


def pair(median: int, slow: int) -> str:
    """Two figures that stay on one line of a table cell."""
    return f"{median:,}&nbsp;/&nbsp;{slow:,}"


def table(entries: Sequence[Entry]) -> str:
    """The leaderboard, best MTVA Score first: each model's score, speed and, once every
    model's calls are priced, its cost per call and MTVA Index. The highlighted organization's
    rows are in bold."""
    priced = all(entry.cost is not None for entry in entries)
    header = ["#", "Model", "Reasoning", "MTVA Score", "First token (ms)", "First sentence (ms)"]
    if priced:
        header += ["Cost per call", "MTVA Index"]
    rows = []
    for rank, entry in enumerate(entries, start=1):
        speed = entry.latency
        cells = [
            str(rank),
            bold(entry, entry.label),
            entry.reasoning,
            bold(entry, f"{entry.scores.mtva_score.value:.1f}"),
            bold(entry, pair(speed.ttft_warm_p50, speed.ttft_warm_p90)),
            bold(entry, pair(speed.first_sentence_warm_p50, speed.first_sentence_warm_p90))
            if isinstance(speed, Speed)
            else "-",
        ]
        if priced and entry.cost is not None and entry.index is not None:
            cells += [
                bold(entry, f"${entry.cost.usd_per_call:.4f}"),
                bold(entry, f"{entry.index:.1f}"),
            ]
        rows.append(cells)
    return markdown_table(header, 2, rows)


def capability_table(entries: Sequence[Entry]) -> str:
    """Each model's MTVA Score beside the four capabilities it is the mean of, best first."""
    header = ["#", "Model", "MTVA Score", *(name for _, name in CAPABILITY_COLUMNS)]
    rows = [
        [
            str(rank),
            bold(entry, entry.label),
            bold(entry, f"{entry.scores.mtva_score.value:.1f}"),
            *(f"{entry.scores.capabilities[key].value:.1f}" for key, _ in CAPABILITY_COLUMNS),
        ]
        for rank, entry in enumerate(entries, start=1)
    ]
    return markdown_table(header, 1, rows)


def fill(readme: str, parts: dict[str, str]) -> str:
    """The README with each marked part replaced; a marker that is missing is an error."""
    for name, body in parts.items():
        pattern = re.compile(
            rf"(<!-- mtva:{name} -->\n)(?:.*?\n)?(<!-- /mtva:{name} -->)", flags=re.DOTALL
        )
        if not pattern.search(readme):
            raise ReadmeError(f"README has no <!-- mtva:{name} --> … <!-- /mtva:{name} --> block")
        readme = pattern.sub(
            lambda match, body=body: match.group(1) + body + "\n" + match.group(2), readme
        )
    return readme


def index_weights() -> str:
    quality, speed, cost = (f"{WEIGHTS[name]:.0%}" for name in ("quality", "speed", "cost"))
    return (
        f"The index is a weighted geometric mean: the MTVA Score counts for {quality}, speed for "
        f"{speed} and cost for {cost}, and a strong part cannot make up for a poor one."
    )


def readme_parts(entries: Sequence[Entry]) -> dict[str, str]:
    return {
        "table": table(entries),
        "capabilities": capability_table(entries),
        "index": index_weights(),
    }
