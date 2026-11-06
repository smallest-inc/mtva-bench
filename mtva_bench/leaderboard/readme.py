"""The README's generated parts: everything between a `<!-- mtva:NAME -->` marker and its
`<!-- /mtva:NAME -->` is rewritten from the leaderboard, so no number in it is typed by hand."""

import re
from collections.abc import Sequence

from mtva_bench.leaderboard.build import Entry
from mtva_bench.leaderboard.charts import highlighted

CAPABILITY_COLUMNS = (
    ("task_completion", "Task"),
    ("tool_use", "Tools"),
    ("instruction_following", "Instructions"),
    ("faithfulness", "Faithfulness"),
)


class ReadmeError(ValueError):
    pass


def table(entries: Sequence[Entry]) -> str:
    """The leaderboard, best MTVA Score first; the highlighted organization's rows in bold."""
    header = ["#", "Model", "Reasoning", "MTVA Score"]
    header += [name for _, name in CAPABILITY_COLUMNS]
    header += ["First token"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join([":-:", ":--", ":--"] + ["--:"] * (len(header) - 3)) + "|",
    ]
    for rank, entry in enumerate(entries, start=1):
        scores = entry.scores

        def bold(text: str, entry: Entry = entry) -> str:
            return f"**{text}**" if highlighted(entry) else text

        cells = [
            str(rank),
            bold(entry.label),
            entry.reasoning,
            bold(f"{scores.mtva_score.value:.1f}"),
            *(f"{scores.capabilities[key].value:.1f}" for key, _ in CAPABILITY_COLUMNS),
            bold(f"{entry.latency.ttft_warm_p50:,} ms"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


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


def readme_parts(entries: Sequence[Entry]) -> dict[str, str]:
    return {"table": table(entries)}
