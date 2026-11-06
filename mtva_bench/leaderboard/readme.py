"""The README's generated parts: everything between a `<!-- mtva:NAME -->` marker and its
`<!-- /mtva:NAME -->` is rewritten from the leaderboard, so no number in it is typed by hand."""

import re
from collections.abc import Sequence

from mtva_bench.leaderboard.build import Entry
from mtva_bench.leaderboard.charts import highlighted, judge_label, lead, preferred, slower_higher
from mtva_bench.run.arena import ArenaSummary

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


def highlights(entries: Sequence[Entry], pairs: Sequence[ArenaSummary]) -> str:
    """What the highlighted model's results show, each line computed from the data and left out
    when the data does not support it."""
    focus = lead(entries)
    if focus is None:
        return ""
    lines = []
    others = sorted((e for e in entries if e is not focus), key=lambda e: e.latency.ttft_warm_p50)
    if others and focus.latency.ttft_warm_p50 < others[0].latency.ttft_warm_p50:
        ratio = others[0].latency.ttft_warm_p50 / focus.latency.ttft_warm_p50
        lines.append(
            f"**Fastest to respond:** {focus.latency.ttft_warm_p50:,} ms median time to first "
            f"token, {ratio:.1f}× sooner than the next fastest model, "  # noqa: RUF001
            f"{others[0].label} ({others[0].latency.ttft_warm_p50:,} ms)."
        )
    mine = [p for p in pairs if p.a.run == focus.summary.run and p.scores is not None]
    won = preferred(mine)
    if mine and len(won) == len(mine):
        rates = [p.scores.wins.value for p in mine if p.scores is not None]
        judge_model = mine[0].provenance.judge_model
        itself = (
            f", including {judge_label(judge_model, entries)}, the judge itself"
            if any(p.b.run == judge_model for p in mine)
            else ""
        )
        lines.append(
            f"**Sounds the most human:** a blind judge preferred it over all {len(mine)} models "
            f"it faced, in {min(rates):.0f}% to {max(rates):.0f}% of calls{itself}."
        )
    ratio = slower_higher(entries, focus)
    if ratio:
        lines.append(
            f"**Quality at speed:** an MTVA Score of {focus.scores.mtva_score.value:.1f}; every "
            f"model that scores higher takes at least {ratio:.1f}× as long to start speaking."  # noqa: RUF001
        )
    return "\n".join(f"- {line}" for line in lines)


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


def readme_parts(entries: Sequence[Entry], pairs: Sequence[ArenaSummary]) -> dict[str, str]:
    return {"highlights": highlights(entries, pairs), "table": table(entries)}
