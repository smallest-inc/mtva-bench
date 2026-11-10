import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from rich.table import Table

from mtva_bench import __version__
from mtva_bench.config import (
    ConfigError,
    ModelSpec,
    RunConfig,
    RunOptions,
    load_benchmark,
    load_model,
    load_run,
)
from mtva_bench.data import (
    Bundle,
    DataError,
    check_bundle,
    download_bundle,
    find_bundle,
    load_bundle,
    system_prompt,
    verify,
)
from mtva_bench.data.bundle import DATA_DIR_ENV, DEFAULT_DIR, file_findings
from mtva_bench.leaderboard.build import (
    LeaderboardError,
    leaderboard_json,
    load_arena,
    load_entries,
)
from mtva_bench.leaderboard.charts import charts
from mtva_bench.leaderboard.readme import ReadmeError, fill, readme_parts
from mtva_bench.llm.batch import Batcher
from mtva_bench.llm.client import LLM, LLMClient
from mtva_bench.llm.endpoints import Endpoint
from mtva_bench.llm.fake import FakeLLM
from mtva_bench.llm.http import LLMError
from mtva_bench.llm.types import BatchStatus, Output, Request
from mtva_bench.report import arena_table, head_to_head_table, runs_table, slice_table, worst_table
from mtva_bench.run import RunLockedError, StaleRunError, evaluate_stage, simulate_stage
from mtva_bench.run.arena import (
    arena_stage,
    judge_spec,
    matchups,
    pair_dir,
    save_arena_summary,
    summarize_arena,
)
from mtva_bench.run.evaluate import EvaluateResult
from mtva_bench.run.latency import LatencyResult, check_scored, latency_stage
from mtva_bench.run.score import Summary, publish_problems, run_config, score_stage, summarize
from mtva_bench.run.store import StoredConversation, StoredVerdict
from mtva_bench.scoring.aggregate import SLICES, CallScore, head_to_head

console = Console(soft_wrap=True)

data_option = click.option(
    "--data",
    "data_dir",
    type=click.Path(path_type=Path, file_okay=False),
    metavar="DIR",
    help="Bundle folder. Default: $MTVA_BENCH_DATA_DIR if set, else data/<version>.",
)
benchmark_option = click.option(
    "--benchmark",
    "benchmark_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("benchmark.yaml"),
    show_default=True,
    metavar="FILE",
    help="Protocol file: caller, judge and the pinned data version.",
)


def at_least_one(_: click.Context, __: click.Parameter, value: int | None) -> int | None:
    if value is not None and value < 1:
        raise click.BadParameter("must be at least 1")
    return value


@click.group(context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100})
@click.version_option(__version__, prog_name="mtva-bench")
def cli() -> None:
    """MTVA-Bench: how well an LLM runs a voice agent on a simulated phone call.

    \b
    Typical use:
      mtva-bench data check                     validate the scenario bundle
      mtva-bench run models/electron-2.0.yaml   run a model; run again to resume
    """


def no_model_call(_: Request) -> Output:
    raise AssertionError("a dry run calls no model")


@asynccontextmanager
async def clients(dry_run: bool, *endpoints: tuple[Endpoint, int]) -> AsyncIterator[list[LLM]]:
    """A client per (endpoint, concurrency). A dry run calls no model, so it gets stand-ins and
    needs no API keys."""
    if dry_run:
        yield [FakeLLM(no_model_call) for _ in endpoints]
        return
    async with AsyncExitStack() as stack:
        yield [
            await stack.enter_async_context(LLMClient(endpoint, concurrency))
            for endpoint, concurrency in endpoints
        ]


def open_bundle(data_dir: Path | None, benchmark_path: Path) -> Bundle:
    """The bundle to use. When none is given and benchmark.yaml pins a version that is not in
    data/ yet, it is downloaded there first."""
    try:
        pinned = load_benchmark(benchmark_path).data if benchmark_path.is_file() else None
        if (
            pinned is not None
            and data_dir is None
            and not os.environ.get(DATA_DIR_ENV)
            and not (DEFAULT_DIR / pinned.version / "bundle.json").exists()
        ):
            console.print(
                f"downloading MTVA-Bench data {pinned.version} into data/", highlight=False
            )
            download_bundle(pinned)
        return load_bundle(find_bundle(data_dir, pinned.version if pinned else None))
    except (ConfigError, DataError) as error:
        raise click.ClickException(str(error)) from error


@cli.group()
def data() -> None:
    """Browse and validate the scenario bundle (agents and scenarios)."""


@data.command(short_help="Download the pinned bundle.")
@benchmark_option
def download(benchmark_path: Path) -> None:
    """Download the bundle benchmark.yaml pins into data/<version>, checking every file and
    the pinned hash before it is used. Other commands do this on their own when the bundle is
    missing."""
    try:
        pinned = load_benchmark(benchmark_path).data
        path = download_bundle(pinned)
    except (ConfigError, DataError) as error:
        raise click.ClickException(str(error)) from error
    console.print(f"[green]✓[/green] {path}", highlight=False)


@data.command(short_help="Validate the bundle.")
@data_option
@benchmark_option
def check(data_dir: Path | None, benchmark_path: Path) -> None:
    """Validate the bundle: every file against the schema, each scenario against its agent,
    file hashes against bundle.json, and the version against benchmark.yaml. Exits with an
    error if anything is wrong."""
    bundle = open_bundle(data_dir, benchmark_path)
    findings = check_bundle(bundle)
    if benchmark_path.is_file() and not file_findings(bundle):
        try:
            verify(bundle, load_benchmark(benchmark_path).data)
        except DataError as error:
            findings.append(str(error))
    for finding in findings:
        console.print(f"[red]✗[/red] {finding}", markup=True, highlight=False)
    summary = (
        f"{bundle.manifest.name} {bundle.version}: {len(bundle.agents)} agents, "
        f"{len(bundle.scenarios)} scenarios, hash {bundle.hash}"
    )
    if findings:
        raise click.ClickException(f"{len(findings)} problems in {summary}")
    console.print(f"[green]✓[/green] {summary}", highlight=False)


@data.command(name="list", short_help="List agents or scenarios.")
@data_option
@benchmark_option
@click.option(
    "--scenarios", "show_scenarios", is_flag=True, help="One row per scenario instead of per agent."
)
@click.option(
    "--agent", "agents", multiple=True, metavar="ID", help="Only this agent. Repeat for more."
)
@click.option(
    "--language",
    "languages",
    multiple=True,
    metavar="NAME",
    help="Only this language, e.g. Hindi. Repeat for more.",
)
@click.option(
    "--difficulty",
    "difficulties",
    multiple=True,
    type=click.Choice(["easy", "medium", "hard"]),
    metavar="LEVEL",
    help="Only easy, medium or hard. Repeat for more.",
)
def list_(
    data_dir: Path | None,
    benchmark_path: Path,
    show_scenarios: bool,
    agents: tuple[str, ...],
    languages: tuple[str, ...],
    difficulties: tuple[str, ...],
) -> None:
    """List the bundle's agents with their language, industry and number of scenarios, or with
    --scenarios one row per scenario with its difficulty and summary."""
    bundle = open_bundle(data_dir, benchmark_path)
    selected = bundle.select(agents=agents, languages=languages, difficulties=difficulties)
    if show_scenarios:
        table = Table("scenario", "difficulty", "language", "summary")
        for scenario in selected:
            table.add_row(
                scenario.scenario_id, scenario.difficulty, scenario.language, scenario.summary
            )
    else:
        counts: dict[str, int] = {}
        for scenario in selected:
            counts[scenario.agent_id] = counts.get(scenario.agent_id, 0) + 1
        table = Table(
            "agent", "language", "industry", "use case", "direction", "tools", "scenarios"
        )
        for agent_id, count in sorted(counts.items()):
            agent = bundle.agents[agent_id]
            table.add_row(
                agent_id,
                agent.language,
                agent.industry,
                agent.use_case,
                agent.direction,
                str(len(agent.tools)),
                str(count),
            )
    console.print(table)
    console.print(f"{table.row_count} rows", highlight=False)


@data.command(short_help="Show an agent or scenario, or its prompt.")
@click.argument("item_id", metavar="ID")
@data_option
@benchmark_option
@click.option(
    "--prompt",
    "show_prompt",
    is_flag=True,
    help="Print the prompt: an agent's own, or a scenario's full system prompt.",
)
@click.option(
    "--no-voice-guide", is_flag=True, help="With --prompt on a scenario: leave out the voice guide."
)
def show(
    item_id: str,
    data_dir: Path | None,
    benchmark_path: Path,
    show_prompt: bool,
    no_voice_guide: bool,
) -> None:
    """Show an agent or a scenario as JSON, or with --prompt the text the model is given."""
    bundle = open_bundle(data_dir, benchmark_path)
    if item_id in bundle.agents:
        agent = bundle.agents[item_id]
        if show_prompt:
            click.echo(agent.prompt)
            return
        summary = agent.model_dump(mode="json", exclude={"prompt", "tools"})
        summary["tools"] = [tool.name for tool in agent.tools]
        summary["scenarios"] = sorted(s.scenario_id for s in bundle.select(agents=[item_id]))
        console.print_json(json.dumps(summary, ensure_ascii=False))
    elif item_id in bundle.scenarios:
        scenario = bundle.scenarios[item_id]
        if show_prompt:
            guide = None if no_voice_guide else bundle.voice_guide
            click.echo(system_prompt(bundle.agent(scenario), scenario, guide))
            return
        console.print_json(scenario.model_dump_json())
    else:
        raise click.ClickException(f"no agent or scenario named {item_id}")


@cli.command(short_help="Run a model through the benchmark.")
@click.argument(
    "model_path", metavar="MODEL_YAML", type=click.Path(path_type=Path, dir_okay=False, exists=True)
)
@benchmark_option
@data_option
@click.option("--name", metavar="NAME", help="Run folder name. Default: the model's name.")
@click.option(
    "--runs-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("runs"),
    show_default=True,
    metavar="DIR",
    help="Where run folders are kept.",
)
@click.option(
    "--limit",
    type=int,
    callback=at_least_one,
    metavar="N",
    help="Only N scenarios: the same seeded sample every time, spread across languages.",
)
@click.option(
    "--agent", "agents", multiple=True, metavar="ID", help="Only this agent. Repeat for more."
)
@click.option(
    "--language",
    "languages",
    multiple=True,
    metavar="NAME",
    help="Only this language, e.g. Hindi. Repeat for more.",
)
@click.option(
    "--scenario",
    "scenarios",
    multiple=True,
    metavar="ID",
    help="Only this scenario. Repeat for more.",
)
@click.option(
    "--concurrency",
    type=int,
    callback=at_least_one,
    metavar="N",
    help="Calls running at once. Default: the model file's concurrency.",
)
@click.option(
    "--judge-concurrency",
    type=int,
    callback=at_least_one,
    default=16,
    show_default=True,
    metavar="N",
    help="Judge requests running at once.",
)
@click.option("--no-judge", is_flag=True, help="Only simulate the calls; judge them later.")
@click.option(
    "--batch",
    is_flag=True,
    help="Judge through the provider's batch API at half price. Waits for the batch, from "
    "minutes to hours; a run stopped while waiting picks the same batch back up.",
)
@click.option(
    "--discard-stale",
    is_flag=True,
    help="Redo saved calls or verdicts made from different inputs, instead of stopping.",
)
@click.option("--dry-run", is_flag=True, help="Show what would run, then stop. Calls no model.")
def run(
    model_path: Path,
    benchmark_path: Path,
    data_dir: Path | None,
    name: str | None,
    runs_dir: Path,
    limit: int | None,
    agents: tuple[str, ...],
    languages: tuple[str, ...],
    scenarios: tuple[str, ...],
    concurrency: int | None,
    judge_concurrency: int,
    no_judge: bool,
    batch: bool,
    discard_stale: bool,
    dry_run: bool,
) -> None:
    """Run a model through the benchmark: simulate each scenario's call, then judge it, and
    save both in runs/<name>/.

    Running the same name again continues the run: finished calls and verdicts are skipped,
    failed ones retried. If saved calls or verdicts were made from different inputs, such as
    another model spec, an edited scenario or another judge, the run stops and says what
    changed.

    \b
    Examples:
      mtva-bench run models/electron-2.0.yaml --dry-run
      mtva-bench run models/electron-2.0.yaml --limit 14
      mtva-bench run models/gpt-6-luna.yaml --language Hindi --language Tamil
    """
    options = RunOptions(
        name=name,
        runs_dir=runs_dir,
        data_dir=data_dir,
        limit=limit,
        agents=list(agents),
        languages=list(languages),
        scenarios=list(scenarios),
        concurrency=concurrency,
        judge_concurrency=judge_concurrency,
        evaluate=not no_judge,
        batch=batch,
        discard_stale=discard_stale,
        dry_run=dry_run,
    )
    try:
        config = load_run(model_path, benchmark_path, options)
        bundle = open_bundle(data_dir, benchmark_path)
        verify(bundle, config.benchmark.data)
    except (ConfigError, DataError) as error:
        raise click.ClickException(str(error)) from error
    console.print(
        f"MTVA-Bench {config.benchmark.version} · bundle {bundle.version} · "
        f"{config.model.name} ({config.model.serving.kind}) · "
        f"{config.run_dir}",
        highlight=False,
    )
    failures: list[StoredConversation] = []
    with Progress(
        TextColumn("simulate"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        disable=dry_run,
    ) as progress:
        task = progress.add_task("simulate", total=None)

        def done(stored: StoredConversation) -> None:
            progress.advance(task)
            if stored.conversation.end_reason == "error":
                failures.append(stored)

        async def go():
            async with clients(
                dry_run,
                (config.model, config.concurrency),
                (config.benchmark.simulation.caller, config.concurrency),
            ) as (model, caller):
                return await simulate_stage(
                    config,
                    bundle,
                    model,
                    caller,
                    on_start=lambda total: progress.update(task, total=total),
                    on_done=done,
                )

        try:
            result = asyncio.run(go())
        except (StaleRunError, RunLockedError, DataError, ValueError) as error:
            raise click.ClickException(str(error)) from error
    for stored in failures:
        conversation = stored.conversation
        where = f"{conversation.scenario_id} r{conversation.rollout}"
        console.print(f"[red]✗[/red] {where}: {conversation.error}", highlight=False)
    verb = "would simulate" if dry_run else "simulated"
    console.print(
        f"{result.planned} rollouts planned · {result.skipped} already done · "
        f"{verb} {result.planned - result.skipped if dry_run else result.simulated} · "
        f"{result.failed} failed",
        highlight=False,
    )
    judged = None if no_judge else judge_calls(config, bundle, dry_run)
    problems = []
    if result.failed:
        problems.append(f"{result.failed} rollouts failed")
    if judged is not None and judged.failed:
        problems.append(f"{judged.failed} verdicts failed")
    if problems:
        raise click.ClickException(f"{' and '.join(problems)}; run again to retry them")
    if judged is not None and not dry_run:
        summary, _ = score_stage(config, bundle)
        show_summaries([summary])


run_argument = click.argument(
    "run_dirs",
    metavar="RUN...",
    nargs=-1,
    required=True,
    type=click.Path(path_type=Path, file_okay=False, exists=True),
)


def score_runs(
    run_dirs: tuple[Path, ...], data_dir: Path | None, benchmark_path: Path, save: bool
) -> list[tuple[Summary, list[CallScore]]]:
    try:
        benchmark = load_benchmark(benchmark_path)
        bundle = open_bundle(data_dir, benchmark_path)
        verify(bundle, benchmark.data)
        configs = [run_config(run_dir, benchmark) for run_dir in run_dirs]
        return [(score_stage if save else summarize)(config, bundle) for config in configs]
    except (ConfigError, DataError) as error:
        raise click.ClickException(str(error)) from error


def show_summaries(summaries: list[Summary]) -> None:
    console.print(runs_table(summaries))
    console.print(
        "[dim]Scores are 0 to 100 with 95% intervals. goal: calls where the caller got what "
        "they came for. clean: calls with no problem found. TTFT: warm time to first token, "
        "p50 and p90, and the cold opening request, in ms.[/dim]",
        highlight=False,
    )
    for summary in summaries:
        coverage = summary.coverage
        if coverage.missing:
            reasons = ", ".join(f"{reason} ({n})" for reason, n in coverage.missing.items())
            console.print(
                f"[yellow]![/yellow] {summary.run}: {coverage.scored} of {coverage.planned} calls "
                f"scored; missing: {reasons}",
                highlight=False,
            )


@cli.command(short_help="Score runs: write each one's summary.json.")
@run_argument
@data_option
@benchmark_option
def score(run_dirs: tuple[Path, ...], data_dir: Path | None, benchmark_path: Path) -> None:
    """Score each run from its saved calls and verdicts and write RUN/summary.json. Calls no
    model.

    Only calls whose conversation and verdict are current under benchmark.yaml count; the
    rest are listed as missing, with the reason. Capability weights come from benchmark.yaml,
    so changing them needs only this command."""
    show_summaries([summary for summary, _ in score_runs(run_dirs, data_dir, benchmark_path, True)])


@cli.command(short_help="Compare runs in the terminal.")
@run_argument
@data_option
@benchmark_option
@click.option(
    "--by",
    "field",
    type=click.Choice([*SLICES, "agent"]),
    help="Also cut the scores by this field.",
)
@click.option(
    "--worst",
    type=int,
    callback=at_least_one,
    metavar="N",
    help="Also list each run's N lowest-scoring calls.",
)
def report(
    run_dirs: tuple[Path, ...],
    data_dir: Path | None,
    benchmark_path: Path,
    field: str | None,
    worst: int | None,
) -> None:
    """Show runs side by side, scored from their saved verdicts. Two runs are also compared
    head to head on the calls both have, with intervals of the paired differences. Writes
    nothing.

    \b
    Examples:
      mtva-bench report runs/*/
      mtva-bench report runs/electron-2.0 runs/gpt-6.1-terra --by language
      mtva-bench report runs/electron-2.0 --worst 20
    """
    scored = score_runs(run_dirs, data_dir, benchmark_path, False)
    show_summaries([summary for summary, _ in scored])
    weights = scored[0][0].weights
    if len(scored) == 2:
        (a, a_calls), (b, b_calls) = scored
        if a_calls and b_calls:
            console.print(head_to_head_table(a.run, b.run, head_to_head(a_calls, b_calls, weights)))
    if field:
        console.print(slice_table({s.run: calls for s, calls in scored if calls}, weights, field))
    if worst:
        for summary, calls in scored:
            console.print(worst_table(calls, weights, worst), f"[dim]{summary.run}[/dim]")


@cli.command(short_help="Put a run's summary on the leaderboard.")
@click.argument(
    "run_dir", metavar="RUN", type=click.Path(path_type=Path, file_okay=False, exists=True)
)
@data_option
@benchmark_option
@click.option(
    "--to",
    "leaderboard_dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("leaderboard/runs"),
    show_default=True,
    help="Where published summaries are kept, under version control.",
)
def publish(
    run_dir: Path, data_dir: Path | None, benchmark_path: Path, leaderboard_dir: Path
) -> None:
    """Score a run and copy its summary to LEADERBOARD_DIR/<run>.json, the file the
    leaderboard and README are built from. The summary holds aggregates only, nothing from a
    scenario or a call. Refused below 99% coverage."""
    [(summary, _)] = score_runs((run_dir,), data_dir, benchmark_path, True)
    problems = publish_problems(summary)
    if problems:
        raise click.ClickException(f"{summary.run} cannot be published: {'; '.join(problems)}")
    leaderboard_dir.mkdir(parents=True, exist_ok=True)
    path = leaderboard_dir / f"{summary.run}.json"
    path.write_text(summary.model_dump_json(indent=1) + "\n", encoding="utf-8")
    show_summaries([summary])
    console.print(f"[green]✓[/green] {path}", highlight=False)


def judge_calls(config: RunConfig, bundle: Bundle, dry_run: bool) -> EvaluateResult:
    failures: list[tuple[str, StoredVerdict]] = []
    batches: set[str] = set()
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        disable=dry_run,
    ) as progress:
        task = progress.add_task("judge   ", total=None)

        def done(stem: str, stored: StoredVerdict) -> None:
            progress.advance(task)
            if stored.error is not None:
                failures.append((stem, stored))

        def waiting(status: BatchStatus) -> None:
            if status.id not in batches:
                batches.add(status.id)
                progress.console.print(f"batch {status.id}", highlight=False)
            counts = status.counts
            left = counts.get("processing", 0)
            word = "results in" if status.ended else f"{left} still processing"
            progress.update(task, description=f"judge    batch: {word} ·")

        async def go():
            settings = config.benchmark.judge
            concurrency = config.options.judge_concurrency
            async with clients(
                dry_run, (settings, concurrency), (settings.fallback or settings, concurrency)
            ) as (judge, fallback):
                return await evaluate_stage(
                    config,
                    bundle,
                    judge,
                    fallback if settings.fallback else None,
                    batch_api=Batcher(judge)
                    if config.options.batch and isinstance(judge, LLMClient)
                    else None,
                    on_start=lambda total: progress.update(task, total=total),
                    on_batch=waiting,
                    on_done=done,
                )

        try:
            result = asyncio.run(go())
        except (StaleRunError, RunLockedError, ValueError, LLMError) as error:
            raise click.ClickException(str(error)) from error
    for stem, stored in failures:
        console.print(f"[red]✗[/red] {stem} verdict: {stored.error}", highlight=False)
    usage = result.usage
    cached = usage.cache_read_tokens / usage.input_tokens if usage.input_tokens else 0
    verb = "would judge" if dry_run else "judged"
    console.print(
        f"{result.planned} calls to judge · {result.skipped} already judged · "
        f"{verb} {result.planned - result.skipped if dry_run else result.judged} · "
        f"{result.failed} failed",
        highlight=False,
    )
    if usage.input_tokens:
        batched = (
            f" · {result.batch_usage.input_tokens:,} in and "
            f"{result.batch_usage.output_tokens:,} out at batch price"
            if result.batch_usage.input_tokens
            else ""
        )
        retried = f" · {result.retried} retried in realtime" if result.retried else ""
        if result.fallback:
            retried += f" · {result.fallback} judged by the fallback judge"
        console.print(
            f"judge tokens: {usage.input_tokens:,} in ({cached:.0%} read from cache, "
            f"{usage.cache_write_tokens:,} written) · {usage.output_tokens:,} out"
            f"{batched}{retried}",
            highlight=False,
        )
    return result


@cli.command(name="leaderboard", short_help="Build the leaderboard and its charts.")
@benchmark_option
@click.option(
    "--runs",
    "runs_dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("leaderboard/runs"),
    show_default=True,
    help="Published summaries to build from.",
)
@click.option(
    "--assets",
    "assets_dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("assets"),
    show_default=True,
    help="Where the charts are written.",
)
@click.option(
    "--readme",
    "readme_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("README.md"),
    show_default=True,
    help="README whose marked parts are rewritten.",
)
@click.option(
    "--check",
    is_flag=True,
    help="Write nothing; fail if the README or a chart is out of date.",
)
def leaderboard_command(
    benchmark_path: Path, runs_dir: Path, assets_dir: Path, readme_path: Path, check: bool
) -> None:
    """Build leaderboard.json, every chart and the README's marked parts from the published
    summaries. Calls no model and reads no run folder. Refuses a summary made under another
    protocol than benchmark.yaml."""
    try:
        benchmark = load_benchmark(benchmark_path)
        entries = load_entries(runs_dir, benchmark)
        pairs = load_arena(runs_dir.parent / "arena", benchmark)
        outputs = {
            runs_dir.parent / "leaderboard.json": leaderboard_json(entries, pairs, benchmark)
        }
        outputs[runs_dir.parent / "leaderboard.json"] += "\n"
        outputs |= {assets_dir / name: svg for name, svg in charts(entries, pairs).items()}
        if readme_path.is_file():
            readme = readme_path.read_text(encoding="utf-8")
            outputs[readme_path] = fill(readme, readme_parts(entries))
    except (ConfigError, LeaderboardError, ReadmeError) as error:
        raise click.ClickException(str(error)) from error
    stale = [
        path
        for path, text in outputs.items()
        if not path.is_file() or path.read_text(encoding="utf-8") != text
    ]
    if check:
        if stale:
            raise click.ClickException(
                "out of date: " + ", ".join(map(str, stale)) + "; run mtva-bench leaderboard"
            )
        console.print(f"[green]✓[/green] {len(outputs)} files up to date", highlight=False)
        return
    for path in stale:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(outputs[path], encoding="utf-8")
    console.print(
        f"[green]✓[/green] {len(entries)} models, {len(pairs)} arena pairs · "
        f"{len(stale)} of {len(outputs)} files updated",
        highlight=False,
    )


@cli.command(short_help="Judge which of two runs sounds more human.")
@click.argument(
    "run_a", metavar="RUN_A", type=click.Path(path_type=Path, file_okay=False, exists=True)
)
@click.argument(
    "run_b", metavar="RUN_B", type=click.Path(path_type=Path, file_okay=False, exists=True)
)
@data_option
@benchmark_option
@click.option(
    "--concurrency",
    type=int,
    default=16,
    show_default=True,
    callback=at_least_one,
    help="Judge requests at once, in realtime.",
)
@click.option(
    "--batch", is_flag=True, help="Judge through the provider's batch api, at half price."
)
@click.option("--discard-stale", is_flag=True, help="Judge again verdicts made from other inputs.")
@click.option("--dry-run", is_flag=True, help="Show what would be judged, then stop.")
@click.option(
    "--show-request",
    "show_request",
    metavar="SCENARIO_ID",
    help="Print what the judge reads for this scenario, then stop. Calls no model.",
)
@click.option(
    "--publish",
    is_flag=True,
    help="Also copy the result to leaderboard/arena/; needs 99% of scenarios judged.",
)
def arena(
    run_a: Path,
    run_b: Path,
    data_dir: Path | None,
    benchmark_path: Path,
    concurrency: int,
    batch: bool,
    discard_stale: bool,
    dry_run: bool,
    show_request: str | None,
    publish: bool,
) -> None:
    """Show the judge the two runs' calls of each scenario, as speech alone and without names,
    and ask which agent sounds more human. Each run's call is shown first in half of the
    scenarios. Verdicts are saved in runs/arena/<a>__vs__<b>/; running again continues.

    \b
    Examples:
      mtva-bench arena runs/electron-2.0 runs/gpt-6.1-sol --show-request <scenario_id>
      mtva-bench arena runs/electron-2.0 runs/gpt-6.1-sol --batch --publish
    """
    try:
        benchmark = load_benchmark(benchmark_path)
        bundle = open_bundle(data_dir, benchmark_path)
        verify(bundle, benchmark.data)
        a, b = run_config(run_a, benchmark), run_config(run_b, benchmark)
        judge_endpoint = judge_spec(a)
    except (ConfigError, DataError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    if show_request:
        found = {m.stem.split("__r")[0]: m for m in matchups(a, b, bundle)}
        if show_request not in found:
            raise click.ClickException(f"{show_request}: not a scenario both runs completed")
        matchup = found[show_request]
        shown = a.name if matchup.first == "a" else b.name
        console.print(f"[dim]Agent A is {shown}[/dim]", highlight=False)
        console.print(matchup.request().messages[0].content, markup=False, highlight=False)
        return
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        disable=dry_run,
    ) as progress:
        task = progress.add_task("judge   ", total=None)

        def waiting(status: BatchStatus) -> None:
            left = status.counts.get("processing", 0)
            word = "results in" if status.ended else f"{left} still processing"
            progress.update(task, description=f"judge    batch {status.id}: {word} ·")

        async def go():
            async with clients(dry_run, (judge_endpoint, concurrency)) as (judge,):
                return await arena_stage(
                    a,
                    b,
                    bundle,
                    judge,
                    batch_api=Batcher(judge) if batch and isinstance(judge, LLMClient) else None,
                    discard_stale=discard_stale,
                    dry_run=dry_run,
                    on_start=lambda total: progress.update(task, total=total),
                    on_batch=waiting,
                    on_done=lambda *_: progress.advance(task),
                )

        try:
            result = asyncio.run(go())
        except (StaleRunError, RunLockedError, ValueError, LLMError) as error:
            raise click.ClickException(str(error)) from error
    verb = "would judge" if dry_run else "judged"
    console.print(
        f"{result.planned} scenarios both runs completed · {result.skipped} already judged · "
        f"{verb} {result.planned - result.skipped if dry_run else result.judged} · "
        f"{result.failed} failed",
        highlight=False,
    )
    if dry_run:
        return
    summary = summarize_arena(a, b, bundle)
    path = save_arena_summary(a, b, summary)
    console.print(arena_table(summary))
    scores = summary.scores
    if scores is not None and scores.first_shown_won is not None:
        console.print(
            f"[dim]The call shown first won {scores.first_shown_won:.0f}% of decisive verdicts; "
            "near 50% means no lean toward either position.[/dim]",
            highlight=False,
        )
    coverage = summary.coverage
    if coverage.missing:
        reasons = ", ".join(f"{reason} ({n})" for reason, n in coverage.missing.items())
        console.print(f"[yellow]![/yellow] missing: {reasons}", highlight=False)
    console.print(f"[green]✓[/green] {path}", highlight=False)
    if publish:
        if coverage.judged < 0.99 * coverage.planned:
            raise click.ClickException(
                f"only {coverage.judged} of {coverage.planned} scenarios judged; "
                "publishing needs 99%"
            )
        target = Path("leaderboard/arena") / f"{pair_dir(Path(), a.name, b.name).name}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(summary.model_dump_json(indent=1) + "\n", encoding="utf-8")
        console.print(f"[green]✓[/green] {target}", highlight=False)


@cli.command(short_help="Time models on the same recorded calls, one request at a time.")
@click.argument(
    "model_paths",
    metavar="MODEL.yaml...",
    nargs=-1,
    required=True,
    type=click.Path(path_type=Path, dir_okay=False, exists=True),
)
@click.option(
    "--region",
    required=True,
    metavar="NAME",
    help="Where the test runs, as a label for the result, e.g. us-east-1.",
)
@click.option(
    "--reference",
    "reference_dir",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    default=Path("runs/gpt-6.1-sol"),
    show_default=True,
    help="The run whose recorded calls are replayed.",
)
@data_option
@benchmark_option
@click.option(
    "--runs-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("runs"),
    show_default=True,
    help="Where replays are saved, under latency/<region>/<model>/.",
)
@click.option(
    "--limit",
    type=int,
    callback=at_least_one,
    metavar="N",
    help="Replay only the first N calls, for a quick check.",
)
@click.option(
    "--publish",
    is_flag=True,
    help="Also copy each result to leaderboard/latency/; needs 99% of calls replayed.",
)
def latency(
    model_paths: tuple[Path, ...],
    region: str,
    reference_dir: Path,
    data_dir: Path | None,
    benchmark_path: Path,
    runs_dir: Path,
    limit: int | None,
    publish: bool,
) -> None:
    """Replay one recorded call per agent from REFERENCE against each model: the caller's lines
    verbatim, the model's own replies and tool calls building the history, one request at a
    time. Measures time to first token, to the first spoken sentence, and generation speed.
    Different models run in parallel. Each model file must match the spec its published summary
    was scored with. Running again continues where it stopped.

    \b
    Examples:
      mtva-bench latency models/electron-2.0.yaml --region ap-south-1 --limit 3
      mtva-bench latency models/gpt-6.1-sol.yaml models/claude-opus-5-5.yaml --region us-east-1
    """
    leaderboard_runs = Path("leaderboard/runs")
    try:
        benchmark = load_benchmark(benchmark_path)
        bundle = open_bundle(data_dir, benchmark_path)
        verify(bundle, benchmark.data)
        specs = [load_model(path) for path in model_paths]
        for spec in specs:
            check_scored(spec, leaderboard_runs)
    except (ConfigError, DataError) as error:
        raise click.ClickException(str(error)) from error

    with Progress(
        TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), console=console
    ) as progress:

        async def one(spec: ModelSpec) -> LatencyResult:
            task = progress.add_task(f"{spec.name:24}", total=None)
            async with LLMClient(spec, 1) as model:
                return await latency_stage(
                    spec,
                    benchmark,
                    bundle,
                    model,
                    reference_dir,
                    region,
                    runs_dir,
                    limit=limit,
                    on_start=lambda total: progress.update(task, total=total),
                    on_done=lambda _: progress.advance(task),
                )

        async def go():
            return await asyncio.gather(*(one(spec) for spec in specs))

        try:
            results = asyncio.run(go())
        except (ConfigError, RunLockedError, LLMError) as error:
            raise click.ClickException(str(error)) from error

    table = Table(
        "model",
        "calls",
        "first token p50",
        "p90",
        "first sentence p50",
        "p90",
        "cold",
        "tok/s",
        title=f"Latency from {region}, in ms",
    )
    for column in table.columns[1:]:
        column.justify = "right"
    for result in results:
        s = result.speed
        table.add_row(
            result.model,
            f"{result.calls}" + (f" [red]({result.failed} failed)[/red]" if result.failed else ""),
            *(
                (
                    f"{s.ttft_warm_p50:,}",
                    f"{s.ttft_warm_p90:,}",
                    f"{s.first_sentence_warm_p50:,}",
                    f"{s.first_sentence_warm_p90:,}",
                    f"{s.first_sentence_cold_p50:,}",
                    f"{s.tokens_per_second_p50:.0f}",
                )
                if s
                else ("-",) * 6
            ),
        )
    console.print(table)
    if publish:
        for result in results:
            planned = result.calls + result.failed
            if result.speed is None or result.calls < 0.99 * planned or limit:
                console.print(
                    f"[yellow]![/yellow] {result.model} not published: needs every call, "
                    "and no --limit",
                    highlight=False,
                )
                continue
            target = Path("leaderboard/latency") / f"{result.model}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(result.model_dump_json(indent=1) + "\n", encoding="utf-8")
            console.print(f"[green]✓[/green] {target}", highlight=False)
