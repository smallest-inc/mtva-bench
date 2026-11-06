import subprocess
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from mtva_bench import __version__
from mtva_bench.config import RunConfig, endpoint_hash
from mtva_bench.data import Bundle
from mtva_bench.prompts import template_hash
from mtva_bench.simulate.call import SIMULATOR_VERSION

PACKAGE_DIR = Path(__file__).resolve().parents[1]
CALLER_PROMPT = PACKAGE_DIR / "simulate" / "caller.md"


class Code(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    commit: str | None
    clean: bool | None


class Judging(BaseModel):
    """What judged a run's conversations. `inputs_hash` covers the judge model, prompt, verdict
    schema and judge code version."""

    model_config = ConfigDict(extra="forbid")

    judge: str
    model: str
    inputs_hash: str
    planned: int
    judged: int
    failed: int
    updated_at: str
    fallback_model: str | None = None
    fallback_judged: int = 0


class Manifest(BaseModel):
    """What produced a run, so it can be verified and compared. It names exactly what every
    conversation in the run is valid under, never the whole protocol file: capability weights
    and arena judges change no conversation or verdict."""

    model_config = ConfigDict(extra="forbid")

    run: str
    code: Code
    benchmark_version: str
    bundle_version: str
    bundle_hash: str
    model: str
    model_hash: str
    caller_hash: str
    caller_prompt_hash: str
    simulator_version: str
    seed: int
    max_turns: int
    max_reply_tokens: int
    rollouts: int
    partial: bool
    planned: int
    conversations: int
    failed: int
    updated_at: str
    judging: Judging | None = None


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(PACKAGE_DIR), *args], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def code() -> Code:
    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain") if commit else None
    return Code(version=__version__, commit=commit, clean=None if status is None else not status)


def build_manifest(
    config: RunConfig,
    bundle: Bundle,
    planned: int,
    conversations: int,
    failed: int,
    judging: Judging | None = None,
) -> Manifest:
    simulation = config.benchmark.simulation
    return Manifest(
        run=config.name,
        code=code(),
        benchmark_version=config.benchmark.version,
        bundle_version=bundle.version,
        bundle_hash=bundle.hash,
        model=config.model.name,
        model_hash=config.model.hash(),
        caller_hash=endpoint_hash(simulation.caller),
        caller_prompt_hash=template_hash(CALLER_PROMPT),
        simulator_version=SIMULATOR_VERSION,
        seed=simulation.seed,
        max_turns=simulation.max_turns,
        max_reply_tokens=simulation.max_reply_tokens,
        rollouts=simulation.rollouts,
        partial=config.options.partial,
        planned=planned,
        conversations=conversations,
        failed=failed,
        updated_at=now(),
        judging=judging,
    )
