import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from mtva_bench.llm.endpoints import Endpoint

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
PROTOCOL_KEYS = frozenset({"caller", "judge", "simulation", "data", "capabilities", "arena"})
# Settings that change how a request is sent, not what comes back, so no hash covers them.
UNHASHED_ENDPOINT_FIELDS = frozenset({"timeout_s", "api_key_env"})
UNHASHED_MODEL_FIELDS = UNHASHED_ENDPOINT_FIELDS | {"name", "display", "serving", "concurrency"}

T = TypeVar("T", bound=BaseModel)


class ConfigError(ValueError):
    pass


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Serving(ConfigModel):
    kind: Literal["api", "self-hosted"]
    region: str | None = None
    hardware: str | None = None


class Display(ConfigModel):
    """How the leaderboard names a model. `reasoning` is the setting it ran at, as a reader would
    say it: none, minimal, low."""

    label: str
    organization: str
    reasoning: str


class ModelSpec(Endpoint):
    """`timeout_s` is far below an endpoint's default: a call turn streams, so a gap that long
    means the request has stalled, and waiting longer only holds a slot the run could use."""

    name: str
    display: Display | None = None
    serving: Serving
    timeout_s: float = Field(default=120, gt=0)
    concurrency: int = Field(default=32, ge=1)
    needs_opening_user_turn: bool = False
    text_with_tool_calls: bool = True

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not NAME.fullmatch(value):
            raise ValueError("must be letters, digits, '.', '_' or '-', starting alphanumeric")
        return value

    def hash(self) -> str:
        return stable_hash(self.model_dump(mode="json", exclude=set(UNHASHED_MODEL_FIELDS)))


class Data(ConfigModel):
    version: str
    sha256: str | None = None
    suites: list[str] = Field(default_factory=lambda: ["core"], min_length=1)
    voice_guide: bool = True


class Simulation(ConfigModel):
    """`max_reply_tokens` caps one model response for every model alike, so a response that
    loops is cut there instead of filling the context window."""

    rollouts: int = Field(default=1, ge=1)
    max_turns: int = Field(default=30, ge=1)
    max_reply_tokens: int = Field(default=4096, ge=1)
    seed: int = 0
    caller: Endpoint


class Judge(Endpoint):
    """`fallback` judges only the calls this judge refuses."""

    repeats: int = Field(default=1, ge=1)
    fallback: Endpoint | None = None

    @field_validator("repeats")
    @classmethod
    def validate_repeats(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("must be odd so the median verdict is a real verdict")
        return value


class Arena(ConfigModel):
    """The judge of the arena's blind head-to-head, and the seed that places each run's call
    first or second."""

    judge: Endpoint | None = None
    seed: int = 0


class Benchmark(ConfigModel):
    version: str
    data: Data
    simulation: Simulation
    judge: Judge
    capabilities: dict[str, float] = Field(min_length=1)
    arena: Arena = Field(default_factory=Arena)

    @field_validator("capabilities")
    @classmethod
    def validate_weights(cls, value: dict[str, float]) -> dict[str, float]:
        invalid = sorted(name for name, weight in value.items() if weight <= 0)
        if invalid:
            raise ValueError(f"weights must be positive: {invalid}")
        return value


class RunOptions(ConfigModel):
    name: str | None = None
    runs_dir: Path = Path("runs")
    data_dir: Path | None = None
    limit: int | None = Field(default=None, ge=1)
    agents: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    scenarios: list[str] = Field(default_factory=list)
    concurrency: int | None = Field(default=None, ge=1)
    judge_concurrency: int = Field(default=16, ge=1)
    batch: bool = False
    evaluate: bool = True
    dry_run: bool = False
    discard_stale: bool = False

    @property
    def partial(self) -> bool:
        return bool(self.limit or self.agents or self.languages or self.scenarios)


class RunConfig(ConfigModel):
    benchmark: Benchmark
    model: ModelSpec
    options: RunOptions = Field(default_factory=RunOptions)

    @property
    def name(self) -> str:
        return self.options.name or self.model.name

    @property
    def run_dir(self) -> Path:
        return self.options.runs_dir / self.name

    @property
    def concurrency(self) -> int:
        return self.options.concurrency or self.model.concurrency


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"{path}: file not found")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a YAML mapping")
    return data


def validate(path: Path, data: dict[str, Any], schema: type[T]) -> T:
    try:
        return schema.model_validate(data)
    except ValidationError as error:
        raise ConfigError(f"{path}: {error}") from error


def load_benchmark(path: Path) -> Benchmark:
    return validate(path, read_yaml(path), Benchmark)


def load_model(path: Path) -> ModelSpec:
    data = read_yaml(path)
    misplaced = sorted(PROTOCOL_KEYS & set(data))
    if misplaced:
        raise ConfigError(
            f"{path}: {misplaced} belong in benchmark.yaml; a model spec only says how to call "
            "the model"
        )
    return validate(path, data, ModelSpec)


def load_run(
    model_path: Path,
    benchmark_path: Path = Path("benchmark.yaml"),
    options: RunOptions | None = None,
) -> RunConfig:
    return RunConfig(
        benchmark=load_benchmark(benchmark_path),
        model=load_model(model_path),
        options=options or RunOptions(),
    )


def endpoint_hash(endpoint: Endpoint, exclude: frozenset[str] = frozenset()) -> str:
    return stable_hash(
        endpoint.model_dump(mode="json", exclude=set(UNHASHED_ENDPOINT_FIELDS | exclude))
    )


def stable_hash(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()
