"""The cost of a run's calls, from the tokens they used and a price file.

    runs/<name>/cost.json    the mean cost of a scored call

A price file is written by whoever runs the command and is not part of the benchmark: prices
change, and only the resulting cost per call is published.

    priced: "2026-11-12"
    models:
      <run name>: {input: 1.00, cached_input: 0.10, output: 4.00, cache_write: 1.25}
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from mtva_bench.config import ConfigError, RunConfig
from mtva_bench.data import Bundle
from mtva_bench.run.score import current_calls
from mtva_bench.run.store import write_atomic
from mtva_bench.scoring.cost import Prices, cost_per_call
from mtva_bench.simulate.types import Record


class PriceList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    priced: str
    models: dict[str, Prices]


class CostResult(Record):
    model: str
    model_hash: str
    priced: str
    calls: int
    usd_per_call: float


def load_prices(path: Path) -> PriceList:
    try:
        return PriceList.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise ConfigError(f"{path}: {error}") from error


def cost_stage(config: RunConfig, bundle: Bundle, prices: PriceList) -> CostResult:
    """Price the run's scored calls and save the result as cost.json."""
    if config.name not in prices.models:
        raise ConfigError(f"the price file has no prices for {config.name}")
    _, conversations, _ = current_calls(config, bundle)
    usd = cost_per_call(conversations, prices.models[config.name])
    if usd is None:
        raise ConfigError(f"{config.run_dir} has no scored calls to price")
    result = CostResult(
        model=config.name,
        model_hash=config.model.hash(),
        priced=prices.priced,
        calls=len(conversations),
        usd_per_call=round(usd, 5),
    )
    write_atomic(config.run_dir / "cost.json", result.model_dump_json(indent=1) + "\n")
    return result
