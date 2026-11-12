import json
import math
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from mtva_bench.cli import cli
from mtva_bench.config import ConfigError, load_benchmark
from mtva_bench.leaderboard.build import LeaderboardError, load_entries
from mtva_bench.leaderboard.charts import charts
from mtva_bench.leaderboard.readme import readme_parts
from mtva_bench.llm.types import Timing, Usage
from mtva_bench.run import RunStore, evaluate_stage
from mtva_bench.run.cost import cost_stage, load_prices
from mtva_bench.scoring.cost import Prices, call_cost, cost_per_call
from mtva_bench.scoring.index import FIRST_TOKEN_MS, USD_PER_CALL, mtva_index, scaled
from mtva_bench.simulate.types import Conversation, ModelCall
from tests.test_capabilities import fake_judge
from tests.test_run import hang_up, make_bundle, make_config, run, unpinned_benchmark

ROOT = Path(__file__).parent.parent


def conversation(*usages: Usage) -> Conversation:
    return Conversation(
        scenario_id="s",
        agent_id="a",
        model="m",
        rollout=0,
        seed=0,
        messages=[],
        model_calls=[
            ModelCall(
                turn=turn,
                round=0,
                timing=Timing(total_ms=1000),
                usage=usage,
                spoke=True,
                tool_calls=0,
                finish_reason="stop",
            )
            for turn, usage in enumerate(usages)
        ],
        end_reason="caller_done",
        duration_ms=1,
    )


def test_a_call_is_priced_by_how_each_token_was_billed() -> None:
    call = conversation(
        Usage(input_tokens=1000, output_tokens=100, cache_read_tokens=600, cache_write_tokens=300),
        Usage(input_tokens=2000, output_tokens=50, cache_read_tokens=2000),
    )
    prices = Prices(input=1.0, cached_input=0.1, output=4.0, cache_write=1.25)
    # Uncached 100, read 600, written 300, out 100; then read 2000, out 50.
    assert call_cost(call, prices) == pytest.approx((100 + 60 + 375 + 400 + 200 + 200) / 1e6)
    # Without a write price, input written to the cache is billed as plain input.
    plain = Prices(input=1.0, cached_input=0.1, output=4.0)
    assert call_cost(call, plain) == pytest.approx((100 + 60 + 300 + 400 + 200 + 200) / 1e6)
    assert cost_per_call([call, conversation()], prices) == pytest.approx(
        call_cost(call, prices) / 2
    )
    assert cost_per_call([], prices) is None


def test_the_index_needs_every_part() -> None:
    fast, slow = FIRST_TOKEN_MS
    cheap, dear = USD_PER_CALL
    assert scaled(fast, fast, slow) == scaled(fast / 2, fast, slow) == 100
    assert scaled(slow, fast, slow) == scaled(slow * 2, fast, slow) == 0
    assert scaled(math.sqrt(fast * slow), fast, slow) == pytest.approx(50)

    assert mtva_index(100, fast, cheap) == pytest.approx(100)
    assert mtva_index(64, fast, cheap) == pytest.approx(80)
    middling = mtva_index(70, 1000, 0.02)
    assert mtva_index(80, 1000, 0.02) > middling
    assert mtva_index(70, 500, 0.02) > middling
    assert mtva_index(70, 1000, 0.01) > middling
    # A strong part does not make up for a poor one.
    assert mtva_index(95, slow, cheap) == 0
    assert mtva_index(95, fast, dear) == 0


async def test_cost_stage_prices_the_scored_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = make_bundle(tmp_path)
    config = make_config(tmp_path)
    await run(config, bundle, hang_up())
    await evaluate_stage(config, bundle, fake_judge([]))
    store = RunStore(config.run_dir)
    requests = 0
    for path in sorted(store.conversations_dir.glob("*.json")):
        stored = store.conversation(path.stem)
        for call in stored.conversation.model_calls:
            call.usage = Usage(input_tokens=1000, output_tokens=10, cache_read_tokens=900)
            requests += 1
        store.save_conversation(stored)

    price_file = tmp_path / "prices.yaml"
    rates = "{input: 2.0, cached_input: 0.5, output: 8.0}"
    price_file.write_text(f'priced: "2026-01-02"\nmodels:\n  {config.name}: {rates}\n')
    result = cost_stage(config, bundle, load_prices(price_file))
    per_request = (100 * 2.0 + 900 * 0.5 + 10 * 8.0) / 1e6
    assert result.calls == 3
    assert result.usd_per_call == pytest.approx(per_request * requests / 3, abs=1e-5)
    assert (result.model_hash, result.priced) == (config.model.hash(), "2026-01-02")
    assert json.loads((config.run_dir / "cost.json").read_text()) == result.model_dump()

    monkeypatch.chdir(tmp_path)
    options = ["--benchmark", str(unpinned_benchmark(tmp_path)), "--data", str(bundle.root)]
    command = ["cost", str(config.run_dir), "--prices", str(price_file), "--publish", *options]
    done = CliRunner().invoke(cli, command)
    assert done.exit_code == 0, done.output
    assert (tmp_path / "leaderboard" / "cost" / f"{config.name}.json").is_file()

    price_file.write_text('priced: "2026-01-02"\nmodels: {}\n')
    with pytest.raises(ConfigError, match="no prices for"):
        cost_stage(config, bundle, load_prices(price_file))
    price_file.write_text("models: {}\n")
    with pytest.raises(ConfigError, match="priced"):
        load_prices(price_file)


def test_the_leaderboard_takes_a_cost_for_every_model_or_none(tmp_path: Path) -> None:
    benchmark = load_benchmark(ROOT / "benchmark.yaml")
    for part in ("runs", "latency", "cost"):
        shutil.copytree(ROOT / "leaderboard" / part, tmp_path / part)
    entries = load_entries(tmp_path / "runs", benchmark)
    assert all(entry.cost is not None and entry.index is not None for entry in entries)
    parts = readme_parts(entries)
    assert parts["table"].splitlines()[0].endswith("| Cost per call | MTVA Index |")
    assert len(parts["table"].splitlines()) == len(parts["capabilities"].splitlines())
    assert (
        parts["capabilities"].splitlines()[0].endswith("| Instruction following | Faithfulness |")
    )
    drawn = charts(entries, [])
    assert "mtva-index.svg" in drawn

    first = next((tmp_path / "cost").glob("*.json"))
    result = json.loads(first.read_text())
    first.write_text(json.dumps({**result, "model_hash": "another spec"}))
    with pytest.raises(LeaderboardError, match="priced with another model spec"):
        load_entries(tmp_path / "runs", benchmark)
    first.write_text(json.dumps({**result, "priced": "2020-01-01"}))
    with pytest.raises(LeaderboardError, match="different days"):
        load_entries(tmp_path / "runs", benchmark)
    first.unlink()
    with pytest.raises(LeaderboardError, match="no cost"):
        load_entries(tmp_path / "runs", benchmark)

    shutil.rmtree(tmp_path / "cost")
    unpriced = load_entries(tmp_path / "runs", benchmark)
    assert all(entry.cost is None and entry.index is None for entry in unpriced)
    assert "MTVA Index" not in readme_parts(unpriced)["table"]
    assert "mtva-index.svg" not in charts(unpriced, [])
