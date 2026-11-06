from pathlib import Path

import pytest

from mtva_bench.config import (
    ConfigError,
    RunOptions,
    load_benchmark,
    load_model,
    load_run,
    stable_hash,
)

ROOT = Path(__file__).parent.parent


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


MODEL = """
name: demo
provider: openai
model: gpt-5.2
params: {temperature: 0.2}
serving: {kind: api}
"""


def test_committed_configs_load() -> None:
    benchmark = load_benchmark(ROOT / "benchmark.yaml")
    assert benchmark.judge.wire_api == "messages"
    for path in (ROOT / "models").glob("*.yaml"):
        assert load_model(path).name == path.stem


def test_model_spec_rejects_protocol_keys(tmp_path: Path) -> None:
    path = write(tmp_path, "m.yaml", MODEL + "judge: {provider: openai, model: x}\n")
    with pytest.raises(ConfigError, match=r"\['judge'\] belong in benchmark.yaml"):
        load_model(path)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ("temprature: 0.2\n", "temprature"),
        ("name: 'has space'\n", "letters, digits"),
    ],
)
def test_model_spec_errors_name_the_file(tmp_path: Path, extra: str, message: str) -> None:
    text = MODEL.replace("name: demo\n", "") if extra.startswith("name") else MODEL
    path = write(tmp_path, "m.yaml", text + extra)
    with pytest.raises(ConfigError, match=message) as raised:
        load_model(path)
    assert str(path) in str(raised.value)


def test_missing_and_empty_files(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="file not found"):
        load_model(tmp_path / "nope.yaml")
    with pytest.raises(ConfigError, match="expected a YAML mapping"):
        load_model(write(tmp_path, "empty.yaml", ""))


def test_model_hash_ignores_execution_fields(tmp_path: Path) -> None:
    base = load_model(write(tmp_path, "a.yaml", MODEL))
    relabelled = load_model(
        write(
            tmp_path,
            "b.yaml",
            MODEL.replace("demo", "demo-2").replace("{kind: api}", "{kind: api, region: eu}")
            + "concurrency: 4\ntimeout_s: 30\n",
        )
    )
    retuned = load_model(write(tmp_path, "c.yaml", MODEL.replace("0.2", "0.3")))
    assert base.hash() == relabelled.hash()
    assert base.hash() != retuned.hash()


def test_benchmark_validation(tmp_path: Path) -> None:
    text = (ROOT / "benchmark.yaml").read_text()
    with pytest.raises(ConfigError, match="must be odd"):
        load_benchmark(write(tmp_path, "even.yaml", text.replace("repeats: 1", "repeats: 2")))
    with pytest.raises(ConfigError, match="weights must be positive"):
        load_benchmark(
            write(tmp_path, "zero.yaml", text.replace("faithfulness: 1", "faithfulness: 0"))
        )


def test_run_config_names_and_overrides(tmp_path: Path) -> None:
    model = write(tmp_path, "m.yaml", MODEL)
    run = load_run(model, ROOT / "benchmark.yaml")
    assert run.name == "demo"
    assert run.run_dir == Path("runs/demo")
    assert run.concurrency == 32
    assert not run.options.partial
    custom = load_run(
        model,
        ROOT / "benchmark.yaml",
        RunOptions(name="demo-smoke", limit=10, concurrency=4, runs_dir=tmp_path),
    )
    assert custom.run_dir == tmp_path / "demo-smoke"
    assert custom.concurrency == 4
    assert custom.options.partial


def test_stable_hash_ignores_key_order() -> None:
    assert stable_hash({"a": 1, "b": "ह"}) == stable_hash({"b": "ह", "a": 1})
