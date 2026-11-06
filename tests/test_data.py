import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from click.testing import CliRunner

from mtva_bench.cli import cli
from mtva_bench.config import Data
from mtva_bench.data import (
    DataError,
    Matcher,
    check_bundle,
    find_bundle,
    hash_files,
    load_bundle,
    matches,
    system_prompt,
    verify,
)
from mtva_bench.data.bundle import DATA_DIR_ENV
from mtva_bench.data.render import render

FIXTURE = Path(__file__).parent / "fixtures" / "bundle"
SCENARIO = "scenarios/demo_dental/demo_dental_01.json"


def copy_bundle(tmp_path: Path, edit: Callable[[dict], object] | None = None) -> Path:
    """The fixture bundle in tmp_path, with `edit` applied to its scenario and the manifest
    rehashed so only the edit itself can produce a finding."""
    root = Path(shutil.copytree(FIXTURE, tmp_path / "bundle"))
    if edit is not None:
        path = root / SCENARIO
        scenario = json.loads(path.read_text())
        edit(scenario)
        path.write_text(json.dumps(scenario))
        manifest = json.loads((root / "bundle.json").read_text())
        manifest["files"] = hash_files(root)
        (root / "bundle.json").write_text(json.dumps(manifest))
    return root


def test_fixture_is_valid() -> None:
    bundle = load_bundle(FIXTURE)
    assert check_bundle(bundle) == []
    verify(bundle, Data(version="1.0.0", sha256=bundle.hash))
    with pytest.raises(DataError, match=r"pins 2\.0\.0"):
        verify(bundle, Data(version="2.0.0"))
    with pytest.raises(DataError, match="has hash"):
        verify(bundle, Data(version="1.0.0", sha256="0" * 64))


def test_load_lists_every_invalid_file(tmp_path: Path) -> None:
    root = copy_bundle(tmp_path, lambda s: s.pop("caller"))
    (root / "agents" / "demo_dental.json").write_text("{")
    with pytest.raises(DataError) as error:
        load_bundle(root)
    assert "agents/demo_dental.json: not valid JSON" in str(error.value)
    assert f"{SCENARIO}: caller: Field required" in str(error.value)


@pytest.mark.parametrize(
    ("edit", "finding"),
    [
        (lambda s: s["variables"].pop("clinic_name"), "no value for ['clinic_name']"),
        (lambda s: s["tools"].pop("end_call"), "tool end_call has no behaviour"),
        (
            lambda s: s["tools"]["book_visit"]["rules"][0]["when"].update(
                date={"ref": "caller.facts.missing"}
            ),
            "rule book_friday: ref caller.facts.missing does not resolve",
        ),
        (
            lambda s: s["evaluation"]["key_actions"][0]["arguments"].pop("date"),
            "key action book: no book_visit rule answers exactly these arguments",
        ),
        (
            lambda s: s["evaluation"]["focus_rules"][0].update(quote="Always confirm."),
            "focus rule confirm_first does not quote the agent prompt verbatim",
        ),
        (
            lambda s: s["tools"]["book_visit"]["rules"][0]["when"].update(time={"present": True}),
            "rule book_friday: time is not a parameter of book_visit",
        ),
    ],
)
def test_check_findings(tmp_path: Path, edit: Callable[[dict], object], finding: str) -> None:
    findings = check_bundle(load_bundle(copy_bundle(tmp_path, edit)))
    assert f"demo_dental_01: {finding}" in findings


def test_check_catches_edited_files(tmp_path: Path) -> None:
    root = copy_bundle(tmp_path)
    (root / "voice_guide.md").write_text("changed")
    (root / "agents" / "extra.json").write_text((root / "agents" / "demo_dental.json").read_text())
    with pytest.raises(DataError, match=r"extra\.json: holds agent demo_dental"):
        load_bundle(root)
    (root / "agents" / "extra.json").unlink()
    bundle = load_bundle(root)
    assert check_bundle(bundle) == ["voice_guide.md: sha256 does not match bundle.json"]
    with pytest.raises(DataError, match=r"voice_guide\.md: sha256 does not match"):
        verify(bundle, Data(version="1.0.0"))


def test_system_prompt() -> None:
    bundle = load_bundle(FIXTURE)
    scenario = bundle.scenarios["demo_dental_01"]
    agent = bundle.agent(scenario)
    prompt = system_prompt(agent, scenario, bundle.voice_guide)
    assert prompt.startswith(
        "## Agent Prompt\n\n[start of agent prompt]\n\n# Front desk for {{clinic_name}}\n"
    )
    assert "[end of agent prompt]\n\n---\n\n## Voice guide" in prompt
    assert "Today is Wednesday, August 19, 2026" in prompt
    assert prompt.endswith("- clinic_name: MetroMolar")
    other_values = {"clinic_name": "X", "current_date": "May 1, 2026", "current_day": "Friday"}
    other = scenario.model_copy(update={"variables": {**scenario.variables, **other_values}})
    full, shared = render(agent, scenario, bundle.voice_guide)
    assert full == prompt and full[shared:].startswith("Today is Wednesday")
    assert render(agent, other, bundle.voice_guide)[0][:shared] == full[:shared]
    bare = system_prompt(agent, scenario, None)
    assert "## Voice guide" not in bare
    assert "You are Female. Speak Hindi by default" in bare


@pytest.mark.parametrize(
    ("matcher", "value", "expected"),
    [
        ({"equals": "Ghodbunder Rd."}, "ghodbunder rd", True),
        ({"equals": 1200}, "1,200", True),
        ({"equals": "10:00"}, "10:30", False),
        ({"equals": "9876543210"}, "98765 43210", True),
        ({"equals": "2026-08-21"}, "21 August 2026", True),
        ({"ref": "caller.facts.preferred_date"}, "21/08/2026", True),
        ({"equals": True}, "true", True),
        ({"any_of": ["Pune", "पुणे"]}, "पुणे", True),
        ({"range": [100, 200]}, "१५०", True),
        ({"range": [100, 200]}, "250", False),
        ({"date": "2026-08-21"}, "August 21st, 2026", True),
        ({"date": "2026-08-21"}, "21/08/2026", True),
        ({"date": "2026-08-21"}, "2026-08-22", False),
        ({"phone": "9876543210"}, "+91 98765-43210", True),
        ({"phone": "9876543210"}, "9876543211", False),
        ({"ref": "caller.facts.phone"}, "9876543210", True),
        ({"present": True}, "  ", False),
        ({"present": True}, None, False),
    ],
)
def test_matchers(matcher: dict, value: object, expected: bool) -> None:
    scenario = load_bundle(FIXTURE).scenarios["demo_dental_01"]
    assert matches(Matcher.model_validate(matcher), value, scenario) is expected


def test_find_bundle_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    with pytest.raises(DataError, match="pass --data"):
        find_bundle(None, None)
    with pytest.raises(DataError, match="from the default folder"):
        find_bundle(None, "0.0.0-missing")
    monkeypatch.setenv(DATA_DIR_ENV, str(FIXTURE))
    assert find_bundle(None, "1.0.0") == FIXTURE
    with pytest.raises(DataError, match=r"from --data\) is not a bundle"):
        find_bundle(tmp_path, "1.0.0")


def test_cli(tmp_path: Path) -> None:
    runner = CliRunner()
    data = ["--data", str(FIXTURE), "--benchmark", str(tmp_path / "none.yaml")]
    result = runner.invoke(cli, ["data", "check", *data])
    assert result.exit_code == 0, result.output
    assert "1 agents, 1 scenarios" in result.output
    result = runner.invoke(cli, ["data", "list", "--scenarios", *data])
    assert "demo_dental_01" in result.output
    result = runner.invoke(cli, ["data", "show", "demo_dental_01", "--prompt", *data])
    assert result.output.startswith("## Agent Prompt")
    result = runner.invoke(
        cli,
        [
            "data",
            "check",
            *data[:1],
            str(copy_bundle(tmp_path, lambda s: s["tools"].pop("end_call"))),
            *data[2:],
        ],
    )
    assert result.exit_code == 1
    assert "tool end_call has no behaviour" in result.output


def test_date_times_and_spoken_ids_match() -> None:
    from datetime import date

    from mtva_bench.data.matchers import dates, equal

    for value in (
        "2026-06-12T21:00:00+05:30",
        "2026-06-12T21:00:00",
        "2026-06-12 21:00",
        "2026-06-12T15:30:00Z",
    ):
        assert dates(value) == {date(2026, 6, 12)}, value
    assert equal("2026-10-16", "2026-10-16T09:00:00-05:00")
    assert equal("MBR-0815533-01", "MBR081553301")
    assert equal("PSH-2022-0815533", "psh 2022 0815533")
    assert not equal("MBR-0815533-01", "MBR-0815533-02")


def bundle_archive(root: Path, extra: dict[str, bytes] | None = None) -> bytes:
    import io
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(root).as_posix())
        for name, content in (extra or {}).items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def test_download_checks_everything_before_the_bundle_is_used(tmp_path: Path) -> None:
    import httpx

    from mtva_bench.config import Data
    from mtva_bench.data import DataError, download_bundle, load_bundle

    bundle = load_bundle(FIXTURE)
    pinned = Data(version=bundle.version, sha256=bundle.hash)
    archive = bundle_archive(FIXTURE)
    served = httpx.MockTransport(lambda _: httpx.Response(200, content=archive))
    path = download_bundle(pinned, tmp_path / "data", served)
    assert load_bundle(path).hash == bundle.hash

    other = Data(version=bundle.version, sha256="0" * 64)
    with pytest.raises(DataError, match="pins"):
        download_bundle(other, tmp_path / "elsewhere", served)
    assert not (tmp_path / "elsewhere" / bundle.version).exists()

    sneaky = httpx.MockTransport(
        lambda _: httpx.Response(200, content=bundle_archive(FIXTURE, {"../escape.txt": b"x"}))
    )
    with pytest.raises(DataError, match="unsafe entry"):
        download_bundle(pinned, tmp_path / "third", sneaky)
    missing = httpx.MockTransport(lambda _: httpx.Response(404))
    with pytest.raises(DataError, match="private; set HF_TOKEN"):
        download_bundle(pinned, tmp_path / "fourth", missing)


def test_download_sends_the_hugging_face_token(tmp_path: Path, monkeypatch) -> None:
    import httpx

    from mtva_bench.config import Data
    from mtva_bench.data import download_bundle, load_bundle

    bundle = load_bundle(FIXTURE)
    archive = bundle_archive(FIXTURE)
    seen: list[str | None] = []

    def private(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, content=archive)

    monkeypatch.setenv("HF_TOKEN", "hf_test")
    pinned = Data(version=bundle.version, sha256=bundle.hash)
    download_bundle(pinned, tmp_path / "data", httpx.MockTransport(private))
    assert seen == ["Bearer hf_test"]
