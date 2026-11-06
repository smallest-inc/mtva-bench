"""Finding, loading, verifying and checking a scenario bundle.

A bundle is a folder:

    bundle.json                       name, version, schema version, sha256 of every file
    voice_guide.md                    the guide appended to every agent prompt
    agents/<agent_id>.json
    scenarios/<agent_id>/<scenario_id>.json

Its hash is the hash of the file list in bundle.json, so it changes when any file does.

A published bundle is downloaded as one archive from the Hugging Face dataset and checked in
full, every file and the hash benchmark.yaml pins, before it takes its place in data/<version>.
"""

import hashlib
import json
import os
import tarfile
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from mtva_bench.config import Data, stable_hash
from mtva_bench.data.render import compose, placeholders
from mtva_bench.data.schema import Agent, Contract, DataError, Matcher, Scenario
from mtva_bench.llm.endpoints import read_env

MANIFEST = "bundle.json"
VOICE_GUIDE = "voice_guide.md"
DATA_DIR_ENV = "MTVA_BENCH_DATA_DIR"
DEFAULT_DIR = Path("data")
DATASET_REPO = "smallestpritish/mtva-bench"
ARCHIVE_URL = (
    "https://huggingface.co/datasets/{repo}/resolve/v{version}/mtva-bench-{version}.tar.gz"
)
DATA_URL_ENV = "MTVA_BENCH_DATA_URL"
# The dataset is private: the download sends this Hugging Face token when it is set.
TOKEN_ENV = "HF_TOKEN"

T = TypeVar("T", bound=BaseModel)


class Manifest(Contract):
    name: str
    version: str
    schema_version: Literal["3.0"]
    files: dict[str, str]


@dataclass(frozen=True)
class Bundle:
    root: Path
    manifest: Manifest
    voice_guide: str
    agents: dict[str, Agent]
    scenarios: dict[str, Scenario]

    @property
    def version(self) -> str:
        return self.manifest.version

    @property
    def hash(self) -> str:
        return stable_hash(self.manifest.files)

    def agent(self, scenario: Scenario) -> Agent:
        return self.agents[scenario.agent_id]

    def select(
        self,
        agents: Iterable[str] = (),
        languages: Iterable[str] = (),
        difficulties: Iterable[str] = (),
        scenarios: Iterable[str] = (),
    ) -> list[Scenario]:
        filters = {
            "agent_id": set(agents),
            "language": set(languages),
            "difficulty": set(difficulties),
            "scenario_id": set(scenarios),
        }
        return [
            scenario
            for scenario in self.scenarios.values()
            if all(
                not wanted or getattr(scenario, key) in wanted for key, wanted in filters.items()
            )
        ]


def find_bundle(explicit: Path | None, version: str | None) -> Path:
    """The bundle folder: `explicit`, else $MTVA_BENCH_DATA_DIR, else data/<version> in the
    current folder."""
    if explicit is not None:
        path, source = explicit, "--data"
    elif os.environ.get(DATA_DIR_ENV):
        path, source = Path(os.environ[DATA_DIR_ENV]), DATA_DIR_ENV
    elif version is not None:
        path, source = DEFAULT_DIR / version, "the default folder"
    else:
        raise DataError(f"no bundle given: pass --data or set {DATA_DIR_ENV}")
    if not (path / MANIFEST).is_file():
        raise DataError(
            f"{path} (from {source}) is not a bundle: it has no {MANIFEST}. Put the bundle "
            f"there, set {DATA_DIR_ENV}, or pass --data"
        )
    return path


def hash_files(root: Path) -> dict[str, str]:
    """sha256 of every bundle file except the manifest, keyed by its path in the bundle."""
    paths = [root / VOICE_GUIDE, *root.glob("agents/*.json"), *root.glob("scenarios/*/*.json")]
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
        if path.is_file()
    }


def read_json(path: Path, root: Path, schema: type[T], errors: list[str]) -> T | None:
    where = path.relative_to(root).as_posix()
    try:
        return schema.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as error:
        errors.append(f"{where}: not valid JSON: {error}")
    except ValidationError as error:
        errors += [
            f"{where}: {'.'.join(str(part) for part in item['loc']) or 'file'}: {item['msg']}"
            for item in error.errors()
        ]
    return None


def load_bundle(root: Path) -> Bundle:
    """Every file parsed against the schema; DataError lists every file that is not."""
    errors: list[str] = []
    manifest = read_json(root / MANIFEST, root, Manifest, errors)
    if not (root / VOICE_GUIDE).is_file():
        errors.append(f"{VOICE_GUIDE} is missing")
    agents: dict[str, Agent] = {}
    for path in sorted(root.glob("agents/*.json")):
        agent = read_json(path, root, Agent, errors)
        if agent is not None and agent.agent_id != path.stem:
            errors.append(f"{path.name}: holds agent {agent.agent_id}")
        elif agent is not None:
            agents[agent.agent_id] = agent
    scenarios: dict[str, Scenario] = {}
    for path in sorted(root.glob("scenarios/*/*.json")):
        scenario = read_json(path, root, Scenario, errors)
        if scenario is None:
            continue
        if (scenario.scenario_id, scenario.agent_id) != (path.stem, path.parent.name):
            errors.append(f"{path.relative_to(root)}: holds scenario {scenario.scenario_id}")
            continue
        scenarios[scenario.scenario_id] = scenario
    if errors or manifest is None:
        raise DataError(f"{root} is not a valid bundle:\n" + "\n".join(errors))
    voice_guide = (root / VOICE_GUIDE).read_text(encoding="utf-8")
    return Bundle(root, manifest, voice_guide, agents, scenarios)


def verify(bundle: Bundle, data: Data) -> None:
    """Refuse a bundle other than the one benchmark.yaml pins, or one whose files no longer
    match its manifest."""
    changed = file_findings(bundle)
    if changed:
        raise DataError("bundle files do not match its manifest:\n" + "\n".join(changed))
    if bundle.version != data.version:
        raise DataError(f"bundle is version {bundle.version}; benchmark.yaml pins {data.version}")
    if data.sha256 is not None and bundle.hash != data.sha256:
        raise DataError(
            f"bundle {bundle.version} has hash {bundle.hash}; benchmark.yaml pins {data.sha256}"
        )


def check_bundle(bundle: Bundle) -> list[str]:
    """Problems that schema validation cannot see: file hashes, and references between a
    scenario, its agent and itself."""
    findings = file_findings(bundle)
    used = {scenario.agent_id for scenario in bundle.scenarios.values()}
    findings += [
        f"agent {agent_id} has no scenarios" for agent_id in sorted(set(bundle.agents) - used)
    ]
    for scenario in bundle.scenarios.values():
        agent = bundle.agents.get(scenario.agent_id)
        if agent is None:
            findings.append(f"{scenario.scenario_id}: unknown agent {scenario.agent_id}")
            continue
        findings += [
            f"{scenario.scenario_id}: {problem}"
            for problem in scenario_findings(scenario, agent, bundle.voice_guide)
        ]
    return findings


def file_findings(bundle: Bundle) -> list[str]:
    listed, actual = bundle.manifest.files, hash_files(bundle.root)
    return [
        *(
            f"{path}: listed in {MANIFEST} but missing"
            for path in sorted(set(listed) - set(actual))
        ),
        *(f"{path}: not listed in {MANIFEST}" for path in sorted(set(actual) - set(listed))),
        *(
            f"{path}: sha256 does not match {MANIFEST}"
            for path in sorted(set(listed) & set(actual))
            if listed[path] != actual[path]
        ),
    ]


def scenario_findings(scenario: Scenario, agent: Agent, voice_guide: str) -> list[str]:
    findings = []
    if scenario.language != agent.language:
        findings.append(f"language {scenario.language} is not the agent's {agent.language}")
    missing = placeholders(compose(agent, voice_guide)) - set(scenario.variables)
    if missing:
        findings.append(f"no value for {sorted(missing)}")

    tool_names = {tool.name for tool in agent.tools}
    if set(scenario.tools) != tool_names:
        findings += [
            f"tool {name} is not declared by the agent"
            for name in sorted(set(scenario.tools) - tool_names)
        ]
        findings += [
            f"tool {name} has no behaviour" for name in sorted(tool_names - set(scenario.tools))
        ]

    rule_ids = [rule.id for behaviour in scenario.tools.values() for rule in behaviour.rules]
    if len(rule_ids) != len(set(rule_ids)):
        findings.append("rule ids are not unique across tools")
    for tool_name, behaviour in scenario.tools.items():
        for rule in behaviour.rules:
            findings += matcher_findings(f"rule {rule.id}", rule.when, tool_name, agent, scenario)
            findings += [
                f"rule {rule.id} requires unknown rule {other}"
                for other in rule.requires
                if other not in rule_ids
            ]

    evaluation = scenario.evaluation
    prompt = " ".join(agent.prompt.split())
    findings += [
        f"focus rule {focus.id} does not quote the agent prompt verbatim"
        for focus in evaluation.focus_rules
        if " ".join(focus.quote.split()) not in prompt
    ]
    focus_ids = {focus.id for focus in evaluation.focus_rules}
    action_ids = [action.id for action in evaluation.key_actions]
    if len(action_ids) != len(set(action_ids)):
        findings.append("key action ids are not unique")
    for action in evaluation.key_actions:
        where = f"key action {action.id}"
        if agent.tool(action.tool) is None:
            findings.append(f"{where}: unknown tool {action.tool}")
            continue
        if action.focus is not None and action.focus not in focus_ids:
            findings.append(f"{where}: unknown focus rule {action.focus}")
        findings += [
            f"{where}: comes after unknown action {other}"
            for other in action.after
            if other not in action_ids
        ]
        findings += matcher_findings(where, action.arguments, action.tool, agent, scenario)
        rules = scenario.tools[action.tool].rules if action.tool in scenario.tools else []
        if (
            action.expect == "call"
            and action.arguments
            and not any(rule.when == action.arguments for rule in rules)
        ):
            findings.append(f"{where}: no {action.tool} rule answers exactly these arguments")
    return findings


def matcher_findings(
    where: str, matchers: dict[str, Matcher], tool_name: str, agent: Agent, scenario: Scenario
) -> list[str]:
    tool = agent.tool(tool_name)
    findings = []
    for name, matcher in matchers.items():
        if tool is not None and name not in tool.properties:
            findings.append(f"{where}: {name} is not a parameter of {tool_name}")
        if matcher.ref is not None:
            try:
                scenario.resolve(matcher.ref)
            except KeyError:
                findings.append(f"{where}: ref {matcher.ref} does not resolve")
    return findings


def archive_url(version: str) -> str:
    return os.environ.get(DATA_URL_ENV) or ARCHIVE_URL.format(repo=DATASET_REPO, version=version)


def extract(archive: Path, into: Path) -> None:
    """Unpack a bundle archive, refusing any entry that is not a plain file or folder inside
    `into`."""
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not (member.isfile() or member.isdir()):
                raise DataError(f"unsafe entry in the bundle archive: {member.name}")
        if hasattr(tarfile, "data_filter"):
            tar.extractall(into, filter="data")
        else:
            tar.extractall(into)


def download_bundle(
    data: Data, root: Path = DEFAULT_DIR, transport: httpx.BaseTransport | None = None
) -> Path:
    """Download the pinned bundle into root/<version>. It takes that place only once every
    file matches its manifest and the bundle matches benchmark.yaml; otherwise nothing is
    left behind."""
    target = root / data.version
    if (target / MANIFEST).is_file():
        raise DataError(f"{target} already holds a bundle")
    url = archive_url(data.version)
    token = read_env(TOKEN_ENV)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix=".download-") as temporary:
        archive = Path(temporary) / "bundle.tar.gz"
        with (
            httpx.Client(
                transport=transport, headers=headers, follow_redirects=True, timeout=120
            ) as client,
            client.stream("GET", url) as response,
        ):
            if response.status_code in (401, 403, 404) and DATA_URL_ENV not in os.environ:
                raise DataError(
                    f"could not download {url}: the dataset is private; set {TOKEN_ENV} to a "
                    f"Hugging Face token with access to {DATASET_REPO}"
                )
            if response.status_code != 200:
                raise DataError(f"could not download {url}: HTTP {response.status_code}")
            with archive.open("wb") as file:
                for chunk in response.iter_bytes():
                    file.write(chunk)
        staged = Path(temporary) / "bundle"
        extract(archive, staged)
        verify(load_bundle(staged), data)
        staged.rename(target)
    return target
