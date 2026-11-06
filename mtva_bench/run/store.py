"""The run folder. RunStore is the only code that knows its paths.

    runs/<name>/
      run.yaml                                  protocol and model spec as resolved, no secrets
      manifest.json                             provenance: code, versions, hashes, seeds
      conversations/<scenario_id>__r<k>.json    one rollout, with the inputs it was made from
      verdicts/<judge>/<scenario_id>__r<k>.json a judge's verdict on it, with its inputs
      verdicts/<judge>/_batch.json              the batch being judged, until its results are in
      summary.json                              scores, intervals, slices and latency
      .lock                                     the process id of the run working here

Every write goes to a temporary file that is then renamed into place, so a killed run never
leaves a half-written file.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from mtva_bench.capabilities.conduct import ConductVerdict
from mtva_bench.llm.types import Usage
from mtva_bench.run.manifest import Manifest
from mtva_bench.simulate.types import Conversation, Record

ROLLOUT_SEPARATOR = "__r"


def stem(scenario_id: str, rollout: int) -> str:
    return f"{scenario_id}{ROLLOUT_SEPARATOR}{rollout}"


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


class RunLockedError(RuntimeError):
    pass


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class StoredConversation(Record):
    """A conversation and the hash of each input it was simulated from."""

    inputs: dict[str, str]
    conversation: Conversation


class VerdictHeader(BaseModel):
    """What a saved verdict was made from and whether it failed, readable even when the verdict
    itself was written under an older schema."""

    inputs: dict[str, str]
    error: str | None = None


class StoredVerdict(Record):
    """A judge's verdict on one rollout, or the error that stopped it, and the hash of each
    input it was judged from. `judge_model` is the model the provider says answered."""

    inputs: dict[str, str]
    judge: str
    verdict: ConductVerdict | None
    error: str | None = None
    usage: list[Usage]
    batch_id: str | None = None
    judge_model: str | None = None


class BatchState(Record):
    """A submitted batch and the inputs of each rollout in it, saved before polling so a killed
    run collects the same batch instead of paying for another."""

    batch_id: str
    inputs: dict[str, dict[str, str]]


class RunStore:
    def __init__(self, root: Path):
        self.root = root

    @property
    def conversations_dir(self) -> Path:
        return self.root / "conversations"

    def conversation_path(self, stem: str) -> Path:
        return self.conversations_dir / f"{stem}.json"

    def conversation(self, stem: str) -> StoredConversation | None:
        path = self.conversation_path(stem)
        if not path.is_file():
            return None
        return StoredConversation.model_validate_json(path.read_text(encoding="utf-8"))

    def save_conversation(self, stored: StoredConversation) -> None:
        conversation = stored.conversation
        path = self.conversation_path(stem(conversation.scenario_id, conversation.rollout))
        write_atomic(path, stored.model_dump_json(indent=1))

    def verdict_path(self, judge: str, stem: str) -> Path:
        return self.root / "verdicts" / judge / f"{stem}.json"

    def verdict(self, judge: str, stem: str) -> StoredVerdict | None:
        path = self.verdict_path(judge, stem)
        if not path.is_file():
            return None
        return StoredVerdict.model_validate_json(path.read_text(encoding="utf-8"))

    def verdict_header(self, judge: str, stem: str) -> VerdictHeader | None:
        path = self.verdict_path(judge, stem)
        if not path.is_file():
            return None
        return VerdictHeader.model_validate_json(path.read_text(encoding="utf-8"))

    def batch_state_path(self, judge: str) -> Path:
        return self.root / "verdicts" / judge / "_batch.json"

    def batch_state(self, judge: str) -> BatchState | None:
        path = self.batch_state_path(judge)
        if not path.is_file():
            return None
        return BatchState.model_validate_json(path.read_text(encoding="utf-8"))

    def save_batch_state(self, judge: str, state: BatchState) -> None:
        write_atomic(self.batch_state_path(judge), state.model_dump_json(indent=1))

    def clear_batch_state(self, judge: str) -> None:
        self.batch_state_path(judge).unlink(missing_ok=True)

    def save_verdict(self, stem: str, stored: StoredVerdict) -> None:
        write_atomic(self.verdict_path(stored.judge, stem), stored.model_dump_json(indent=1))

    @contextmanager
    def lock(self) -> Iterator[None]:
        """Hold the folder for this process: two runs in one folder would pay for the same
        calls and overwrite each other. A lock left by a process that died is taken over."""
        path = self.root / ".lock"
        self.root.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                owner = int(path.read_text() or 0)
                if alive(owner):
                    raise RunLockedError(
                        f"{self.root} is in use by another run (process {owner})"
                    ) from None
                path.unlink(missing_ok=True)
        else:
            raise RunLockedError(f"could not lock {self.root}")
        try:
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            yield
        finally:
            path.unlink(missing_ok=True)

    def save_run_yaml(self, data: dict) -> None:
        write_atomic(
            self.root / "run.yaml", yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        )

    def manifest(self) -> Manifest | None:
        """The saved manifest; None when there is none, or one in an older format, which the
        next stage writes afresh."""
        path = self.root / "manifest.json"
        if not path.is_file():
            return None
        try:
            return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError:
            return None

    def save_manifest(self, manifest: Manifest) -> None:
        write_atomic(self.root / "manifest.json", manifest.model_dump_json(indent=1))

    @property
    def summary_path(self) -> Path:
        return self.root / "summary.json"

    def save_summary(self, text: str) -> None:
        write_atomic(self.summary_path, text + "\n")
