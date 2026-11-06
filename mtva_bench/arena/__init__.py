"""The arena: which of two agents sounds more human, judged blind on the same scenarios.

The judge reads the two calls of one scenario as speech alone: no system prompt, tools or model
names, so it can only judge how each agent sounds. Judges tend to favour whichever answer they
read first, so each run's call is shown first in exactly half of the scenarios, picked by a
seeded shuffle; the lean cancels out across the set instead of being paid for twice.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from mtva_bench.llm.types import Message, Request, Response
from mtva_bench.prompts import load_prompt
from mtva_bench.simulate.call import OPENING_TURN
from mtva_bench.simulate.types import Conversation, Record

PROMPT = Path(__file__).parent / "prompt.md"
ATTEMPTS = 2
SCHEMA = {
    "type": "object",
    "properties": {
        "reasons": {"type": "string"},
        "verdict": {"type": "string", "enum": ["A", "B", "tie"]},
    },
    "required": ["reasons", "verdict"],
    "additionalProperties": False,
}


class ArenaVerdict(Record):
    reasons: str
    verdict: Literal["A", "B", "tie"]


class ArenaError(ValueError):
    pass


def spoken(conversation: Conversation) -> str:
    """The call as the caller heard it: one line per turn, tool calls left out. A turn in which
    the agent said nothing at all shows as silence."""
    lines: list[str] = []
    said: list[str] | None = None
    messages = conversation.messages[1:]
    if messages and messages[0].role == "user" and messages[0].content == OPENING_TURN:
        messages = messages[1:]

    def close() -> None:
        if said is not None:
            lines.append("Agent: " + (" ".join(said) if said else "(silence)"))

    for message in messages:
        if message.role == "user":
            close()
            said = None
            lines.append(f"Caller: {message.content or ''}")
        elif message.role == "assistant":
            said = [] if said is None else said
            if message.content and message.content.strip():
                said.append(message.content.strip())
    close()
    return "\n".join(lines)


def first_in_half(stems: list[str], seed: int) -> set[str]:
    """The half of the scenarios in which the first run's call is shown as Agent A."""
    ordered = sorted(stems, key=lambda stem: hashlib.sha256(f"{seed}:{stem}".encode()).hexdigest())
    return set(ordered[: len(ordered) // 2])


def build_request(
    call_a: Conversation, call_b: Conversation, cache_ttl: Literal["5m", "1h"] = "5m"
) -> Request:
    text = load_prompt(PROMPT).render(call_a=spoken(call_a), call_b=spoken(call_b))
    return Request(
        messages=[Message(role="user", content=text)],
        json_schema=SCHEMA,
        cache_ttl=cache_ttl,
    )


def read_answer(response: Response) -> ArenaVerdict:
    if response.finish_reason != "stop":
        raise ArenaError(f"the judge stopped early ({response.finish_reason})")
    try:
        return ArenaVerdict.model_validate(json.loads(response.text or ""))
    except (json.JSONDecodeError, ValidationError) as error:
        raise ArenaError(f"the judge's answer is not a verdict: {error}") from error
