"""The phone line between caller and agent: what the agent receives of what the caller said.

`asr: light` drops or doubles one word of the first caller turn, chosen by seed and never a
word of a caller fact; `asr: authored` applies the scenario's events on their turns;
`vad: natural_splits` delivers an utterance split at `<pause>` as separate messages.
"""

import random
import re
from collections.abc import Iterable

from mtva_bench.data.schema import Channel
from mtva_bench.simulate.types import ChannelEdit

PAUSE = "<pause>"
MAX_PAUSES = 2
WORD = re.compile(r"[^\W\d_]{4,}")


def deliver(
    utterance: str, channel: Channel, turn: int, seed: int, protected: Iterable[str]
) -> tuple[list[str], list[ChannelEdit]]:
    """The fragments the agent receives, and the edits applied on the way."""
    heard, edits = utterance, []
    if channel.asr == "light" and turn == 0:
        guarded = " ".join(protected)
        candidates = [word for word in WORD.findall(heard) if word not in guarded]
        if candidates:
            source = random.Random(seed).choice(candidates)
            kind = "drop" if seed % 2 == 0 else "double"
            target = "" if kind == "drop" else f"{source} {source}"
            heard = heard.replace(source, target, 1)
            edits.append(ChannelEdit(type=kind, source=source, target=target, applied=True))
    for event in (event for event in channel.events if event.turn == turn):
        target = {
            "drop": "",
            "double": f"{event.source} {event.source}",
            "substitute": event.target or "",
        }[event.type]
        applied = event.source in heard
        if applied:
            heard = heard.replace(event.source, target, 1)
        edits.append(
            ChannelEdit(type=event.type, source=event.source, target=target, applied=applied)
        )
    if channel.vad != "natural_splits":
        return [" ".join(heard.replace(PAUSE, " ").split())], edits
    fragments = [" ".join(fragment.split()) for fragment in heard.split(PAUSE)]
    return [fragment for fragment in fragments if fragment], edits
