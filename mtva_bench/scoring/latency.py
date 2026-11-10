"""Time to first token of the model under test.

A call's first request reads the whole system prompt cold; every later one reuses what the
provider cached, as it would on a live line. So the opening request is reported on its own and
the warm ones are the headline. Tool rounds count as warm requests: the caller waits through
them too.
"""

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from mtva_bench.scoring.stats import percentile
from mtva_bench.simulate.types import Conversation, ModelCall


class Latency(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ttft_warm_p50: int
    ttft_warm_p90: int
    ttft_cold_p50: int
    warm_requests: int
    cold_requests: int


def is_cold(turn: int, round_: int) -> bool:
    return turn == 0 and round_ == 0


def ttft(conversations: Iterable[Conversation]) -> Latency | None:
    """None when no request streamed a first token."""
    warm: list[float] = []
    cold: list[float] = []
    for conversation in conversations:
        for call in conversation.model_calls:
            if call.timing.first_token_ms is not None:
                (cold if is_cold(call.turn, call.round) else warm).append(
                    call.timing.first_token_ms
                )
    if not warm or not cold:
        return None
    return Latency(
        ttft_warm_p50=round(percentile(warm, 50)),
        ttft_warm_p90=round(percentile(warm, 90)),
        ttft_cold_p50=round(percentile(cold, 50)),
        warm_requests=len(warm),
        cold_requests=len(cold),
    )


class Speed(BaseModel):
    """A dedicated latency test, per agent turn: time from the caller's last word to the first
    token, and to the first complete spoken sentence, which is when text-to-speech can start;
    tool rounds before the agent speaks count towards it, the mock tools' own time does not.
    The opening turn is cold and reported apart. Tokens per second is the generation rate once
    a reply has started; it stays with the run and is left out of what is published."""

    model_config = ConfigDict(extra="forbid")

    ttft_warm_p50: int
    ttft_warm_p90: int
    ttft_cold_p50: int
    first_sentence_warm_p50: int
    first_sentence_warm_p90: int
    first_sentence_cold_p50: int
    tokens_per_second_p50: float | None
    warm_turns: int
    cold_turns: int


def turn_times(conversation: Conversation) -> list[tuple[int, float | None, float | None]]:
    """(turn, time to first token, time to first sentence) of each agent turn."""
    calls: dict[int, list[ModelCall]] = {}
    for call in conversation.model_calls:
        calls.setdefault(call.turn, []).append(call)
    times = []
    for turn, rounds in sorted(calls.items()):
        first_token = rounds[0].timing.first_token_ms
        waited = 0.0
        first_sentence = None
        for call in sorted(rounds, key=lambda c: c.round):
            if call.timing.first_sentence_ms is not None:
                first_sentence = waited + call.timing.first_sentence_ms
                break
            waited += call.timing.total_ms
        times.append((turn, first_token, first_sentence))
    return times


def speed(conversations: Iterable[Conversation]) -> Speed | None:
    """None when no warm and cold turn both streamed a first token and a sentence."""
    times = {"ttft": ([], []), "sentence": ([], [])}
    rates: list[float] = []
    for conversation in conversations:
        for turn, first_token, first_sentence in turn_times(conversation):
            index = 0 if turn else 1
            if first_token is not None:
                times["ttft"][index].append(first_token)
            if first_sentence is not None:
                times["sentence"][index].append(first_sentence)
        for call in conversation.model_calls:
            generating = call.timing.total_ms - (call.timing.first_token_ms or call.timing.total_ms)
            if call.spoke and generating > 0 and call.usage.output_tokens > 1:
                rates.append((call.usage.output_tokens - 1) / generating * 1000)
    (warm, cold), (warm_sentence, cold_sentence) = times["ttft"], times["sentence"]
    if not (warm and cold and warm_sentence and cold_sentence and rates):
        return None
    return Speed(
        ttft_warm_p50=round(percentile(warm, 50)),
        ttft_warm_p90=round(percentile(warm, 90)),
        ttft_cold_p50=round(percentile(cold, 50)),
        first_sentence_warm_p50=round(percentile(warm_sentence, 50)),
        first_sentence_warm_p90=round(percentile(warm_sentence, 90)),
        first_sentence_cold_p50=round(percentile(cold_sentence, 50)),
        tokens_per_second_p50=round(percentile(rates, 50), 1),
        warm_turns=len(warm),
        cold_turns=len(cold),
    )
