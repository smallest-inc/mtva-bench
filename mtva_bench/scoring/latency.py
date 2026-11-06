"""Time to first token of the model under test.

A call's first request reads the whole system prompt cold; every later one reuses what the
provider cached, as it would on a live line. So the opening request is reported on its own and
the warm ones are the headline. Tool rounds count as warm requests: the caller waits through
them too.
"""

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from mtva_bench.scoring.stats import percentile
from mtva_bench.simulate.types import Conversation


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
