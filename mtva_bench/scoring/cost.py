"""What a call costs: the tokens the model under test used, at its provider's prices.

Every request resends the system prompt and the call so far, so most input is read from the
provider's cache. Input is billed at one of three rates: read from the cache, written to it, or
neither.
"""

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from mtva_bench.simulate.types import Conversation


class Prices(BaseModel):
    """US dollars per million tokens. `cache_write` is None where a provider bills input it
    caches like any other input."""

    model_config = ConfigDict(extra="forbid")

    input: float
    cached_input: float
    output: float
    cache_write: float | None = None


def call_cost(conversation: Conversation, prices: Prices) -> float:
    """The call's cost in US dollars, over every request to the model under test."""
    write = prices.input if prices.cache_write is None else prices.cache_write
    total = 0.0
    for call in conversation.model_calls:
        usage = call.usage
        uncached = usage.input_tokens - usage.cache_read_tokens - usage.cache_write_tokens
        total += (
            uncached * prices.input
            + usage.cache_read_tokens * prices.cached_input
            + usage.cache_write_tokens * write
            + usage.output_tokens * prices.output
        )
    return total / 1_000_000


def cost_per_call(conversations: Iterable[Conversation], prices: Prices) -> float | None:
    """The mean cost of a call in US dollars; None without calls."""
    costs = [call_cost(conversation, prices) for conversation in conversations]
    return sum(costs) / len(costs) if costs else None
