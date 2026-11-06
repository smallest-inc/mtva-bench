"""Win, tie and loss rates of one run against another, from its point of view, with clustered
bootstrap intervals by agent."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from mtva_bench.scoring.aggregate import DIGITS, Interval, Model, interval
from mtva_bench.scoring.stats import clustered_bootstrap, mean

Side = Literal["a", "b"]


@dataclass(frozen=True)
class Outcome:
    """One judged scenario: who won, and whose call the judge read first."""

    stem: str
    agent_id: str
    language: str
    winner: Literal["a", "b", "tie"]
    first: Side


class Rates(Model):
    """Shares of calls in percent, from A's point of view."""

    calls: int
    wins: float
    ties: float
    losses: float


class ArenaScores(Model):
    """`first_shown_won` is the share of decisive verdicts won by the call the judge read
    first; near 50 means the judge did not lean toward either position."""

    calls: int
    wins: Interval
    ties: Interval
    losses: Interval
    by_language: dict[str, Rates]
    first_shown_won: float | None


def columns(outcomes: Sequence[Outcome]) -> dict[str, list[float]]:
    return {
        name: [100.0 * (outcome.winner == side) for outcome in outcomes]
        for name, side in (("wins", "a"), ("ties", "tie"), ("losses", "b"))
    }


def rates(outcomes: Sequence[Outcome]) -> Rates:
    data = columns(outcomes)
    return Rates(
        calls=len(outcomes), **{name: round(mean(values), DIGITS) for name, values in data.items()}
    )


def arena_scores(outcomes: Sequence[Outcome]) -> ArenaScores:
    if not outcomes:
        raise ValueError("no judged scenarios to score")
    data = columns(outcomes)
    cis = clustered_bootstrap([outcome.agent_id for outcome in outcomes], data)
    languages: dict[str, list[Outcome]] = {}
    for outcome in outcomes:
        languages.setdefault(outcome.language, []).append(outcome)
    decisive = [outcome for outcome in outcomes if outcome.winner != "tie"]
    return ArenaScores(
        calls=len(outcomes),
        **{name: interval(values, cis[name]) for name, values in data.items()},
        by_language={language: rates(group) for language, group in sorted(languages.items())},
        first_shown_won=round(100 * mean([float(o.winner == o.first) for o in decisive]), DIGITS)
        if decisive
        else None,
    )
