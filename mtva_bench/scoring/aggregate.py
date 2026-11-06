"""From verdicts to capability scores, the MTVA Score, slices and head-to-head comparisons.

A capability's 1 to 5 score maps linearly onto 0 to 100. The MTVA Score is the weighted mean of
the capability scores. Every number carries a clustered bootstrap interval, and an interval of a
difference comes from resampling the paired differences, not from two separate intervals.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from mtva_bench.capabilities.conduct import ConductVerdict, scores
from mtva_bench.data.schema import Agent, Scenario
from mtva_bench.scoring.stats import clustered_bootstrap, mean
from mtva_bench.simulate.types import Conversation

# The cuts a summary publishes. A call can also be cut by agent, which only the run folder shows.
SLICES = ("language", "industry", "difficulty", "direction")
ORDER = {"difficulty": ("easy", "medium", "hard")}
GOAL_REACHED = frozenset({"reached", "alternative"})
DIGITS = 1


def to_100(score: int) -> float:
    return (score - 1) * 25


@dataclass(frozen=True)
class CallScore:
    """One judged call. `raw` holds each capability's 1 to 5 score."""

    stem: str
    agent_id: str
    slices: dict[str, str]
    raw: dict[str, int]
    outcome: str
    ending: str

    def capability(self, name: str) -> float:
        return to_100(self.raw[name])

    def mtva(self, weights: Mapping[str, float]) -> float:
        total = sum(weights.values())
        return sum(weight * self.capability(name) for name, weight in weights.items()) / total

    @property
    def goal_reached(self) -> bool:
        return self.outcome in GOAL_REACHED

    @property
    def clean(self) -> bool:
        """Every capability at 5: the caller got what they came for and nothing went wrong."""
        return all(score == 5 for score in self.raw.values())


def score_call(
    stem: str,
    verdict: ConductVerdict,
    scenario: Scenario,
    agent: Agent,
    conversation: Conversation,
) -> CallScore:
    return CallScore(
        stem=stem,
        agent_id=agent.agent_id,
        slices={
            "language": scenario.language,
            "industry": agent.industry,
            "difficulty": scenario.difficulty,
            "direction": agent.direction,
            "agent": agent.agent_id,
        },
        raw=scores(verdict, scenario),
        outcome=verdict.task_completion.outcome,
        ending=conversation.end_reason,
    )


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Interval(Model):
    value: float
    ci: tuple[float, float]


class Slice(Model):
    calls: int
    mtva_score: float
    capabilities: dict[str, float]


class Scores(Model):
    """`goal_reached` and `clean_calls` are shares of calls, in percent."""

    calls: int
    mtva_score: Interval
    capabilities: dict[str, Interval]
    goal_reached: Interval
    clean_calls: Interval
    outcomes: dict[str, int]
    endings: dict[str, int]
    slices: dict[str, dict[str, Slice]]


def columns(calls: Sequence[CallScore], weights: Mapping[str, float]) -> dict[str, list[float]]:
    return {
        "mtva_score": [call.mtva(weights) for call in calls],
        **{name: [call.capability(name) for call in calls] for name in weights},
        "goal_reached": [100.0 * call.goal_reached for call in calls],
        "clean_calls": [100.0 * call.clean for call in calls],
    }


def interval(values: Sequence[float], ci: tuple[float, float]) -> Interval:
    return Interval(
        value=round(mean(values), DIGITS), ci=(round(ci[0], DIGITS), round(ci[1], DIGITS))
    )


def aggregate(calls: Sequence[CallScore], weights: Mapping[str, float]) -> Scores:
    if not calls:
        raise ValueError("no judged calls to score")
    data = columns(calls, weights)
    cis = clustered_bootstrap([call.agent_id for call in calls], data)
    return Scores(
        calls=len(calls),
        mtva_score=interval(data["mtva_score"], cis["mtva_score"]),
        capabilities={name: interval(data[name], cis[name]) for name in weights},
        goal_reached=interval(data["goal_reached"], cis["goal_reached"]),
        clean_calls=interval(data["clean_calls"], cis["clean_calls"]),
        outcomes=dict(sorted(Counter(call.outcome for call in calls).items())),
        endings=dict(sorted(Counter(call.ending for call in calls).items())),
        slices={field: slices(calls, weights, field) for field in SLICES},
    )


def slices(
    calls: Sequence[CallScore], weights: Mapping[str, float], field: str
) -> dict[str, Slice]:
    groups: dict[str, list[CallScore]] = {}
    for call in calls:
        groups.setdefault(call.slices[field], []).append(call)
    return {
        value: Slice(
            calls=len(group),
            mtva_score=round(mean([call.mtva(weights) for call in group]), DIGITS),
            capabilities={
                name: round(mean([call.capability(name) for call in group]), DIGITS)
                for name in weights
            },
        )
        for value, group in sorted(groups.items(), key=lambda item: order(field, item[0]))
    }


def order(field: str, value: str) -> tuple[int, str]:
    known = ORDER.get(field, ())
    return (known.index(value) if value in known else len(known), value)


class HeadToHead(Model):
    """B against A on the calls both judged: each delta is B minus A, and `wins` counts the
    calls where B's MTVA Score is higher."""

    calls: int
    mtva_score: Interval
    capabilities: dict[str, Interval]
    wins: int
    ties: int
    losses: int


def head_to_head(
    a: Sequence[CallScore], b: Sequence[CallScore], weights: Mapping[str, float]
) -> HeadToHead:
    by_stem = {call.stem: call for call in a}
    pairs = [(by_stem[call.stem], call) for call in b if call.stem in by_stem]
    if not pairs:
        raise ValueError("the two runs share no judged calls")
    first = columns([pair[0] for pair in pairs], weights)
    second = columns([pair[1] for pair in pairs], weights)
    deltas = {
        name: [y - x for x, y in zip(first[name], second[name], strict=True)]
        for name in ["mtva_score", *weights]
    }
    cis = clustered_bootstrap([pair[0].agent_id for pair in pairs], deltas)
    signs = Counter((d > 0) - (d < 0) for d in deltas["mtva_score"])
    return HeadToHead(
        calls=len(pairs),
        mtva_score=interval(deltas["mtva_score"], cis["mtva_score"]),
        capabilities={name: interval(deltas[name], cis[name]) for name in weights},
        wins=signs[1],
        ties=signs[0],
        losses=signs[-1],
    )
