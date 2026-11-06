"""Code checks of a scenario's key actions against the tool calls an agent made.

Only calls that ran count: a call the mock tools rejected for breaking the tool's schema did
nothing. A `call` action is met by a call to its tool whose arguments all match, made after a
matching call of each action it must follow; when an action it must follow was never met, its
order cannot be checked and is not held against it. A `never` action is met when its tool was
never called.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from mtva_bench.data.matchers import matches
from mtva_bench.data.schema import Agent, KeyAction, Matcher, Scenario
from mtva_bench.simulate.tools import argument_errors
from mtva_bench.simulate.types import ToolEvent

Status = Literal["met", "wrong_arguments", "out_of_order", "not_called", "called"]


@dataclass(frozen=True)
class Call:
    index: int
    turn: int
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Mismatch:
    name: str
    expected: str
    actual: Any


@dataclass
class CheckResult:
    action: KeyAction
    status: Status
    turn: int | None = None
    mismatches: list[Mismatch] = field(default_factory=list)
    rejected: int = 0


def executed_calls(agent: Agent, events: list[ToolEvent]) -> dict[str, list[Call]]:
    """Each tool's calls that ran, in order; `turn` is the agent turn counted from 1."""
    calls: dict[str, list[Call]] = {}
    for index, event in enumerate(events):
        try:
            arguments = json.loads(event.arguments or "{}")
        except json.JSONDecodeError:
            continue
        tool = agent.tool(event.name)
        if not isinstance(arguments, dict) or tool is None:
            continue
        if argument_errors(tool.parameters, arguments):
            continue
        calls.setdefault(event.name, []).append(Call(index, event.turn + 1, arguments))
    return calls


def describe(matcher: Matcher, scenario: Scenario) -> str:
    if matcher.equals is not None:
        return json.dumps(matcher.equals, ensure_ascii=False)
    if matcher.ref is not None:
        return json.dumps(scenario.resolve(matcher.ref), ensure_ascii=False)
    if matcher.any_of is not None:
        return "one of " + json.dumps(matcher.any_of, ensure_ascii=False)
    if matcher.range is not None:
        return f"between {matcher.range[0]:g} and {matcher.range[1]:g}"
    if matcher.date is not None:
        return f"the date {matcher.date}"
    if matcher.phone is not None:
        return f"the phone number {matcher.phone}"
    return "any value"


def mismatches(action: KeyAction, call: Call, scenario: Scenario) -> list[Mismatch]:
    return [
        Mismatch(name, describe(matcher, scenario), call.arguments.get(name))
        for name, matcher in action.arguments.items()
        if not matches(matcher, call.arguments.get(name), scenario)
    ]


def check_actions(agent: Agent, scenario: Scenario, events: list[ToolEvent]) -> list[CheckResult]:
    calls = executed_calls(agent, events)
    rejected = {
        name: sum(event.name == name for event in events) - len(calls.get(name, []))
        for name in {event.name for event in events}
    }
    actions = scenario.evaluation.key_actions
    first_match: dict[str, Call] = {}
    for action in actions:
        matching = [
            call for call in calls.get(action.tool, []) if not mismatches(action, call, scenario)
        ]
        if action.expect == "call" and matching:
            first_match[action.id] = matching[0]

    results = []
    for action in actions:
        made = calls.get(action.tool, [])
        result = CheckResult(action, "met", rejected=rejected.get(action.tool, 0))
        if action.expect == "never":
            if made:
                result.status, result.turn = "called", made[0].turn
        elif not made:
            result.status = "not_called"
        elif action.id not in first_match:
            closest = min(made, key=lambda call: len(mismatches(action, call, scenario)))
            result.status, result.turn = "wrong_arguments", closest.turn
            result.mismatches = mismatches(action, closest, scenario)
        else:
            earliest = max(
                (first_match[other].index for other in action.after if other in first_match),
                default=-1,
            )
            in_order = [
                call
                for call in made
                if call.index > earliest and not mismatches(action, call, scenario)
            ]
            chosen = in_order[0] if in_order else first_match[action.id]
            result.status = "met" if in_order else "out_of_order"
            result.turn = chosen.turn
        results.append(result)
    return results
