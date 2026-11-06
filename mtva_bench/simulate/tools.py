"""Deterministic mock tools: the scenario's rules decide every reply.

A call is answered by the first rule of its tool whose matchers all match the call's
arguments and whose required rules have answered an earlier call. That rule's first matching
calls get its `first_replies` in order, every later one its `reply`; a rule is never used up,
so a corrected retry is answered like a first try. A call no rule matches gets `otherwise`.
A call whose arguments break the tool's schema (a missing required parameter, a wrong JSON type,
a value outside an enum) is not executed: it gets the error a live voice-agent runtime returns,
naming what is wrong, so the model can correct it.
"""

import json
from collections import Counter
from typing import Any

from mtva_bench.data.matchers import matches
from mtva_bench.data.schema import Agent, Rule, Scenario

JSON_TYPES = {
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: isinstance(value, int | float) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "array": lambda value: isinstance(value, list),
    "object": lambda value: isinstance(value, dict),
}


def argument_errors(parameters: dict[str, Any], arguments: dict[str, Any]) -> str | None:
    """What is wrong with `arguments` for a tool with `parameters`, worded as a live voice-agent
    runtime words them; None when they pass."""
    properties = parameters.get("properties", {})
    errors = [
        f"Missing required parameter: '{name}'"
        for name in parameters.get("required", [])
        if name not in arguments
    ]
    for name, value in arguments.items():
        spec = properties.get(name)
        if spec is None:
            continue
        expected = spec.get("type")
        check = JSON_TYPES.get(expected) if isinstance(expected, str) else None
        if value is not None and check and not check(value):
            errors.append(
                f"Parameter '{name}' expected type '{expected}', got '{type(value).__name__}'"
            )
            continue
        enum = spec.get("enum")
        if enum and value not in enum:
            errors.append(f"Parameter '{name}' value '{value}' not in allowed values: {enum}")
    return "; ".join(errors) if errors else None


class MockTools:
    def __init__(self, agent: Agent, scenario: Scenario):
        self.agent = agent
        self.scenario = scenario
        self.answered: Counter[str] = Counter()

    def is_terminal(self, name: str) -> bool:
        return name in self.agent.terminal_tools

    def call(self, name: str, arguments: str) -> tuple[dict[str, Any], str | None]:
        """The reply to one call, and the id of the rule that answered it (None: otherwise
        or an invalid call)."""
        behaviour = self.scenario.tools.get(name)
        if behaviour is None:
            return {"error": f"there is no tool named {name}"}, None
        try:
            parsed = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return {"error": "the arguments are not valid JSON"}, None
        if not isinstance(parsed, dict):
            return {"error": "the arguments must be a JSON object"}, None
        tool = self.agent.tool(name)
        invalid = argument_errors(tool.parameters, parsed) if tool is not None else None
        if invalid:
            return {"error": f"Invalid arguments: {invalid}"}, None
        rule = next((rule for rule in behaviour.rules if self.applies(rule, parsed)), None)
        if rule is None:
            return behaviour.otherwise, None
        count = self.answered[rule.id]
        self.answered[rule.id] += 1
        if count < len(rule.first_replies):
            return rule.first_replies[count], rule.id
        return rule.reply, rule.id

    def applies(self, rule: Rule, arguments: dict[str, Any]) -> bool:
        return all(self.answered[other] for other in rule.requires) and all(
            matches(matcher, arguments.get(name), self.scenario)
            for name, matcher in rule.when.items()
        )
