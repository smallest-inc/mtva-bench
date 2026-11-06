import re
from datetime import date
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    model_serializer,
    model_validator,
)

SCHEMA_VERSION = "3.0"
REF_ROOTS = ("caller.facts.", "world.", "variables.")
LOCAL_ID = re.compile(r"^[a-z][a-z0-9_]*$")


class DataError(ValueError):
    pass


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def local_id(value: str) -> str:
    if not LOCAL_ID.fullmatch(value):
        raise ValueError(f"'{value}' must be lowercase letters, digits and underscores")
    return value


LocalId = Annotated[str, AfterValidator(local_id)]


class Tool(Contract):
    name: str
    description: str
    parameters: dict[str, Any]

    @property
    def properties(self) -> dict[str, Any]:
        return self.parameters.get("properties", {})


class Agent(Contract):
    schema_version: Literal["3.0"]
    agent_id: str
    name: str
    industry: str
    use_case: str
    call_type: str
    direction: Literal["inbound", "outbound"]
    language: str
    supported_languages: list[str]
    gender: Literal["Male", "Female"]
    prompt: str
    tools: list[Tool]
    terminal_tools: list[str]

    @model_validator(mode="after")
    def validate_agent(self) -> "Agent":
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique")
        unknown = set(self.terminal_tools) - set(names)
        if unknown:
            raise ValueError(f"terminal tools are not declared tools: {sorted(unknown)}")
        if self.language not in self.supported_languages:
            raise ValueError(f"{self.language} is not in supported_languages")
        return self

    def tool(self, name: str) -> Tool | None:
        return next((tool for tool in self.tools if tool.name == name), None)


class Matcher(Contract):
    equals: str | int | float | bool | None = None
    ref: str | None = None
    any_of: list[str | int | float | bool] | None = None
    range: tuple[float, float] | None = None
    date: str | None = None
    phone: str | None = None
    present: Literal[True] | None = None

    @model_validator(mode="after")
    def validate_operation(self) -> "Matcher":
        used = [name for name in type(self).model_fields if getattr(self, name) is not None]
        if len(used) != 1:
            raise ValueError(f"a matcher uses exactly one operation, got {used or 'none'}")
        if self.range is not None and self.range[0] > self.range[1]:
            raise ValueError("range minimum exceeds maximum")
        if self.date is not None:
            date.fromisoformat(self.date)
        if self.any_of == []:
            raise ValueError("any_of needs at least one value")
        return self

    @model_serializer(mode="wrap")
    def serialize_operation(self, handler: Any) -> dict[str, Any]:
        return {name: value for name, value in handler(self).items() if value is not None}


class Rule(Contract):
    id: LocalId
    when: dict[str, Matcher] = Field(default_factory=dict)
    requires: list[str] = Field(default_factory=list)
    first_replies: list[dict[str, Any]] = Field(default_factory=list)
    reply: dict[str, Any]


class ToolBehaviour(Contract):
    rules: list[Rule] = Field(default_factory=list)
    otherwise: dict[str, Any]


class Beat(Contract):
    id: LocalId
    when: str = Field(min_length=1)
    does: str = Field(min_length=1)


class Caller(Contract):
    profile: str = Field(min_length=1)
    facts: dict[str, Any]
    unknowns: list[str] = Field(default_factory=list)
    agenda: list[Beat] = Field(min_length=3, max_length=10)
    done_when: str = Field(min_length=1)


class ChannelEvent(Contract):
    type: Literal["drop", "double", "substitute"]
    turn: int = Field(ge=0)
    source: str = Field(min_length=1)
    target: str | None = None

    @model_validator(mode="after")
    def validate_target(self) -> "ChannelEvent":
        if (self.type == "substitute") != (self.target is not None):
            raise ValueError("target is required for substitute and only for substitute")
        return self


class Channel(Contract):
    asr: Literal["clean", "light", "authored"] = "clean"
    vad: Literal["clean", "natural_splits"] = "clean"
    events: list[ChannelEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_events(self) -> "Channel":
        if (self.asr == "authored") != bool(self.events):
            raise ValueError("events are required for asr 'authored' and only for it")
        return self


class Goal(Contract):
    outcome: str = Field(min_length=1)
    alternatives: list[str] = Field(default_factory=list)
    caller_must_hear: list[str] = Field(default_factory=list)


class KeyAction(Contract):
    id: LocalId
    tool: str
    expect: Literal["call", "never"]
    importance: int = Field(ge=1, le=3)
    arguments: dict[str, Matcher] = Field(default_factory=dict)
    after: list[str] = Field(default_factory=list)
    focus: str | None = None
    note: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_action(self) -> "KeyAction":
        if self.expect == "never" and (self.arguments or self.after):
            raise ValueError("a 'never' action takes no arguments or ordering")
        if (self.expect == "never" or self.after) and not self.focus:
            raise ValueError("a 'never' action or an ordering must cite a focus rule")
        return self


class FocusRule(Contract):
    id: LocalId
    quote: str = Field(min_length=1)
    check: str = Field(min_length=1)


class Evaluation(Contract):
    goal: Goal
    key_actions: list[KeyAction] = Field(default_factory=list, max_length=5)
    focus_rules: list[FocusRule] = Field(min_length=1, max_length=3)


class Scenario(Contract):
    schema_version: Literal["3.0"]
    scenario_id: str
    agent_id: str
    language: str
    difficulty: Literal["easy", "medium", "hard"]
    summary: str = Field(min_length=1)
    variables: dict[str, str]
    caller: Caller
    channel: Channel = Field(default_factory=Channel)
    world: dict[str, Any]
    tools: dict[str, ToolBehaviour]
    evaluation: Evaluation

    def resolve(self, reference: str) -> Any:
        """The value at a `caller.facts.`, `world.` or `variables.` path; KeyError if absent."""
        root = next((root for root in REF_ROOTS if reference.startswith(root)), None)
        if root is None:
            raise KeyError(reference)
        value: Any = {
            "caller.facts.": self.caller.facts,
            "world.": self.world,
            "variables.": self.variables,
        }[root]
        try:
            for part in reference[len(root) :].split("."):
                value = value[int(part)] if isinstance(value, list) else value[part]
        except (KeyError, IndexError, ValueError, TypeError) as error:
            raise KeyError(reference) from error
        return value
