"""The conduct judge: one request per call judges task completion, tool use, instruction
following and faithfulness.

The four read the same evidence, so one judge reads the call once and files each problem under
exactly one capability, which neither counts it twice nor lets it fall between two judges. The
judge writes evidence before verdicts and judges task completion last.

The judge gives verdicts and severities, and `scores` turns them into 1 to 5 in code: two runs
of the judge agree far more often on verdicts than on scores it picks itself.

Its answer is checked in code: every focus rule, key action and must-hear item answered once,
every turn a real agent turn, a severity on exactly the focus rules that were not followed. A
quote that cannot be found where it claims to come from is kept and marked, not dropped.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from mtva_bench.capabilities.conduct.checks import CheckResult, check_actions, describe
from mtva_bench.capabilities.conduct.transcript import ending, transcript
from mtva_bench.data.matchers import normalize
from mtva_bench.data.schema import Agent, Scenario
from mtva_bench.llm.client import LLM
from mtva_bench.llm.http import LLMError
from mtva_bench.llm.types import Message, Request, Response, Usage
from mtva_bench.prompts import load_prompt
from mtva_bench.simulate.types import Conversation, Record

PROMPT = Path(__file__).parent / "prompt.md"
CACHE_MARK = "<<<cache>>>\n"
ATTEMPTS = 2
REFUSED = "the judge refused to judge this call"
# Shown to the judge for a model that cannot speak and call a tool in one message, so it does not
# blame the agent for the silence that leaves around a tool call.
SILENT_TOOLS = (
    "[This model cannot speak and call a tool in the same message, so it says nothing while it "
    "calls a tool. That silence is a limit of the model, not a choice of the agent: do not count "
    "a tool call made without words around it against the agent, under any capability. Its "
    "goodbye or handoff line before a tool that ends the call was given with that tool and is "
    "shown above as speech.]"
)

Severity = Literal["critical", "major", "minor"]
CacheTtl = Literal["5m", "1h"]
SEVERITY_OF_IMPORTANCE: dict[int, Severity] = {3: "critical", 2: "major", 1: "minor"}
ONE_LOWER: dict[Severity, Severity] = {"critical": "major", "major": "minor", "minor": "minor"}


class FocusVerdict(Record):
    id: str
    evidence: str
    turns: list[int]
    verdict: Literal["followed", "partly", "broken", "not_reached"]
    severity: Severity | None


class ActionVerdict(Record):
    id: str
    evidence: str
    verdict: Literal["met", "partly", "not_met"]


class Finding(Record):
    capability: Literal["instruction_following", "tool_use", "faithfulness"]
    turn: int
    quote: str
    what_happened: str
    severity: Severity
    quote_found: bool | None = None


class Heard(Record):
    item: int
    heard: Literal["yes", "partly", "no"]
    turn: int | None


class TaskCompletion(Record):
    caller_heard: list[Heard]
    evidence: str
    outcome: Literal["reached", "alternative", "partial", "not_reached", "wrong_result"]


class ConductVerdict(Record):
    focus_rules: list[FocusVerdict]
    key_actions: list[ActionVerdict]
    findings: list[Finding]
    task_completion: TaskCompletion
    simulation_problem: str | None
    summary: str


def obj(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def enum(*values: Any) -> dict[str, Any]:
    return {"enum": list(values)}


def nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


TEXT = {"type": "string"}
TURN = {"type": "integer"}
SEVERITY = enum("critical", "major", "minor")
# One schema for every call: a schema per scenario would be compiled again for each.
SCHEMA = obj(
    {
        "focus_rules": {
            "type": "array",
            "items": obj(
                {
                    "id": TEXT,
                    "evidence": TEXT,
                    "turns": {"type": "array", "items": TURN},
                    "verdict": enum("followed", "partly", "broken", "not_reached"),
                    "severity": nullable(SEVERITY),
                }
            ),
        },
        "key_actions": {
            "type": "array",
            "items": obj(
                {"id": TEXT, "evidence": TEXT, "verdict": enum("met", "partly", "not_met")}
            ),
        },
        "findings": {
            "type": "array",
            "items": obj(
                {
                    "capability": enum("instruction_following", "tool_use", "faithfulness"),
                    "turn": TURN,
                    "quote": TEXT,
                    "what_happened": TEXT,
                    "severity": SEVERITY,
                }
            ),
        },
        "task_completion": obj(
            {
                "caller_heard": {
                    "type": "array",
                    "items": obj(
                        {
                            "item": {"type": "integer"},
                            "heard": enum("yes", "partly", "no"),
                            "turn": nullable(TURN),
                        }
                    ),
                },
                "evidence": TEXT,
                "outcome": enum("reached", "alternative", "partial", "not_reached", "wrong_result"),
            }
        ),
        "simulation_problem": nullable(TEXT),
        "summary": TEXT,
    }
)


def severity_score(severities: list[Severity]) -> int:
    """5 with no problems, 4 with minor ones only, 3 with one major or three minor, 2 with one
    critical or two major, 1 with two critical."""
    critical, major, minor = (severities.count(level) for level in ("critical", "major", "minor"))
    if critical >= 2:
        return 1
    if critical or major >= 2:
        return 2
    if major or minor >= 3:
        return 3
    return 4 if minor else 5


def task_score(task: TaskCompletion) -> int:
    if task.outcome in ("reached", "alternative"):
        return 5 if all(heard.heard == "yes" for heard in task.caller_heard) else 4
    return {"partial": 3, "not_reached": 2, "wrong_result": 1}[task.outcome]


def scores(verdict: ConductVerdict, scenario: Scenario) -> dict[str, int]:
    """Each capability's 1 to 5 score. A key action that is not met counts at the severity of
    its importance, one that is partly met one level lower."""
    importance = {action.id: action.importance for action in scenario.evaluation.key_actions}
    problems: dict[str, list[Severity]] = {
        "instruction_following": [
            focus.severity for focus in verdict.focus_rules if focus.severity is not None
        ],
        "tool_use": [],
        "faithfulness": [],
    }
    for action in verdict.key_actions:
        level = SEVERITY_OF_IMPORTANCE[importance[action.id]]
        if action.verdict == "not_met":
            problems["tool_use"].append(level)
        elif action.verdict == "partly":
            problems["tool_use"].append(ONE_LOWER[level])
    for finding in verdict.findings:
        problems[finding.capability].append(finding.severity)
    return {
        "task_completion": task_score(verdict.task_completion),
        **{capability: severity_score(found) for capability, found in problems.items()},
    }


class VerdictError(ValueError):
    pass


def check_text(result: CheckResult) -> str:
    tool = f"`{result.action.tool}`"
    rejected = (
        f" {result.rejected} call(s) to it were rejected for invalid arguments and did not run."
        if result.rejected
        else ""
    )
    if result.status == "met":
        if result.action.expect == "never":
            return f"{tool} was never called."
        return f"a matching call ran at Agent [{result.turn}]."
    if result.status == "called":
        return f"{tool} was called at Agent [{result.turn}]."
    if result.status == "not_called":
        return f"no call to {tool} ran.{rejected}"
    if result.status == "out_of_order":
        return (
            f"a matching call ran at Agent [{result.turn}], but not after the actions it must "
            "follow."
        )
    differences = "; ".join(
        f"`{m.name}` was {json.dumps(m.actual, ensure_ascii=False)} where {m.expected} was expected"
        for m in result.mismatches
    )
    return f"the closest call, at Agent [{result.turn}], differs: {differences}.{rejected}"


def request_messages(
    agent: Agent,
    scenario: Scenario,
    conversation: Conversation,
    checks: list[CheckResult],
    text_with_tool_calls: bool = True,
) -> list[Message]:
    """The rendered prompt.md as one user turn, cut at its cache mark. The agent's system
    prompt is the one saved with the conversation, exactly as the model saw it."""
    evaluation = scenario.evaluation
    actions = [
        {
            "id": result.action.id,
            "expect": result.action.expect,
            "tool": result.action.tool,
            "importance": result.action.importance,
            "focus": result.action.focus,
            "note": result.action.note,
            "arguments": ", ".join(
                f"`{name}` = {describe(matcher, scenario)}"
                for name, matcher in result.action.arguments.items()
            ),
            "after": ", ".join(f"`{other}`" for other in result.action.after),
            "result": check_text(result),
        }
        for result in checks
    ]
    system = conversation.messages[0]
    prompt = system.content or ""
    shared = system.cache_prefix or 0
    tools = [
        {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
        for tool in agent.tools
    ]
    text = load_prompt(PROMPT).render(
        tools=json.dumps(tools, ensure_ascii=False, indent=1),
        terminal_tools=agent.terminal_tools,
        shared_prompt=prompt[:shared],
        call_prompt=prompt[shared:],
        transcript=transcript(conversation)[0]
        + ("" if text_with_tool_calls else f"\n\n{SILENT_TOOLS}"),
        ending=ending(conversation),
        focus_rules=evaluation.focus_rules,
        key_actions=actions,
        goal=evaluation.goal,
    )
    parts = text.split(CACHE_MARK)
    if len(parts) != 2:
        raise ValueError(f"{PROMPT} must render with exactly one cache mark, got {len(parts) - 1}")
    shared, call = parts
    return [
        Message(role="user", content=shared, cache_breakpoint=True),
        Message(role="user", content=call),
    ]


def spoken(conversation: Conversation) -> str:
    return " ".join(turn.text for turn in conversation.agent_turns)


def ids(items: list[Any]) -> list[str]:
    return [item.id for item in items]


def validate(
    raw: str, agent: Agent, scenario: Scenario, conversation: Conversation, turns: int
) -> ConductVerdict:
    try:
        verdict = ConductVerdict.model_validate_json(raw)
    except ValidationError as error:
        raise VerdictError(f"the answer does not fit the schema: {error}") from error
    evaluation = scenario.evaluation
    problems = []
    for kind, answered, expected in (
        ("focus rules", ids(verdict.focus_rules), ids(evaluation.focus_rules)),
        ("key actions", ids(verdict.key_actions), ids(evaluation.key_actions)),
        (
            "must-hear items",
            [h.item for h in verdict.task_completion.caller_heard],
            list(range(1, len(evaluation.goal.caller_must_hear) + 1)),
        ),
    ):
        if sorted(answered) != sorted(expected):
            problems.append(f"{kind} answered {answered}, expected {expected}")
    for focus in verdict.focus_rules:
        if (focus.verdict in ("partly", "broken")) != (focus.severity is not None):
            problems.append(
                f"focus rule {focus.id} is {focus.verdict}: a severity goes with partly or "
                "broken and only with them"
            )
    # Something the agent failed to do is placed where it should have happened, which is the
    # turn after its last when it ended the call first; what the caller heard was said in a turn.
    missed = [turn for focus in verdict.focus_rules for turn in focus.turns]
    missed += [finding.turn for finding in verdict.findings]
    heard = [h.turn for h in verdict.task_completion.caller_heard if h.turn is not None]
    bad = sorted({turn for turn in missed if not 1 <= turn <= turns + 1})
    bad += sorted({turn for turn in heard if not 1 <= turn <= turns} - set(bad))
    if bad:
        problems.append(f"turns {bad} are not agent turns (1 to {turns})")
    if problems:
        raise VerdictError("; ".join(problems))

    prompt = " ".join(
        normalize(text)
        for text in [conversation.messages[0].content or "", *(t.description for t in agent.tools)]
    )
    sources = {
        "instruction_following": prompt,
        "tool_use": prompt + " " + normalize(transcript(conversation)[0]),
        "faithfulness": normalize(spoken(conversation)),
    }
    for finding in verdict.findings:
        finding.quote_found = normalize(finding.quote) in sources[finding.capability]
    return verdict


@dataclass
class JudgeResult:
    """`model` is the model whose answer became the verdict; `refused` is set when the judge
    declined the call, and `fallback` when the fallback judge was asked instead."""

    verdict: ConductVerdict | None = None
    error: str | None = None
    usage: list[Usage] = field(default_factory=list)
    model: str | None = None
    refused: bool = False
    fallback: bool = False


def build_request(
    agent: Agent,
    scenario: Scenario,
    conversation: Conversation,
    cache_ttl: CacheTtl = "5m",
    text_with_tool_calls: bool = True,
) -> Request:
    checks = check_actions(agent, scenario, conversation.tool_events)
    return Request(
        messages=request_messages(agent, scenario, conversation, checks, text_with_tool_calls),
        json_schema=SCHEMA,
        cache_ttl=cache_ttl,
    )


def read_answer(
    response: Response, agent: Agent, scenario: Scenario, conversation: Conversation
) -> ConductVerdict:
    if response.finish_reason == "refusal":
        raise VerdictError(REFUSED)
    if response.finish_reason != "stop":
        raise VerdictError(f"the judge stopped early ({response.finish_reason})")
    return validate(response.text or "", agent, scenario, conversation, transcript(conversation)[1])


async def ask(
    llm: LLM,
    request: Request,
    attempts: int,
    result: JudgeResult,
    agent: Agent,
    scenario: Scenario,
    conversation: Conversation,
) -> None:
    for _ in range(attempts):
        try:
            response = await llm.stream(request)
        except LLMError as error:
            result.error = f"{type(error).__name__}: {error}"
            continue
        result.usage.append(response.usage)
        try:
            result.verdict = read_answer(response, agent, scenario, conversation)
        except VerdictError as error:
            result.error = str(error)
            # The same request to the same model is refused again.
            if response.finish_reason == "refusal":
                result.refused = True
                return
            continue
        result.error, result.model = None, response.model
        return


async def judge(
    llm: LLM,
    agent: Agent,
    scenario: Scenario,
    conversation: Conversation,
    attempts: int = ATTEMPTS,
    fallback: LLM | None = None,
    refused: bool = False,
    text_with_tool_calls: bool = True,
) -> JudgeResult:
    """Judge one call. A failed request or an answer that does not check out is tried again
    up to `attempts` in all; after that the error is returned, not raised. A call the judge
    refuses, or `refused` already, goes to `fallback` when there is one."""
    request = build_request(
        agent, scenario, conversation, text_with_tool_calls=text_with_tool_calls
    )
    result = JudgeResult(refused=refused)
    if not refused:
        await ask(llm, request, attempts, result, agent, scenario, conversation)
    if result.verdict is None and result.refused and fallback is not None:
        result.fallback = True
        await ask(fallback, request, ATTEMPTS, result, agent, scenario, conversation)
    return result
