"""The system prompt the model under test receives for one scenario.

The agent's own prompt is fenced and keeps its `{{name}}` placeholders; the bundle's voice
guide follows it with its own values filled in place, then a Variables section lists the
value of each agent-prompt placeholder. Everything before that section is the same for every
scenario of an agent, so providers can cache it across calls, as production serves it.
Without the voice guide, a short call context stands in for it, since the date, time, gender
and languages are part of the task.
"""

import re
from pathlib import Path

from mtva_bench.data.schema import Agent, DataError, Scenario
from mtva_bench.prompts import load_prompt

PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
AGENT_PROMPT_END = "[end of agent prompt]"
# Filled from the agent into every scenario (the bundle format), so the same across its calls.
AGENT_VARIABLES = frozenset({"default_language", "supported_languages_text", "agent_gender"})
HERE = Path(__file__).parent


def placeholders(text: str) -> set[str]:
    return set(PLACEHOLDER.findall(text))


def compose(agent: Agent, voice_guide: str | None) -> str:
    """The system prompt with every `{{name}}` still open, the Variables section included."""
    guide = voice_guide if voice_guide is not None else (HERE / "call_context.md").read_text()
    names = sorted(placeholders(agent.prompt) - placeholders(guide))
    variables = load_prompt(HERE / "variables.md").render(
        variables="\n".join(f"- {name}: {{{{{name}}}}}" for name in names)
    )
    return load_prompt(HERE / "system_prompt.md").render(
        agent_prompt=agent.prompt.strip(), guide=guide.strip(), variables=variables.strip()
    )


def render(agent: Agent, scenario: Scenario, voice_guide: str | None) -> tuple[str, int]:
    """The system prompt, and the length of its start that every call of the agent shares:
    everything before the line that holds the first value set per call."""
    text = compose(agent, voice_guide)
    missing = placeholders(text) - set(scenario.variables)
    if missing:
        raise DataError(f"{scenario.scenario_id}: no value for {sorted(missing)}")
    fenced, end, rest = text.partition(AGENT_PROMPT_END)
    head = fenced + end
    filled, shared, last = [], None, 0
    for match in PLACEHOLDER.finditer(rest):
        filled.append(rest[last : match.start()])
        if shared is None and match.group(1) not in AGENT_VARIABLES:
            before = head + "".join(filled)
            shared = before.rfind("\n") + 1
        filled.append(scenario.variables[match.group(1)])
        last = match.end()
    prompt = (head + "".join(filled) + rest[last:]).strip()
    return prompt, len(prompt) if shared is None else shared


def system_prompt(agent: Agent, scenario: Scenario, voice_guide: str | None) -> str:
    return render(agent, scenario, voice_guide)[0]
