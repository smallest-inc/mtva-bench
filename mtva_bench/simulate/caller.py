import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from mtva_bench.data.schema import Agent, Scenario
from mtva_bench.llm.client import LLM
from mtva_bench.llm.types import Message, Request, Usage
from mtva_bench.prompts import load_prompt
from mtva_bench.simulate.agent import cached

MAX_ATTEMPTS = 3
SILENCE = "(silence)"
REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "utterance": {"type": "string"},
        "beats": {"type": "array", "items": {"type": "string"}},
        "done": {"type": "boolean"},
    },
    "required": ["utterance", "beats", "done"],
    "additionalProperties": False,
}
INTERNAL_WORDS = ("caller.facts", "done_when", "utterance", "agenda beat")


class CallerError(RuntimeError):
    pass


class CallerReply(BaseModel):
    model_config = ConfigDict(extra="forbid")

    utterance: str
    beats: list[str]
    done: bool


class Caller:
    """The simulated person on the line. It hears only what the agent says, never tool calls."""

    def __init__(self, llm: LLM, agent: Agent, scenario: Scenario):
        self.llm = llm
        self.order = {beat.id: index for index, beat in enumerate(scenario.caller.agenda)}
        self.advanced: set[str] = set()
        self.usage: list[Usage] = []
        variables = scenario.variables
        system = load_prompt(Path(__file__).with_name("caller.md")).render(
            direction=agent.direction,
            profile=scenario.caller.profile.strip(),
            facts=json.dumps(scenario.caller.facts, indent=2, ensure_ascii=False),
            unknowns=scenario.caller.unknowns,
            agenda=scenario.caller.agenda,
            done_when=scenario.caller.done_when,
            language=scenario.language,
            natural_splits=scenario.channel.vad == "natural_splits",
            current_day=variables.get("current_day", ""),
            current_date=variables.get("current_date", ""),
            current_time=variables.get("current_time", ""),
        )
        self.history = [Message(role="system", content=system)]

    def hear(self, agent_text: str) -> None:
        self.history.append(Message(role="user", content=agent_text.strip() or SILENCE))

    async def reply(self, seed: int) -> CallerReply:
        problems: list[str] = []
        for attempt in range(MAX_ATTEMPTS):
            request = Request(
                messages=cached(self.history), json_schema=REPLY_SCHEMA, seed=seed + attempt
            )
            response = await self.llm.complete(request)
            self.usage.append(response.usage)
            try:
                reply = CallerReply.model_validate_json(response.text or "")
            except ValidationError as error:
                problems.append(f"invalid reply: {error.errors()[0]['msg']}")
                continue
            problem = self.problem(reply)
            if problem:
                problems.append(problem)
                continue
            # A beat the caller already covered, or one that does not exist, is only a wrong label:
            # the utterance still stands, as when the agent asks something a second time.
            new = {beat for beat in reply.beats if beat in self.order} - self.advanced
            reply.beats = sorted(new, key=self.order.__getitem__)
            self.advanced.update(reply.beats)
            self.history.append(Message(role="assistant", content=reply.model_dump_json()))
            return reply
        raise CallerError(f"no usable caller reply in {MAX_ATTEMPTS} attempts: {problems}")

    def problem(self, reply: CallerReply) -> str | None:
        if not reply.utterance.strip():
            return "empty utterance"
        leaked = [word for word in INTERNAL_WORDS if word in reply.utterance.casefold()]
        if leaked:
            return f"utterance exposes {leaked}"
        return None
