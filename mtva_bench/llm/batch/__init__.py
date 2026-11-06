"""Batch backends: submit, poll and collect provider batch jobs."""

import re
from types import ModuleType

from mtva_bench.llm.batch import anthropic, openai
from mtva_bench.llm.client import LLMClient
from mtva_bench.llm.http import LLMError, with_retries
from mtva_bench.llm.types import BatchStatus, Request, Response, Timing

BACKENDS: dict[str, ModuleType] = {"anthropic": anthropic, "openai": openai}
REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class Batcher:
    def __init__(self, client: LLMClient):
        if client.endpoint.provider not in BACKENDS:
            raise ValueError(f"provider '{client.endpoint.provider}' has no batch api")
        self.client = client
        self.backend = BACKENDS[client.endpoint.provider]
        self.path = client.protocol.PATH

    async def submit(self, requests: dict[str, Request]) -> str:
        invalid = [key for key in requests if not REQUEST_ID.fullmatch(key)]
        if invalid:
            raise ValueError(f"batch request ids must match {REQUEST_ID.pattern}: {invalid[:3]}")
        bodies = {key: self.client.body(request, stream=False) for key, request in requests.items()}
        return await self.backend.submit(self.client.http, self.path, bodies)

    async def status(self, batch_id: str) -> BatchStatus:
        return await with_retries(
            lambda: self.backend.status(self.client.http, self.path, batch_id)
        )

    async def results(self, batch_id: str) -> dict[str, Response | LLMError]:
        raw = await with_retries(
            lambda: self.backend.results(self.client.http, self.path, batch_id)
        )
        return {key: self.response(body) for key, body in raw.items()}

    def response(self, body: dict | LLMError) -> Response | LLMError:
        if isinstance(body, LLMError):
            return body
        try:
            output = self.client.protocol.decode(body)
        except LLMError as error:
            return error
        return Response(**dict(output), timing=Timing(total_ms=0.0))
