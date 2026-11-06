import asyncio
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import httpx

from mtva_bench.llm.endpoints import Api, Endpoint
from mtva_bench.llm.http import HttpClient, with_retries
from mtva_bench.llm.protocols import WireApi, chat, messages, responses
from mtva_bench.llm.timing import StreamClock, elapsed_ms
from mtva_bench.llm.types import Request, Response, Timing

PROTOCOLS: dict[Api, WireApi] = {"chat": chat, "responses": responses, "messages": messages}


class LLM(Protocol):
    async def complete(self, request: Request) -> Response: ...

    async def stream(self, request: Request) -> Response: ...


class LLMClient:
    def __init__(
        self,
        endpoint: Endpoint,
        concurrency: int = 8,
        env_file: Path = Path(".env"),
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.endpoint = endpoint
        self.protocol = PROTOCOLS[endpoint.wire_api]
        self.http = HttpClient(
            endpoint.url,
            self.protocol.headers(endpoint.api_key(env_file)),
            endpoint.timeout_s,
            transport,
        )
        self._slots = asyncio.Semaphore(concurrency)

    def body(self, request: Request, stream: bool) -> dict[str, Any]:
        body = merge(
            self.endpoint.params, self.protocol.encode(request, self.endpoint.model, stream)
        )
        for name in self.endpoint.unsupported:
            body.pop(name, None)
        return body

    async def complete(self, request: Request) -> Response:
        body = self.body(request, stream=False)

        async def attempt() -> Response:
            async with self._slots:
                start = perf_counter()
                data = await self.http.post(self.protocol.PATH, body)
                output = self.protocol.decode(data)
                return Response(**dict(output), timing=Timing(total_ms=elapsed_ms(start)))

        return await with_retries(attempt)

    async def stream(self, request: Request) -> Response:
        body = self.body(request, stream=True)

        async def attempt() -> Response:
            async with self._slots:
                decoder = self.protocol.StreamDecoder()
                clock = StreamClock()
                async for event in self.http.events(self.protocol.PATH, body):
                    clock.observe(decoder.feed(event))
                return Response(**dict(decoder.output()), timing=clock.timing())

        return await with_retries(attempt)

    async def aclose(self) -> None:
        await self.http.aclose()

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = value
    return merged
