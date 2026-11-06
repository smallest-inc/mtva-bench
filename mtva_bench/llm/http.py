import asyncio
import json
import random
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx

RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})

T = TypeVar("T")


class LLMError(Exception):
    def __init__(
        self,
        message: str,
        status: int | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.retry_after = retry_after


@dataclass(frozen=True)
class SSEEvent:
    event: str | None
    data: dict[str, Any]


class HttpClient:
    def __init__(
        self,
        base_url: str,
        headers: dict[str, str],
        timeout_s: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._client = httpx.AsyncClient(
            base_url=base_url, headers=headers, timeout=timeout_s, transport=transport
        )
        self.base_path = httpx.URL(base_url).path.rstrip("/")

    async def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        return (await self._send("POST", path, json=body)).json()

    async def upload(
        self, path: str, filename: str, content: bytes, fields: dict[str, str]
    ) -> dict[str, Any]:
        return (
            await self._send("POST", path, files={"file": (filename, content)}, data=fields)
        ).json()

    async def get(self, path: str) -> dict[str, Any]:
        return (await self._send("GET", path)).json()

    async def get_text(self, path: str) -> str:
        return (await self._send("GET", path)).text

    async def events(self, path: str, body: dict[str, Any]) -> AsyncIterator[SSEEvent]:
        try:
            async with self._client.stream("POST", path, json=body) as response:
                if response.is_error:
                    await response.aread()
                    raise status_error(response)
                event: str | None = None
                data: list[str] = []
                async for line in response.aiter_lines():
                    if line.startswith("event:"):
                        event = line[6:].strip()
                    elif line.startswith("data:"):
                        data.append(line[5:].lstrip())
                    elif not line:
                        if data and (parsed := parse_event(event, data)):
                            yield parsed
                        event, data = None, []
                if data and (parsed := parse_event(event, data)):
                    yield parsed
        except httpx.TransportError as error:
            raise LLMError(f"{type(error).__name__}: {error}", retryable=True) from error
        except json.JSONDecodeError as error:
            # An event cut off mid-JSON is a broken connection, not a bad reply.
            raise LLMError(f"malformed stream event: {error}", retryable=True) from error

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _send(self, method: str, path: str, **content: Any) -> httpx.Response:
        try:
            response = await self._client.request(method, path, **content)
        except httpx.TransportError as error:
            raise LLMError(f"{type(error).__name__}: {error}", retryable=True) from error
        if response.is_error:
            raise status_error(response)
        return response


def parse_event(event: str | None, data: list[str]) -> SSEEvent | None:
    payload = "\n".join(data)
    if payload == "[DONE]":
        return None
    return SSEEvent(event, json.loads(payload))


CONTEXT_LIMIT = re.compile(
    r"maximum context length|context_length_exceeded|context window|prompt is too long"
    r"|input tokens exceeds|exceeds the maximum number of tokens",
    re.IGNORECASE,
)


def context_limit(error: LLMError) -> bool:
    """Whether the provider rejected the request because the prompt no longer fits the
    model's context window."""
    return error.status == 400 and bool(CONTEXT_LIMIT.search(str(error)))


def status_error(response: httpx.Response) -> LLMError:
    retry_after = response.headers.get("retry-after")
    return LLMError(
        f"HTTP {response.status_code} from {response.request.url}: {response.text[:2000]}",
        status=response.status_code,
        retryable=response.status_code in RETRYABLE_STATUS,
        retry_after=float(retry_after) if retry_after and retry_after.isdigit() else None,
    )


async def with_retries(
    attempt: Callable[[], Awaitable[T]],
    max_attempts: int = 6,
    base_delay_s: float = 1.0,
    max_delay_s: float = 60.0,
) -> T:
    for number in range(1, max_attempts + 1):
        try:
            return await attempt()
        except LLMError as error:
            if not error.retryable or number == max_attempts:
                raise
            backoff = min(max_delay_s, base_delay_s * 2 ** (number - 1))
            await asyncio.sleep(error.retry_after or backoff * random.uniform(0.5, 1.0))
    raise AssertionError("unreachable")
