"""Wire formats: encode a Request into a provider body, decode a body or stream into a Response."""

from collections.abc import Callable
from typing import Any, Protocol

from mtva_bench.llm.http import SSEEvent
from mtva_bench.llm.types import Delta, Output, Request


class Decoder(Protocol):
    def feed(self, event: SSEEvent) -> Delta | None: ...

    def output(self) -> Output: ...


class WireApi(Protocol):
    @property
    def PATH(self) -> str: ...

    @property
    def StreamDecoder(self) -> Callable[[], Decoder]: ...

    def headers(self, api_key: str | None) -> dict[str, str]: ...

    def encode(self, request: Request, model: str, stream: bool) -> dict[str, Any]: ...

    def decode(self, body: dict[str, Any]) -> Output: ...
