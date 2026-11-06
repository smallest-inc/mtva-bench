from collections.abc import Callable, Iterable

from mtva_bench.llm.types import Output, Request, Response, Timing

Responder = Callable[[Request], Output | str]


class FakeLLM:
    def __init__(self, responder: Responder):
        self.responder = responder
        self.requests: list[Request] = []

    async def complete(self, request: Request) -> Response:
        self.requests.append(request)
        output = self.responder(request)
        if isinstance(output, str):
            output = Output(text=output)
        spoken = 0.0 if output.text else None
        timing = Timing(
            first_token_ms=0.0,
            first_speech_token_ms=spoken,
            first_sentence_ms=spoken,
            total_ms=0.0,
        )
        return Response(**dict(output), timing=timing)

    async def stream(self, request: Request) -> Response:
        return await self.complete(request)


def scripted(outputs: Iterable[Output | str]) -> Responder:
    queue = list(outputs)

    def respond(_: Request) -> Output | str:
        if not queue:
            raise AssertionError("scripted responder ran out of outputs")
        return queue.pop(0)

    return respond
