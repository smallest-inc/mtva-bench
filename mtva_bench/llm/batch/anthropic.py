import json
from typing import Any

from mtva_bench.llm.http import HttpClient, LLMError
from mtva_bench.llm.types import BatchStatus


async def submit(http: HttpClient, path: str, bodies: dict[str, dict[str, Any]]) -> str:
    requests = [{"custom_id": key, "params": body} for key, body in bodies.items()]
    return (await http.post(f"{path}/batches", {"requests": requests}))["id"]


async def status(http: HttpClient, path: str, batch_id: str) -> BatchStatus:
    data = await http.get(f"{path}/batches/{batch_id}")
    return BatchStatus(
        id=batch_id, ended=data["processing_status"] == "ended", counts=data["request_counts"]
    )


async def results(
    http: HttpClient, path: str, batch_id: str
) -> dict[str, dict[str, Any] | LLMError]:
    data = await http.get(f"{path}/batches/{batch_id}")
    collected: dict[str, dict[str, Any] | LLMError] = {}
    for line in (await http.get_text(data["results_url"])).splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        result = item["result"]
        if result["type"] == "succeeded":
            collected[item["custom_id"]] = result["message"]
        else:
            collected[item["custom_id"]] = LLMError(
                f"batch request {result['type']}: {result.get('error')}", retryable=True
            )
    return collected
