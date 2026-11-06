import json
from typing import Any

from mtva_bench.llm.http import HttpClient, LLMError
from mtva_bench.llm.types import BatchStatus

FINAL_STATES = frozenset({"completed", "failed", "expired", "cancelled"})


async def submit(http: HttpClient, path: str, bodies: dict[str, dict[str, Any]]) -> str:
    url = http.base_path + path
    lines = "\n".join(
        json.dumps({"custom_id": key, "method": "POST", "url": url, "body": body})
        for key, body in bodies.items()
    )
    upload = await http.upload("/files", "batch.jsonl", lines.encode(), {"purpose": "batch"})
    batch = await http.post(
        "/batches",
        {"input_file_id": upload["id"], "endpoint": url, "completion_window": "24h"},
    )
    return batch["id"]


async def status(http: HttpClient, path: str, batch_id: str) -> BatchStatus:
    data = await http.get(f"/batches/{batch_id}")
    return BatchStatus(
        id=batch_id,
        ended=data["status"] in FINAL_STATES,
        counts=data.get("request_counts") or {},
    )


async def results(
    http: HttpClient, path: str, batch_id: str
) -> dict[str, dict[str, Any] | LLMError]:
    data = await http.get(f"/batches/{batch_id}")
    collected: dict[str, dict[str, Any] | LLMError] = {}
    for file_id in (data.get("output_file_id"), data.get("error_file_id")):
        if not file_id:
            continue
        for line in (await http.get_text(f"/files/{file_id}/content")).splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            response = item.get("response") or {}
            if response.get("status_code") == 200:
                collected[item["custom_id"]] = response["body"]
            else:
                collected[item["custom_id"]] = LLMError(
                    f"batch request failed: {item.get('error') or response.get('body')}",
                    status=response.get("status_code"),
                    retryable=True,
                )
    return collected
