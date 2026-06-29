from __future__ import annotations
import asyncio
import logging
import os
import random
from typing import Any
import httpx
from dotenv import load_dotenv
from fastapi import HTTPException
from fastapi.responses import Response

load_dotenv()

logger = logging.getLogger(__name__)

ALGORITHM_SERVICE_URL = os.getenv("ALGORITHM_SERVICE_URL", "http://localhost:8001")
POST_TIMEOUT_S = float(os.getenv("ALGORITHM_POST_TIMEOUT_S", "240"))
GET_TIMEOUT_S = float(os.getenv("ALGORITHM_GET_TIMEOUT_S", "60"))
EXPORT_TIMEOUT_S = float(os.getenv("ALGORITHM_EXPORT_TIMEOUT_S", "180"))
MAX_RETRIES = int(os.getenv("ALGORITHM_MAX_RETRIES", "2"))
RETRYABLE_STATUS = {502, 503, 504}


class AlgorithmServiceError(RuntimeError):
    pass


def _is_retryable_status(status_code: int) -> bool:
    return status_code in RETRYABLE_STATUS


async def _backoff_sleep(attempt: int) -> None:
    # Exponential backoff with jitter: 0.5s, 1s, 2s, ...
    delay = min(2 ** attempt * 0.5, 4.0) + random.uniform(0, 0.25)
    await asyncio.sleep(delay)


async def _request_with_retry(
    method: str,
    url: str,
    *,
    timeout: float,
    json: dict[str, Any] | None = None,
) -> httpx.Response:
    last_exc: Exception | None = None
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await client.request(method, url, json=json)
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                if attempt >= MAX_RETRIES:
                    raise AlgorithmServiceError(
                        f"Failed to call algorithm service after {attempt + 1} attempts: {exc}"
                    ) from exc
                logger.warning(
                    "Algorithm service %s %s transport error (attempt %d/%d): %s",
                    method, url, attempt + 1, MAX_RETRIES + 1, exc,
                )
                await _backoff_sleep(attempt)
                continue

            if _is_retryable_status(response.status_code) and attempt < MAX_RETRIES:
                logger.warning(
                    "Algorithm service %s %s returned %d (attempt %d/%d)",
                    method, url, response.status_code, attempt + 1, MAX_RETRIES + 1,
                )
                await _backoff_sleep(attempt)
                continue

            return response

    # Defensive — loop should always either return or raise.
    raise AlgorithmServiceError(
        f"Failed to call algorithm service: {last_exc}"
    )


async def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{ALGORITHM_SERVICE_URL}{path}"
    response = await _request_with_retry("POST", url, timeout=POST_TIMEOUT_S, json=payload)
    if response.status_code >= 400:
        raise AlgorithmServiceError(
            f"Algorithm service returned {response.status_code}: {response.text}"
        )
    return response.json()


async def _get(path: str) -> dict[str, Any]:
    url = f"{ALGORITHM_SERVICE_URL}{path}"
    response = await _request_with_retry("GET", url, timeout=GET_TIMEOUT_S)
    if response.status_code >= 400:
        raise AlgorithmServiceError(
            f"Algorithm service returned {response.status_code}: {response.text}"
        )
    return response.json()


async def health() -> dict[str, Any]:
    return await _get("/health")


async def retrieve_sites(payload: dict[str, Any]) -> dict[str, Any]:
    return await _post("/retrieve-sites", payload)


async def log_feedback(payload: dict[str, Any]) -> dict[str, Any]:
    return await _post("/feedback", payload)


async def create_report_job(payload: dict[str, Any]) -> dict[str, Any]:
    return await _post("/report-jobs", payload)


async def get_report_job(report_id: str) -> dict[str, Any]:
    return await _get(f"/report-jobs/{report_id}")


async def export_report(payload: dict) -> Response:
    """
    Proxy stateless report export to the algorithm service.

    The algorithm service returns either:
    - application/pdf
    - text/markdown
    """
    url = f"{ALGORITHM_SERVICE_URL}/export-report"

    response = await _request_with_retry(
        "POST", url, timeout=EXPORT_TIMEOUT_S, json=payload
    )

    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail=response.text,
        )

    content_type = response.headers.get("content-type", "application/octet-stream")
    content_disposition = response.headers.get(
        "content-disposition",
        'attachment; filename="smart_developer_report.pdf"',
    )

    return Response(
        content=response.content,
        media_type=content_type,
        headers={
            "Content-Disposition": content_disposition,
        },
    )