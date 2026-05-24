from __future__ import annotations
import os
from typing import Any
import httpx
from dotenv import load_dotenv
from fastapi import HTTPException
from fastapi.responses import Response

load_dotenv()

ALGORITHM_SERVICE_URL = os.getenv("ALGORITHM_SERVICE_URL", "http://localhost:8001")


class AlgorithmServiceError(RuntimeError):
    pass


async def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{ALGORITHM_SERVICE_URL}{path}"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        raise AlgorithmServiceError(
            f"Algorithm service returned {exc.response.status_code}: {detail}"
        ) from exc
    except httpx.HTTPError as exc:
        raise AlgorithmServiceError(f"Failed to call algorithm service: {exc}") from exc


async def _get(path: str) -> dict[str, Any]:
    url = f"{ALGORITHM_SERVICE_URL}{path}"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        raise AlgorithmServiceError(
            f"Algorithm service returned {exc.response.status_code}: {detail}"
        ) from exc
    except httpx.HTTPError as exc:
        raise AlgorithmServiceError(f"Failed to call algorithm service: {exc}") from exc


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

    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(url, json=payload)

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