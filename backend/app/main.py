from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app.schemas import FeedbackRequest, ReportRequest, SearchRequest
from backend.app.services.algorithm_client import (
    AlgorithmServiceError,
    create_report_job,
    get_report_job,
    health,
    log_feedback,
    retrieve_sites,
)

app = FastAPI(
    title="Smart Developer Backend Gateway",
    version="0.1.0",
    description="Product-facing backend gateway for the Smart Developer demo platform.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _normalise_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).upper().strip()


def _apply_strict_locality_guard(response: dict, locality: str | None) -> dict:
    """
    Demo/product safety guard.

    If a user explicitly enters a locality, do not show unrelated global results.
    This prevents the UI from looking random when the algorithm service falls
    back to global candidates.
    """
    if not locality or not locality.strip():
        return response

    locality_norm = _normalise_text(locality)

    results = response.get("results", [])
    if not isinstance(results, list):
        return response

    matched_results = []
    for item in results:
        address_text = " ".join(
            [
                _normalise_text(item.get("base_site_address")),
                _normalise_text(item.get("address")),
            ]
        )

        if locality_norm in address_text:
            matched_results.append(item)

    metadata = response.setdefault("metadata", {})
    metadata["frontend_location_filter_requested"] = True
    metadata["frontend_location_query"] = locality_norm
    metadata["frontend_location_original_result_count"] = len(results)
    metadata["frontend_location_match_count"] = len(matched_results)

    if len(matched_results) == 0:
        metadata["frontend_location_guard_applied"] = True
        metadata["frontend_location_warning"] = (
            f"No exact address/locality matches found for '{locality_norm}'. "
            "Showing no results instead of unrelated global recommendations."
        )
        response["results"] = []
        metadata["result_count"] = 0
        return response

    metadata["frontend_location_guard_applied"] = True
    metadata["frontend_location_warning"] = None
    response["results"] = matched_results
    metadata["result_count"] = len(matched_results)
    return response


@app.get("/health")
async def gateway_health():
    try:
        algorithm_health = await health()
    except AlgorithmServiceError as exc:
        return {
            "status": "degraded",
            "gateway": "ready",
            "algorithm_service": "unavailable",
            "error": str(exc),
        }

    return {
        "status": "ready",
        "gateway": "ready",
        "algorithm_service": algorithm_health,
    }


@app.post("/api/search")
async def search_sites(payload: SearchRequest):
    try:
        response = await retrieve_sites(payload.model_dump())
        response = _apply_strict_locality_guard(response, payload.locality)
        return response
    except AlgorithmServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/feedback")
async def feedback(payload: FeedbackRequest):
    try:
        return await log_feedback(payload.model_dump())
    except AlgorithmServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/reports")
async def reports(payload: ReportRequest):
    try:
        return await create_report_job(payload.model_dump())
    except AlgorithmServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/reports/{report_id}")
async def report_status(report_id: str):
    try:
        return await get_report_job(report_id)
    except AlgorithmServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc