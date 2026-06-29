from __future__ import annotations
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.app.routers import (
    ai_property_summary,
    property_image,
    recommendation_feedback,
)

DEFAULT_ALLOWED_ORIGINS = [
    "https://smart-developer-frontend.vercel.app",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5173",
]


def _load_allowed_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "").strip()
    if not raw:
        return DEFAULT_ALLOWED_ORIGINS
    return [origin.strip() for origin in raw.split(",") if origin.strip()]
from backend.app.schemas import (
    FeedbackRequest,
    ReportRequest,
    SearchRequest,
    ExportReportRequest
)
from backend.app.services.algorithm_client import (
    AlgorithmServiceError,
    create_report_job,
    get_report_job,
    export_report,
    health,
    log_feedback,
    retrieve_sites,
)
from backend.app.services.recommendation_feedback import (
    attach_recommendation_feedback_prompt,
)

app = FastAPI(
    title="Smart Developer Backend Gateway",
    version="0.1.0",
    description="Product-facing backend gateway for the Smart Developer demo platform.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_load_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(recommendation_feedback.router)
app.include_router(ai_property_summary.router)
app.include_router(property_image.router)


def _normalise_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).upper().strip()


def _annotate_locality_metadata(response: dict, locality: str | None) -> dict:
    """
    Attach locality-filter metadata for observability.

    The algorithm service already hard-filters candidates by locality in
    `_build_candidate_pool` and returns an empty result set when nothing matches,
    so the gateway no longer re-filters. We still annotate metadata so the
    frontend can detect when a locality query produced zero matches.
    """
    if not locality or not locality.strip():
        return response

    locality_norm = _normalise_text(locality)
    results = response.get("results", []) if isinstance(response.get("results"), list) else []

    metadata = response.setdefault("metadata", {})
    metadata["frontend_location_filter_requested"] = True
    metadata["frontend_location_query"] = locality_norm
    metadata["frontend_location_match_count"] = len(results)

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
        response = _annotate_locality_metadata(response, payload.locality)
        response = attach_recommendation_feedback_prompt(response)
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


@app.post("/api/export-report")
async def export_report_endpoint(payload: ExportReportRequest):
    return await export_report(payload.model_dump())
