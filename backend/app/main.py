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
        return await retrieve_sites(payload.model_dump())
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