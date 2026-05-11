from __future__ import annotations
import time
from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI
from pydantic import BaseModel, Field
from algorithm.src.mlops.logger import log_retrieval_response, log_user_feedback
from algorithm.src.inference.predictor import (
    DEFAULT_RERANKING_MODEL,
    DEFAULT_RETRIEVAL_MODEL,
    PredictionRequest,
    SmartDeveloperPredictor,
)


PRODUCT_RESULT_FIELDS = [
    "RID",
    "address",
    "base_site_address",
    "primary_zoning_code",
    "primary_zoning_class",
    "zoning_band",
    "lot_size_band",
    "lot_size_proxy_sqm",
    "constraint_severity_band",
    "station_distance_band",
    "distance_to_station_m",
    "within_800m_catchment",
    "heritage_flag",
    "flood_flag",
    "bushfire_flag",
    "top_strategy",
    "top_strategy_score",
    "strategy_score",
    "retrieval_similarity",
    "fusion_score",
    "dcn_prob",
    "dcn_rank_score",
    "explanation",
]


def _filter_product_response(response: dict[str, Any]) -> dict[str, Any]:
    filtered = dict(response)
    filtered["results"] = [
        {k: item.get(k) for k in PRODUCT_RESULT_FIELDS if k in item}
        for item in response.get("results", [])
    ]
    return filtered


class RetrieveSitesPayload(BaseModel):
    strategy: str
    query_text: str

    top_k: int = Field(default=5, ge=1, le=50)
    recall_k: int = Field(default=1000, ge=10, le=10000)

    with_explanations: bool = False

    retrieval_model: str = DEFAULT_RETRIEVAL_MODEL
    use_dcn_reranker: bool = True
    reranking_model: str = DEFAULT_RERANKING_MODEL

    alpha: float = 0.5
    beta: float = 0.5
    dedupe_by_address: bool = True

    locality: str | None = None
    address_contains: str | None = None

    # MLOps logging metadata
    user_id: str | None = None
    session_id: str | None = None
    log_request: bool = True
    debug: bool = False


class FeedbackPayload(BaseModel):
    request_id: str
    event_type: str

    rid: str | int | None = None
    rank_position: int | None = None
    event_value: dict[str, Any] | None = None
    user_note: str | None = None

    user_id: str | None = None
    session_id: str | None = None


class ServiceState:
    predictor: SmartDeveloperPredictor | None = None
    startup_latency_ms: float | None = None
    warmup_latency_ms: float | None = None
    is_ready: bool = False


state = ServiceState()


def _warmup_predictor(predictor: SmartDeveloperPredictor) -> float:
    warmup_request = PredictionRequest(
        strategy="low_rise_apartment",
        query_text=(
            "I want a site for low-rise apartment redevelopment near a train station, "
            "with high development zoning, a large site, and limited planning constraints."
        ),
        top_k=1,
        recall_k=1000,
        with_explanations=False,
        retrieval_model=DEFAULT_RETRIEVAL_MODEL,
        use_dcn_reranker=True,
        reranking_model=DEFAULT_RERANKING_MODEL,
        locality="WAITARA",
    )

    t0 = time.perf_counter()
    predictor.predict(warmup_request)
    return round((time.perf_counter() - t0) * 1000, 2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_t0 = time.perf_counter()

    predictor = SmartDeveloperPredictor()
    state.predictor = predictor

    # Force model/candidate/DCN loading before serving real requests.
    state.warmup_latency_ms = _warmup_predictor(predictor)
    state.startup_latency_ms = round((time.perf_counter() - startup_t0) * 1000, 2)
    state.is_ready = True

    yield

    state.is_ready = False
    state.predictor = None


app = FastAPI(
    title="Smart Developer Algorithm Service",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ready" if state.is_ready else "starting",
        "service_startup_latency_ms": state.startup_latency_ms,
        "model_load_and_warmup_latency_ms": state.warmup_latency_ms,
        "warm_request_expected": state.is_ready,
        "retrieval_model": DEFAULT_RETRIEVAL_MODEL,
        "reranking_model": DEFAULT_RERANKING_MODEL,
    }


@app.post("/retrieve-sites")
def retrieve_sites(payload: RetrieveSitesPayload) -> dict[str, Any]:
    if state.predictor is None or not state.is_ready:
        return {
            "status": "error",
            "message": "Predictor is not ready.",
        }

    request = PredictionRequest(
        strategy=payload.strategy,
        query_text=payload.query_text,
        top_k=payload.top_k,
        recall_k=payload.recall_k,
        with_explanations=payload.with_explanations,
        retrieval_model=payload.retrieval_model,
        use_dcn_reranker=payload.use_dcn_reranker,
        reranking_model=payload.reranking_model,
        alpha=payload.alpha,
        beta=payload.beta,
        dedupe_by_address=payload.dedupe_by_address,
        locality=payload.locality,
        address_contains=payload.address_contains,
    )

    response = state.predictor.predict(request)
    response["service"] = {
        "mode": "warm_predictor_singleton",
        "service_startup_latency_ms": state.startup_latency_ms,
        "model_load_and_warmup_latency_ms": state.warmup_latency_ms,
    }

    if payload.log_request:
        try:
            log_retrieval_response(
                response,
                user_id=payload.user_id,
                session_id=payload.session_id,
            )
            response["logging"] = {
                "enabled": True,
                "status": "logged",
            }
        except Exception as exc:
            response["logging"] = {
                "enabled": True,
                "status": "failed",
                "error": str(exc),
            }

    if not payload.debug:
        response = _filter_product_response(response)

    return response


@app.post("/feedback")
def feedback(payload: FeedbackPayload) -> dict[str, Any]:
    try:
        feedback_id = log_user_feedback(
            request_id=payload.request_id,
            rid=payload.rid,
            rank_position=payload.rank_position,
            event_type=payload.event_type,
            event_value={
                **(payload.event_value or {}),
                "user_id": payload.user_id,
                "session_id": payload.session_id,
            },
            user_note=payload.user_note,
        )

        return {
            "status": "logged",
            "feedback_id": feedback_id,
            "request_id": payload.request_id,
        }

    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
            "request_id": payload.request_id,
        }