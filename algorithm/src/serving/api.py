from __future__ import annotations
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Literal
from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field
import tempfile
from pathlib import Path
import pandas as pd

from algorithm.src.explanation.report import ReportConfig, build_site_report
from algorithm.src.explanation.report_export import export_markdown_report_to_pdf
from algorithm.src.explanation.template_generator import add_template_explanations

try:
    from algorithm.src.mlops.db import is_database_enabled
    from algorithm.src.mlops.logger import log_retrieval_response, log_user_feedback
    from algorithm.src.mlops.report_jobs import generate_report_from_request_id, get_report_job
except Exception as exc:
    MLOPS_IMPORT_ERROR = str(exc)
    MLOPS_AVAILABLE = False

    def is_database_enabled() -> bool:
        return False

    def log_retrieval_response(*args, **kwargs):
        raise RuntimeError(f"MLOps logging unavailable: {MLOPS_IMPORT_ERROR}")

    def log_user_feedback(*args, **kwargs):
        raise RuntimeError(f"MLOps logging unavailable: {MLOPS_IMPORT_ERROR}")

    def generate_report_from_request_id(*args, **kwargs):
        raise RuntimeError(f"Report jobs unavailable: {MLOPS_IMPORT_ERROR}")

    def get_report_job(*args, **kwargs):
        raise RuntimeError(f"Report jobs unavailable: {MLOPS_IMPORT_ERROR}")
else:
    MLOPS_IMPORT_ERROR = None
    MLOPS_AVAILABLE = True

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
    "latitude",
    "longitude",
    "geometry_type",
    "geocode_source",
    "geocode_confidence",

    "agent_opportunity_score",
    "agent_rank_position",

    # Geographical factors
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

    # Government Policy factors
    "policy_upside_score",
    "policy_signal_band",
    "policy_matched_rules",
    "policy_matched_policies",
    "policy_matched_policy_names",
    "policy_evidence",
    "policy_evidence_count",
    "policy_explanation",

    # Value/Cost factors
    "locality",
    "locality_median_sale_price",
    "locality_sales_count",
    "locality_price_confidence",

    "ml_estimated_market_value",
    "ml_value_lower_bound",
    "ml_value_upper_bound",
    "ml_value_error_pct",
    "ml_value_confidence",
    "ml_value_model",

    "estimated_acquisition_cost",
    "estimated_acquisition_cost_source",
    "gross_floor_area_proxy_sqm",
    "base_construction_cost",
    "estimated_development_cost",
    "estimated_soft_cost",
    "estimated_contingency",
    "estimated_total_project_cost",
    "cost_band",
    "cost_risk_score",
    "cost_efficiency_score",
    "value_potential_score",
    "value_potential_band",
    "cost_value_explanation",

    "predicted_market_growth_3m",
    "market_trend_multiplier",
    "market_trend_score",
    "market_trend_band",
    "market_trend_source",
    "market_trend_model",
    "trend_adjusted_ml_market_value",
    "market_trend_raw_prediction",
    "market_trend_scaled_prediction",
    "market_trend_was_clipped",

    "construction_cost_trend_quarter",
    "predicted_construction_cost_growth_qoq",
    "construction_cost_escalation_multiplier",
    "construction_cost_trend_score",
    "construction_cost_trend_band",
    "combined_construction_cost_index",
    "cost_trend_model",
    "trend_adjusted_development_cost",

    # Explanation/Reporting
    "ranking_profile",
    "fast_explanation",
    "explanation",
    "agent_pitch",
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
    recall_k: int = Field(default=1000, ge=10, le=20000)

    with_explanations: bool = False
    use_template_explanations: bool = True

    retrieval_model: str = DEFAULT_RETRIEVAL_MODEL
    use_dcn_reranker: bool = True
    reranking_model: str = DEFAULT_RERANKING_MODEL

    alpha: float = 0.5
    beta: float = 0.5
    dedupe_by_address: bool = True

    locality: str | None = None
    address_contains: str | None = None

    ranking_profile: Literal[
        "balanced",
        "policy_upside",
        "budget_sensitive",
        "high_value",
    ] = "balanced"

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


class ReportJobPayload(BaseModel):
    request_id: str
    explanation_mode: str = "template"

    output_markdown: bool = True
    output_pdf: bool = True

    audience: str = "developer"
    title: str = "Smart Developer Site Recommendation Report"


class ExportReportPayload(BaseModel):
    strategy: str
    query_text: str
    results: list[dict[str, Any]]

    title: str = "Smart Developer Site Recommendation Report"
    audience: str = "developer / real estate agent"

    output_format: Literal["pdf", "markdown"] = "pdf"
    max_rows: int = Field(default=5, ge=1, le=20)

    include_explanations: bool = True
    include_risks: bool = True
    include_table: bool = True
    include_policy: bool = True
    include_economics: bool = True
    include_policy_evidence: bool = True


class ServiceState:
    predictor: SmartDeveloperPredictor | None = None
    startup_latency_ms: float | None = None
    warmup_latency_ms: float | None = None
    is_ready: bool = False


state = ServiceState()


def _get_or_create_predictor() -> SmartDeveloperPredictor:
    if state.predictor is None:
        state.predictor = SmartDeveloperPredictor()
    return state.predictor


def _warmup_predictor(predictor: SmartDeveloperPredictor) -> float:
    warmup_request = PredictionRequest(
        strategy="low_rise_apartment",
        query_text=(
            "I want a site for low-rise apartment redevelopment near a train station, "
            "with high development zoning, a large site, and limited planning constraints."
        ),
        top_k=1,
        recall_k=10000,
        with_explanations=False,
        retrieval_model=DEFAULT_RETRIEVAL_MODEL,
        use_dcn_reranker=True,
        reranking_model=DEFAULT_RERANKING_MODEL,
        locality="WAITARA",
        ranking_profile="balanced"
    )

    t0 = time.perf_counter()
    predictor.predict(warmup_request)
    return round((time.perf_counter() - t0) * 1000, 2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_t0 = time.perf_counter()

    lazy_load_predictor = os.getenv("LAZY_LOAD_PREDICTOR", "false").lower() == "true"
    skip_warmup = os.getenv("SKIP_STARTUP_WARMUP", "false").lower() == "true"

    if lazy_load_predictor:
        # Cloud/demo mode:
        # Open the HTTP port immediately. The predictor/model artifacts will be
        # loaded on the first real retrieval request.
        state.predictor = None
        state.warmup_latency_ms = None
    else:
        predictor = SmartDeveloperPredictor()
        state.predictor = predictor

        if skip_warmup:
            state.warmup_latency_ms = None
        else:
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
        "predictor_loaded": state.predictor is not None,
        "lazy_load_predictor": os.getenv("LAZY_LOAD_PREDICTOR", "false").lower() == "true",
        "skip_startup_warmup": os.getenv("SKIP_STARTUP_WARMUP", "false").lower() == "true",
        "service_startup_latency_ms": state.startup_latency_ms,
        "model_load_and_warmup_latency_ms": state.warmup_latency_ms,
        "warm_request_expected": state.is_ready,
        "mlops_logging_enabled": MLOPS_AVAILABLE and is_database_enabled(),
        "mlops_available": MLOPS_AVAILABLE,
        "mlops_import_error": MLOPS_IMPORT_ERROR,
        "production_models": {
            "retrieval_model": DEFAULT_RETRIEVAL_MODEL,
            "reranking_model": DEFAULT_RERANKING_MODEL,
            "market_value_model": "xgb_market_value_v1",
        },
        "ranking_profiles": [
            "balanced",
            "policy_upside",
            "budget_sensitive",
            "high_value",
        ],
        "economics_enabled": True,
        "policy_rag_enabled": True,
    }


@app.post("/retrieve-sites")
def retrieve_sites(payload: RetrieveSitesPayload) -> dict[str, Any]:
    if not state.is_ready:
        return {
            "status": "error",
            "message": "Service is not ready.",
        }

    predictor = _get_or_create_predictor()

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
        ranking_profile=payload.ranking_profile,
    )

    response = predictor.predict(request)

    if payload.use_template_explanations:
        response["results"] = add_template_explanations(
            response.get("results", []),
            strategy=payload.strategy,
            output_field="fast_explanation",
        )
        response["metadata"]["use_template_explanations"] = True
    else:
        response["metadata"]["use_template_explanations"] = False

    response["service"] = {
        "mode": "warm_predictor_singleton",
        "service_startup_latency_ms": state.startup_latency_ms,
        "model_load_and_warmup_latency_ms": state.warmup_latency_ms,
    }

    if payload.log_request and MLOPS_AVAILABLE and is_database_enabled():
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
    else:
        response["logging"] = {
            "enabled": False,
            "status": "skipped",
            "reason": (
                "DATABASE_URL is not set or MLOps logging is unavailable."
            ),
        }

    if not payload.debug:
        response = _filter_product_response(response)

    return response


@app.post("/feedback")
def feedback(payload: FeedbackPayload) -> dict[str, Any]:
    if not (MLOPS_AVAILABLE and is_database_enabled()):
        return {
            "status": "disabled",
            "message": "Feedback logging is disabled because DATABASE_URL is not configured.",
            "request_id": payload.request_id,
        }
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


@app.post("/report-jobs")
def create_report_job_endpoint(payload: ReportJobPayload) -> dict[str, Any]:
    if not (MLOPS_AVAILABLE and is_database_enabled()):
        return {
            "status": "disabled",
            "message": "Report jobs are disabled because DATABASE_URL is not configured.",
            "request_id": payload.request_id,
        }
    try:
        result = generate_report_from_request_id(
            request_id=payload.request_id,
            explanation_mode=payload.explanation_mode,
            output_markdown=payload.output_markdown,
            output_pdf=payload.output_pdf,
            audience=payload.audience,
            title=payload.title,
        )
        return result

    except Exception as exc:
        return {
            "status": "failed",
            "request_id": payload.request_id,
            "error": str(exc),
        }


@app.post("/export-report")
def export_report(payload: ExportReportPayload):
    """
    Generate a report directly from the provided result payload.

    This endpoint does not require DATABASE_URL or MLOps logging.
    It is intended for demo / stateless report export.
    """
    if not payload.results:
        return {
            "status": "failed",
            "error": "No results were provided for report export.",
        }

    df = pd.DataFrame(payload.results)

    report = build_site_report(
        results=df,
        strategy=payload.strategy,
        query_text=payload.query_text,
        config=ReportConfig(
            title=payload.title,
            audience=payload.audience,
            include_explanations=payload.include_explanations,
            include_risks=payload.include_risks,
            include_table=payload.include_table,
            include_policy=payload.include_policy,
            include_economics=payload.include_economics,
            include_policy_evidence=payload.include_policy_evidence,
            max_rows=payload.max_rows,
        ),
    )

    safe_strategy = payload.strategy.replace("/", "_").replace(" ", "_")
    filename_stem = f"smart_developer_{safe_strategy}_report"

    if payload.output_format == "markdown":
        return PlainTextResponse(
            report,
            media_type="text/markdown",
            headers={
                "Content-Disposition": f'attachment; filename="{filename_stem}.md"'
            },
        )

    temp_dir = Path(tempfile.gettempdir()) / "smart_developer_reports"
    temp_dir.mkdir(parents=True, exist_ok=True)

    output_path = temp_dir / f"{filename_stem}.pdf"
    export_markdown_report_to_pdf(report, output_path)

    return FileResponse(
        path=str(output_path),
        media_type="application/pdf",
        filename=f"{filename_stem}.pdf",
    )


@app.get("/report-jobs/{report_id}")
def get_report_job_endpoint(report_id: str) -> dict[str, Any]:
    if not (MLOPS_AVAILABLE and is_database_enabled()):
        return {
            "status": "disabled",
            "message": "Report jobs are disabled because DATABASE_URL is not configured.",
            "report_id": report_id,
        }

    try:
        return get_report_job(report_id)

    except Exception as exc:
        return {
            "status": "failed",
            "report_id": report_id,
            "error": str(exc),
        }