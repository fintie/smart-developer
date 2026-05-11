from __future__ import annotations
import uuid
from typing import Any
from algorithm.src.mlops.db import get_session
from algorithm.src.mlops.models import (
    RetrievalRequestLog,
    RetrievalResultLog,
    UserFeedbackLog,
)


def _to_str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def log_retrieval_response(
    response: dict[str, Any],
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    planner_payload: dict[str, Any] | None = None,
) -> None:
    """
    Persist one retrieval response into:
    - retrieval_requests
    - retrieval_results

    This function expects the backend-facing predictor response shape:
    {
        "request_id": ...,
        "request": {...},
        "metadata": {...},
        "results": [...]
    }
    """
    request_id = response["request_id"]
    request_payload = response.get("request", {})
    metadata = response.get("metadata", {})
    results = response.get("results", [])

    request_row = RetrievalRequestLog(
        request_id=request_id,
        user_id=user_id,
        session_id=session_id,
        strategy=metadata.get("strategy") or request_payload.get("strategy"),
        query_text=metadata.get("query_text") or request_payload.get("query_text"),
        locality=metadata.get("locality") or request_payload.get("locality"),
        address_contains=metadata.get("address_contains") or request_payload.get("address_contains"),
        top_k=int(metadata.get("top_k") or request_payload.get("top_k") or 0),
        recall_k=int(metadata.get("recall_k") or request_payload.get("recall_k") or 0),
        retrieval_model=metadata.get("retrieval_model") or request_payload.get("retrieval_model"),
        reranking_model=metadata.get("reranking_model") or request_payload.get("reranking_model"),
        use_dcn_reranker=bool(
            metadata.get("use_dcn_reranker", request_payload.get("use_dcn_reranker", True))
        ),
        with_explanations=bool(
            metadata.get("with_explanations", request_payload.get("with_explanations", False))
        ),
        dedupe_by_address=bool(
            metadata.get("dedupe_by_address", request_payload.get("dedupe_by_address", True))
        ),
        latency_ms=metadata.get("latency_ms"),
        result_count=metadata.get("result_count") or response.get("result_count"),
        request_payload=request_payload,
        planner_payload=planner_payload,
        metadata_payload=metadata,
    )

    result_rows: list[RetrievalResultLog] = []

    for idx, item in enumerate(results, start=1):
        final_rank_score = (
            item.get("dcn_rank_score")
            if item.get("dcn_rank_score") is not None
            else item.get("fusion_rank_score")
        )

        result_rows.append(
            RetrievalResultLog(
                request_id=request_id,
                rid=_to_str_or_none(item.get("RID") or item.get("rid")) or "",
                rank_position=idx,
                address=item.get("address"),
                base_site_address=item.get("base_site_address"),
                primary_zoning_code=item.get("primary_zoning_code"),
                zoning_band=item.get("zoning_band"),
                lot_size_band=item.get("lot_size_band"),
                constraint_severity_band=item.get("constraint_severity_band"),
                station_distance_band=item.get("station_distance_band"),
                strategy_score=item.get("strategy_score"),
                retrieval_similarity=item.get("retrieval_similarity"),
                fusion_score=item.get("fusion_score"),
                dcn_prob=item.get("dcn_prob"),
                final_rank_score=final_rank_score,
                result_payload=item,
            )
        )

    with get_session() as session:
        session.add(request_row)
        session.add_all(result_rows)


def log_user_feedback(
    *,
    request_id: str,
    event_type: str,
    rid: str | int | None = None,
    rank_position: int | None = None,
    event_value: dict[str, Any] | None = None,
    user_note: str | None = None,
    feedback_id: str | None = None,
) -> str:
    """
    Persist a user feedback event.

    event_type examples:
    - view
    - click
    - save
    - dismiss
    - select
    - export_report
    - download_pdf
    - manual_positive
    - manual_negative
    """
    feedback_id = feedback_id or f"fb_{uuid.uuid4().hex[:12]}"

    row = UserFeedbackLog(
        feedback_id=feedback_id,
        request_id=request_id,
        rid=_to_str_or_none(rid),
        rank_position=rank_position,
        event_type=event_type,
        event_value=event_value,
        user_note=user_note,
    )

    with get_session() as session:
        session.add(row)

    return feedback_id