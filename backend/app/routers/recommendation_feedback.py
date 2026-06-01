from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.schemas import RecommendationFeedbackRequest
from backend.app.services.algorithm_client import AlgorithmServiceError, log_feedback
from backend.app.services.recommendation_feedback import (
    RATING_LABELS,
    build_recommendation_feedback_event_value,
)


router = APIRouter(prefix="/api", tags=["recommendation-feedback"])


@router.post("/recommendation-feedback")
async def recommendation_feedback(payload: RecommendationFeedbackRequest):
    event_value = build_recommendation_feedback_event_value(
        rating=payload.rating,
        user_id=payload.user_id,
        session_id=payload.session_id,
    )

    try:
        result = await log_feedback(
            {
                "request_id": payload.request_id,
                "event_type": "recommendation_satisfaction_rating",
                "rid": None,
                "rank_position": None,
                "event_value": event_value,
                "user_note": payload.user_note,
                "user_id": payload.user_id,
                "session_id": payload.session_id,
            }
        )
    except AlgorithmServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        **result,
        "rating": payload.rating,
        "rating_label": RATING_LABELS[payload.rating],
    }
