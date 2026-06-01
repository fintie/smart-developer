from __future__ import annotations


RATING_LABELS = {
    1: "very_unsatisfied",
    2: "unsatisfied",
    3: "neutral",
    4: "satisfied",
    5: "very_satisfied",
}


def attach_recommendation_feedback_prompt(response: dict) -> dict:
    response["feedback_prompt"] = {
        "enabled": True,
        "type": "recommendation_satisfaction",
        "title": "How satisfied are you with these recommendations?",
        "scale": {
            "min": 1,
            "max": 5,
            "labels": RATING_LABELS,
        },
        "submit_endpoint": "/api/recommendation-feedback",
    }
    return response


def build_recommendation_feedback_event_value(
    *,
    rating: int,
    user_id: str | None,
    session_id: str | None,
) -> dict:
    return {
        "rating": rating,
        "rating_label": RATING_LABELS[rating],
        "rating_scale_min": 1,
        "rating_scale_max": 5,
        "feedback_scope": "recommendation_set",
        "user_id": user_id,
        "session_id": session_id,
    }
