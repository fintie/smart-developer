from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    request_id: str
    event_type: str
    rid: str | int | None = None
    rank_position: int | None = None
    event_value: dict[str, Any] | None = None
    user_note: str | None = None
    user_id: str | None = "demo_user"
    session_id: str | None = "frontend_demo"


class RecommendationFeedbackRequest(BaseModel):
    request_id: str
    rating: int = Field(ge=1, le=5)
    user_note: str | None = None
    user_id: str | None = "demo_user"
    session_id: str | None = "frontend_demo"
