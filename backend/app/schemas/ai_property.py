from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AIPropertySummaryRequest(BaseModel):
    query_text: str
    site: dict[str, Any]
    user_requirements: str | None = None
    user_id: str | None = "demo_user"
    session_id: str | None = "frontend_demo"


class AIPropertyValueEstimate(BaseModel):
    label: str
    amount: float | None = None
    confidence: str | None = None
    range_label: str | None = None
    explanation: str


class AIPropertySummary(BaseModel):
    headline: str
    basic_info: list[str] = Field(default_factory=list)
    requirement_match: str
    value_estimate: AIPropertyValueEstimate
    opportunity_notes: list[str] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)
    disclaimer: str


class AIExternalSource(BaseModel):
    title: str
    link: str
    snippet: str | None = None


class AIPropertySuggestion(BaseModel):
    headline: str
    suggestion: str
    external_context_used: bool = False
    source_notes: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)


class AIPropertySummaryResponse(BaseModel):
    source: str
    model: str | None = None
    summary: AIPropertySummary
    ai_suggestion: AIPropertySuggestion
    external_sources: list[AIExternalSource] = Field(default_factory=list)
