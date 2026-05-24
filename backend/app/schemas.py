from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    strategy: str = "single_dwelling_rebuild"
    query_text: str
    top_k: int = Field(default=5, ge=1, le=20)
    recall_k: int = Field(default=1000, ge=10, le=20000)

    locality: str | None = None
    address_contains: str | None = None

    ranking_profile: Literal[
        "balanced",
        "policy_upside",
        "budget_sensitive",
        "high_value",
    ] = "balanced"

    user_id: str | None = "demo_user"
    session_id: str | None = "frontend_demo"

    with_explanations: bool = False
    use_template_explanations: bool = True
    log_request: bool = True
    debug: bool = False


class FeedbackRequest(BaseModel):
    request_id: str
    event_type: str
    rid: str | int | None = None
    rank_position: int | None = None
    event_value: dict[str, Any] | None = None
    user_note: str | None = None
    user_id: str | None = "demo_user"
    session_id: str | None = "frontend_demo"


class ReportRequest(BaseModel):
    request_id: str
    explanation_mode: str = "template"
    output_markdown: bool = True
    output_pdf: bool = True
    audience: str = "developer"
    title: str = "Smart Developer Site Recommendation Report"


class ExportReportRequest(BaseModel):
    strategy: str
    query_text: str
    results: list[dict[str, Any]]

    title: str = "Smart Developer Site Recommendation Report"
    audience: str = "developer / real estate agent"

    output_format: Literal["pdf", "markdown"] = "pdf"
    max_rows: int = 5

    include_explanations: bool = True
    include_risks: bool = True
    include_table: bool = True
    include_policy: bool = True
    include_economics: bool = True
    include_policy_evidence: bool = True