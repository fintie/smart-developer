from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


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
