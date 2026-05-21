from __future__ import annotations
from typing import Any, List
from pydantic import BaseModel, Field


class ExplanationPayload(BaseModel):
    strategy: str
    decision_band: str

    positive_evidence: List[str] = Field(default_factory=list)
    negative_evidence: List[str] = Field(default_factory=list)
    cautions: List[str] = Field(default_factory=list)

    # New policy-aware explanation fields.
    policy_signal_band: str | None = None
    policy_upside_score: float | None = None
    policy_matched_policies: List[str] = Field(default_factory=list)
    policy_evidence: List[dict[str, Any]] = Field(default_factory=list)

    # New economics-aware explanation fields.
    value_potential_score: float | None = None
    value_potential_band: str | None = None
    cost_efficiency_score: float | None = None
    cost_risk_score: float | None = None
    cost_band: str | None = None

    ml_estimated_market_value: float | None = None
    trend_adjusted_ml_market_value: float | None = None
    estimated_total_project_cost: float | None = None

    market_trend_band: str | None = None
    predicted_market_growth_3m: float | None = None
    construction_cost_trend_band: str | None = None
    predicted_construction_cost_growth_qoq: float | None = None