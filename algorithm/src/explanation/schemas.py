from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field


class ExplanationPayload(BaseModel):
    strategy: str
    decision_band: str
    positive_evidence: List[str] = Field(default_factory=list)
    negative_evidence: List[str] = Field(default_factory=list)
    cautions: List[str] = Field(default_factory=list)