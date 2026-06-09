from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.schemas import AIPropertySummaryRequest
from backend.app.services.ai_property_summary import (
    AIPropertySummaryError,
    generate_ai_property_summary,
)

router = APIRouter(prefix="/api", tags=["ai-property-summary"])


@router.post("/ai-property-summary")
async def ai_property_summary(payload: AIPropertySummaryRequest):
    try:
        return await generate_ai_property_summary(
            query_text=payload.query_text,
            site=payload.site,
            user_requirements=payload.user_requirements,
        )
    except AIPropertySummaryError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
