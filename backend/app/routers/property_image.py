from __future__ import annotations

from fastapi import APIRouter, Query

from backend.app.services.property_image import get_property_street_view_image

router = APIRouter(prefix="/api", tags=["property-image"])


@router.get("/property-image")
async def property_image(
    address: str = Query(min_length=3),
    latitude: float | None = None,
    longitude: float | None = None,
):
    return await get_property_street_view_image(
        address=address,
        latitude=latitude,
        longitude=longitude,
    )
