from __future__ import annotations

import io
import math
from typing import Any

import httpx
from fastapi import HTTPException
from fastapi.responses import Response
from PIL import Image, ImageDraw

OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256
OUTPUT_WIDTH = 960
OUTPUT_HEIGHT = 300
ZOOM = 17


async def get_property_street_view_image(
    *,
    address: str,
    latitude: float | None = None,
    longitude: float | None = None,
) -> Response:
    if latitude is None or longitude is None:
        raise HTTPException(
            status_code=404,
            detail="Map preview requires latitude and longitude.",
        )

    try:
        image = await _build_osm_tile_preview(latitude=latitude, longitude=longitude)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=404,
            detail="OpenStreetMap preview is unavailable for this property.",
        ) from exc

    output = io.BytesIO()
    image.save(output, format="PNG")
    return Response(content=output.getvalue(), media_type="image/png")


def _lon_lat_to_world_pixels(
    *,
    latitude: float,
    longitude: float,
    zoom: int,
) -> tuple[float, float]:
    sin_lat = math.sin(math.radians(max(min(latitude, 85.05112878), -85.05112878)))
    scale = TILE_SIZE * (2**zoom)
    x = (longitude + 180.0) / 360.0 * scale
    y = (
        0.5
        - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)
    ) * scale
    return x, y


async def _build_osm_tile_preview(*, latitude: float, longitude: float) -> Image.Image:
    center_x, center_y = _lon_lat_to_world_pixels(
        latitude=latitude,
        longitude=longitude,
        zoom=ZOOM,
    )
    left = center_x - OUTPUT_WIDTH / 2
    top = center_y - OUTPUT_HEIGHT / 2
    right = center_x + OUTPUT_WIDTH / 2
    bottom = center_y + OUTPUT_HEIGHT / 2
    min_tile_x = math.floor(left / TILE_SIZE)
    max_tile_x = math.floor((right - 1) / TILE_SIZE)
    min_tile_y = math.floor(top / TILE_SIZE)
    max_tile_y = math.floor((bottom - 1) / TILE_SIZE)
    tile_count = 2**ZOOM
    canvas = Image.new(
        "RGB",
        (
            (max_tile_x - min_tile_x + 1) * TILE_SIZE,
            (max_tile_y - min_tile_y + 1) * TILE_SIZE,
        ),
        "#f1f5f9",
    )

    async with httpx.AsyncClient(
        timeout=20.0,
        headers={"User-Agent": "SmartDeveloperDemo/0.1"},
    ) as client:
        for tile_x in range(min_tile_x, max_tile_x + 1):
            for tile_y in range(min_tile_y, max_tile_y + 1):
                if tile_y < 0 or tile_y >= tile_count:
                    continue
                response = await client.get(
                    OSM_TILE_URL.format(
                        z=ZOOM,
                        x=tile_x % tile_count,
                        y=tile_y,
                    )
                )
                response.raise_for_status()
                tile = Image.open(io.BytesIO(response.content)).convert("RGB")
                canvas.paste(
                    tile,
                    (
                        (tile_x - min_tile_x) * TILE_SIZE,
                        (tile_y - min_tile_y) * TILE_SIZE,
                    ),
                )

    crop_left = int(left - min_tile_x * TILE_SIZE)
    crop_top = int(top - min_tile_y * TILE_SIZE)
    image = canvas.crop(
        (
            crop_left,
            crop_top,
            crop_left + OUTPUT_WIDTH,
            crop_top + OUTPUT_HEIGHT,
        )
    )
    draw = ImageDraw.Draw(image)
    marker_x = OUTPUT_WIDTH // 2
    marker_y = OUTPUT_HEIGHT // 2
    draw.ellipse(
        (marker_x - 13, marker_y - 13, marker_x + 13, marker_y + 13),
        fill="#2563eb",
        outline="#ffffff",
        width=4,
    )
    draw.ellipse(
        (marker_x - 5, marker_y - 5, marker_x + 5, marker_y + 5),
        fill="#ffffff",
    )
    return image
