from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _point_from_geometry(geometry: dict[str, Any]) -> tuple[float | None, float | None, str | None]:
    """
    Returns (longitude, latitude, geometry_type).

    GeoJSON coordinates are usually [lon, lat] when outSR=4326.
    For polygons, use a simple centroid approximation from all coordinate points.
    This is enough for frontend map pins.
    """
    if not geometry:
        return None, None, None

    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")

    if not coords:
        return None, None, geom_type

    if geom_type == "Point":
        try:
            lon, lat = coords[0], coords[1]
            return float(lon), float(lat), geom_type
        except Exception:
            return None, None, geom_type

    points: list[tuple[float, float]] = []

    def collect_points(obj: Any) -> None:
        if (
            isinstance(obj, list)
            and len(obj) >= 2
            and isinstance(obj[0], (int, float))
            and isinstance(obj[1], (int, float))
        ):
            points.append((float(obj[0]), float(obj[1])))
            return

        if isinstance(obj, list):
            for item in obj:
                collect_points(item)

    collect_points(coords)

    if not points:
        return None, None, geom_type

    lon = sum(p[0] for p in points) / len(points)
    lat = sum(p[1] for p in points) / len(points)

    return lon, lat, geom_type


def parse_geojson(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text())

    rows = []

    for feature in data.get("features", []):
        props = feature.get("properties", {}) or {}
        geometry = feature.get("geometry", {}) or {}

        rid = props.get("RID")
        lon, lat, geom_type = _point_from_geometry(geometry)

        if rid is None:
            continue

        rows.append(
            {
                "RID": str(rid).replace(".0", "").strip(),
                "latitude": lat,
                "longitude": lon,
                "geometry_type": geom_type,
                "geocode_source": "NSW Land Parcel Property Theme",
                "geocode_confidence": "property_rid_match",
                "source_file": path.name,
                "property_address": props.get("address"),
                "property_type": props.get("propertytype"),
                "shape_area": props.get("Shape__Area"),
                "shape_length": props.get("Shape__Length"),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/raw/nsw_property/chunks")
    parser.add_argument("--output", default="data/processed/geospatial/property_coordinates.parquet")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    paths = sorted(input_dir.glob("*.geojson"))
    print(f"Found GeoJSON files: {len(paths):,}")

    frames = []

    for i, path in enumerate(paths, 1):
        if i <= 5 or i % 50 == 0 or i == len(paths):
            print(f"[{i}/{len(paths)}] {path}")

        try:
            df = parse_geojson(path)
        except Exception as exc:
            print(f"Failed to parse {path}: {exc}")
            continue

        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError("No coordinate rows parsed.")

    coords = pd.concat(frames, ignore_index=True)

    before = len(coords)
    coords = coords.dropna(subset=["latitude", "longitude"])
    coords = coords.drop_duplicates(subset=["RID"])
    after = len(coords)

    coords.to_parquet(output_path, index=False)

    print()
    print(f"Raw coordinate rows: {before:,}")
    print(f"Saved coordinate rows: {after:,}")
    print(f"Saved to: {output_path}")
    print()
    print(coords.head(20).to_string(index=False))


if __name__ == "__main__":
    main()