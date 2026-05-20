from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def normalise_rid(series: pd.Series) -> pd.Series:
    return (
        series
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/processed/retrieval/candidate_sites.parquet")
    parser.add_argument("--coordinates", default="data/processed/geospatial/property_coordinates.parquet")
    parser.add_argument("--output", default="data/processed/retrieval/candidate_sites_geo.parquet")
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    coords_path = Path(args.coordinates)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_parquet(candidates_path)
    coords = pd.read_parquet(coords_path)

    candidates = candidates.copy()
    coords = coords.copy()

    candidates["RID"] = normalise_rid(candidates["RID"])
    coords["RID"] = normalise_rid(coords["RID"])

    keep_coord_cols = [
        "RID",
        "latitude",
        "longitude",
        "geometry_type",
        "geocode_source",
        "geocode_confidence",
        "property_address",
        "property_type",
        "shape_area",
        "shape_length",
    ]
    keep_coord_cols = [c for c in keep_coord_cols if c in coords.columns]

    out = candidates.merge(
        coords[keep_coord_cols],
        on="RID",
        how="left",
        validate="many_to_one",
    )

    matched = out["latitude"].notna() & out["longitude"].notna()
    match_rate = matched.mean()

    out.to_parquet(output_path, index=False)

    print(f"Candidates: {len(candidates):,}")
    print(f"Coordinate rows: {len(coords):,}")
    print(f"Matched candidates: {matched.sum():,}")
    print(f"Match rate: {match_rate:.2%}")
    print(f"Saved to: {output_path}")

    print()
    print(out[
        [
            "RID",
            "address",
            "latitude",
            "longitude",
            "geometry_type",
            "geocode_confidence",
        ]
    ].head(20).to_string(index=False))


if __name__ == "__main__":
    main()