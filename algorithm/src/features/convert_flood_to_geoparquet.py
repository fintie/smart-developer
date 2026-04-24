from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

CHUNKS_DIR = ROOT / "data" / "raw" / "nsw_flood" / "chunks"
OUTPUT_DIR = ROOT / "data" / "processed" / "nsw_flood"
OUTPUT_PATH = OUTPUT_DIR / "flood.parquet"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chunk_files = sorted(CHUNKS_DIR.glob("*.geojson"))
    if not chunk_files:
        raise RuntimeError(f"No chunk files found in {CHUNKS_DIR}")

    frames = []
    failed = []

    for i, chunk_file in enumerate(chunk_files, start=1):
        try:
            print(f"[{i}/{len(chunk_files)}] Reading {chunk_file.name}")
            gdf = gpd.read_file(chunk_file)
            frames.append(gdf)
        except Exception as e:
            print(f"FAILED: {chunk_file.name} -> {e}")
            failed.append(chunk_file.name)

    if not frames:
        raise RuntimeError("No flood chunk files could be read.")

    print("Concatenating flood frames...")
    merged = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs=frames[0].crs,
    )

    print("Rows:", len(merged))
    print("Columns:", list(merged.columns))
    print("CRS:", merged.crs)

    print(f"Writing GeoParquet: {OUTPUT_PATH}")
    merged.to_parquet(OUTPUT_PATH, index=False)

    print("Verifying parquet...")
    check = gpd.read_parquet(OUTPUT_PATH)
    print("Verified rows:", len(check))
    print("Verified CRS:", check.crs)

    if failed:
        print("\nFailed chunks:")
        for name in failed[:20]:
            print(" -", name)
        print(f"Total failed chunks: {len(failed)}")
    else:
        print("\nAll flood chunks read successfully.")


if __name__ == "__main__":
    main()