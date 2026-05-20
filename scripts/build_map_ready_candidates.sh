#!/usr/bin/env bash
set -euo pipefail

PROPERTY_CHUNKS_DIR="${PROPERTY_CHUNKS_DIR:-data/raw/nsw_property/chunks}"
PROPERTY_COORDS_PATH="${PROPERTY_COORDS_PATH:-data/processed/geospatial/property_coordinates.parquet}"
CANDIDATES_PATH="${CANDIDATES_PATH:-data/processed/retrieval/candidate_sites.parquet}"
OUTPUT_PATH="${OUTPUT_PATH:-data/processed/retrieval/candidate_sites_geo.parquet}"

echo
echo "============================================================"
echo "Building property coordinate lookup"
echo "============================================================"

python -m algorithm.src.data.build_property_coordinates \
  --input-dir "$PROPERTY_CHUNKS_DIR" \
  --output "$PROPERTY_COORDS_PATH"

echo
echo "============================================================"
echo "Joining coordinates into candidate sites"
echo "============================================================"

python -m algorithm.src.data.build_candidate_sites_geo \
  --candidates "$CANDIDATES_PATH" \
  --coordinates "$PROPERTY_COORDS_PATH" \
  --output "$OUTPUT_PATH"

echo
echo "============================================================"
echo "Map-ready candidate dataset completed"
echo "============================================================"
echo "Output: $OUTPUT_PATH"