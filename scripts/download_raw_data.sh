#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# ArcGIS raw data download script
#
# Modes:
# 1. Full raw dataset download:
#    - addressing / zoning / bushfire / flood / heritage / property
#
# 2. Candidate-only property geometry download:
#    - reads data/processed/retrieval/candidate_rids.txt
#    - downloads only property geometry for current recommendation candidates
#    - used to build latitude/longitude for map-ready recommendations
#
# Recommended for normal development:
#   DOWNLOAD_PROPERTY_BY_CANDIDATE_RIDS=true
#   DOWNLOAD_PROPERTY=false
#   DOWNLOAD_ADDRESSING=false
# ============================================================

# ----------------------------
# Full raw dataset downloads
# ----------------------------
DOWNLOAD_ADDRESSING=false
DOWNLOAD_ZONING=false
DOWNLOAD_BUSHFIRE=false
DOWNLOAD_FLOOD=false
DOWNLOAD_HERITAGE=false
DOWNLOAD_PROPERTY=false

# ----------------------------
# Candidate-only property geometry
# ----------------------------
DOWNLOAD_PROPERTY_BY_CANDIDATE_RIDS=true

# RID list generated from:
#   data/processed/retrieval/candidate_sites.parquet
RID_FILE="../../data/processed/retrieval/candidate_rids.txt"
RID_CHUNK_SIZE=200

# ----------------------------
# Rust downloader options
# ----------------------------
MAX_CONCURRENCY=4
REQUEST_PAUSE_MS=50
TIMEOUT_SECS=60

# Optional smoke test.
# Empty means no limit.
# Example:
#   LIMIT=2 ./scripts/download_raw_data.sh
LIMIT="${LIMIT:-}"

# Rust project location.
# Change this if your crate folder is different.
RUST_DOWNLOADER_DIR="rust_data_loaders/nsw_addressing_downloader"

# ============================================================
# Helper
# ============================================================

run_dataset() {
  local dataset="$1"

  echo
  echo "============================================================"
  echo "Downloading full dataset: $dataset"
  echo "============================================================"

  local args=(
    --dataset "$dataset"
    --max-concurrency "$MAX_CONCURRENCY"
    --request-pause-ms "$REQUEST_PAUSE_MS"
    --timeout-secs "$TIMEOUT_SECS"
  )

  if [ -n "$LIMIT" ]; then
    args+=(--limit "$LIMIT")
  fi

  cargo run --release -- "${args[@]}"
}

run_candidate_property_geometry() {
  echo
  echo "============================================================"
  echo "Downloading candidate-only property geometry"
  echo "============================================================"
  echo "RID file: $RID_FILE"
  echo "RID chunk size: $RID_CHUNK_SIZE"

  if [ ! -f "$RID_FILE" ]; then
    echo "RID file not found: $RID_FILE"
    echo
    echo "Generate it from repo root with:"
    echo "python - <<'PY'"
    echo "import pandas as pd"
    echo "from pathlib import Path"
    echo "df = pd.read_parquet('data/processed/retrieval/candidate_sites.parquet')"
    echo "out = Path('data/processed/retrieval/candidate_rids.txt')"
    echo "out.parent.mkdir(parents=True, exist_ok=True)"
    echo "rids = df['RID'].dropna().astype(str).str.replace(r'\\.0$', '', regex=True).drop_duplicates().sort_values()"
    echo "out.write_text('\\n'.join(rids) + '\\n')"
    echo "print('Saved:', out)"
    echo "print('RID count:', len(rids))"
    echo "PY"
    exit 1
  fi

  local args=(
    --dataset property
    --rid-file "$RID_FILE"
    --rid-chunk-size "$RID_CHUNK_SIZE"
    --max-concurrency "$MAX_CONCURRENCY"
    --request-pause-ms "$REQUEST_PAUSE_MS"
    --timeout-secs "$TIMEOUT_SECS"
  )

  if [ -n "$LIMIT" ]; then
    args+=(--limit "$LIMIT")
  fi

  cargo run --release -- "${args[@]}"
}

# ============================================================
# Main
# Assumes current directory is repo root.
# ============================================================

if [ ! -d "$RUST_DOWNLOADER_DIR" ]; then
  echo "Rust downloader directory not found: $RUST_DOWNLOADER_DIR"
  echo "Please update RUST_DOWNLOADER_DIR in this script."
  exit 1
fi

cd "$RUST_DOWNLOADER_DIR"

if [ "$DOWNLOAD_ADDRESSING" = true ]; then
  run_dataset "addressing"
fi

if [ "$DOWNLOAD_ZONING" = true ]; then
  run_dataset "zoning"
fi

if [ "$DOWNLOAD_BUSHFIRE" = true ]; then
  run_dataset "bushfire"
fi

if [ "$DOWNLOAD_FLOOD" = true ]; then
  run_dataset "flood"
fi

if [ "$DOWNLOAD_HERITAGE" = true ]; then
  run_dataset "heritage"
fi

if [ "$DOWNLOAD_PROPERTY" = true ]; then
  run_dataset "property"
fi

if [ "$DOWNLOAD_PROPERTY_BY_CANDIDATE_RIDS" = true ]; then
  run_candidate_property_geometry
fi

echo
echo "============================================================"
echo "Selected raw data downloads completed."
echo "============================================================"
