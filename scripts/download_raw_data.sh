#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# ArcGIS raw data download script
# Edit the flags below to choose which datasets to download.
# ============================================================

DOWNLOAD_ADDRESSING=true
DOWNLOAD_ZONING=true
DOWNLOAD_BUSHFIRE=true
DOWNLOAD_FLOOD=true
DOWNLOAD_HERITAGE=true
DOWNLOAD_PROPERTY=true

# Rust downloader options
MAX_CONCURRENCY=4
REQUEST_PAUSE_MS=50

# ============================================================
# Helper
# ============================================================

run_dataset() {
  local dataset="$1"
  echo
  echo "============================================================"
  echo "Downloading dataset: $dataset"
  echo "============================================================"
  cargo run --release -- \
    --dataset "$dataset" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --request-pause-ms "$REQUEST_PAUSE_MS"
}

# ============================================================
# Main
# Assumes current directory is repo root
# and Rust project lives at rust_data_loaders/arcgis_downloader
# ============================================================

cd rust_data_loaders/arcgis_downloader

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

echo
echo "============================================================"
echo "Selected raw data downloads completed."
echo "============================================================"