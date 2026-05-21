#!/usr/bin/env bash
set -euo pipefail

SOURCES_PATH="${SOURCES_PATH:-algorithm/configs/policies/policy_sources.yaml}"
INDEX_DIR="${INDEX_DIR:-algorithm/artifacts/policy_index/chroma}"
RAW_DIR="${RAW_DIR:-algorithm/artifacts/policy_index/raw}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"

python -m algorithm.src.policy.build_policy_index \
  --sources "$SOURCES_PATH" \
  --index-dir "$INDEX_DIR" \
  --raw-dir "$RAW_DIR" \
  --embedding-model "$EMBEDDING_MODEL" \
  --reset
