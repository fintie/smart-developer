#!/usr/bin/env bash
set -euo pipefail

# =========================
# Smart Developer Feedback Dataset Export
# =========================

OUTPUT_DIR="algorithm/artifacts/feedback"

FULL_OUTPUT="$OUTPUT_DIR/feedback_reranker_dataset.parquet"
LABELLED_OUTPUT="$OUTPUT_DIR/feedback_reranker_dataset_labelled.parquet"
WEAK_OUTPUT="$OUTPUT_DIR/feedback_reranker_dataset_weak.parquet"

mkdir -p "$OUTPUT_DIR"

echo "Exporting full feedback dataset..."
python -m algorithm.src.mlops.build_feedback_dataset \
  --output "$FULL_OUTPUT"

echo
echo "Exporting labelled-only feedback dataset..."
python -m algorithm.src.mlops.build_feedback_dataset \
  --labelled-only \
  --output "$LABELLED_OUTPUT"

echo
echo "Exporting weak-negative feedback dataset..."
python -m algorithm.src.mlops.build_feedback_dataset \
  --weak-negative-unfeedbacked \
  --output "$WEAK_OUTPUT"

echo
echo "Feedback dataset export complete."
echo "Full:          $FULL_OUTPUT"
echo "Labelled only: $LABELLED_OUTPUT"
echo "Weak labels:   $WEAK_OUTPUT"