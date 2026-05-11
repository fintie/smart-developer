#!/usr/bin/env bash
set -euo pipefail

# =========================
# Smart Developer Demo Report
# Edit these fields only
# =========================

STRATEGY="low_rise_apartment"
QUERY_TEXT="I want a site for low-rise apartment redevelopment near a train station, with high development zoning, a large site, and limited planning constraints."

TOP_K=5
RECALL_K=1000

RETRIEVAL_EXPERIMENT="two_tower_v1"
DCN_EXPERIMENT="dcn_reranker_v1"

USE_DCN_RERANKER=true
WITH_EXPLANATIONS=true
NO_DEDUPE=false

# Optional location filters.
# Leave empty string "" to disable.
LOCALITY="WAITARA"
ADDRESS_CONTAINS=""

# Report options.
AUDIENCE="developer"
TITLE="Smart Developer Site Recommendation Report"

# Leave empty string "" to print to terminal instead of saving.
OUTPUT_PATH="algorithm/artifacts/reports/demo_report.md"
OUTPUT_PDF_PATH="algorithm/artifacts/reports/demo_report.pdf"

# =========================
# Build command
# =========================

CMD=(
  python -m algorithm.demo_report
  --strategy "$STRATEGY"
  --query-text "$QUERY_TEXT"
  --top-k "$TOP_K"
  --recall-k "$RECALL_K"
  --retrieval-experiment "$RETRIEVAL_EXPERIMENT"
  --dcn-experiment "$DCN_EXPERIMENT"
  --audience "$AUDIENCE"
  --title "$TITLE"
)

if [ "$USE_DCN_RERANKER" = false ]; then
  CMD+=(--no-dcn-reranker)
fi

if [ "$WITH_EXPLANATIONS" = false ]; then
  CMD+=(--no-explanations)
fi

if [ "$NO_DEDUPE" = true ]; then
  CMD+=(--no-dedupe)
fi

if [ -n "$LOCALITY" ]; then
  CMD+=(--locality "$LOCALITY")
fi

if [ -n "$ADDRESS_CONTAINS" ]; then
  CMD+=(--address-contains "$ADDRESS_CONTAINS")
fi

if [ -n "$OUTPUT_PATH" ]; then
  CMD+=(--output "$OUTPUT_PATH")
fi

if [ -n "$OUTPUT_PDF_PATH" ]; then
  CMD+=(--output-pdf "$OUTPUT_PDF_PATH")
fi

echo "Running demo report..."
echo "Strategy:           $STRATEGY"
echo "Top K:              $TOP_K"
echo "Recall K:           $RECALL_K"
echo "Retrieval model:    $RETRIEVAL_EXPERIMENT"
echo "Use DCN reranker:   $USE_DCN_RERANKER"
echo "DCN experiment:     $DCN_EXPERIMENT"
echo "With explanations:  $WITH_EXPLANATIONS"
echo "Dedupe disabled:    $NO_DEDUPE"
echo "Locality filter:    ${LOCALITY:-none}"
echo "Address contains:   ${ADDRESS_CONTAINS:-none}"
echo "Output path:        ${OUTPUT_PATH:-terminal}"
echo "Output PDF path:    ${OUTPUT_PDF_PATH:-none}"
echo

"${CMD[@]}"