#!/usr/bin/env bash
set -euo pipefail

# =========================
# Smart Developer Demo Retrieval
# Edit these fields only
# =========================

EXPERIMENT="two_tower_v1"
STRATEGY="single_dwelling_rebuild"
QUERY_TEXT="I want a site for detached house redevelopment on standard residential land, with low planning constraints and a suitable lot size."

TOP_K=5
RECALL_K=200
ALPHA=0.5
BETA=0.5

WITH_EXPLANATIONS=true
USE_DCN_RERANKER=true
DCN_EXPERIMENT="dcn_reranker_v1"
NO_DEDUPE=false

# =========================
# Build command
# =========================

CMD=(
  python -m algorithm.demo_retrieval
  --experiment "$EXPERIMENT"
  --strategy "$STRATEGY"
  --query-text "$QUERY_TEXT"
  --top-k "$TOP_K"
  --recall-k "$RECALL_K"
  --alpha "$ALPHA"
  --beta "$BETA"
  --dcn-experiment "$DCN_EXPERIMENT"
)

if [ "$WITH_EXPLANATIONS" = true ]; then
  CMD+=(--with-explanations)
fi

if [ "$USE_DCN_RERANKER" = false ]; then
  CMD+=(--no-dcn-reranker)
fi

if [ "$NO_DEDUPE" = true ]; then
  CMD+=(--no-dedupe)
fi

echo "Running demo retrieval..."
echo "Experiment:        $EXPERIMENT"
echo "Strategy:          $STRATEGY"
echo "Top K:             $TOP_K"
echo "Recall K:          $RECALL_K"
echo "Alpha/Beta:        $ALPHA / $BETA"
echo "With explanations: $WITH_EXPLANATIONS"
echo "Use DCN reranker:  $USE_DCN_RERANKER"
echo "DCN experiment:    $DCN_EXPERIMENT"
echo "Dedupe disabled:   $NO_DEDUPE"
echo

"${CMD[@]}"