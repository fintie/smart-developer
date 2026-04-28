#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Config
# Can be overridden, e.g.
# TWO_TOWER_EXPERIMENT=two_tower_v1 DCN_EXPERIMENT=dcn_reranker_v1 ./scripts/run_training.sh
# ============================================================

TWO_TOWER_EXPERIMENT="${TWO_TOWER_EXPERIMENT:-two_tower_v1}"
DCN_EXPERIMENT="${DCN_EXPERIMENT:-dcn_reranker_v1}"
RECALL_EXPERIMENT="${RECALL_EXPERIMENT:-two_tower_v1}"

RUN_FEATURES="${RUN_FEATURES:-true}"
RUN_SCORING="${RUN_SCORING:-true}"
RUN_CANDIDATES="${RUN_CANDIDATES:-true}"
RUN_TRAINING_PAIRS="${RUN_TRAINING_PAIRS:-true}"
RUN_TWO_TOWER_TRAIN="${RUN_TWO_TOWER_TRAIN:-true}"
RUN_TWO_TOWER_EVAL="${RUN_TWO_TOWER_EVAL:-true}"
RUN_RERANKER_DATASET="${RUN_RERANKER_DATASET:-true}"
RUN_DCN_TRAIN="${RUN_DCN_TRAIN:-true}"
RUN_DCN_EVAL="${RUN_DCN_EVAL:-true}"

echo "============================================================"
echo "Smart Developer training pipeline"
echo "============================================================"
echo "Two-tower experiment : $TWO_TOWER_EXPERIMENT"
echo "DCN experiment       : $DCN_EXPERIMENT"
echo "Recall experiment    : $RECALL_EXPERIMENT"
echo "============================================================"

# ------------------------------------------------------------
# Step 1
# ------------------------------------------------------------
if [ "$RUN_FEATURES" = true ]; then
  echo
  echo "============================================================"
  echo "Step 1/9: Building site features"
  echo "============================================================"
  python -m algorithm.src.features.build_features
else
  echo "Skipping Step 1/9: build_features"
fi

# ------------------------------------------------------------
# Step 2
# ------------------------------------------------------------
if [ "$RUN_SCORING" = true ]; then
  echo
  echo "============================================================"
  echo "Step 2/9: Building strategy scores"
  echo "============================================================"
  python -m algorithm.src.scoring.scoring
else
  echo "Skipping Step 2/9: scoring"
fi

# ------------------------------------------------------------
# Step 3
# ------------------------------------------------------------
if [ "$RUN_CANDIDATES" = true ]; then
  echo
  echo "============================================================"
  echo "Step 3/9: Building retrieval candidate sites"
  echo "============================================================"
  python -m algorithm.src.retrieval.build_candidate_sites
else
  echo "Skipping Step 3/9: build_candidate_sites"
fi

# ------------------------------------------------------------
# Step 4
# ------------------------------------------------------------
if [ "$RUN_TRAINING_PAIRS" = true ]; then
  echo
  echo "============================================================"
  echo "Step 4/9: Building training pairs"
  echo "============================================================"
  python -m algorithm.src.retrieval.build_training_pairs
else
  echo "Skipping Step 4/9: build_training_pairs"
fi

# ------------------------------------------------------------
# Step 5
# ------------------------------------------------------------
if [ "$RUN_TWO_TOWER_TRAIN" = true ]; then
  echo
  echo "============================================================"
  echo "Step 5/9: Training two-tower model"
  echo "============================================================"
  python -m algorithm.src.models.train_two_tower_v1 --experiment "$TWO_TOWER_EXPERIMENT"
else
  echo "Skipping Step 5/9: train_two_tower_v1"
fi

# ------------------------------------------------------------
# Step 6
# ------------------------------------------------------------
if [ "$RUN_TWO_TOWER_EVAL" = true ]; then
  echo
  echo "============================================================"
  echo "Step 6/9: Evaluating two-tower model"
  echo "============================================================"
  python -m algorithm.src.models.evaluate_two_tower --experiment "$TWO_TOWER_EXPERIMENT"
else
  echo "Skipping Step 6/9: evaluate_two_tower"
fi

# ------------------------------------------------------------
# Step 7
# ------------------------------------------------------------
if [ "$RUN_RERANKER_DATASET" = true ]; then
  echo
  echo "============================================================"
  echo "Step 7/9: Building reranker dataset"
  echo "============================================================"
  python -m algorithm.src.retrieval.build_reranker_dataset \
    --experiment "$DCN_EXPERIMENT" \
    --recall-experiment "$RECALL_EXPERIMENT"
else
  echo "Skipping Step 7/9: build_reranker_dataset"
fi

# ------------------------------------------------------------
# Step 8
# ------------------------------------------------------------
if [ "$RUN_DCN_TRAIN" = true ]; then
  echo
  echo "============================================================"
  echo "Step 8/9: Training DCN reranker"
  echo "============================================================"
  python -m algorithm.src.models.train_dcn_reranker --experiment "$DCN_EXPERIMENT"
else
  echo "Skipping Step 8/9: train_dcn_reranker"
fi

# ------------------------------------------------------------
# Step 9
# ------------------------------------------------------------
if [ "$RUN_DCN_EVAL" = true ]; then
  echo
  echo "============================================================"
  echo "Step 9/9: Evaluating DCN reranker"
  echo "============================================================"
  python -m algorithm.src.models.evaluate_dcn_reranker \
    --experiment "$DCN_EXPERIMENT" \
    --recall-experiment "$RECALL_EXPERIMENT"
else
  echo "Skipping Step 9/9: evaluate_dcn_reranker"
fi

echo
echo "============================================================"
echo "Pipeline completed successfully."
echo "============================================================"