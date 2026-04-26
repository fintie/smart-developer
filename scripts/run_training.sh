#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo "Step 1/5: Building site features"
echo "============================================================"
python -m algorithm.src.features.build_features

echo "============================================================"
echo "Step 2/5: Building retrieval candidate sites"
echo "============================================================"
python -m algorithm.src.retrieval.build_candidate_sites

echo "============================================================"
echo "Step 3/5: Building training pairs"
echo "============================================================"
python -m algorithm.src.retrieval.build_training_pairs

echo "============================================================"
echo "Step 4/5: Training two-tower model"
echo "============================================================"
python -m algorithm.src.models.train_two_tower

echo "============================================================"
echo "Step 5/5: Evaluating two-tower model"
echo "============================================================"
python -m algorithm.src.models.evaluate_two_tower

echo "============================================================"
echo "Pipeline completed successfully."
echo "============================================================"