#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ABS_DOWNLOAD="${RUN_ABS_DOWNLOAD:-false}"
RUN_VALUE_MODEL="${RUN_VALUE_MODEL:-true}"
RUN_TREND_MODEL="${RUN_TREND_MODEL:-true}"
RUN_COST_INDEX="${RUN_COST_INDEX:-true}"

echo "== Smart Developer Economics Pipeline =="
echo "ROOT_DIR=$ROOT_DIR"
echo "RUN_ABS_DOWNLOAD=$RUN_ABS_DOWNLOAD"
echo "RUN_VALUE_MODEL=$RUN_VALUE_MODEL"
echo "RUN_TREND_MODEL=$RUN_TREND_MODEL"
echo "RUN_COST_INDEX=$RUN_COST_INDEX"

mkdir -p data/processed/economics/trend
mkdir -p algorithm/artifacts/economics

if [ "$RUN_ABS_DOWNLOAD" = "true" ]; then
  echo
  echo "== Download ABS WPI/PPI raw data =="
  python -m algorithm.src.economics.trend.download_abs_cost_indices --source all
else
  echo
  echo "== Skipping ABS download =="
fi

if [ "$RUN_VALUE_MODEL" = "true" ]; then
  echo
  echo "== Build sales training data =="
  python -m algorithm.src.economics.value_model.build_sales_training_data

  echo
  echo "== Train XGBoost market value model =="
  python -m algorithm.src.economics.value_model.train_xgb_value_model
else
  echo
  echo "== Skipping value model =="
fi

if [ "$RUN_TREND_MODEL" = "true" ]; then
  echo
  echo "== Build suburb monthly market features =="
  python -m algorithm.src.economics.trend.build_suburb_monthly_market

  echo
  echo "== Train rolling ridge market trend model =="
  python -m algorithm.src.economics.trend.train_market_trend_regression
else
  echo
  echo "== Skipping market trend model =="
fi

if [ "$RUN_COST_INDEX" = "true" ]; then
  echo
  echo "== Build construction cost indices =="
  python -m algorithm.src.economics.trend.build_construction_cost_indices
else
  echo
  echo "== Skipping construction cost index =="
fi

echo
echo "== Economics pipeline completed successfully =="