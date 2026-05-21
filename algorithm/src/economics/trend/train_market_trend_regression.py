from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_INPUT = Path("data/processed/economics/trend/suburb_monthly_market.parquet")
DEFAULT_ARTIFACT_DIR = Path("algorithm/artifacts/economics/market_trend_regression_v1")

MODEL_NAME = "rolling_ridge_market_trend_v1"
TARGET_COL = "target_growth_3m"
CLIPPED_TARGET_COL = "target_growth_3m_clipped"
TARGET_CLIP = 0.30
PREDICTION_CLIP = 0.04


REGRESSION_FEATURES = [
    "sales_count",
    "median_sale_price",
    "log_median_sale_price",
    "growth_1m",
    "growth_3m",
    "rolling_median_3m",
    "rolling_median_6m",
    "rolling_sales_count_3m",
    "rolling_sales_count_6m",
    "lag_growth_1m_1",
    "lag_growth_1m_2",
    "lag_growth_1m_3",
    "lag_growth_3m_1",
    "lag_growth_3m_2",
    "lag_growth_3m_3",
    "lag_log_price_1",
    "lag_log_price_2",
    "lag_log_price_3",
    "lag_sales_count_1",
    "lag_sales_count_2",
    "lag_sales_count_3",
    "rolling_growth_1m_mean_3",
    "rolling_growth_3m_mean_3",
    "rolling_sales_count_mean_3",
]


def clean_text_series(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    work["month"] = pd.to_datetime(work["month"], errors="coerce")
    work["suburb"] = clean_text_series(work["suburb"])
    work["postcode"] = clean_text_series(work["postcode"])

    numeric_base_cols = [
        "sales_count",
        "median_sale_price",
        "mean_sale_price",
        "p25_sale_price",
        "p75_sale_price",
        "log_median_sale_price",
        "growth_1m",
        "growth_3m",
        "rolling_median_3m",
        "rolling_median_6m",
        "rolling_sales_count_3m",
        "rolling_sales_count_6m",
        TARGET_COL,
    ]

    for col in numeric_base_cols:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work[work["month"].notna()].copy()
    work = work[work["suburb"] != ""].copy()
    work = work[work[TARGET_COL].notna()].copy()

    work[CLIPPED_TARGET_COL] = work[TARGET_COL].clip(-TARGET_CLIP, TARGET_CLIP)

    work = work.sort_values(["suburb", "postcode", "month"]).copy()
    group_cols = ["suburb", "postcode"]

    for lag in [1, 2, 3]:
        work[f"lag_growth_1m_{lag}"] = work.groupby(group_cols)["growth_1m"].shift(lag)
        work[f"lag_growth_3m_{lag}"] = work.groupby(group_cols)["growth_3m"].shift(lag)
        work[f"lag_log_price_{lag}"] = work.groupby(group_cols)["log_median_sale_price"].shift(lag)
        work[f"lag_sales_count_{lag}"] = work.groupby(group_cols)["sales_count"].shift(lag)

    work["rolling_growth_1m_mean_3"] = (
        work.groupby(group_cols)["growth_1m"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )

    work["rolling_growth_3m_mean_3"] = (
        work.groupby(group_cols)["growth_3m"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )

    work["rolling_sales_count_mean_3"] = (
        work.groupby(group_cols)["sales_count"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )

    for col in REGRESSION_FEATURES:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work[REGRESSION_FEATURES] = (
        work[REGRESSION_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    return work.reset_index(drop=True)


def evaluate_predictions(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    pred = np.clip(pred, -PREDICTION_CLIP, PREDICTION_CLIP)

    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred, squared=False)),
        "direction_accuracy": float(np.mean(np.sign(y_true) == np.sign(pred))),
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred)),
        "pred_min": float(np.min(pred)),
        "pred_max": float(np.max(pred)),
    }


def train_time_split(df: pd.DataFrame) -> tuple[Pipeline, dict[str, Any]]:
    split_month = df["month"].quantile(0.8)

    train = df[df["month"] <= split_month].copy()
    test = df[df["month"] > split_month].copy()

    X_train = train[REGRESSION_FEATURES].to_numpy(dtype=np.float32)
    y_train = train[CLIPPED_TARGET_COL].to_numpy(dtype=np.float32)

    X_test = test[REGRESSION_FEATURES].to_numpy(dtype=np.float32)
    y_test = test[CLIPPED_TARGET_COL].to_numpy(dtype=np.float32)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    metrics = {
        "split_type": "time_quantile_80_20",
        "split_month": pd.Timestamp(split_month).isoformat(),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_start": train["month"].min().isoformat() if len(train) else None,
        "train_end": train["month"].max().isoformat() if len(train) else None,
        "test_start": test["month"].min().isoformat() if len(test) else None,
        "test_end": test["month"].max().isoformat() if len(test) else None,
        "train": evaluate_predictions(y_train, train_pred),
        "test": evaluate_predictions(y_test, test_pred),
    }

    return model, metrics


def rolling_walk_forward_eval(
    df: pd.DataFrame,
    base_model: Pipeline,
    min_train_months: int = 12,
    train_window_months: int = 24,
) -> dict[str, Any]:
    df = df.sort_values("month").copy()
    months = sorted(df["month"].dropna().unique())

    rows: list[dict[str, Any]] = []

    for test_month_raw in months[min_train_months:]:
        test_month = pd.Timestamp(test_month_raw)
        train_start = test_month - pd.DateOffset(months=train_window_months)

        train = df[
            (df["month"] < test_month)
            & (df["month"] >= train_start)
        ].copy()

        test = df[df["month"] == test_month].copy()

        if len(train) < 200 or len(test) < 10:
            continue

        X_train = train[REGRESSION_FEATURES].to_numpy(dtype=np.float32)
        y_train = train[CLIPPED_TARGET_COL].to_numpy(dtype=np.float32)

        X_test = test[REGRESSION_FEATURES].to_numpy(dtype=np.float32)
        y_test = test[CLIPPED_TARGET_COL].to_numpy(dtype=np.float32)

        model = clone(base_model)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        pred = np.clip(pred, -PREDICTION_CLIP, PREDICTION_CLIP)

        for i, (_, row) in enumerate(test.iterrows()):
            rows.append(
                {
                    "month": row["month"],
                    "suburb": row["suburb"],
                    "postcode": row["postcode"],
                    "actual": float(y_test[i]),
                    "pred": float(pred[i]),
                }
            )

    pred_df = pd.DataFrame(rows)

    if pred_df.empty:
        return {
            "n": 0,
            "months": 0,
            "mae": None,
            "rmse": None,
            "direction_accuracy": None,
        }

    actual = pred_df["actual"].to_numpy(dtype=float)
    pred = pred_df["pred"].to_numpy(dtype=float)

    pos = pred_df[np.sign(pred_df["actual"]) > 0]
    neg = pred_df[np.sign(pred_df["actual"]) < 0]

    return {
        "n": int(len(pred_df)),
        "months": int(pred_df["month"].nunique()),
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(mean_squared_error(actual, pred, squared=False)),
        "direction_accuracy": float(np.mean(np.sign(actual) == np.sign(pred))),
        "positive_actual_count": int(len(pos)),
        "positive_direction_accuracy": float((np.sign(pos["pred"]) > 0).mean()) if len(pos) else None,
        "negative_actual_count": int(len(neg)),
        "negative_direction_accuracy": float((np.sign(neg["pred"]) < 0).mean()) if len(neg) else None,
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
        "actual_mean": float(actual.mean()),
        "actual_std": float(actual.std()),
        "pred_clip_lower_rate": float((pred <= -PREDICTION_CLIP).mean()),
        "pred_clip_upper_rate": float((pred >= PREDICTION_CLIP).mean()),
    }


def fit_final_model(df: pd.DataFrame) -> Pipeline:
    X = df[REGRESSION_FEATURES].to_numpy(dtype=np.float32)
    y = df[CLIPPED_TARGET_COL].to_numpy(dtype=np.float32)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]
    )

    model.fit(X, y)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    args = parser.parse_args()

    input_path = Path(args.input)
    artifact_dir = Path(args.artifact_dir)

    raw = pd.read_parquet(input_path)
    df = prepare_training_frame(raw)

    print(f"Loaded rows: {len(raw):,}")
    print(f"Training rows: {len(df):,}")
    print(f"Date range: {df['month'].min()} -> {df['month'].max()}")
    print(f"Suburbs: {df['suburb'].nunique():,}")

    model_for_eval, time_split_metrics = train_time_split(df)

    rolling_metrics = rolling_walk_forward_eval(
        df,
        base_model=Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        min_train_months=12,
        train_window_months=24,
    )

    final_model = fit_final_model(df)

    artifact_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, artifact_dir / "model.joblib")

    feature_payload = {
        "feature_columns": REGRESSION_FEATURES,
        "target_column": CLIPPED_TARGET_COL,
        "raw_target_column": TARGET_COL,
        "target_clip": TARGET_CLIP,
        "prediction_clip": PREDICTION_CLIP,
    }

    with (artifact_dir / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(feature_payload, f, indent=2)

    metrics = {
        "model_name": MODEL_NAME,
        "time_split": time_split_metrics,
        "rolling_walk_forward": rolling_metrics,
    }

    with (artifact_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    model_card = {
        "model_name": MODEL_NAME,
        "model_type": "Ridge regression with rolling-window walk-forward validation",
        "purpose": (
            "Estimate short-term suburb-level market momentum from recent transaction patterns. "
            "The model is used as a trend adjustment signal, not as a formal property price forecast."
        ),
        "training_data": str(input_path),
        "features": REGRESSION_FEATURES,
        "target": CLIPPED_TARGET_COL,
        "prediction_clip": [-PREDICTION_CLIP, PREDICTION_CLIP],
        "limitations": [
            "Suburb-month sales data is sparse and irregular.",
            "Monthly suburb medians can be noisy due to transaction mix.",
            "Predictions should be interpreted as indicative local market momentum only.",
            "Not suitable for valuation, lending, legal, or investment advice without professional verification.",
        ],
    }

    with (artifact_dir / "model_card.json").open("w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)

    print()
    print(f"Saved model artifact to: {artifact_dir}")
    print()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()