from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


DEFAULT_CONFIG_PATH = Path("algorithm/configs/economics/xgb_market_value.yaml")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def median_ape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_pipeline(config: dict[str, Any]) -> Pipeline:
    categorical = config["features"]["categorical"]
    numeric = config["features"]["numeric"]
    xgb_cfg = config["xgboost"]

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=10,
                    sparse_output=True,
                ),
            ),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical),
            ("numeric", numeric_transformer, numeric),
        ],
        remainder="drop",
    )

    model = XGBRegressor(
        n_estimators=int(xgb_cfg.get("n_estimators", 600)),
        max_depth=int(xgb_cfg.get("max_depth", 6)),
        learning_rate=float(xgb_cfg.get("learning_rate", 0.04)),
        subsample=float(xgb_cfg.get("subsample", 0.85)),
        colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.85)),
        objective=str(xgb_cfg.get("objective", "reg:squarederror")),
        tree_method=str(xgb_cfg.get("tree_method", "hist")),
        random_state=int(config["training"].get("random_state", 42)),
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def evaluate_prices(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    return {
        "mae_dollars": float(mean_absolute_error(y_true, y_pred)),
        "median_ae_dollars": float(median_absolute_error(y_true, y_pred)),
        "rmse_dollars": rmse(y_true, y_pred),
        "mape": safe_mape(y_true, y_pred),
        "median_ape": median_ape(y_true, y_pred),
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
    }


def filter_training_dates(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    df = df.copy()

    if "sale_date" not in df.columns:
        raise ValueError("Training data must contain sale_date")

    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
    df = df[df["sale_date"].notna()].copy()

    training_cfg = config.get("training", {})
    min_sale_date = training_cfg.get("min_sale_date")
    max_sale_date = training_cfg.get("max_sale_date")

    if min_sale_date:
        df = df[df["sale_date"] >= pd.Timestamp(min_sale_date)].copy()

    if max_sale_date == "today":
        today = pd.Timestamp.today().normalize()
        df = df[df["sale_date"] <= today].copy()
    elif max_sale_date:
        df = df[df["sale_date"] <= pd.Timestamp(max_sale_date)].copy()

    df = df.sort_values("sale_date").reset_index(drop=True)

    if df.empty:
        raise ValueError("No rows remaining after date filtering")

    return df


def make_train_test_split(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_cfg = config["training"]

    split_strategy = training_cfg.get("split_strategy", "time")
    test_fraction = float(training_cfg.get("test_fraction", 0.2))
    random_state = int(training_cfg.get("random_state", 42))

    if not 0.05 <= test_fraction <= 0.5:
        raise ValueError("test_fraction should be between 0.05 and 0.5")

    df = df.copy()
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
    df = df[df["sale_date"].notna()].copy()
    df = df.sort_values("sale_date").reset_index(drop=True)

    if split_strategy == "time":
        split_idx = int(len(df) * (1.0 - test_fraction))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

    elif split_strategy == "random":
        shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        split_idx = int(len(shuffled) * (1.0 - test_fraction))
        train_df = shuffled.iloc[:split_idx].copy()
        test_df = shuffled.iloc[split_idx:].copy()

    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy}")

    if len(train_df) < 1000 or len(test_df) < 1000:
        print("Warning: train/test split too small, falling back to random 80/20 split.")
        shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        split_idx = int(len(shuffled) * 0.8)
        train_df = shuffled.iloc[:split_idx].copy()
        test_df = shuffled.iloc[split_idx:].copy()

    return train_df, test_df


def validate_required_columns(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> None:
    missing = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def print_price_distribution(df: pd.DataFrame, label: str) -> None:
    if "sale_price" not in df.columns:
        return

    prices = pd.to_numeric(df["sale_price"], errors="coerce").dropna()
    if prices.empty:
        return

    print()
    print(f"{label} sale_price distribution:")
    print(f"  count:  {len(prices):,}")
    print(f"  median: ${prices.median():,.0f}")
    print(f"  mean:   ${prices.mean():,.0f}")
    print(f"  p25:    ${prices.quantile(0.25):,.0f}")
    print(f"  p75:    ${prices.quantile(0.75):,.0f}")
    print(f"  p90:    ${prices.quantile(0.90):,.0f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_yaml(config_path)

    training_path = Path(config["data"]["training_output_path"])
    artifact_dir = Path(config["model"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if not training_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {training_path}. "
            "Run: python -m algorithm.src.economics.value_model.build_sales_training_data"
        )

    df = pd.read_parquet(training_path)
    print(f"Loaded training rows: {len(df):,}")

    df = filter_training_dates(df, config)

    print(f"Filtered rows: {len(df):,}")
    print("Filtered date range:", df["sale_date"].min(), "->", df["sale_date"].max())

    categorical = config["features"]["categorical"]
    numeric = config["features"]["numeric"]
    feature_cols = categorical + numeric
    target_col = config["model"]["target_column"]

    validate_required_columns(df, feature_cols, target_col)

    train_df, test_df = make_train_test_split(df, config)

    print(f"Training rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print("Train date range:", train_df["sale_date"].min(), "->", train_df["sale_date"].max())
    print("Test date range:", test_df["sale_date"].min(), "->", test_df["sale_date"].max())

    print_price_distribution(train_df, "Train")
    print_price_distribution(test_df, "Test")

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].to_numpy()

    X_test = test_df[feature_cols]
    y_test = test_df[target_col].to_numpy()

    pipeline = build_pipeline(config)

    print()
    print("Training XGBoost market value model...")
    pipeline.fit(X_train, y_train)

    pred_train = pipeline.predict(X_train)
    pred_test = pipeline.predict(X_test)

    train_metrics = evaluate_prices(y_train, pred_train)
    test_metrics = evaluate_prices(y_test, pred_test)

    metrics = {
        "model_name": config["model"]["model_name"],
        "target": target_col,
        "config_path": str(config_path),
        "training_data_path": str(training_path),
        "n_total_after_filter": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "date_filter": {
            "min_sale_date": config.get("training", {}).get("min_sale_date"),
            "max_sale_date": config.get("training", {}).get("max_sale_date"),
        },
        "split": {
            "strategy": config.get("training", {}).get("split_strategy", "time"),
            "test_fraction": float(config.get("training", {}).get("test_fraction", 0.2)),
        },
        "train_date_range": {
            "min": str(train_df["sale_date"].min()),
            "max": str(train_df["sale_date"].max()),
        },
        "test_date_range": {
            "min": str(test_df["sale_date"].min()),
            "max": str(test_df["sale_date"].max()),
        },
        "train_sale_price_distribution": {
            "median": float(train_df["sale_price"].median()),
            "mean": float(train_df["sale_price"].mean()),
            "p25": float(train_df["sale_price"].quantile(0.25)),
            "p75": float(train_df["sale_price"].quantile(0.75)),
            "p90": float(train_df["sale_price"].quantile(0.90)),
        },
        "test_sale_price_distribution": {
            "median": float(test_df["sale_price"].median()),
            "mean": float(test_df["sale_price"].mean()),
            "p25": float(test_df["sale_price"].quantile(0.25)),
            "p75": float(test_df["sale_price"].quantile(0.75)),
            "p90": float(test_df["sale_price"].quantile(0.90)),
        },
        "train": train_metrics,
        "test": test_metrics,
        "feature_columns": feature_cols,
        "categorical_features": categorical,
        "numeric_features": numeric,
    }

    model_path = artifact_dir / "model.joblib"
    metrics_path = artifact_dir / "metrics.json"
    feature_path = artifact_dir / "feature_columns.json"
    model_card_path = artifact_dir / "model_card.json"

    joblib.dump(pipeline, model_path)
    save_json(metrics_path, metrics)
    save_json(feature_path, feature_cols)

    model_card = {
        "model_name": config["model"]["model_name"],
        "model_type": "XGBoostRegressor",
        "target": "log1p(sale_price)",
        "intended_use": (
            "Indicative NSW market value / acquisition cost proxy for "
            "Smart Developer opportunity screening."
        ),
        "not_intended_use": (
            "Formal valuation, lending assessment, legal advice, investment advice, "
            "or financial advice."
        ),
        "training_data": str(training_path),
        "artifact_path": str(model_path),
        "metrics_path": str(metrics_path),
        "feature_columns_path": str(feature_path),
        "date_filter": metrics["date_filter"],
        "split": metrics["split"],
        "test_metrics": test_metrics,
        "limitations": [
            "NSW PSI data may be incomplete, delayed, or contain unusual transactions.",
            "The current model uses suburb, postcode, date, property class, and rolling market statistics.",
            "The current model does not yet include bedrooms, bathrooms, parking, land size, building area, licensed comparable-sales features, or full parcel-level planning attributes.",
            "Predictions should be treated as indicative screening proxies only.",
        ],
    }
    save_json(model_card_path, model_card)

    print()
    print("Saved model:", model_path)
    print("Saved metrics:", metrics_path)
    print("Saved feature columns:", feature_path)
    print("Saved model card:", model_card_path)

    print()
    print("Train metrics:")
    print(json.dumps(train_metrics, indent=2))

    print()
    print("Test metrics:")
    print(json.dumps(test_metrics, indent=2))

    median_ape_value = test_metrics.get("median_ape", float("nan"))
    r2_value = test_metrics.get("r2_log", float("nan"))

    print()
    print("Model readiness check:")
    if median_ape_value <= 0.30 and r2_value > 0.20:
        print("  Status: usable for MVP screening, with clear uncertainty disclaimers.")
    elif median_ape_value <= 0.40 and r2_value > 0.0:
        print("  Status: borderline. Use only as a weak market value signal.")
    else:
        print("  Status: not ready for recommendation ranking. Keep locality median fallback.")


if __name__ == "__main__":
    main()
