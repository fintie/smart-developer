from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd


DEFAULT_MARKET_PATH = Path(
    "data/processed/economics/trend/suburb_monthly_market.parquet"
)

DEFAULT_ARTIFACT_DIR = Path(
    "algorithm/artifacts/economics/market_trend_regression_v1"
)


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).upper().strip().split())


class MarketTrendPredictor:
    def __init__(
        self,
        market_path: Path | str = DEFAULT_MARKET_PATH,
        artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR,
    ):
        self.market_path = Path(market_path)
        self.artifact_dir = Path(artifact_dir)

        self.model = self._load_model()
        self.feature_payload = self._load_feature_payload()
        self.feature_columns = list(self.feature_payload["feature_columns"])
        self.prediction_clip = float(self.feature_payload.get("prediction_clip", 0.04))
        self.prediction_scale = float(self.feature_payload.get("prediction_scale", 0.6))

        self.df = self._load_market_data()

    def _load_model(self):
        model_path = self.artifact_dir / "model.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Market trend model artifact not found: {model_path}. "
                "Run: python -m algorithm.src.economics.trend.train_market_trend_regression"
            )

        return joblib.load(model_path)

    def _load_feature_payload(self) -> dict[str, Any]:
        path = self.artifact_dir / "feature_columns.json"

        if not path.exists():
            raise FileNotFoundError(f"Feature config not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_market_data(self) -> pd.DataFrame:
        if not self.market_path.exists():
            raise FileNotFoundError(f"Market trend file not found: {self.market_path}")

        df = pd.read_parquet(self.market_path).copy()

        df["month"] = pd.to_datetime(df["month"], errors="coerce")
        df["suburb"] = df["suburb"].map(clean_text)
        df["postcode"] = df["postcode"].map(clean_text)

        numeric_cols = [
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
            "target_growth_3m",
        ]

        for col in numeric_cols:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[df["month"].notna()].copy()
        df = df[df["suburb"] != ""].copy()

        return df.sort_values(["suburb", "postcode", "month"]).reset_index(drop=True)

    @staticmethod
    def _band_from_growth(growth: float) -> str:
        if growth >= 0.025:
            return "positive"
        if growth >= 0.0075:
            return "slightly_positive"
        if growth > -0.0075:
            return "stable"
        if growth > -0.025:
            return "slightly_negative"
        return "negative"

    @staticmethod
    def _score_from_growth(growth: float) -> float:
        return round(float(np.clip(50.0 + growth * 500.0, 0.0, 100.0)), 2)

    def _latest_rows_for_location(
        self,
        suburb: str | None,
        postcode: str | None,
    ) -> tuple[pd.DataFrame, str]:
        suburb_norm = clean_text(suburb)
        postcode_norm = clean_text(postcode)

        if suburb_norm:
            rows = self.df[self.df["suburb"] == suburb_norm].copy()
            if not rows.empty:
                return rows, "suburb_model"

        if postcode_norm:
            rows = self.df[self.df["postcode"] == postcode_norm].copy()
            if not rows.empty:
                return rows, "postcode_model"

        return self.df.copy(), "global_model"

    def _build_feature_row_from_history(self, rows: pd.DataFrame) -> pd.DataFrame:
        rows = rows.sort_values("month").copy()

        if rows.empty:
            return pd.DataFrame([{col: 0.0 for col in self.feature_columns}])

        latest = rows.iloc[-1].copy()
        feature_row: dict[str, float] = {}

        base_cols = [
            "sales_count",
            "median_sale_price",
            "log_median_sale_price",
            "growth_1m",
            "growth_3m",
            "rolling_median_3m",
            "rolling_median_6m",
            "rolling_sales_count_3m",
            "rolling_sales_count_6m",
        ]

        for col in base_cols:
            value = latest.get(col, 0.0)
            feature_row[col] = float(value) if pd.notna(value) else 0.0

        def lag_value(col: str, lag: int) -> float:
            if len(rows) <= lag:
                return 0.0
            value = rows.iloc[-1 - lag].get(col, 0.0)
            return float(value) if pd.notna(value) else 0.0

        for lag in [1, 2, 3]:
            feature_row[f"lag_growth_1m_{lag}"] = lag_value("growth_1m", lag)
            feature_row[f"lag_growth_3m_{lag}"] = lag_value("growth_3m", lag)
            feature_row[f"lag_log_price_{lag}"] = lag_value("log_median_sale_price", lag)
            feature_row[f"lag_sales_count_{lag}"] = lag_value("sales_count", lag)

        previous = rows.iloc[:-1].tail(3).copy()

        if previous.empty:
            feature_row["rolling_growth_1m_mean_3"] = 0.0
            feature_row["rolling_growth_3m_mean_3"] = 0.0
            feature_row["rolling_sales_count_mean_3"] = 0.0
        else:
            feature_row["rolling_growth_1m_mean_3"] = float(
                pd.to_numeric(previous["growth_1m"], errors="coerce").fillna(0.0).mean()
            )
            feature_row["rolling_growth_3m_mean_3"] = float(
                pd.to_numeric(previous["growth_3m"], errors="coerce").fillna(0.0).mean()
            )
            feature_row["rolling_sales_count_mean_3"] = float(
                pd.to_numeric(previous["sales_count"], errors="coerce").fillna(0.0).mean()
            )

        for col in self.feature_columns:
            feature_row.setdefault(col, 0.0)

        out = pd.DataFrame([feature_row])[self.feature_columns]
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def predict(
        self,
        *,
        suburb: str | None = None,
        postcode: str | None = None,
    ) -> dict[str, Any]:
        rows, source = self._latest_rows_for_location(suburb=suburb, postcode=postcode)

        feature_row = self._build_feature_row_from_history(rows)

        raw_pred = float(self.model.predict(feature_row.to_numpy(dtype=np.float32))[0])
        scaled_pred = raw_pred * self.prediction_scale
        predicted_growth = float(
            np.clip(scaled_pred, -self.prediction_clip, self.prediction_clip)
        )

        multiplier = float(np.exp(predicted_growth))
        band = self._band_from_growth(predicted_growth)
        score = self._score_from_growth(predicted_growth)

        latest_month = None
        if not rows.empty:
            latest_month = rows["month"].max()

        return {
            "predicted_market_growth_3m": round(predicted_growth, 5),
            "market_trend_multiplier": round(multiplier, 5),
            "market_trend_score": score,
            "market_trend_band": band,
            "market_trend_source": source,
            "market_trend_model": "rolling_ridge_market_trend_v1",
            "market_trend_latest_month": latest_month,
            "market_trend_raw_prediction": round(raw_pred, 5),
            "market_trend_scaled_prediction": round(scaled_pred, 5),
            "market_trend_was_clipped": bool(abs(scaled_pred) >= self.prediction_clip),
        }