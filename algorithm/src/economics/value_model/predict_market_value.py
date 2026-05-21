from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from algorithm.src.economics.common import (
    extract_locality_candidates,
    normalise_upper,
    safe_float,
)


DEFAULT_MODEL_DIR = Path("algorithm/artifacts/economics/xgb_market_value_v1")


class MarketValuePredictor:
    def __init__(self, model_dir: Path | str = DEFAULT_MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "model.joblib"
        self.feature_path = self.model_dir / "feature_columns.json"
        self.metrics_path = self.model_dir / "metrics.json"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Market value model not found: {self.model_path}")

        self.pipeline = joblib.load(self.model_path)
        self.feature_columns = json.loads(self.feature_path.read_text(encoding="utf-8"))
        self.metrics = (
            json.loads(self.metrics_path.read_text(encoding="utf-8"))
            if self.metrics_path.exists()
            else {}
        )

    def build_feature_row(
        self,
        *,
        site: dict[str, Any],
        locality_row: dict[str, Any] | None,
        sale_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        if sale_date is None:
            sale_date = pd.Timestamp.today()

        address = site.get("base_site_address") or site.get("address")
        locality = site.get("locality")

        if not locality:
            candidates = extract_locality_candidates(address)
            locality = candidates[0] if candidates else ""

        postcode = ""
        if locality_row is not None:
            postcode = str(locality_row.get("postcode", "") or "")

        contract_year = str(int(sale_date.year))
        contract_month = str(int(sale_date.month))
        contract_quarter = str(int((sale_date.month - 1) // 3 + 1))

        months_since_start = (sale_date.year - 2025) * 12 + sale_date.month

        median = safe_float(locality_row.get("median_sale_price"), 0.0) if locality_row else 0.0
        sales_count = safe_float(locality_row.get("sales_count"), 0.0) if locality_row else 0.0
        p25 = safe_float(locality_row.get("p25_sale_price"), median) if locality_row else median
        p75 = safe_float(locality_row.get("p75_sale_price"), median) if locality_row else median

        row = {
            "suburb": normalise_upper(locality),
            "postcode": normalise_upper(postcode),
            "property_class_code": "R",
            "property_class": "RESIDENCE",
            "contract_year": contract_year,
            "contract_month": contract_month,
            "contract_quarter": contract_quarter,
            "months_since_start": months_since_start,
            "suburb_sales_count_12m": sales_count,
            "suburb_median_price_12m": median,
            "suburb_mean_price_12m": median,
            "suburb_p25_price_12m": p25,
            "suburb_p75_price_12m": p75,
            "postcode_sales_count_12m": sales_count,
            "postcode_median_price_12m": median,
        }

        return pd.DataFrame([{col: row.get(col) for col in self.feature_columns}])

    def predict(
        self,
        *,
        site: dict[str, Any],
        locality_row: dict[str, Any] | None,
    ) -> dict[str, Any]:
        X = self.build_feature_row(site=site, locality_row=locality_row)

        pred_log = float(self.pipeline.predict(X)[0])
        pred_value = float(np.expm1(pred_log))

        test_metrics = self.metrics.get("test", {})
        error_pct = float(
            test_metrics.get("median_ape")
            or test_metrics.get("mape")
            or 0.30
        )

        lower = pred_value * (1.0 - error_pct)
        upper = pred_value * (1.0 + error_pct)

        if error_pct <= 0.15:
            confidence = "high"
        elif error_pct <= 0.30:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "ml_estimated_market_value": round(pred_value, 2),
            "ml_value_lower_bound": round(max(0.0, lower), 2),
            "ml_value_upper_bound": round(upper, 2),
            "ml_value_error_pct": round(error_pct, 4),
            "ml_value_confidence": confidence,
            "ml_value_model": self.model_dir.name,
        }