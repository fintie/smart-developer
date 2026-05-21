from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd


DEFAULT_COST_INDEX_PATH = Path(
    "data/processed/economics/trend/construction_cost_indices.parquet"
)


class CostTrendPredictor:
    def __init__(self, cost_index_path: Path | str = DEFAULT_COST_INDEX_PATH):
        self.cost_index_path = Path(cost_index_path)
        self.df = self._load()

    def _load(self) -> pd.DataFrame:
        if not self.cost_index_path.exists():
            raise FileNotFoundError(
                f"Construction cost index file not found: {self.cost_index_path}"
            )

        df = pd.read_parquet(self.cost_index_path).copy()
        df["quarter"] = pd.to_datetime(df["quarter"], errors="coerce")
        df = df[df["quarter"].notna()].sort_values("quarter").reset_index(drop=True)

        if df.empty:
            raise ValueError(f"Construction cost index file is empty: {self.cost_index_path}")

        return df

    def latest(self) -> dict[str, Any]:
        row = self.df.iloc[-1].to_dict()

        growth = row.get("predicted_construction_cost_growth_qoq")
        multiplier = row.get("construction_cost_escalation_multiplier")
        score = row.get("construction_cost_trend_score")
        band = row.get("construction_cost_trend_band")

        return {
            "construction_cost_trend_quarter": row.get("quarter"),
            "predicted_construction_cost_growth_qoq": float(growth)
            if pd.notna(growth)
            else 0.0,
            "construction_cost_escalation_multiplier": float(multiplier)
            if pd.notna(multiplier)
            else 1.0,
            "construction_cost_trend_score": float(score)
            if pd.notna(score)
            else 50.0,
            "construction_cost_trend_band": str(band) if pd.notna(band) else "stable",
            "combined_construction_cost_index": float(
                row.get("combined_construction_cost_index")
            )
            if pd.notna(row.get("combined_construction_cost_index"))
            else None,
            "cost_trend_model": str(
                row.get("cost_trend_model") or "construction_cost_index_v1"
            ),
        }