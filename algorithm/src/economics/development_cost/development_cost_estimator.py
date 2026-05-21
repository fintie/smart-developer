from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
from algorithm.src.economics.common import money_or_none, normalise_text, safe_float


DEFAULT_ASSUMPTIONS_PATH = Path("algorithm/configs/economics/development_cost_assumptions.yaml")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class DevelopmentCostEstimator:
    def __init__(self, assumptions_path: Path | str = DEFAULT_ASSUMPTIONS_PATH):
        self.assumptions_path = Path(assumptions_path)
        self.config = load_yaml(self.assumptions_path)

    def estimate(self, site: dict[str, Any], strategy: str) -> dict[str, Any]:
        assumptions = self.config["assumptions"]

        lot_size = safe_float(site.get("lot_size_proxy_sqm"), 0.0)
        constraint_band = normalise_text(site.get("constraint_severity_band")).lower()

        fsr = float(self.config["strategy_fsr"].get(strategy, 0.8))
        build_cost = float(self.config["build_cost_per_sqm"].get(strategy, 3000))
        complexity = float(self.config["strategy_complexity_multiplier"].get(strategy, 1.0))
        constraint_multiplier = float(
            self.config["constraint_cost_multiplier"].get(constraint_band, 1.0)
        )

        escalation = float(assumptions.get("labour_material_escalation_multiplier", 1.0))
        soft_cost_ratio = float(assumptions.get("soft_cost_ratio", 0.15))
        contingency_ratio = float(assumptions.get("contingency_ratio", 0.10))

        gross_floor_area_proxy = None
        base_construction_cost = None
        estimated_development_cost = None
        soft_cost = None
        contingency = None

        if lot_size > 0 and fsr > 0 and build_cost > 0:
            gross_floor_area_proxy = lot_size * fsr
            base_construction_cost = gross_floor_area_proxy * build_cost

            estimated_development_cost = (
                base_construction_cost
                * complexity
                * constraint_multiplier
                * escalation
            )

            soft_cost = estimated_development_cost * soft_cost_ratio
            contingency = estimated_development_cost * contingency_ratio

        return {
            "gross_floor_area_proxy_sqm": round(gross_floor_area_proxy, 2)
            if gross_floor_area_proxy is not None
            else None,
            "base_construction_cost": money_or_none(base_construction_cost),
            "estimated_development_cost": money_or_none(estimated_development_cost),
            "estimated_soft_cost": money_or_none(soft_cost),
            "estimated_contingency": money_or_none(contingency),
            "development_cost_assumption_version": self.config.get("version"),
            "labour_material_escalation_multiplier": escalation,
        }

    def cost_band(self, total_project_cost: float | None, strategy: str) -> str:
        if total_project_cost is None:
            return "unknown"

        thresholds = self.config["strategy_cost_bands"].get(
            strategy,
            [2_000_000, 5_000_000, 10_000_000],
        )

        medium_threshold, high_threshold, very_high_threshold = thresholds

        if total_project_cost >= very_high_threshold:
            return "very_high"
        if total_project_cost >= high_threshold:
            return "high"
        if total_project_cost >= medium_threshold:
            return "medium"
        return "low"

    def cost_risk_score(
        self,
        total_project_cost: float | None,
        strategy: str,
        constraint_band: str,
    ) -> float:
        band = self.cost_band(total_project_cost, strategy)

        if band == "very_high":
            score = 85.0
        elif band == "high":
            score = 68.0
        elif band == "medium":
            score = 50.0
        elif band == "low":
            score = 30.0
        else:
            score = 50.0

        constraint_band = normalise_text(constraint_band).lower()

        if constraint_band in {"moderate", "medium"}:
            score += 8.0
        elif constraint_band == "high":
            score += 18.0

        return round(min(100.0, max(0.0, score)), 2)