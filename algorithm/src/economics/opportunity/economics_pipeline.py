from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd
from algorithm.src.economics.common import (
    band_from_score,
    extract_locality_candidates,
    money_or_none,
    normalise_upper,
    safe_float,
    safe_int,
)
from algorithm.src.economics.development_cost.development_cost_estimator import (
    DevelopmentCostEstimator,
)
from algorithm.src.economics.value_model.predict_market_value import MarketValuePredictor
from algorithm.src.economics.trend.predict_cost_trend import CostTrendPredictor
from algorithm.src.economics.trend.predict_market_trend import MarketTrendPredictor


DEFAULT_LOCALITY_SALES_PATH = Path("data/processed/cost/locality_sales_summary.parquet")

REDEVELOPMENT_STRATEGIES = {
    "low_rise_apartment",
    "townhouse_multi_dwelling",
    "assembly_opportunity",
}

ACQUISITION_MULTIPLIER_BY_STRATEGY_AND_LOT = {
    "low_rise_apartment": {
        "s": 2.0,
        "m": 4.0,
        "l": 8.0,
        "xl": 12.0,
    },
    "townhouse_multi_dwelling": {
        "s": 1.5,
        "m": 3.0,
        "l": 5.0,
        "xl": 8.0,
    },
    "assembly_opportunity": {
        "s": 2.0,
        "m": 4.0,
        "l": 8.0,
        "xl": 14.0,
    },
}


def _estimate_site_acquisition_proxy(
    *,
    strategy: str,
    site: dict[str, Any],
    locality_median: float,
    ml_market_value: float,
) -> tuple[float | None, str]:
    lot_band = str(site.get("lot_size_band") or "").lower()
    lot_size = safe_float(site.get("lot_size_proxy_sqm"), 0.0)

    if ml_market_value <= 0 and locality_median <= 0:
        return None, "unavailable"

    # For large redevelopment candidates, transaction-level ML value is usually
    # too small to represent acquisition of the whole site/assembly.
    if strategy in REDEVELOPMENT_STRATEGIES and lot_size >= 1200:
        multiplier = ACQUISITION_MULTIPLIER_BY_STRATEGY_AND_LOT.get(strategy, {}).get(
            lot_band,
            4.0,
        )

        if locality_median > 0:
            scaled = locality_median * multiplier
            if ml_market_value > 0:
                return max(scaled, ml_market_value), "scaled_locality_market_proxy"
            return scaled, "scaled_locality_market_proxy"

    if ml_market_value > 0:
        return ml_market_value, "ml_transaction_value_model"

    if locality_median > 0:
        return locality_median, "locality_median_sale_price"

    return None, "unavailable"


def _score_cost_efficiency(
    total_project_cost: float | None,
    value_potential_score: float,
    strategy: str,
) -> float:
    if total_project_cost is None or total_project_cost <= 0:
        return 50.0

    # Strategy-specific reference cost.
    reference_cost = {
        "single_dwelling_rebuild": 3_000_000,
        "granny_flat": 600_000,
        "dual_occupancy": 4_000_000,
        "townhouse_multi_dwelling": 12_000_000,
        "low_rise_apartment": 50_000_000,
        "assembly_opportunity": 45_000_000,
        "land_bank_hold": 5_000_000,
    }.get(strategy, 10_000_000)

    cost_ratio = total_project_cost / reference_cost

    if cost_ratio <= 0.75:
        cost_score = 90.0
    elif cost_ratio <= 1.0:
        cost_score = 75.0
    elif cost_ratio <= 1.5:
        cost_score = 55.0
    elif cost_ratio <= 2.0:
        cost_score = 35.0
    else:
        cost_score = 20.0

    # Blend with value potential, so cheap but low-value sites don't dominate.
    return round(0.65 * cost_score + 0.35 * value_potential_score, 2)


class EconomicsPipeline:
    def __init__(
        self,
        locality_sales_path: Path | str = DEFAULT_LOCALITY_SALES_PATH,
        use_ml_market_value: bool = False,
        market_value_predictor: MarketValuePredictor | None = None,
        use_trend_adjustment: bool = True,
        market_trend_predictor: MarketTrendPredictor | None = None,
        cost_trend_predictor: CostTrendPredictor | None = None,
    ):
        self.locality_sales_path = Path(locality_sales_path)
        self.locality_sales = self._load_locality_sales(self.locality_sales_path)
        self.development_cost_estimator = DevelopmentCostEstimator()

        self.use_ml_market_value = use_ml_market_value
        self.market_value_predictor = market_value_predictor

        if self.use_ml_market_value and self.market_value_predictor is None:
            self.market_value_predictor = MarketValuePredictor()

        self.use_trend_adjustment = use_trend_adjustment
        self.market_trend_predictor = market_trend_predictor
        self.cost_trend_predictor = cost_trend_predictor

        if self.use_trend_adjustment and self.market_trend_predictor is None:
            self.market_trend_predictor = MarketTrendPredictor()

        if self.use_trend_adjustment and self.cost_trend_predictor is None:
            self.cost_trend_predictor = CostTrendPredictor()

    def _load_locality_sales(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()

        df = pd.read_parquet(path).copy()

        if "suburb" in df.columns:
            df["suburb_norm"] = df["suburb"].astype(str).str.upper().str.strip()
        elif "locality" in df.columns:
            df["suburb_norm"] = df["locality"].astype(str).str.upper().str.strip()
        else:
            df["suburb_norm"] = ""

        return df

    def _lookup_locality_row(self, address: Any) -> tuple[str | None, dict[str, Any] | None]:
        if self.locality_sales.empty:
            return None, None

        for locality in extract_locality_candidates(address):
            matched = self.locality_sales[self.locality_sales["suburb_norm"] == locality]
            if not matched.empty:
                row = matched.sort_values("sales_count", ascending=False).iloc[0].to_dict()
                return locality, row

        candidates = extract_locality_candidates(address)
        return (candidates[0] if candidates else None), None

    def _score_value_potential(
        self,
        site: dict[str, Any],
        locality_row: dict[str, Any] | None,
        strategy: str,
    ) -> float:
        score = 30.0

        zoning = normalise_upper(site.get("primary_zoning_code"))
        zoning_band = str(site.get("zoning_band") or "").lower()
        lot_band = str(site.get("lot_size_band") or "").lower()
        constraint = str(site.get("constraint_severity_band") or "").lower()

        within_800m = bool(safe_int(site.get("within_800m_catchment"), 0))
        policy_score = safe_float(site.get("policy_upside_score"), 0.0)

        if zoning in {"R4", "MU1"} or zoning_band in {"high_dev", "mixed_use"}:
            score += 14.0
        elif zoning == "R3" or zoning_band == "med_dev":
            score += 9.0
        elif zoning == "R2":
            score += 4.0

        if lot_band == "xl":
            score += 12.0
        elif lot_band == "l":
            score += 8.0
        elif lot_band == "m":
            score += 4.0

        if within_800m:
            score += 8.0

        score += min(12.0, policy_score * 0.12)

        if locality_row is not None:
            sales_count = safe_int(locality_row.get("sales_count"), 0)
            median_price = safe_float(locality_row.get("median_sale_price"), 0.0)

            if sales_count >= 100:
                score += 5.0
            elif sales_count >= 50:
                score += 4.0
            elif sales_count >= 20:
                score += 2.0

            if median_price >= 2_000_000:
                score += 6.0
            elif median_price >= 1_200_000:
                score += 4.0
            elif median_price >= 800_000:
                score += 2.0

        if constraint in {"moderate", "medium"}:
            score -= 8.0
        elif constraint == "high":
            score -= 20.0

        if strategy == "land_bank_hold":
            score = min(score, 65.0)

        return round(min(100.0, max(0.0, score)), 2)

    def score_site(self, site: dict[str, Any], strategy: str) -> dict[str, Any]:
        address = site.get("base_site_address") or site.get("address")
        locality, locality_row = self._lookup_locality_row(address)

        market_trend_payload: dict[str, Any] = {}
        cost_trend_payload: dict[str, Any] = {}

        postcode = ""
        if locality_row is not None:
            postcode = str(locality_row.get("postcode", "") or "")

        if self.use_trend_adjustment:
            if self.market_trend_predictor is not None:
                try:
                    market_trend_payload = self.market_trend_predictor.predict(
                        suburb=locality,
                        postcode=postcode,
                    )
                except Exception as exc:
                    market_trend_payload = {
                        "predicted_market_growth_3m": 0.0,
                        "market_trend_multiplier": 1.0,
                        "market_trend_score": 50.0,
                        "market_trend_band": "unavailable",
                        "market_trend_source": "error",
                        "market_trend_model": "unavailable",
                        "market_trend_error": str(exc),
                    }

            if self.cost_trend_predictor is not None:
                try:
                    cost_trend_payload = self.cost_trend_predictor.latest()
                except Exception as exc:
                    cost_trend_payload = {
                        "predicted_construction_cost_growth_qoq": 0.0,
                        "construction_cost_escalation_multiplier": 1.0,
                        "construction_cost_trend_score": 50.0,
                        "construction_cost_trend_band": "unavailable",
                        "cost_trend_model": "unavailable",
                        "cost_trend_error": str(exc),
                    }

        enriched_site = dict(site)
        enriched_site["locality"] = locality

        ml_value_payload: dict[str, Any] = {}

        if self.use_ml_market_value and self.market_value_predictor is not None:
            try:
                ml_value_payload = self.market_value_predictor.predict(
                    site=enriched_site,
                    locality_row=locality_row,
                )
            except Exception as exc:
                ml_value_payload = {
                    "ml_estimated_market_value": None,
                    "ml_value_lower_bound": None,
                    "ml_value_upper_bound": None,
                    "ml_value_error_pct": None,
                    "ml_value_confidence": "unavailable",
                    "ml_value_model": "unavailable",
                    "ml_value_error": str(exc),
                }

        locality_median = (
            safe_float(locality_row.get("median_sale_price"), 0.0)
            if locality_row is not None
            else 0.0
        )

        ml_market_value = safe_float(ml_value_payload.get("ml_estimated_market_value"), 0.0)
        market_trend_multiplier = safe_float(
            market_trend_payload.get("market_trend_multiplier"),
            1.0,
        )

        trend_adjusted_market_value = None
        if ml_market_value > 0:
            trend_adjusted_market_value = ml_market_value * market_trend_multiplier

        acquisition, acquisition_source = _estimate_site_acquisition_proxy(
            strategy=strategy,
            site=enriched_site,
            locality_median=locality_median,
            ml_market_value=trend_adjusted_market_value or ml_market_value,
        )

        dev = self.development_cost_estimator.estimate(enriched_site, strategy=strategy)
        construction_cost_escalation_multiplier = safe_float(
            cost_trend_payload.get("construction_cost_escalation_multiplier"),
            1.0,
        )

        if construction_cost_escalation_multiplier <= 0:
            construction_cost_escalation_multiplier = 1.0

        if construction_cost_escalation_multiplier != 1.0:
            for key in [
                "base_construction_cost",
                "estimated_development_cost",
                "estimated_soft_cost",
                "estimated_contingency",
            ]:
                value = safe_float(dev.get(key), 0.0)
                if value > 0:
                    dev[key] = round(value * construction_cost_escalation_multiplier, 2)

        dev["trend_adjusted_development_cost"] = dev.get("estimated_development_cost")

        development_cost = safe_float(dev.get("estimated_development_cost"), 0.0)
        soft_cost = safe_float(dev.get("estimated_soft_cost"), 0.0)
        contingency = safe_float(dev.get("estimated_contingency"), 0.0)

        if acquisition is not None:
            total_project_cost = acquisition + development_cost + soft_cost + contingency
        elif development_cost > 0:
            total_project_cost = development_cost + soft_cost + contingency
        else:
            total_project_cost = None

        cost_band = self.development_cost_estimator.cost_band(total_project_cost, strategy)
        cost_risk = self.development_cost_estimator.cost_risk_score(
            total_project_cost,
            strategy,
            str(site.get("constraint_severity_band") or ""),
        )

        value_score = self._score_value_potential(enriched_site, locality_row, strategy)
        value_band = band_from_score(value_score)

        cost_efficiency_score = _score_cost_efficiency(
            total_project_cost=total_project_cost,
            value_potential_score=value_score,
            strategy=strategy,
        )

        explanation_parts = []

        if locality and locality_row is not None:
            explanation_parts.append(
                f"Locality sales proxy: {locality} has a median sale price of approximately "
                f"${locality_median:,.0f} based on {safe_int(locality_row.get('sales_count'), 0)} PSI sale records."
            )

        if acquisition_source == "ml_transaction_value_model":
            explanation_parts.append(
                f"Acquisition value uses the ML transaction-level market value estimate of approximately ${acquisition:,.0f}."
            )
        elif acquisition_source == "scaled_locality_market_proxy":
            explanation_parts.append(
                f"Acquisition value uses a scaled locality market proxy of approximately ${acquisition:,.0f}, "
                "because the site appears to be a larger redevelopment candidate where a single transaction-level ML estimate may understate site acquisition requirements."
            )
        elif acquisition_source == "locality_median_sale_price":
            explanation_parts.append(
                f"Acquisition value uses the locality median sale price proxy of approximately ${acquisition:,.0f}."
            )
        else:
            explanation_parts.append("Acquisition value could not be estimated from the available data.")

        if total_project_cost is not None:
            explanation_parts.append(
                f"Indicative total project cost is approximately ${total_project_cost:,.0f}, "
                f"including acquisition proxy, development cost, soft costs and contingency. Cost band: {cost_band}."
            )

        explanation_parts.append(
            f"Value potential is rated {value_band} with a score of {value_score:.1f}, based on zoning, "
            f"site scale, transport access, policy signal, local sales data and constraints."
        )

        explanation_parts.append(
            "These figures are indicative screening proxies only and are not formal valuation, quantity surveying, financial or investment advice."
        )

        return {
            "locality": locality,
            "locality_median_sale_price": money_or_none(locality_median),
            "locality_sales_count": safe_int(locality_row.get("sales_count"), 0)
            if locality_row is not None
            else None,
            "locality_price_confidence": locality_row.get("confidence")
            if locality_row is not None
            else None,
            **ml_value_payload,
            **market_trend_payload,
            "trend_adjusted_ml_market_value": money_or_none(trend_adjusted_market_value),
            "estimated_acquisition_cost": money_or_none(acquisition),
            "estimated_acquisition_cost_source": acquisition_source,
            **cost_trend_payload,
            **dev,
            "estimated_total_project_cost": money_or_none(total_project_cost),
            "cost_band": cost_band,
            "cost_risk_score": cost_risk,
            "cost_efficiency_score": cost_efficiency_score,
            "value_potential_score": value_score,
            "value_potential_band": value_band,
            "cost_value_explanation": " ".join(explanation_parts),
        }

    def score_dataframe(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        if df.empty:
            return df

        rows = []
        for record in df.to_dict(orient="records"):
            rows.append(self.score_site(record, strategy=strategy))

        econ_df = pd.DataFrame(rows, index=df.index)
        return pd.concat([df.reset_index(drop=True), econ_df.reset_index(drop=True)], axis=1)