from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pandas as pd


DEFAULT_LOCALITY_SALES_PATH = Path("data/processed/cost/locality_sales_summary.parquet")


ASSUMED_FSR_BY_STRATEGY = {
    "single_dwelling_rebuild": 0.45,
    "granny_flat": 0.25,
    "dual_occupancy": 0.60,
    "townhouse_multi_dwelling": 0.85,
    "low_rise_apartment": 1.50,
    "assembly_opportunity": 1.30,
    "land_bank_hold": 0.0,
}


BUILD_COST_PER_SQM = {
    "single_dwelling_rebuild": 2600,
    "granny_flat": 2400,
    "dual_occupancy": 2800,
    "townhouse_multi_dwelling": 3200,
    "low_rise_apartment": 3800,
    "assembly_opportunity": 3500,
    "land_bank_hold": 0,
}


STRATEGY_COMPLEXITY_MULTIPLIER = {
    "single_dwelling_rebuild": 1.00,
    "granny_flat": 0.90,
    "dual_occupancy": 1.05,
    "townhouse_multi_dwelling": 1.15,
    "low_rise_apartment": 1.25,
    "assembly_opportunity": 1.20,
    "land_bank_hold": 0.10,
}


CONSTRAINT_COST_MULTIPLIER = {
    "none": 1.00,
    "low": 1.00,
    "moderate": 1.10,
    "medium": 1.10,
    "high": 1.25,
}


SOFT_COST_RATIO = 0.15
CONTINGENCY_RATIO = 0.10


@dataclass(frozen=True)
class CostValueResult:
    locality: str | None
    locality_median_sale_price: float | None
    locality_sales_count: int | None
    locality_price_confidence: str | None
    estimated_acquisition_cost: float | None
    estimated_development_cost: float | None
    estimated_total_project_cost: float | None
    cost_band: str
    cost_risk_score: float
    value_potential_score: float
    value_potential_band: str
    cost_value_explanation: str


def _normalise_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalise_upper(value: Any) -> str:
    return _normalise_text(value).upper()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def _money_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    if value <= 0:
        return None
    return round(float(value), 2)


def extract_locality_from_address(address: Any) -> str | None:
    """
    Very simple NSW-style address locality extractor.

    Assumption:
      candidate address often ends with suburb/locality, e.g.
      '21-37 WAITARA AVENUE WAITARA'

    This is not perfect, but it is sufficient for MVP because the PSI summary
    is keyed by suburb/locality.
    """
    text = _normalise_upper(address)
    if not text:
        return None

    # Remove unit-like prefix noise but preserve final locality tokens.
    text = re.sub(r"\s+", " ", text).strip()

    # Known issue: street names can contain locality-like words.
    # MVP heuristic: use the last 1-3 uppercase word tokens and try lookup later.
    tokens = re.findall(r"[A-Z][A-Z\-']+", text)
    if not tokens:
        return None

    # Return full tail candidates later in lookup.
    return " ".join(tokens[-3:])


def _cost_band(total_project_cost: float | None) -> str:
    if total_project_cost is None:
        return "unknown"
    if total_project_cost >= 10_000_000:
        return "very_high"
    if total_project_cost >= 5_000_000:
        return "high"
    if total_project_cost >= 2_000_000:
        return "medium"
    return "low"


def _score_cost_risk(total_project_cost: float | None, constraint_band: str, strategy: str) -> float:
    if total_project_cost is None:
        return 50.0

    if total_project_cost >= 10_000_000:
        score = 85.0
    elif total_project_cost >= 5_000_000:
        score = 70.0
    elif total_project_cost >= 2_000_000:
        score = 50.0
    else:
        score = 30.0

    constraint_band = constraint_band.lower()
    if constraint_band in {"moderate", "medium"}:
        score += 8.0
    elif constraint_band == "high":
        score += 18.0

    if strategy in {"low_rise_apartment", "assembly_opportunity"}:
        score += 5.0

    return round(min(100.0, max(0.0, score)), 2)


def _band_from_value_score(score: float) -> str:
    if score >= 75:
        return "very_high"
    if score >= 60:
        return "high"
    if score >= 40:
        return "medium"
    if score > 0:
        return "low"
    return "unknown"


def _score_value_potential(site: dict[str, Any], locality_row: dict[str, Any] | None, strategy: str) -> float:
    score = 40.0

    zoning = _normalise_upper(site.get("primary_zoning_code"))
    zoning_band = _normalise_text(site.get("zoning_band")).lower()
    lot_band = _normalise_text(site.get("lot_size_band")).lower()
    constraint = _normalise_text(site.get("constraint_severity_band")).lower()

    within_800m = bool(_safe_int(site.get("within_800m_catchment"), 0))
    policy_score = _safe_float(site.get("policy_upside_score"), 0.0)

    if zoning in {"R4", "MU1"} or zoning_band in {"high_dev", "mixed_use"}:
        score += 18.0
    elif zoning in {"R3"} or zoning_band == "med_dev":
        score += 12.0
    elif zoning in {"R2"}:
        score += 6.0

    if lot_band == "xl":
        score += 14.0
    elif lot_band == "l":
        score += 10.0
    elif lot_band == "m":
        score += 5.0

    if within_800m:
        score += 10.0

    score += min(15.0, policy_score * 0.15)

    if locality_row is not None:
        sales_count = _safe_int(locality_row.get("sales_count"), 0)
        median_price = _safe_float(locality_row.get("median_sale_price"), 0.0)

        if sales_count >= 50:
            score += 5.0
        elif sales_count >= 20:
            score += 3.0

        if median_price >= 1_500_000:
            score += 8.0
        elif median_price >= 1_000_000:
            score += 5.0
        elif median_price >= 750_000:
            score += 3.0

    if constraint in {"moderate", "medium"}:
        score -= 8.0
    elif constraint == "high":
        score -= 20.0

    if strategy == "land_bank_hold":
        score = min(score, 65.0)

    return round(min(100.0, max(0.0, score)), 2)


def _build_explanation(
    *,
    locality: str | None,
    locality_row: dict[str, Any] | None,
    strategy: str,
    acquisition: float | None,
    development: float | None,
    total: float | None,
    cost_band: str,
    value_score: float,
    value_band: str,
) -> str:
    parts = []

    if locality and locality_row is not None:
        sales_count = _safe_int(locality_row.get("sales_count"), 0)
        median = _safe_float(locality_row.get("median_sale_price"), 0.0)
        parts.append(
            f"Locality sales proxy: {locality} has a median sale price of approximately ${median:,.0f} "
            f"based on {sales_count} recent PSI sale records."
        )
    elif locality:
        parts.append(
            f"No matching PSI locality sales summary was found for {locality}; cost/value estimates use conservative fallback assumptions."
        )
    else:
        parts.append(
            "No locality could be reliably extracted from the address; cost/value estimates use conservative fallback assumptions."
        )

    if total is not None:
        parts.append(
            f"Indicative project cost is estimated at approximately ${total:,.0f}, including acquisition proxy, "
            f"construction proxy, soft costs, and contingency. Cost band: {cost_band}."
        )
    else:
        parts.append(
            "Indicative total project cost could not be estimated due to missing lot size or cost assumptions."
        )

    parts.append(
        f"Value potential is rated {value_band} with a score of {value_score:.1f}, based on zoning, site scale, "
        f"transport access, policy signal, local sales proxy, and planning constraints."
    )

    parts.append(
        "These figures are indicative screening proxies only and are not formal valuation, feasibility, quantity surveying, or financial advice."
    )

    return " ".join(parts)


class CostValueEstimator:
    def __init__(self, locality_sales_path: Path | str = DEFAULT_LOCALITY_SALES_PATH):
        self.locality_sales_path = Path(locality_sales_path)
        self.locality_sales = self._load_locality_sales(self.locality_sales_path)

    def _load_locality_sales(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()

        df = pd.read_parquet(path)
        df = df.copy()

        # Expected columns from current summary:
        # suburb, postcode, sales_count, median_sale_price, confidence, ...
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

        text = _normalise_upper(address)
        tokens = re.findall(r"[A-Z][A-Z\-']+", text)

        # Try longest tail first: e.g. "SYDNEY OLYMPIC PARK", then "OLYMPIC PARK", then "PARK".
        candidates = []
        for n in [4, 3, 2, 1]:
            if len(tokens) >= n:
                candidates.append(" ".join(tokens[-n:]))

        for locality in candidates:
            matched = self.locality_sales[self.locality_sales["suburb_norm"] == locality]
            if not matched.empty:
                row = matched.sort_values("sales_count", ascending=False).iloc[0].to_dict()
                return locality, row

        return extract_locality_from_address(address), None

    def score_site(self, site: dict[str, Any], strategy: str) -> dict[str, Any]:
        address = site.get("base_site_address") or site.get("address")
        locality, locality_row = self._lookup_locality_row(address)

        lot_size = _safe_float(site.get("lot_size_proxy_sqm"), 0.0)
        constraint_band = _normalise_text(site.get("constraint_severity_band")).lower()

        fsr = ASSUMED_FSR_BY_STRATEGY.get(strategy, 0.8)
        build_cost = BUILD_COST_PER_SQM.get(strategy, 3000)
        complexity = STRATEGY_COMPLEXITY_MULTIPLIER.get(strategy, 1.0)
        constraint_multiplier = CONSTRAINT_COST_MULTIPLIER.get(constraint_band, 1.0)

        acquisition: float | None = None
        if locality_row is not None:
            acquisition = _safe_float(locality_row.get("median_sale_price"), 0.0)
            if acquisition <= 0:
                acquisition = None

        development: float | None = None
        total: float | None = None

        if lot_size > 0 and build_cost > 0 and fsr > 0:
            gross_floor_area_proxy = lot_size * fsr
            development = gross_floor_area_proxy * build_cost * complexity * constraint_multiplier

            soft_cost = SOFT_COST_RATIO * development
            contingency = CONTINGENCY_RATIO * development

            if acquisition is not None:
                total = acquisition + development + soft_cost + contingency
            else:
                total = development + soft_cost + contingency

        cost_band = _cost_band(total)
        cost_risk = _score_cost_risk(total, constraint_band, strategy)
        value_score = _score_value_potential(site, locality_row, strategy)
        value_band = _band_from_value_score(value_score)

        explanation = _build_explanation(
            locality=locality,
            locality_row=locality_row,
            strategy=strategy,
            acquisition=acquisition,
            development=development,
            total=total,
            cost_band=cost_band,
            value_score=value_score,
            value_band=value_band,
        )

        return {
            "locality": locality,
            "locality_median_sale_price": _money_or_none(acquisition),
            "locality_sales_count": _safe_int(locality_row.get("sales_count"), 0) if locality_row else None,
            "locality_price_confidence": locality_row.get("confidence") if locality_row else None,
            "estimated_acquisition_cost": _money_or_none(acquisition),
            "estimated_development_cost": _money_or_none(development),
            "estimated_total_project_cost": _money_or_none(total),
            "cost_band": cost_band,
            "cost_risk_score": cost_risk,
            "value_potential_score": value_score,
            "value_potential_band": value_band,
            "cost_value_explanation": explanation,
        }

    def score_dataframe(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        if df.empty:
            return df

        rows = []
        for record in df.to_dict(orient="records"):
            rows.append(self.score_site(record, strategy=strategy))

        cost_df = pd.DataFrame(rows, index=df.index)
        return pd.concat([df.reset_index(drop=True), cost_df.reset_index(drop=True)], axis=1)