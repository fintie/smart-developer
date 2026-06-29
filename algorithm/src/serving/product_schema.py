"""
Public field schema for the product-facing /retrieve-sites response.

When the predictor pipeline adds a new column that should be exposed to the
product frontend, add the field name here. Internal/debug-only columns should
be left out so they don't leak into the public response.

The schema is kept as a tuple (for stable ordering) and a frozenset (for fast
membership tests).
"""

from __future__ import annotations

from typing import Any

PRODUCT_RESULT_FIELDS: tuple[str, ...] = (
    "RID",
    "address",
    "base_site_address",
    "latitude",
    "longitude",
    "geometry_type",
    "geocode_source",
    "geocode_confidence",
    "agent_opportunity_score",
    "agent_rank_position",

    # Geographical factors
    "primary_zoning_code",
    "primary_zoning_class",
    "zoning_band",
    "lot_size_band",
    "lot_size_proxy_sqm",
    "constraint_severity_band",
    "station_distance_band",
    "distance_to_station_m",
    "within_800m_catchment",
    "heritage_flag",
    "flood_flag",
    "bushfire_flag",
    "top_strategy",
    "top_strategy_score",
    "strategy_score",

    # Government Policy factors
    "policy_upside_score",
    "policy_signal_band",
    "policy_matched_rules",
    "policy_matched_policies",
    "policy_matched_policy_names",
    "policy_evidence",
    "policy_evidence_count",
    "policy_explanation",

    # Value/Cost factors
    "locality",
    "locality_median_sale_price",
    "locality_sales_count",
    "locality_price_confidence",
    "ml_estimated_market_value",
    "ml_value_lower_bound",
    "ml_value_upper_bound",
    "ml_value_error_pct",
    "ml_value_confidence",
    "ml_value_model",
    "estimated_acquisition_cost",
    "estimated_acquisition_cost_source",
    "gross_floor_area_proxy_sqm",
    "base_construction_cost",
    "estimated_development_cost",
    "estimated_soft_cost",
    "estimated_contingency",
    "estimated_total_project_cost",
    "cost_band",
    "cost_risk_score",
    "cost_efficiency_score",
    "value_potential_score",
    "value_potential_band",
    "cost_value_explanation",

    "predicted_market_growth_3m",
    "market_trend_multiplier",
    "market_trend_score",
    "market_trend_band",
    "market_trend_source",
    "market_trend_model",
    "trend_adjusted_ml_market_value",
    "market_trend_raw_prediction",
    "market_trend_scaled_prediction",
    "market_trend_was_clipped",

    "construction_cost_trend_quarter",
    "predicted_construction_cost_growth_qoq",
    "construction_cost_escalation_multiplier",
    "construction_cost_trend_score",
    "construction_cost_trend_band",
    "combined_construction_cost_index",
    "cost_trend_model",
    "trend_adjusted_development_cost",

    # Explanation/Reporting
    "ranking_profile",
    "fast_explanation",
    "explanation",
    "agent_pitch",
)

PRODUCT_RESULT_FIELD_SET: frozenset[str] = frozenset(PRODUCT_RESULT_FIELDS)


def filter_product_response(response: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `response` with each result row trimmed to public fields."""
    filtered = dict(response)
    filtered["results"] = [
        {k: v for k, v in item.items() if k in PRODUCT_RESULT_FIELD_SET}
        for item in response.get("results", [])
    ]
    return filtered
