from algorithm.src.economics.opportunity.economics_pipeline import EconomicsPipeline

required_fields = [
    "ml_estimated_market_value",
    "trend_adjusted_ml_market_value",
    "market_trend_model",
    "predicted_market_growth_3m",
    "construction_cost_escalation_multiplier",
    "predicted_construction_cost_growth_qoq",
    "estimated_total_project_cost",
    "cost_efficiency_score",
    "value_potential_score",
]

site = {
    "address": "95 BONAR STREET WOLLI CREEK",
    "base_site_address": "95 BONAR STREET WOLLI CREEK",
    "primary_zoning_code": "R4",
    "lot_size_band": "xl",
    "constraint_severity_band": "low",
    "station_distance_band": "within_800m",
    "policy_upside_score": 73.5,
}

pipeline = EconomicsPipeline(
    use_ml_market_value=True,
    use_trend_adjustment=True,
)

result = pipeline.score_site(site, strategy="low_rise_apartment")

missing = [field for field in required_fields if field not in result]

if missing:
    raise RuntimeError(f"Missing fields: {missing}")

print("Economics pipeline validation OK")
for field in required_fields:
    print(field, "=", result.get(field))