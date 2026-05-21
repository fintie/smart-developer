from __future__ import annotations
import pandas as pd
from algorithm.src.economics.opportunity.economics_pipeline import EconomicsPipeline
from algorithm.src.policy.policy_scorer import PolicyScorer
from algorithm.src.ranking.opportunity_fusion import apply_opportunity_fusion


def main() -> None:
    df = pd.read_parquet("data/processed/retrieval/candidate_sites_geo.parquet")

    sample = df[
        (df["primary_zoning_code"].isin(["R4", "MU1", "R3"]))
        & (df["within_800m_catchment"].astype(bool))
    ].head(20)

    strategy = "low_rise_apartment"

    strategy_score_col = f"{strategy}_score"
    if strategy_score_col not in sample.columns:
        raise ValueError(f"Missing strategy score column: {strategy_score_col}")

    sample = sample.copy()
    sample["strategy_score"] = sample[strategy_score_col]

    policy_scorer = PolicyScorer(enable_rag_evidence=False)
    economics = EconomicsPipeline(use_ml_market_value=True)

    enriched = policy_scorer.score_dataframe(sample, strategy=strategy)
    enriched = economics.score_dataframe(enriched, strategy=strategy)
    fused = apply_opportunity_fusion(enriched)

    cols = [
        "RID",
        "address",
        "primary_zoning_code",
        "lot_size_band",
        "strategy_score",
        "policy_upside_score",
        "locality",
        "ml_estimated_market_value",
        "ml_value_confidence",
        "estimated_acquisition_cost",
        "estimated_acquisition_cost_source",
        "estimated_total_project_cost",
        "cost_band",
        "cost_risk_score",
        "value_potential_score",
        "value_potential_band",
        "agent_opportunity_score",
    ]

    existing_cols = [c for c in cols if c in fused.columns]
    missing_cols = [c for c in cols if c not in fused.columns]

    if missing_cols:
        print("Missing columns:", missing_cols)

    print(fused[existing_cols].head(20).to_string(index=False))
    print()
    print("Example explanation:")
    print(fused.iloc[0]["cost_value_explanation"])


if __name__ == "__main__":
    main()
