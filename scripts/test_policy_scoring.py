from __future__ import annotations
import pandas as pd
from algorithm.src.policy.policy_scorer import PolicyScorer
from algorithm.src.ranking.opportunity_fusion import apply_opportunity_fusion


STRATEGY_SCORE_COLUMNS = {
    "single_dwelling_rebuild": "single_dwelling_rebuild_score",
    "assembly_opportunity": "assembly_opportunity_score",
    "granny_flat": "granny_flat_score",
    "land_bank_hold": "land_bank_hold_score",
    "townhouse_multi_dwelling": "townhouse_multi_dwelling_score",
    "low_rise_apartment": "low_rise_apartment_score",
    "dual_occupancy": "dual_occupancy_score",
}


def attach_strategy_score(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    out = df.copy()

    score_col = STRATEGY_SCORE_COLUMNS.get(strategy)
    if score_col is None:
        raise ValueError(f"Unknown strategy: {strategy}")

    if score_col not in out.columns:
        raise KeyError(f"Missing strategy score column: {score_col}")

    out["strategy_score"] = pd.to_numeric(out[score_col], errors="coerce").fillna(0.0)
    return out


def main() -> None:
    path = "data/processed/retrieval/candidate_sites_geo.parquet"
    strategy = "low_rise_apartment"

    df = pd.read_parquet(path)
    df = attach_strategy_score(df, strategy)

    # Test on likely policy-relevant candidates.
    sample = df[
        (df["within_800m_catchment"].astype(bool))
        & (df["primary_zoning_code"].isin(["R2", "R3", "R4", "MU1"]))
    ].head(50)

    print("Sample rows:", len(sample))

    scorer = PolicyScorer()
    enriched = scorer.score_dataframe(sample, strategy=strategy)
    fused = apply_opportunity_fusion(enriched)

    cols = [
        "RID",
        "address",
        "primary_zoning_code",
        "lot_size_band",
        "within_800m_catchment",
        "constraint_severity_band",
        "strategy_score",
        "policy_upside_score",
        "policy_signal_band",
        "policy_matched_rules",
        "agent_opportunity_score",
    ]

    print(fused[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()