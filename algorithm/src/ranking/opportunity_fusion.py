from __future__ import annotations
from typing import Any
import pandas as pd


FUSION_WEIGHT_PROFILES = {
    "balanced": {
        "base": 0.38,
        "policy": 0.22,
        "value": 0.20,
        "cost_efficiency": 0.12,
        "budget": 0.03,
        "cost_penalty": 0.05,
    },
    "policy_upside": {
        "base": 0.32,
        "policy": 0.40,
        "value": 0.16,
        "cost_efficiency": 0.05,
        "budget": 0.02,
        "cost_penalty": 0.05,
    },
    "budget_sensitive": {
        "base": 0.30,
        "policy": 0.15,
        "value": 0.12,
        "cost_efficiency": 0.25,
        "budget": 0.08,
        "cost_penalty": 0.10,
    },
    "high_value": {
        "base": 0.32,
        "policy": 0.18,
        "value": 0.36,
        "cost_efficiency": 0.08,
        "budget": 0.03,
        "cost_penalty": 0.03,
    },
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _normalise_minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)

    min_v = float(s.min())
    max_v = float(s.max())

    if max_v <= min_v:
        return pd.Series([0.5] * len(s), index=s.index)

    return (s - min_v) / (max_v - min_v)


def _get_base_score(df: pd.DataFrame) -> pd.Series:
    """
    Product-facing base score.

    Prefer strategy_score because it is already an interpretable 0-100 fit score.
    DCN scores are useful for learned ordering, but they are often tightly clustered
    and should not be min-max normalised over a small top-k pool for product scoring.
    """
    if "strategy_score" in df.columns:
        return (
            pd.to_numeric(df["strategy_score"], errors="coerce")
            .fillna(50.0)
            .clip(0, 100)
            / 100.0
        )

    for col in ["dcn_rank_score", "dcn_prob", "fusion_score"]:
        if col in df.columns:
            return _normalise_minmax(df[col])

    return pd.Series([0.5] * len(df), index=df.index)


def _score_0_100(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default / 100.0] * len(df), index=df.index)

    return pd.to_numeric(df[col], errors="coerce").fillna(default).clip(0, 100) / 100.0


def apply_opportunity_fusion(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
    ranking_profile: str = "balanced",
) -> pd.DataFrame:
    if df.empty:
        return df

    if weights is None:
        weights = FUSION_WEIGHT_PROFILES.get(
            ranking_profile,
            FUSION_WEIGHT_PROFILES["balanced"],
        )

    out = df.copy()

    if "policy_upside_score" not in out.columns:
        out["policy_upside_score"] = 0.0

    if "value_potential_score" not in out.columns:
        out["value_potential_score"] = 50.0

    if "budget_fit_score" not in out.columns:
        out["budget_fit_score"] = 50.0

    if "cost_risk_score" not in out.columns:
        out["cost_risk_score"] = 50.0

    if "strategy_score" not in out.columns:
        if "top_strategy_score" in out.columns:
            out["strategy_score"] = out["top_strategy_score"]
        else:
            out["strategy_score"] = 50.0

    base = _get_base_score(out)
    policy = _score_0_100(out, "policy_upside_score", default=0.0)
    value = _score_0_100(out, "value_potential_score", default=50.0)
    budget = _score_0_100(out, "budget_fit_score", default=50.0)
    cost_risk = _score_0_100(out, "cost_risk_score", default=50.0)
    cost_efficiency = _score_0_100(out, "cost_efficiency_score", default=50.0)

    opportunity = (
        weights.get("base", 0.38) * base
        + weights.get("policy", 0.22) * policy
        + weights.get("value", 0.20) * value
        + weights.get("cost_efficiency", 0.12) * cost_efficiency
        + weights.get("budget", 0.03) * budget
        - weights.get("cost_penalty", 0.05) * cost_risk
    )

    out["agent_opportunity_score"] = (opportunity * 100).round(2)
    out["ranking_profile"] = ranking_profile

    sort_cols = ["agent_opportunity_score"]
    ascending = [False]

    if "policy_upside_score" in out.columns:
        sort_cols.append("policy_upside_score")
        ascending.append(False)

    if "strategy_score" in out.columns:
        sort_cols.append("strategy_score")
        ascending.append(False)

    out = out.sort_values(
        by=sort_cols,
        ascending=ascending,
        na_position="last",
    ).reset_index(drop=True)

    out["agent_rank_position"] = range(1, len(out) + 1)

    return out


def build_agent_pitch(site: dict[str, Any]) -> str:
    address = site.get("base_site_address") or site.get("address") or "This property"
    strategy_score = _safe_float(site.get("strategy_score"))
    policy_score = _safe_float(site.get("policy_upside_score"))
    policy_band = site.get("policy_signal_band") or "unknown"

    zoning = site.get("primary_zoning_code") or "available zoning"
    lot_band = site.get("lot_size_band") or "available lot size"
    constraint = site.get("constraint_severity_band") or "available constraint"

    market_value = _safe_float(site.get("ml_estimated_market_value"))
    value_confidence = site.get("ml_value_confidence")
    value_score = _safe_float(site.get("value_potential_score"))
    value_band = site.get("value_potential_band")
    cost_band = site.get("cost_band")
    total_cost = _safe_float(site.get("estimated_total_project_cost"))
    cost_efficiency = _safe_float(site.get("cost_efficiency_score"))
    ranking_profile = site.get("ranking_profile")
    evidence_count = int(_safe_float(site.get("policy_evidence_count"), 0))

    parts = [
        f"{address} may be attractive to developer or investor buyers because it combines "
        f"{zoning} zoning, a {lot_band} site profile, and {constraint} identified constraints."
    ]

    if strategy_score > 0:
        parts.append(f"The requested strategy fit score is {strategy_score:.1f}.")

    if policy_score > 0:
        parts.append(
            f"It has a {policy_band} policy signal with a policy upside score of {policy_score:.1f}, "
            "which may strengthen the redevelopment pitch."
        )
    else:
        parts.append(
            "No major policy-driven uplift signal was identified from the current structured policy screening rules."
        )

    if evidence_count > 0:
        parts.append(
            f"The policy signal is supported by {evidence_count} retrieved NSW Planning evidence snippets from official policy sources."
        )

    if market_value > 0:
        parts.append(
            f"The ML market value model estimates a typical transaction-level market value in this locality of approximately ${market_value:,.0f}"
            + (f" with {value_confidence} confidence." if value_confidence else ".")
        )

    if value_score > 0 and value_band:
        parts.append(
            f"The cost/value screen rates value potential as {value_band} with a score of {value_score:.1f}."
        )

    if total_cost > 0 and cost_band:
        parts.append(
            f"Indicative total project cost is approximately ${total_cost:,.0f}, placing it in the {cost_band} cost band."
        )

    if cost_efficiency > 0:
        parts.append(
            f"The cost efficiency score is {cost_efficiency:.1f}, reflecting the relationship between estimated project cost and value potential."
        )

    if ranking_profile == "budget_sensitive":
        parts.append(
            "Because the ranking profile is budget-sensitive, sites with stronger cost efficiency are prioritised over higher-cost opportunities."
        )
    elif ranking_profile == "policy_upside":
        parts.append(
            "Because the ranking profile emphasises policy upside, sites with stronger planning-policy signals are prioritised."
        )
    elif ranking_profile == "high_value":
        parts.append(
            "Because the ranking profile emphasises value potential, sites with stronger market and redevelopment value signals are prioritised."
        )

    parts.append(
        "This is an indicative opportunity screen only and should be verified with planning, valuation, legal, and feasibility advice before being used in a transaction."
    )

    return " ".join(parts)


def add_agent_pitch(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for item in results:
        enriched = dict(item)
        enriched["agent_pitch"] = build_agent_pitch(enriched)
        out.append(enriched)
    return out