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
        "base": 0.22,
        "policy": 0.10,
        "value": 0.10,
        "cost_efficiency": 0.38,
        "budget": 0.05,
        "cost_penalty": 0.15,
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

    if "cost_efficiency_score" not in out.columns:
        out["cost_efficiency_score"] = 50.0

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

    if ranking_profile == "budget_sensitive":
        cost_band = (
            out["cost_band"].fillna("").astype(str).str.lower()
            if "cost_band" in out.columns
            else pd.Series([""] * len(out), index=out.index)
        )

        cost_risk_raw = pd.to_numeric(
            out["cost_risk_score"],
            errors="coerce",
        ).fillna(50.0)

        cost_eff_raw = pd.to_numeric(
            out["cost_efficiency_score"],
            errors="coerce",
        ).fillna(50.0)

        very_high_cost_mask = cost_band.eq("very_high") | (cost_risk_raw >= 80)
        high_cost_mask = cost_band.eq("high") | (cost_risk_raw >= 70)
        low_efficiency_mask = cost_eff_raw < 50

        opportunity = opportunity.copy()

        # Product intent: Budget-sensitive ranking should strongly demote
        # high-capital / low-efficiency opportunities even if strategy fit is high.
        opportunity = opportunity - very_high_cost_mask.astype(float) * 0.12
        opportunity = opportunity - high_cost_mask.astype(float) * 0.06
        opportunity = opportunity - low_efficiency_mask.astype(float) * 0.06

    out["agent_opportunity_score"] = (opportunity * 100).round(2)
    out["ranking_profile"] = ranking_profile

    if ranking_profile == "budget_sensitive":
        sort_cols = [
            "agent_opportunity_score",
            "cost_efficiency_score",
            "cost_risk_score",
            "estimated_total_project_cost",
            "strategy_score",
        ]
        ascending = [False, False, True, True, False]
    else:
        sort_cols = ["agent_opportunity_score"]
        ascending = [False]

        if "policy_upside_score" in out.columns:
            sort_cols.append("policy_upside_score")
            ascending.append(False)

        if "strategy_score" in out.columns:
            sort_cols.append("strategy_score")
            ascending.append(False)

    sort_cols_available = []
    ascending_available = []

    for col, asc in zip(sort_cols, ascending):
        if col in out.columns:
            sort_cols_available.append(col)
            ascending_available.append(asc)

    out = out.sort_values(
        by=sort_cols_available,
        ascending=ascending_available,
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
    trend_adjusted_market_value = _safe_float(site.get("trend_adjusted_ml_market_value"))
    value_confidence = site.get("ml_value_confidence")

    value_score = _safe_float(site.get("value_potential_score"))
    value_band = site.get("value_potential_band")

    total_cost = _safe_float(site.get("estimated_total_project_cost"))
    cost_band = site.get("cost_band")
    cost_efficiency = _safe_float(site.get("cost_efficiency_score"))

    ranking_profile = site.get("ranking_profile")
    evidence_count = int(_safe_float(site.get("policy_evidence_count"), 0))

    market_trend_band = site.get("market_trend_band")
    market_growth = _safe_float(site.get("predicted_market_growth_3m"))

    cost_trend_band = site.get("construction_cost_trend_band")
    cost_growth = _safe_float(site.get("predicted_construction_cost_growth_qoq"))

    parts: list[str] = []

    # Opening summary
    parts.append(
        f"{address} is a strong candidate for further review under the selected strategy. "
        f"It combines {zoning} zoning, a {lot_band} site profile, and {constraint} identified planning constraints."
    )

    # Strategy and policy
    if strategy_score > 0:
        parts.append(
            f"The strategy fit score is {strategy_score:.1f}, indicating a strong match against the requested development intent."
        )

    if policy_score > 0:
        policy_sentence = (
            f"The site also shows a {policy_band} policy signal, with a policy upside score of {policy_score:.1f}."
        )

        if evidence_count > 0:
            policy_sentence += (
                f" This is supported by {evidence_count} retrieved NSW Planning evidence snippets from official policy sources."
            )

        parts.append(policy_sentence)
    else:
        parts.append(
            "No major policy-driven uplift signal was identified from the current structured policy screening rules."
        )

    # Market value and local trend
    if market_value > 0:
        sentence = (
            f"The ML value model estimates a typical transaction-level market value in this locality of approximately ${market_value:,.0f}"
        )

        if value_confidence:
            sentence += f" with {value_confidence} confidence"

        if trend_adjusted_market_value > 0 and abs(trend_adjusted_market_value - market_value) > 1:
            sentence += (
                f"; after applying the local market trend adjustment, this becomes approximately ${trend_adjusted_market_value:,.0f}"
            )

        sentence += "."
        parts.append(sentence)

    if market_trend_band:
        parts.append(
            f"Recent local transaction data suggests a {market_trend_band} short-term market trend, "
            f"with an indicative 3-month movement of {market_growth * 100:.1f}%."
        )

    # Cost/value and construction trend
    if value_score > 0 and value_band:
        parts.append(
            f"The economics screen rates value potential as {value_band}, with a value score of {value_score:.1f}."
        )

    if total_cost > 0 and cost_band:
        parts.append(
            f"The indicative total project cost is approximately ${total_cost:,.0f}, placing it in the {cost_band} cost band."
        )

    if cost_trend_band:
        parts.append(
            f"Construction cost conditions are currently {cost_trend_band}, with an indicative next-quarter cost movement of {cost_growth * 100:.1f}%."
        )

    if cost_efficiency > 0:
        parts.append(
            f"The cost efficiency score is {cost_efficiency:.1f}, reflecting the balance between estimated project cost and redevelopment value potential."
        )

    # Ranking profile explanation
    if ranking_profile == "budget_sensitive":
        parts.append(
            "Because the ranking profile is budget-sensitive, this result is assessed more heavily on cost efficiency and capital intensity."
        )
    elif ranking_profile == "policy_upside":
        parts.append(
            "Because the ranking profile emphasises policy upside, this result is weighted more heavily toward planning-policy support and redevelopment uplift signals."
        )
    elif ranking_profile == "high_value":
        parts.append(
            "Because the ranking profile emphasises value potential, this result is weighted more heavily toward market value, redevelopment upside, and site-scale signals."
        )
    elif ranking_profile == "balanced":
        parts.append(
            "Under the balanced ranking profile, the final score combines strategy fit, policy upside, market value, cost efficiency, and cost risk."
        )

    # Disclaimer
    parts.append(
        "This is an indicative opportunity screen only. It should be verified with planning, valuation, legal, and feasibility advice before being used for a transaction or investment decision."
    )

    return " ".join(parts)


def add_agent_pitch(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for item in results:
        enriched = dict(item)
        enriched["agent_pitch"] = build_agent_pitch(enriched)
        out.append(enriched)
    return out