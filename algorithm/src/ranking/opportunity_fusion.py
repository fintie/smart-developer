from __future__ import annotations
from typing import Any
import pandas as pd


DEFAULT_FUSION_WEIGHTS = {
    "base": 0.55,
    "policy": 0.35,
    "value": 0.05,
    "budget": 0.03,
    "cost_penalty": 0.02,
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
) -> pd.DataFrame:
    """
    Adds agent_opportunity_score and reranks rows.

    Policy score is expected to come from PolicyScorer.
    Cost/value scores can be added later. Missing value/cost/budget scores are
    handled with neutral defaults so this function can be used immediately.
    """
    if df.empty:
        return df

    weights = weights or DEFAULT_FUSION_WEIGHTS
    out = df.copy()

    base = _get_base_score(out)
    policy = _score_0_100(out, "policy_upside_score", default=0.0)
    value = _score_0_100(out, "value_potential_score", default=50.0)
    budget = _score_0_100(out, "budget_fit_score", default=50.0)
    cost_risk = _score_0_100(out, "cost_risk_score", default=50.0)

    opportunity = (
        weights.get("base", 0.50) * base
        + weights.get("policy", 0.25) * policy
        + weights.get("value", 0.15) * value
        + weights.get("budget", 0.05) * budget
        - weights.get("cost_penalty", 0.05) * cost_risk
    )

    out["agent_opportunity_score"] = (opportunity * 100).round(2)

    sort_cols = ["agent_opportunity_score", "policy_upside_score"]
    ascending = [False, False]

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

    parts = [
        f"{address} may be attractive to developer or investor buyers because it combines {zoning} zoning, a {lot_band} site profile, and {constraint} identified constraints.",
    ]

    if strategy_score > 0:
        parts.append(f"The requested strategy fit score is {strategy_score:.1f}.")

    if policy_score > 0:
        parts.append(
            f"It also has a {policy_band} policy signal with a policy upside score of {policy_score:.1f}, which may strengthen the agent pitch around redevelopment upside."
        )
    else:
        parts.append(
            "No major policy-driven uplift signal was identified from the current structured policy screening rules."
        )

    parts.append(
        "This is an indicative opportunity screen only and should be verified with planning, legal, and feasibility advice before being used in a transaction."
    )

    return " ".join(parts)


def add_agent_pitch(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for item in results:
        enriched = dict(item)
        enriched["agent_pitch"] = build_agent_pitch(enriched)
        out.append(enriched)
    return out