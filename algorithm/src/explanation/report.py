from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass
class ReportConfig:
    title: str = "Smart Developer Site Recommendation Report"
    audience: str = "developer / real estate agent"
    include_explanations: bool = True
    include_risks: bool = True
    include_table: bool = True
    include_policy: bool = True
    include_economics: bool = True
    include_policy_evidence: bool = True
    max_rows: int = 10


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _fmt_float(value: Any, digits: int = 2) -> str:
    if _is_missing(value):
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _fmt_int(value: Any) -> str:
    if _is_missing(value):
        return "N/A"
    try:
        return str(int(float(value)))
    except Exception:
        return str(value)


def _fmt_money(value: Any) -> str:
    if _is_missing(value):
        return "N/A"
    try:
        amount = float(value)
        if amount <= 0:
            return "N/A"
        return f"${amount:,.0f}"
    except Exception:
        return str(value)


def _fmt_pct(value: Any) -> str:
    if _is_missing(value):
        return "N/A"
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return str(value)


def _strategy_label(strategy: str) -> str:
    labels = {
        "single_dwelling_rebuild": "Single Dwelling Rebuild",
        "assembly_opportunity": "Assembly Opportunity",
        "granny_flat": "Granny Flat",
        "land_bank_hold": "Land Bank / Hold",
        "townhouse_multi_dwelling": "Townhouse / Multi-Dwelling",
        "low_rise_apartment": "Low-Rise Apartment",
        "dual_occupancy": "Dual Occupancy",
    }
    return labels.get(strategy, strategy.replace("_", " ").title())


def _score_col(strategy: str) -> str:
    return f"{strategy}_score"


def _risk_summary(row: pd.Series) -> list[str]:
    risks: list[str] = []

    if int(row.get("heritage_flag", 0) or 0) == 1:
        sig = row.get("heritage_max_significance")
        if not _is_missing(sig):
            risks.append(f"Heritage constraint identified ({sig}).")
        else:
            risks.append("Heritage constraint identified.")

    if int(row.get("flood_flag", 0) or 0) == 1:
        flood_class = row.get("primary_flood_class")
        if not _is_missing(flood_class):
            risks.append(f"Flood planning constraint identified ({flood_class}).")
        else:
            risks.append("Flood planning constraint identified.")

    bushfire_level = int(row.get("bushfire_risk_level", 0) or 0)
    if bushfire_level > 0:
        risks.append(f"Bushfire risk level: {bushfire_level}.")

    if int(row.get("mixed_zoning_flag", 0) or 0) == 1:
        risks.append("Mixed zoning context may require closer planning review.")

    constraint_band = row.get("constraint_severity_band")
    if not _is_missing(constraint_band) and str(constraint_band).lower() in {"moderate", "medium", "high"}:
        risks.append(f"Overall planning constraint band is {constraint_band}.")

    cost_band = row.get("cost_band")
    if not _is_missing(cost_band) and str(cost_band).lower() in {"high", "very_high"}:
        risks.append(f"Indicative project cost is in the {cost_band} band and requires detailed feasibility review.")

    cost_risk = row.get("cost_risk_score")
    try:
        if float(cost_risk) >= 70:
            risks.append(f"Cost risk score is elevated ({_fmt_float(cost_risk, 1)}).")
    except Exception:
        pass

    if not risks:
        risks.append("No major heritage, flood, bushfire, or high-cost risk flags were identified in the current feature bundle.")

    return risks


def _site_strengths(row: pd.Series, strategy: str) -> list[str]:
    strengths: list[str] = []

    zoning = row.get("primary_zoning_code")
    zoning_band = row.get("zoning_band")
    if not _is_missing(zoning):
        strengths.append(f"Zoning context: {zoning}" + (f" ({zoning_band})." if not _is_missing(zoning_band) else "."))

    lot_band = row.get("lot_size_band")
    lot_size = row.get("lot_size_proxy_sqm")
    if not _is_missing(lot_band):
        strengths.append(
            f"Site scale: {lot_band} lot-size band, approx. {_fmt_float(lot_size, 0)} sqm."
        )

    station_band = row.get("station_distance_band")
    distance = row.get("distance_to_station_m")
    if not _is_missing(station_band):
        strengths.append(
            f"Transport access: {station_band}, approx. {_fmt_float(distance, 0)} m to rail/metro station."
        )

    score_col = _score_col(strategy)
    if score_col in row.index:
        strengths.append(f"Strategy score: {_fmt_float(row.get(score_col), 1)}.")

    opportunity_score = row.get("agent_opportunity_score")
    if not _is_missing(opportunity_score):
        strengths.append(f"Agent opportunity score: {_fmt_float(opportunity_score, 1)}.")

    return strengths


def _policy_summary(row: pd.Series) -> list[str]:
    lines: list[str] = []

    policy_score = row.get("policy_upside_score")
    policy_band = row.get("policy_signal_band")
    matched_names = row.get("policy_matched_policy_names")
    matched_rules = row.get("policy_matched_rules")
    explanation = row.get("policy_explanation")

    if not _is_missing(policy_score):
        lines.append(
            f"Policy signal: {policy_band or 'detected'} with upside score {_fmt_float(policy_score, 1)}."
        )

    if isinstance(matched_names, list) and matched_names:
        lines.append("Matched policies: " + ", ".join(str(x) for x in matched_names[:5]) + ".")

    if isinstance(matched_rules, list) and matched_rules:
        lines.append("Matched policy rules: " + ", ".join(str(x) for x in matched_rules[:5]) + ".")

    if not _is_missing(explanation):
        lines.append(str(explanation))

    if not lines:
        lines.append("No major policy-driven uplift signal was identified from the structured policy screening layer.")

    return lines


def _policy_evidence_lines(row: pd.Series, max_items: int = 3) -> list[str]:
    evidence = row.get("policy_evidence")

    if not isinstance(evidence, list) or not evidence:
        return ["No retrieved policy evidence snippets were attached for this site."]

    lines: list[str] = []
    for idx, item in enumerate(evidence[:max_items], start=1):
        if not isinstance(item, dict):
            lines.append(f"{idx}. {item}")
            continue

        title = item.get("title") or item.get("source_title") or item.get("policy_name") or "Policy evidence"
        policy_id = item.get("policy_id")
        snippet = item.get("text") or item.get("snippet") or item.get("content") or ""
        url = item.get("url") or item.get("source_url")

        line = f"{idx}. **{title}**"
        if policy_id:
            line += f" (`{policy_id}`)"
        if snippet:
            line += f": {str(snippet).strip()[:350]}"
        if url:
            line += f" Source: {url}"
        lines.append(line)

    return lines


def _economics_summary(row: pd.Series) -> list[str]:
    lines: list[str] = []

    ml_value = row.get("ml_estimated_market_value")
    trend_value = row.get("trend_adjusted_ml_market_value")
    confidence = row.get("ml_value_confidence")

    if not _is_missing(ml_value):
        sentence = f"ML transaction-level market value estimate: {_fmt_money(ml_value)}"
        if not _is_missing(confidence):
            sentence += f" ({confidence} confidence)"
        sentence += "."
        lines.append(sentence)

    if not _is_missing(trend_value):
        lines.append(f"Trend-adjusted ML value estimate: {_fmt_money(trend_value)}.")

    market_band = row.get("market_trend_band")
    market_growth = row.get("predicted_market_growth_3m")
    if not _is_missing(market_band) or not _is_missing(market_growth):
        lines.append(
            f"Local market trend: {market_band or 'N/A'}, indicative 3-month movement {_fmt_pct(market_growth)}."
        )

    acquisition = row.get("estimated_acquisition_cost")
    acquisition_source = row.get("estimated_acquisition_cost_source")
    if not _is_missing(acquisition):
        lines.append(
            f"Estimated acquisition proxy: {_fmt_money(acquisition)}"
            + (f" ({acquisition_source})." if not _is_missing(acquisition_source) else ".")
        )

    dev_cost = row.get("estimated_development_cost")
    total_cost = row.get("estimated_total_project_cost")
    cost_band = row.get("cost_band")
    if not _is_missing(dev_cost):
        lines.append(f"Estimated development cost: {_fmt_money(dev_cost)}.")
    if not _is_missing(total_cost):
        lines.append(
            f"Estimated total project cost: {_fmt_money(total_cost)}"
            + (f" ({cost_band} cost band)." if not _is_missing(cost_band) else ".")
        )

    cost_trend_band = row.get("construction_cost_trend_band")
    cost_growth = row.get("predicted_construction_cost_growth_qoq")
    if not _is_missing(cost_trend_band) or not _is_missing(cost_growth):
        lines.append(
            f"Construction cost trend: {cost_trend_band or 'N/A'}, indicative next-quarter movement {_fmt_pct(cost_growth)}."
        )

    value_score = row.get("value_potential_score")
    value_band = row.get("value_potential_band")
    cost_efficiency = row.get("cost_efficiency_score")
    cost_risk = row.get("cost_risk_score")

    if not _is_missing(value_score):
        lines.append(f"Value potential: {_fmt_float(value_score, 1)}" + (f" ({value_band})." if not _is_missing(value_band) else "."))
    if not _is_missing(cost_efficiency):
        lines.append(f"Cost efficiency score: {_fmt_float(cost_efficiency, 1)}.")
    if not _is_missing(cost_risk):
        lines.append(f"Cost risk score: {_fmt_float(cost_risk, 1)}.")

    if not lines:
        lines.append("No economics layer outputs were attached for this site.")

    return lines


def _make_markdown_table(df: pd.DataFrame, strategy: str, max_rows: int = 10) -> str:
    score_col = _score_col(strategy)
    site_col = "base_site_address" if "base_site_address" in df.columns else "address"

    cols = [
        site_col,
        "primary_zoning_code",
        "lot_size_band",
        "station_distance_band",
        "constraint_severity_band",
        score_col,
        "agent_opportunity_score",
        "policy_upside_score",
        "value_potential_score",
        "cost_efficiency_score",
        "estimated_total_project_cost",
        "market_trend_band",
        "construction_cost_trend_band",
    ]

    cols = [c for c in cols if c in df.columns]

    if not cols:
        return ""

    view = df[cols].head(max_rows).copy()

    rename_map = {
        site_col: "Site",
        "primary_zoning_code": "Zoning",
        "lot_size_band": "Lot Band",
        "station_distance_band": "Station Access",
        "constraint_severity_band": "Constraint",
        score_col: "Strategy Score",
        "agent_opportunity_score": "Opportunity",
        "policy_upside_score": "Policy",
        "value_potential_score": "Value",
        "cost_efficiency_score": "Cost Eff.",
        "estimated_total_project_cost": "Total Cost",
        "market_trend_band": "Market Trend",
        "construction_cost_trend_band": "Cost Trend",
    }

    view = view.rename(columns=rename_map)

    for col in ["Strategy Score", "Opportunity", "Policy", "Value", "Cost Eff."]:
        if col in view.columns:
            view[col] = view[col].apply(lambda x: _fmt_float(x, 1))

    if "Total Cost" in view.columns:
        view["Total Cost"] = view["Total Cost"].apply(_fmt_money)

    return view.to_markdown(index=False)


def _executive_summary(results: pd.DataFrame, strategy: str, query_text: str) -> str:
    score_col = _score_col(strategy)

    if len(results) == 0:
        return (
            "No candidate sites were returned for this query. "
            "The query may need to be broadened or the strategy constraints relaxed."
        )

    top = results.iloc[0]
    mean_score = results[score_col].mean() if score_col in results.columns else None
    mean_opportunity = results["agent_opportunity_score"].mean() if "agent_opportunity_score" in results.columns else None
    mean_policy = results["policy_upside_score"].mean() if "policy_upside_score" in results.columns else None
    mean_cost_eff = results["cost_efficiency_score"].mean() if "cost_efficiency_score" in results.columns else None

    summary = [
        f"This report evaluates candidate sites for **{_strategy_label(strategy)}**.",
        f"The user intent was: _{query_text}_",
        f"The current shortlist contains **{len(results)}** recommended site(s).",
    ]

    if mean_score is not None:
        summary.append(f"The average strategy score across the shortlist is **{_fmt_float(mean_score, 1)}**.")

    if mean_opportunity is not None:
        summary.append(f"The average agent opportunity score is **{_fmt_float(mean_opportunity, 1)}**.")

    if mean_policy is not None:
        summary.append(f"The average policy upside score is **{_fmt_float(mean_policy, 1)}**.")

    if mean_cost_eff is not None:
        summary.append(f"The average cost efficiency score is **{_fmt_float(mean_cost_eff, 1)}**.")

    top_site = top.get("base_site_address", top.get("address"))
    if not _is_missing(top_site):
        summary.append(f"The top-ranked candidate site is **{top_site}**.")

    ranking_profile = top.get("ranking_profile")
    if not _is_missing(ranking_profile):
        summary.append(f"The active ranking profile is **{ranking_profile}**.")

    return " ".join(summary)


def build_site_report(
    results: pd.DataFrame,
    strategy: str,
    query_text: str,
    config: ReportConfig | None = None,
) -> str:
    """
    Build a markdown developer/investor-style report from retrieval results.

    Args:
        results: Retrieval result dataframe from HybridRetriever / predictor.
        strategy: Development strategy name.
        query_text: Original user query.
        config: Report formatting config.

    Returns:
        Markdown report string.
    """
    if config is None:
        config = ReportConfig()

    results = results.copy().reset_index(drop=True)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines: list[str] = []

    lines.append(f"# {config.title}")
    lines.append("")
    lines.append(f"**Generated at:** {generated_at}")
    lines.append(f"**Strategy:** {_strategy_label(strategy)}")
    lines.append(f"**Audience:** {config.audience}")
    lines.append("")

    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append(_executive_summary(results, strategy, query_text))
    lines.append("")

    if config.include_table:
        lines.append("## 2. Shortlisted Sites")
        lines.append("")
        table = _make_markdown_table(results, strategy, max_rows=config.max_rows)
        if table:
            lines.append(table)
        else:
            lines.append("No table columns available.")
        lines.append("")

    lines.append("## 3. Site-Level Rationale")
    lines.append("")

    if len(results) == 0:
        lines.append("No sites available for detailed rationale.")
    else:
        for idx, row in results.head(config.max_rows).iterrows():
            rank = idx + 1

            site_address = row.get("base_site_address", row.get("address", "Unknown site"))
            source_address = row.get("address", None)

            lines.append(f"### {rank}. {site_address}")

            if source_address and not _is_missing(source_address) and source_address != site_address:
                lines.append("")
                lines.append(f"_Source address example: {source_address}_")

            lines.append("")

            lines.append("**Key strengths:**")
            for strength in _site_strengths(row, strategy):
                lines.append(f"- {strength}")
            lines.append("")

            if config.include_policy:
                lines.append("**Policy and planning signal:**")
                for item in _policy_summary(row):
                    lines.append(f"- {item}")
                lines.append("")

                if config.include_policy_evidence:
                    lines.append("**Retrieved policy evidence:**")
                    for item in _policy_evidence_lines(row):
                        lines.append(f"- {item}")
                    lines.append("")

            if config.include_economics:
                lines.append("**Economics and feasibility:**")
                for item in _economics_summary(row):
                    lines.append(f"- {item}")
                lines.append("")

            if config.include_risks:
                lines.append("**Risks / checks:**")
                for risk in _risk_summary(row):
                    lines.append(f"- {risk}")
                lines.append("")

            agent_pitch = row.get("agent_pitch")
            if config.include_explanations and not _is_missing(agent_pitch):
                lines.append("**Agent-facing summary:**")
                lines.append("")
                lines.append(str(agent_pitch))
                lines.append("")
            elif config.include_explanations and "fast_explanation" in row.index and not _is_missing(row.get("fast_explanation")):
                lines.append("**Fast explanation:**")
                lines.append("")
                lines.append(str(row.get("fast_explanation")))
                lines.append("")
            elif config.include_explanations and "explanation" in row.index and not _is_missing(row.get("explanation")):
                lines.append("**Planning rationale:**")
                lines.append("")
                lines.append(str(row.get("explanation")))
                lines.append("")

    lines.append("## 4. Suggested Next Checks")
    lines.append("")
    lines.append("- Verify zoning permissibility and relevant local planning controls using official planning instruments.")
    lines.append("- Review the policy evidence snippets against the full source documents before relying on a policy uplift assumption.")
    lines.append("- Review parcel geometry, access, frontage, easements, and title constraints.")
    lines.append("- Confirm heritage, flood, bushfire, and environmental overlays through official sources.")
    lines.append("- Validate the ML market value estimate against comparable sales and a licensed valuation where required.")
    lines.append("- Validate the development cost estimate with quantity surveyor, architect, builder, or feasibility consultant input.")
    lines.append("- Stress-test feasibility under different market trend and construction cost escalation assumptions.")
    lines.append("- If proceeding, conduct planner / architect / feasibility review before acquisition or design work.")
    lines.append("")

    lines.append("## 5. Notes")
    lines.append("")
    lines.append(
        "This report is generated from the current prototype feature bundle, retrieval model, reranker, "
        "policy screening layer, policy evidence retriever, market value model, cost estimator, "
        "market trend model, and construction cost trend layer. It should be treated as a screening "
        "and prioritisation tool, not as formal planning, valuation, legal, financial, or investment advice."
    )

    return "\n".join(lines)