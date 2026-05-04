from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Any
import pandas as pd


@dataclass
class ReportConfig:
    title: str = "Smart Developer Site Recommendation Report"
    audience: str = "developer"
    include_explanations: bool = True
    include_risks: bool = True
    include_table: bool = True


def _fmt_float(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _fmt_int(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return str(int(value))
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
        if pd.notna(sig):
            risks.append(f"Heritage constraint identified ({sig}).")
        else:
            risks.append("Heritage constraint identified.")

    if int(row.get("flood_flag", 0) or 0) == 1:
        flood_class = row.get("primary_flood_class")
        if pd.notna(flood_class):
            risks.append(f"Flood planning constraint identified ({flood_class}).")
        else:
            risks.append("Flood planning constraint identified.")

    bushfire_level = int(row.get("bushfire_risk_level", 0) or 0)
    if bushfire_level > 0:
        risks.append(f"Bushfire risk level: {bushfire_level}.")

    if int(row.get("mixed_zoning_flag", 0) or 0) == 1:
        risks.append("Mixed zoning context may require closer planning review.")

    if not risks:
        risks.append("No major heritage, flood, or bushfire constraints identified in the current feature bundle.")

    return risks


def _site_strengths(row: pd.Series, strategy: str) -> list[str]:
    strengths: list[str] = []

    zoning = row.get("primary_zoning_code")
    zoning_band = row.get("zoning_band")
    if pd.notna(zoning):
        strengths.append(f"Zoning context: {zoning} ({zoning_band}).")

    lot_band = row.get("lot_size_band")
    lot_size = row.get("lot_size_proxy_sqm")
    if pd.notna(lot_band):
        strengths.append(
            f"Site scale: {lot_band} lot-size band, approx. {_fmt_float(lot_size, 0)} sqm."
        )

    station_band = row.get("station_distance_band")
    distance = row.get("distance_to_station_m")
    if pd.notna(station_band):
        strengths.append(
            f"Transport access: {station_band}, approx. {_fmt_float(distance, 0)} m to rail/metro station."
        )

    score_col = _score_col(strategy)
    if score_col in row.index:
        strengths.append(f"Strategy score: {_fmt_float(row.get(score_col), 1)}.")

    #if "dcn_prob" in row.index and pd.notna(row.get("dcn_prob")):
     #   strengths.append(f"DCN reranker confidence: {_fmt_float(row.get('dcn_prob'), 3)}.")

    return strengths


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
    }

    view = view.rename(columns=rename_map)

    for col in ["Strategy Score", "DCN Score"]:
        if col in view.columns:
            view[col] = view[col].apply(lambda x: _fmt_float(x, 3 if col == "DCN Score" else 1))

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

    summary = [
        f"This report evaluates candidate sites for **{_strategy_label(strategy)}**.",
        f"The user intent was: _{query_text}_",
        f"The current shortlist contains **{len(results)}** recommended site(s).",
    ]

    if mean_score is not None:
        summary.append(f"The average strategy score across the shortlist is **{_fmt_float(mean_score, 1)}**.")

    top_site = top.get("base_site_address", top.get("address"))
    if pd.notna(top_site):
        summary.append(f"The top-ranked candidate site is **{top_site}**.")

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
        table = _make_markdown_table(results, strategy)
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
        for idx, row in results.iterrows():
            rank = idx + 1

            site_address = row.get("base_site_address", row.get("address", "Unknown site"))
            source_address = row.get("address", None)

            lines.append(f"### {rank}. {site_address}")

            if source_address and pd.notna(source_address) and source_address != site_address:
                lines.append("")
                lines.append(f"_Source address example: {source_address}_")

            lines.append("")

            lines.append("**Key strengths:**")
            for strength in _site_strengths(row, strategy):
                lines.append(f"- {strength}")
            lines.append("")

            if config.include_risks:
                lines.append("**Risks / checks:**")
                for risk in _risk_summary(row):
                    lines.append(f"- {risk}")
                lines.append("")

            if config.include_explanations and "explanation" in row.index and pd.notna(row.get("explanation")):
                lines.append("**Planning rationale:**")
                lines.append("")
                lines.append(str(row.get("explanation")))
                lines.append("")

    lines.append("## 4. Suggested Next Checks")
    lines.append("")
    lines.append("- Verify zoning permissibility and relevant local planning controls.")
    lines.append("- Review parcel geometry, access, frontage, easements, and title constraints.")
    lines.append("- Confirm heritage, flood, bushfire, and environmental overlays through official sources.")
    lines.append("- Compare the shortlist with recent comparable developments and market demand.")
    lines.append("- If proceeding, conduct planner / architect / feasibility review before acquisition or design work.")
    lines.append("")

    lines.append("## 5. Notes")
    lines.append("")
    lines.append(
        "This report is generated from the current prototype feature bundle, retrieval model, DCN reranker, "
        "and local explanation layer. It should be treated as a screening and prioritisation tool, not as formal planning advice."
    )

    return "\n".join(lines)