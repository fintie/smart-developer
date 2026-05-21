from __future__ import annotations
from typing import Any


STRATEGY_LABELS = {
    "single_dwelling_rebuild": "detached house redevelopment",
    "dual_occupancy": "dual occupancy",
    "granny_flat": "granny flat",
    "townhouse_multi_dwelling": "townhouse or multi-dwelling redevelopment",
    "low_rise_apartment": "low-rise apartment redevelopment",
    "land_bank_hold": "land banking or long-term hold",
    "assembly_opportunity": "site assembly",
}


def _yes_no_flag(value: Any) -> bool:
    if value is None:
        return False
    try:
        return int(value) == 1
    except Exception:
        return bool(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _fmt_money(value: Any) -> str | None:
    amount = _safe_float(value, 0.0)
    if amount <= 0:
        return None
    return f"${amount:,.0f}"


def _fmt_percent(value: Any) -> str | None:
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return None


def _fmt_distance(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return f"{float(value):.0f} m"
    except Exception:
        return None


def _strategy_label(strategy: str) -> str:
    return STRATEGY_LABELS.get(strategy, strategy.replace("_", " "))


def _zoning_phrase(site: dict[str, Any], strategy: str) -> str | None:
    zoning_code = site.get("primary_zoning_code")
    zoning_class = site.get("primary_zoning_class")
    zoning_band = site.get("zoning_band")

    if not zoning_code:
        return None

    if strategy == "single_dwelling_rebuild":
        if zoning_code in {"R1", "R2", "R3"} or zoning_band in {"low_dev", "medium_dev"}:
            return f"{zoning_code} zoning supports a residential redevelopment context"
        return f"{zoning_code} zoning should be reviewed for detached dwelling suitability"

    if strategy in {"low_rise_apartment", "townhouse_multi_dwelling", "assembly_opportunity"}:
        if zoning_band in {"medium_dev", "high_dev"} or zoning_code in {"R3", "R4", "MU1"}:
            return f"{zoning_code} zoning provides a supportive redevelopment context"
        return f"{zoning_code} zoning may require closer review for this intensity"

    if strategy == "granny_flat":
        return f"{zoning_code} zoning provides the starting planning context"

    if strategy == "land_bank_hold":
        return f"{zoning_code} zoning gives the site a clear planning context"

    if zoning_class:
        return f"{zoning_code} zoning ({zoning_class}) is a key planning signal"

    return f"{zoning_code} zoning is a key planning signal"


def _lot_phrase(site: dict[str, Any], strategy: str) -> str | None:
    lot_band = site.get("lot_size_band")
    lot_size = site.get("lot_size_proxy_sqm")

    if not lot_band:
        return None

    size_text = None
    try:
        if lot_size is not None:
            size_text = f"approximately {float(lot_size):.0f} sqm"
    except Exception:
        size_text = None

    if strategy == "single_dwelling_rebuild":
        if lot_band in {"m", "l", "xl"}:
            base = "the lot size appears suitable for detached house redevelopment"
        else:
            base = "the lot size may need closer feasibility review"
    elif strategy in {"low_rise_apartment", "townhouse_multi_dwelling", "assembly_opportunity"}:
        if lot_band in {"l", "xl"}:
            base = "the larger site scale may support more intensive redevelopment options"
        elif lot_band == "m":
            base = "the medium site scale may support redevelopment depending on controls"
        else:
            base = "the site scale may limit more intensive redevelopment"
    elif strategy == "granny_flat":
        base = "the lot size is relevant for secondary dwelling feasibility"
    else:
        base = "the lot size is an important feasibility signal"

    if size_text:
        return f"{base} ({size_text})"
    return base


def _transport_phrase(site: dict[str, Any]) -> str | None:
    band = site.get("station_distance_band")
    distance = _fmt_distance(site.get("distance_to_station_m"))

    if not band:
        return None

    if band == "within_800m":
        if distance:
            return f"the site has strong rail or metro accessibility at about {distance}"
        return "the site is within an 800 m rail or metro catchment"

    if band == "800m_2km":
        if distance:
            return f"the site has reasonable transport access at about {distance} from a station"
        return "the site has reasonable access to rail or metro transport"

    if band == "2km_5km":
        return "the site has moderate station access, which may be less important for some strategies"

    if band in {"over_5km", "over_10km"}:
        return "station access appears weaker and should be considered in market feasibility"

    return f"station access is classified as {band}"


def _constraint_phrases(site: dict[str, Any]) -> tuple[list[str], list[str]]:
    strengths: list[str] = []
    risks: list[str] = []

    constraint_band = site.get("constraint_severity_band")

    heritage = _yes_no_flag(site.get("heritage_flag"))
    flood = _yes_no_flag(site.get("flood_flag"))
    bushfire = _yes_no_flag(site.get("bushfire_flag"))

    if constraint_band == "low" and not any([heritage, flood, bushfire]):
        strengths.append(
            "no major heritage, flood, or bushfire constraints were identified from the available screening data"
        )
    else:
        if constraint_band:
            risks.append(f"the overall constraint level is classified as {constraint_band}")
        if heritage:
            risks.append("heritage constraints may require closer review")
        if flood:
            risks.append("flood-related constraints may require closer review")
        if bushfire:
            risks.append("bushfire-related constraints may require closer review")

    mixed_zoning = _yes_no_flag(site.get("mixed_zoning_flag"))
    if mixed_zoning:
        risks.append("mixed zoning context may require closer planning review")

    return strengths, risks


def _policy_phrase(site: dict[str, Any]) -> str | None:
    score = _safe_float(site.get("policy_upside_score"), 0.0)
    band = site.get("policy_signal_band")
    evidence_count = int(_safe_float(site.get("policy_evidence_count"), 0.0))

    if score <= 0:
        return None

    phrase = f"policy screening shows a {band or 'detected'} policy signal with an upside score of {score:.1f}"

    if evidence_count > 0:
        phrase += f", supported by {evidence_count} retrieved NSW Planning evidence snippets"

    return phrase


def _economics_phrase(site: dict[str, Any]) -> str | None:
    value_score = _safe_float(site.get("value_potential_score"), 0.0)
    value_band = site.get("value_potential_band")
    cost_efficiency = _safe_float(site.get("cost_efficiency_score"), 0.0)
    total_cost = _fmt_money(site.get("estimated_total_project_cost"))
    cost_band = site.get("cost_band")

    chunks: list[str] = []

    if value_score > 0 and value_band:
        chunks.append(f"value potential is rated {value_band} ({value_score:.1f})")

    if cost_efficiency > 0:
        chunks.append(f"cost efficiency is {cost_efficiency:.1f}")

    if total_cost and cost_band:
        chunks.append(f"indicative total project cost is {total_cost} ({cost_band} cost band)")

    if not chunks:
        return None

    return "; ".join(chunks)


def _market_trend_phrase(site: dict[str, Any]) -> str | None:
    band = site.get("market_trend_band")
    growth = _fmt_percent(site.get("predicted_market_growth_3m"))
    trend_value = _fmt_money(site.get("trend_adjusted_ml_market_value"))
    raw_value = _fmt_money(site.get("ml_estimated_market_value"))

    chunks: list[str] = []

    if band and growth:
        chunks.append(f"recent transaction data suggests a {band} short-term market trend ({growth} indicative 3-month movement)")

    if raw_value and trend_value:
        chunks.append(f"ML value estimate adjusts from {raw_value} to {trend_value} after local trend adjustment")

    if not chunks:
        return None

    return "; ".join(chunks)


def _cost_trend_phrase(site: dict[str, Any]) -> str | None:
    band = site.get("construction_cost_trend_band")
    growth = _fmt_percent(site.get("predicted_construction_cost_growth_qoq"))

    if band and growth:
        return f"construction cost conditions are {band}, with an indicative next-quarter movement of {growth}"

    return None


def build_template_explanation(site: dict[str, Any], strategy: str) -> str:
    """
    Build a fast deterministic explanation for a ranked site.

    This is designed for product site cards and API responses.
    It does not call an LLM and is safe for low-latency serving.
    """
    label = _strategy_label(strategy)

    strengths: list[str] = []
    risks: list[str] = []

    zoning = _zoning_phrase(site, strategy)
    if zoning:
        strengths.append(zoning)

    lot = _lot_phrase(site, strategy)
    if lot:
        strengths.append(lot)

    transport = _transport_phrase(site)
    if transport:
        strengths.append(transport)

    constraint_strengths, constraint_risks = _constraint_phrases(site)
    strengths.extend(constraint_strengths)
    risks.extend(constraint_risks)

    policy = _policy_phrase(site)
    economics = _economics_phrase(site)
    market_trend = _market_trend_phrase(site)
    cost_trend = _cost_trend_phrase(site)

    score = site.get("strategy_score")
    score_phrase = None
    try:
        if score is not None:
            score_phrase = f"The requested strategy fit score is {float(score):.1f}."
    except Exception:
        score_phrase = None

    if strengths:
        strength_text = "; ".join(strengths[:4])
        sentence = f"This site appears suitable for {label} because {strength_text}."
    else:
        sentence = f"This site is a candidate for {label}, but its key feasibility drivers should be reviewed."

    if policy:
        sentence += f" From a policy perspective, {policy}."

    if economics:
        sentence += f" The economics screen indicates that {economics}."

    if market_trend:
        sentence += f" Market trend layer: {market_trend}."

    if cost_trend:
        sentence += f" Cost trend layer: {cost_trend}."

    if risks:
        risk_text = "; ".join(risks[:2])
        sentence += f" Key checks include: {risk_text}."

    if score_phrase:
        sentence += f" {score_phrase}"

    return sentence


def add_template_explanations(
    results: list[dict[str, Any]],
    strategy: str,
    output_field: str = "fast_explanation",
) -> list[dict[str, Any]]:
    """
    Return a copy of result dictionaries with deterministic explanations attached.
    """
    enriched: list[dict[str, Any]] = []

    for item in results:
        new_item = dict(item)
        new_item[output_field] = build_template_explanation(new_item, strategy)
        enriched.append(new_item)

    return enriched