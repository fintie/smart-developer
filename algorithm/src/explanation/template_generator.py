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

    if band == "over_5km" or band == "over_10km":
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