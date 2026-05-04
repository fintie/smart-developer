from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any


STRATEGIES = [
    "single_dwelling_rebuild",
    "assembly_opportunity",
    "granny_flat",
    "land_bank_hold",
    "townhouse_multi_dwelling",
    "low_rise_apartment",
    "dual_occupancy",
]


@dataclass
class QueryPlan:
    original_query: str
    input_strategy: str | None
    selected_strategy: str
    sanitised_query: str
    rewritten_query: str
    warnings: list[str]
    suggested_alternatives: list[str]
    matched_signals: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _contains_any(text: str, terms: list[str]) -> list[str]:
    text_l = text.lower()
    return [term for term in terms if term in text_l]


STRATEGY_SIGNAL_TERMS: dict[str, list[str]] = {
    "single_dwelling_rebuild": [
        "house",
        "detached",
        "single dwelling",
        "knockdown rebuild",
        "rebuild",
        "standard residential",
        "family home",
    ],
    "dual_occupancy": [
        "dual occupancy",
        "duplex",
        "dual occ",
        "two dwellings",
        "two homes",
        "infill",
    ],
    "granny_flat": [
        "granny flat",
        "secondary dwelling",
        "backyard dwelling",
        "additional dwelling",
    ],
    "townhouse_multi_dwelling": [
        "townhouse",
        "townhouses",
        "multi dwelling",
        "multi-dwelling",
        "terrace",
        "medium density",
    ],
    "low_rise_apartment": [
        "apartment",
        "apartments",
        "low-rise apartment",
        "low rise apartment",
        "unit development",
        "higher density",
        "high density",
        "near station",
        "train station",
        "metro",
        "high development zoning",
    ],
    "land_bank_hold": [
        "land bank",
        "landbank",
        "hold",
        "future upside",
        "long-term",
        "long term",
        "strategic",
        "future development",
    ],
    "assembly_opportunity": [
        "assembly",
        "amalgamation",
        "site assembly",
        "aggregate",
        "adjoining lots",
        "neighbouring lots",
        "multiple lots",
    ],
}


STRATEGY_PRIORS: dict[str, str] = {
    "single_dwelling_rebuild": (
        "strategy single_dwelling_rebuild | detached house rebuild | "
        "standard residential land preferred | low constraint preferred | "
        "suitable lot size | station access helpful but not required"
    ),
    "dual_occupancy": (
        "strategy dual_occupancy | residential infill | "
        "moderate to large lot size preferred | low constraint preferred | "
        "low to medium density residential context"
    ),
    "granny_flat": (
        "strategy granny_flat | secondary dwelling | "
        "existing residential land | suitable backyard or lot size | "
        "low constraint preferred"
    ),
    "townhouse_multi_dwelling": (
        "strategy townhouse_multi_dwelling | medium density redevelopment | "
        "larger residential lot preferred | access helpful | "
        "low to moderate constraint preferred"
    ),
    "low_rise_apartment": (
        "strategy low_rise_apartment | higher intensity redevelopment | "
        "high development zoning preferred | large site preferred | "
        "strong station access preferred | low constraint preferred"
    ),
    "land_bank_hold": (
        "strategy land_bank_hold | long-term development upside | "
        "planning or location upside preferred | access helpful | "
        "some planning complexity acceptable"
    ),
    "assembly_opportunity": (
        "strategy assembly_opportunity | site assembly or amalgamation | "
        "multiple adjoining lots or large combined site preferred | "
        "redevelopment upside preferred"
    ),
}


HIGH_INTENSITY_TERMS = [
    "apartment",
    "apartments",
    "high density",
    "higher density",
    "high development zoning",
    "near station",
    "train station",
    "metro",
    "large site",
    "high rise",
    "low-rise",
    "low rise",
]

LOW_DENSITY_TERMS = [
    "house",
    "detached",
    "single dwelling",
    "family home",
    "standard residential",
    "knockdown rebuild",
]


def infer_strategy(query_text: str) -> tuple[str, dict[str, list[str]]]:
    matched: dict[str, list[str]] = {}

    for strategy, terms in STRATEGY_SIGNAL_TERMS.items():
        hits = _contains_any(query_text, terms)
        if hits:
            matched[strategy] = hits

    if not matched:
        return "land_bank_hold", matched

    # Simple scoring: count matched terms
    ranked = sorted(
        matched.items(),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )
    return ranked[0][0], matched


def validate_strategy(strategy: str) -> None:
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Expected one of: {', '.join(STRATEGIES)}"
        )


def detect_conflicts(strategy: str, query_text: str) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    alternatives: list[str] = []

    high_hits = _contains_any(query_text, HIGH_INTENSITY_TERMS)
    low_hits = _contains_any(query_text, LOW_DENSITY_TERMS)

    if strategy == "single_dwelling_rebuild" and high_hits:
        warnings.append(
            "The query contains high-intensity redevelopment signals, which may conflict "
            "with the selected single dwelling rebuild strategy."
        )
        alternatives.extend(["low_rise_apartment", "land_bank_hold"])

    if strategy in {"low_rise_apartment", "townhouse_multi_dwelling"} and low_hits:
        warnings.append(
            "The query contains low-density house redevelopment signals, which may conflict "
            "with the selected higher-intensity redevelopment strategy."
        )
        alternatives.append("single_dwelling_rebuild")

    if strategy == "granny_flat" and _contains_any(query_text, ["apartment", "high density", "large site"]):
        warnings.append(
            "The query appears broader or more intensive than a granny flat strategy."
        )
        alternatives.extend(["dual_occupancy", "townhouse_multi_dwelling"])

    # Preserve order and remove duplicates
    alternatives = list(dict.fromkeys([a for a in alternatives if a != strategy]))
    return warnings, alternatives


CONFLICT_REWRITE_RULES: dict[str, list[tuple[str, str]]] = {
    "single_dwelling_rebuild": [
        ("near a train station", "with transport access helpful but not required"),
        ("near train station", "with transport access helpful but not required"),
        ("close to train station", "with transport access helpful but not required"),
        ("close to station", "with transport access helpful but not required"),
        ("near station", "with transport access helpful but not required"),
        ("high development zoning", "suitable residential zoning"),
        ("high density", "standard residential density"),
        ("higher density", "standard residential density"),
        ("large site", "suitable lot size"),
        ("large lot", "suitable lot size"),
        ("apartment", "detached house"),
        ("apartments", "detached houses"),
    ],
    "low_rise_apartment": [
        ("detached house", "low-rise apartment"),
        ("single dwelling", "low-rise apartment"),
        ("family home", "residential apartment"),
        ("standard residential land", "higher-intensity residential land"),
    ],
    "dual_occupancy": [
        ("apartment", "dual occupancy"),
        ("apartments", "dual occupancy"),
        ("high density", "residential infill"),
        ("high development zoning", "suitable residential zoning"),
    ],
    "granny_flat": [
        ("apartment", "secondary dwelling"),
        ("apartments", "secondary dwelling"),
        ("high density", "low-intensity residential infill"),
        ("large site", "suitable residential lot"),
    ],
}


def sanitise_query_for_strategy(strategy: str, query_text: str) -> str:
    """
    Softly rewrite user text so it is less contradictory with the selected strategy.

    This is intentionally simple and deterministic. It does not remove the user's
    intent completely; it just neutralises phrases that strongly pull retrieval
    toward another strategy.
    """
    cleaned = query_text

    for source, target in CONFLICT_REWRITE_RULES.get(strategy, []):
        cleaned = cleaned.replace(source, target)
        cleaned = cleaned.replace(source.title(), target)
        cleaned = cleaned.replace(source.capitalize(), target)

    return cleaned


def rewrite_query(strategy: str, query_text: str) -> str:
    prior = STRATEGY_PRIORS[strategy]
    sanitised_query = sanitise_query_for_strategy(strategy, query_text)
    return f"{prior} | sanitised user query: {sanitised_query}"


def plan_query(
    query_text: str,
    strategy: str | None = None,
    allow_infer_strategy: bool = True,
) -> QueryPlan:
    query_text = query_text.strip()
    if not query_text:
        raise ValueError("query_text cannot be empty.")

    inferred_strategy, matched_signals = infer_strategy(query_text)

    if strategy is not None and strategy.strip():
        selected_strategy = strategy.strip()
        validate_strategy(selected_strategy)
    else:
        if not allow_infer_strategy:
            raise ValueError("strategy is required when allow_infer_strategy=False.")
        selected_strategy = inferred_strategy

    warnings, alternatives = detect_conflicts(selected_strategy, query_text)

    if strategy is not None and strategy.strip() and inferred_strategy != selected_strategy and matched_signals:
        alternatives = list(dict.fromkeys([inferred_strategy] + alternatives))
        warnings.append(
            f"The query signals also match '{inferred_strategy}', which differs from "
            f"the selected strategy '{selected_strategy}'."
        )

    sanitised = sanitise_query_for_strategy(selected_strategy, query_text)
    rewritten = f"{STRATEGY_PRIORS[selected_strategy]} | sanitised user query: {sanitised}"

    return QueryPlan(
        original_query=query_text,
        input_strategy=strategy,
        selected_strategy=selected_strategy,
        sanitised_query=sanitised,
        rewritten_query=rewritten,
        warnings=warnings,
        suggested_alternatives=alternatives,
        matched_signals=matched_signals,
    )