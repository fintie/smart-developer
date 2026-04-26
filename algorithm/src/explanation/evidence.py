from __future__ import annotations
from typing import Any
import pandas as pd
from algorithm.src.explanation.schemas import ExplanationPayload


def get_decision_band(score: float) -> str:
    if score >= 80:
        return "strong_fit"
    if score >= 65:
        return "good_fit"
    if score >= 45:
        return "moderate_fit"
    if score >= 25:
        return "weak_fit"
    return "poor_fit"


def _zoning_positive(strategy: str, zoning_code: str | None) -> list[str]:
    if zoning_code is None or pd.isna(zoning_code):
        return []

    z = str(zoning_code)

    mapping = {
        "single_dwelling_rebuild": {
            "R1": "The zoning is supportive of standard residential redevelopment.",
            "R2": "The zoning is supportive of standard residential redevelopment.",
            "R5": "The zoning supports lower-intensity residential use.",
            "RU5": "The zoning supports lower-intensity residential use.",
        },
        "assembly_opportunity": {
            "MU1": "The zoning supports higher-intensity redevelopment potential.",
            "R4": "The zoning supports higher-intensity redevelopment potential.",
            "R3": "The zoning supports medium-density redevelopment potential.",
            "E1": "The zoning suggests broader strategic redevelopment potential.",
            "E2": "The zoning suggests broader strategic redevelopment potential.",
            "E3": "The zoning suggests broader strategic redevelopment potential.",
            "SP5": "The zoning supports a more intensive urban redevelopment context.",
        },
        "granny_flat": {
            "R1": "The zoning is compatible with lower-intensity residential expansion.",
            "R2": "The zoning is compatible with lower-intensity residential expansion.",
            "R5": "The zoning is compatible with lower-intensity residential expansion.",
            "R3": "The zoning may still support a smaller-scale additional dwelling outcome.",
        },
        "land_bank_hold": {
            "MU1": "The zoning provides future redevelopment upside.",
            "R4": "The zoning provides future redevelopment upside.",
            "R3": "The zoning provides future redevelopment upside.",
            "E1": "The zoning suggests longer-term strategic value.",
            "E2": "The zoning suggests longer-term strategic value.",
            "E3": "The zoning suggests longer-term strategic value.",
            "SP5": "The zoning suggests longer-term strategic value.",
        },
        "townhouse_multi_dwelling": {
            "R3": "The zoning is supportive of medium-density redevelopment.",
            "MU1": "The zoning is supportive of medium-density redevelopment.",
            "R4": "The zoning may support denser redevelopment outcomes.",
            "R1": "The zoning may still offer some medium-density potential.",
        },
        "low_rise_apartment": {
            "R4": "The zoning is supportive of higher-intensity redevelopment.",
            "MU1": "The zoning is supportive of higher-intensity redevelopment.",
            "SP5": "The zoning is supportive of higher-intensity redevelopment.",
            "E1": "The zoning suggests a more urban redevelopment context.",
            "E2": "The zoning suggests a more urban redevelopment context.",
            "E3": "The zoning suggests a more urban redevelopment context.",
        },
        "dual_occupancy": {
            "R1": "The zoning is supportive of moderate residential intensification.",
            "R2": "The zoning is supportive of moderate residential intensification.",
            "R3": "The zoning may support moderate residential intensification.",
            "R5": "The zoning may support moderate residential intensification.",
        },
    }

    return [mapping.get(strategy, {}).get(z)] if mapping.get(strategy, {}).get(z) else []


def _lot_size_positive(strategy: str, lot_size_sqm: float | None) -> list[str]:
    if lot_size_sqm is None or pd.isna(lot_size_sqm):
        return []

    area = float(lot_size_sqm)

    if strategy in {"low_rise_apartment", "assembly_opportunity"} and area >= 5000:
        return ["The site area appears relatively large for a more intensive redevelopment pathway."]
    if strategy in {"townhouse_multi_dwelling", "land_bank_hold"} and area >= 2000:
        return ["The site area appears supportive of a larger redevelopment envelope."]
    if strategy in {"granny_flat", "dual_occupancy"} and area >= 1000:
        return ["The lot size appears supportive of an additional residential use."]
    if strategy == "single_dwelling_rebuild" and area >= 800:
        return ["The site area appears adequate for a conventional residential rebuild."]
    return []


def _access_positive(strategy: str, within_800m: Any, distance_to_station_m: Any) -> list[str]:
    if pd.notna(within_800m) and int(within_800m) == 1:
        if strategy in {"low_rise_apartment", "assembly_opportunity", "townhouse_multi_dwelling", "land_bank_hold"}:
            return ["The site has strong rail or metro accessibility."]
        return ["The site benefits from nearby rail or metro access."]

    if pd.notna(distance_to_station_m) and float(distance_to_station_m) <= 2000:
        if strategy in {"low_rise_apartment", "assembly_opportunity", "townhouse_multi_dwelling", "land_bank_hold"}:
            return ["The site has reasonably good rail or metro access."]
    return []


def _negative_evidence(row: pd.Series) -> list[str]:
    negatives: list[str] = []

    if int(row.get("heritage_flag", 0) or 0) == 1:
        sig = row.get("heritage_max_significance")
        if pd.notna(sig):
            negatives.append(f"Heritage constraints are present, with {sig.lower()} significance indicated.")
        else:
            negatives.append("Heritage constraints are present and may reduce redevelopment flexibility.")

    if int(row.get("flood_flag", 0) or 0) == 1:
        flood_class = row.get("primary_flood_class")
        if pd.notna(flood_class):
            negatives.append(f"Flood-related planning constraints are present ({flood_class}).")
        else:
            negatives.append("Flood-related planning constraints are present.")

    bushfire_level = int(row.get("bushfire_risk_level", 0) or 0)
    if bushfire_level >= 3:
        negatives.append("Bushfire exposure appears elevated and may materially constrain redevelopment.")
    elif bushfire_level == 2:
        negatives.append("Bushfire exposure is present and may increase development complexity.")
    elif bushfire_level == 1:
        negatives.append("Some bushfire-related constraints are present.")

    return negatives


def _cautions(row: pd.Series) -> list[str]:
    cautions: list[str] = []

    if int(row.get("mixed_zoning_flag", 0) or 0) == 1:
        cautions.append("The site sits in a mixed zoning context and may require closer planning review.")

    if pd.notna(row.get("lot_size_proxy_sqm")) and float(row["lot_size_proxy_sqm"]) > 1e7:
        cautions.append("The site area appears unusually large and may warrant manual validation.")

    return cautions


def _prioritized_negative_evidence(row: pd.Series) -> list[str]:
    negatives = _negative_evidence(row)
    cautions = _cautions(row)

    # strong constraints：heritage / flood / bushfire
    if negatives:
        ranked = negatives[:2]

        if int(row.get("mixed_zoning_flag", 0) or 0) == 1:
            ranked.append("The mixed zoning context may still warrant closer planning review.")

        return ranked[:3]

    return cautions[:2]


def build_explanation_payload(row: pd.Series, strategy: str) -> ExplanationPayload:
    score_col = f"{strategy}_score"
    score = float(row.get(score_col, 0.0))

    positives: list[str] = []
    positives += _zoning_positive(strategy, row.get("primary_zoning_code"))
    positives += _lot_size_positive(strategy, row.get("lot_size_proxy_sqm"))
    positives += _access_positive(strategy, row.get("within_800m_catchment"), row.get("distance_to_station_m"))

    negatives = _prioritized_negative_evidence(row)
    cautions: list[str] = []

    return ExplanationPayload(
        strategy=strategy,
        decision_band=get_decision_band(score),
        positive_evidence=positives[:3],
        negative_evidence=negatives[:3],
        cautions=cautions[:2],
    )