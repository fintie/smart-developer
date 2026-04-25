from __future__ import annotations
from pathlib import Path
from typing import Any
import math
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "algorithm" / "configs" / "strategies.yaml"


def load_strategy_config(path: Path=CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_lot_size_band(
    area: float | int | None,
    band_config: dict[str, list[Any]]
) -> str:
    if area is None or pd.isna(area):
        return "unknown"

    for band_name, bounds in band_config.items():
        lower, upper = bounds
        lower_ok = area >= lower
        upper_ok = True if upper is None else area < upper
        if lower_ok and upper_ok:
            return band_name

    return "unknown"


def zoning_score(
    zoning_code: str | None,
    zoning_cfg: dict[str, list[str]]
) -> float:
    if zoning_code is None or pd.isna(zoning_code):
        return 0.02

    if zoning_code in zoning_cfg.get("strong_positive", []):
        return 0.40
    if zoning_code in zoning_cfg.get("medium_positive", []):
        return 0.25
    if zoning_code in zoning_cfg.get("weak_positive", []):
        return 0.12
    if zoning_code in zoning_cfg.get("negative", []):
        return 0.01
    return 0.06


def lot_size_score(
    lot_size_sqm: float | int | None,
    preferred_bands: list[str],
    min_soft_sqm: float,
    strong_sqm: float,
    band_config: dict[str, list[Any]],
) -> float:
    if lot_size_sqm is None or pd.isna(lot_size_sqm):
        return 0.0

    band = get_lot_size_band(float(lot_size_sqm), band_config)
    score = 0.0

    if band in preferred_bands:
        score += 0.10

    if lot_size_sqm >= strong_sqm:
        score += 0.20
    elif lot_size_sqm >= min_soft_sqm:
        score += 0.10

    return score


def accessibility_score(
    distance_to_station_m: float | int | None,
    within_800m_catchment: int | float | None,
    weight: float,
    shared_cfg: dict[str, Any],
) -> float:
    if distance_to_station_m is None or pd.isna(distance_to_station_m):
        return 0.0

    bonus_cfg = shared_cfg["accessibility_bonus"]
    bonus = 0.0

    if within_800m_catchment == 1:
        bonus = bonus_cfg["within_800m"]
    elif distance_to_station_m <= 2000:
        bonus = bonus_cfg["within_2km"]
    elif distance_to_station_m <= 5000:
        bonus = bonus_cfg["within_5km"]

    return bonus * weight


def constraint_penalty(
    row: pd.Series,
    strategy_cfg: dict[str, Any],
    shared_cfg: dict[str, Any],
) -> float:
    penalty = 0.0
    c_cfg = strategy_cfg["constraints"]

    if int(row.get("heritage_flag", 0) or 0) == 1:
        penalty += 0.10 * float(c_cfg["heritage_weight"])
        sig = row.get("heritage_max_significance")
        if pd.notna(sig):
            penalty += float(shared_cfg["heritage_significance_penalty"].get(str(sig), 0.0)) * float(
                c_cfg["heritage_weight"]
            )

    if int(row.get("flood_flag", 0) or 0) == 1:
        penalty += float(shared_cfg["flood_penalty"]["default"]) * float(c_cfg["flood_weight"])

    bushfire_level = int(row.get("bushfire_risk_level", 0) or 0)
    penalty += float(shared_cfg["bushfire_risk_penalty"].get(bushfire_level, 0.0)) * float(
        c_cfg["bushfire_weight"]
    )

    return min(penalty, 0.85)


def mixed_zoning_bonus(row: pd.Series) -> float:
    return 0.05 if int(row.get("mixed_zoning_flag", 0) or 0) == 1 else 0.0


def build_strategy_score(row: pd.Series, strategy_name: str, cfg: dict[str, Any]) -> float:
    strategy_cfg = cfg["strategies"][strategy_name]
    shared_cfg = cfg["shared"]

    z_score = zoning_score(row.get("primary_zoning_code"), strategy_cfg["zoning"])

    l_cfg = strategy_cfg["lot_size"]
    l_score = lot_size_score(
        lot_size_sqm=row.get("lot_size_proxy_sqm"),
        preferred_bands=l_cfg["preferred_bands"],
        min_soft_sqm=float(l_cfg["min_soft_sqm"]),
        strong_sqm=float(l_cfg["strong_sqm"]),
        band_config=shared_cfg["lot_size_bands"],
    )

    a_cfg = strategy_cfg["accessibility"]
    a_score = accessibility_score(
        distance_to_station_m=row.get("distance_to_station_m"),
        within_800m_catchment=row.get("within_800m_catchment"),
        weight=float(a_cfg["weight"]),
        shared_cfg=shared_cfg,
    )

    c_penalty = constraint_penalty(row, strategy_cfg, shared_cfg)
    m_bonus = mixed_zoning_bonus(row)

    raw = z_score + l_score + a_score + m_bonus - c_penalty
    return float(max(0.0, min(1.0, raw)) * 100.0)


def score_row(row: pd.Series, cfg: dict[str, Any] | None = None) -> dict[str, float]:
    if cfg is None:
        cfg = load_strategy_config()

    out: dict[str, float] = {}
    for strategy_name in cfg["strategies"]:
        out[f"{strategy_name}_score"] = build_strategy_score(row, strategy_name, cfg)
    return out


def score_dataframe(df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    if cfg is None:
        cfg = load_strategy_config()

    scored = df.copy()

    for strategy_name in cfg["strategies"]:
        col = f"{strategy_name}_score"
        scored[col] = scored.apply(
            lambda row: build_strategy_score(row, strategy_name, cfg),
            axis=1,
        )

    return scored


if __name__ == "__main__":
    sample_path = ROOT / "data" / "processed" / "site_features" / "property_site_features_v1_sample.parquet"
    df = pd.read_parquet(sample_path)
    scored_df = score_dataframe(df)
    print(scored_df.filter(like="_score").head())