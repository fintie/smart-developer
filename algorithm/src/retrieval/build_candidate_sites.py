from __future__ import annotations
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = ROOT / "data" / "processed" / "site_features" / "property_site_features_v1_sample_scored.parquet"
OUTPUT_DIR = ROOT / "data" / "processed" / "retrieval"
OUTPUT_PATH = OUTPUT_DIR / "candidate_sites.parquet"


STRATEGY_SCORE_COLS = [
    "single_dwelling_rebuild_score",
    "assembly_opportunity_score",
    "granny_flat_score",
    "land_bank_hold_score",
    "townhouse_multi_dwelling_score",
    "low_rise_apartment_score",
    "dual_occupancy_score",
]


def safe_token(value, default: str = "unknown") -> str:
    if value is None or pd.isna(value):
        return default
    return str(value)


def lot_size_band(area: float | int | None) -> str:
    if area is None or pd.isna(area):
        return "unknown"
    if area < 500:
        return "xs"
    if area < 1000:
        return "s"
    if area < 2000:
        return "m"
    if area < 5000:
        return "l"
    return "xl"


def station_distance_band(dist: float | int | None) -> str:
    if dist is None or pd.isna(dist):
        return "unknown"
    if dist <= 800:
        return "within_800m"
    if dist <= 2000:
        return "800m_2km"
    if dist <= 5000:
        return "2km_5km"
    if dist <= 10000:
        return "5km_10km"
    return "over_10km"


def constraint_severity(row: pd.Series) -> str:
    score = 0

    if int(row.get("heritage_flag", 0) or 0) == 1:
        score += 2
        sig = row.get("heritage_max_significance")
        if pd.notna(sig) and str(sig) in {"State", "National"}:
            score += 1

    if int(row.get("flood_flag", 0) or 0) == 1:
        score += 2

    bushfire = int(row.get("bushfire_risk_level", 0) or 0)
    score += bushfire

    if score == 0:
        return "low"
    if score <= 2:
        return "moderate"
    if score <= 4:
        return "high"
    return "severe"


def zoning_band(z: str | None) -> str:
    if z is None or pd.isna(z):
        return "unknown"

    z = str(z)
    if z in {"MU1", "R4", "SP5", "E1", "E2", "E3"}:
        return "high_dev"
    if z in {"R3", "R1"}:
        return "mid_dev"
    if z in {"R2", "R5", "RU5"}:
        return "low_dev"
    if z in {"C2", "C3", "C4", "RE1", "RU1", "RU4", "SP2", "W1", "W2", "W3", "W4"}:
        return "restricted"
    return "other"


def build_site_summary_text(row: pd.Series) -> str:
    parts: list[str] = []

    parts.append(f"zoning_code {safe_token(row.get('primary_zoning_code'))}")
    parts.append(f"zoning_band {safe_token(row.get('zoning_band'))}")
    parts.append(f"lot_size_band {safe_token(row.get('lot_size_band'))}")

    mixed = "yes" if int(row.get("mixed_zoning_flag", 0) or 0) == 1 else "no"
    parts.append(f"mixed_zoning {mixed}")

    heritage = "yes" if int(row.get("heritage_flag", 0) or 0) == 1 else "no"
    parts.append(f"heritage {heritage}")

    if heritage == "yes":
        sig = row.get("heritage_max_significance")
        if pd.notna(sig):
            parts.append(f"heritage_significance {str(sig).lower()}")

    flood = "yes" if int(row.get("flood_flag", 0) or 0) == 1 else "no"
    parts.append(f"flood {flood}")

    flood_class = row.get("primary_flood_class")
    if pd.notna(flood_class):
        parts.append(f"flood_class {str(flood_class).lower()}")

    parts.append(f"bushfire_risk {int(row.get('bushfire_risk_level', 0) or 0)}")
    parts.append(f"station_distance_band {safe_token(row.get('station_distance_band'))}")

    within = "yes" if int(row.get("within_800m_catchment", 0) or 0) == 1 else "no"
    parts.append(f"within_800m {within}")

    parts.append(f"constraint_severity {safe_token(row.get('constraint_severity_band'))}")

    return " | ".join(parts)


def build_candidate_text_debug(row: pd.Series) -> str:
    parts: list[str] = []

    top_strategy = row.get("top_strategy")
    top_score = row.get("top_strategy_score")

    if pd.notna(top_strategy):
        parts.append(f"top_strategy {top_strategy}")

    high_signals: list[str] = []
    if pd.notna(top_strategy) and pd.notna(top_score) and float(top_score) >= 45:
        high_signals.append(str(top_strategy))

    for col in STRATEGY_SCORE_COLS:
        val = row.get(col)
        if pd.notna(val) and float(val) >= 60:
            high_signals.append(col.replace("_score", ""))

    high_signals = sorted(set(high_signals))
    if high_signals:
        parts.append("strategy_signals " + " ".join(high_signals))
    else:
        parts.append("strategy_signals none")

    parts.append(f"zoning_code {safe_token(row.get('primary_zoning_code'))}")
    parts.append(f"zoning_band {safe_token(row.get('zoning_band'))}")
    parts.append(f"lot_size_band {safe_token(row.get('lot_size_band'))}")
    parts.append(f"station_distance_band {safe_token(row.get('station_distance_band'))}")
    parts.append(f"constraint_severity {safe_token(row.get('constraint_severity_band'))}")

    mixed = "yes" if int(row.get("mixed_zoning_flag", 0) or 0) == 1 else "no"
    parts.append(f"mixed_zoning {mixed}")

    heritage = "yes" if int(row.get("heritage_flag", 0) or 0) == 1 else "no"
    parts.append(f"heritage {heritage}")

    flood = "yes" if int(row.get("flood_flag", 0) or 0) == 1 else "no"
    parts.append(f"flood {flood}")

    parts.append(f"bushfire_risk {int(row.get('bushfire_risk_level', 0) or 0)}")

    within = "yes" if int(row.get("within_800m_catchment", 0) or 0) == 1 else "no"
    parts.append(f"within_800m {within}")

    return " | ".join(parts)


def build_candidate_text_clean(row: pd.Series) -> str:
    parts: list[str] = []

    parts.append(f"zoning_code {safe_token(row.get('primary_zoning_code'))}")
    parts.append(f"zoning_band {safe_token(row.get('zoning_band'))}")
    parts.append(f"lot_size_band {safe_token(row.get('lot_size_band'))}")
    parts.append(f"station_distance_band {safe_token(row.get('station_distance_band'))}")
    parts.append(f"constraint_severity {safe_token(row.get('constraint_severity_band'))}")

    mixed = "yes" if int(row.get("mixed_zoning_flag", 0) or 0) == 1 else "no"
    parts.append(f"mixed_zoning {mixed}")

    heritage = "yes" if int(row.get("heritage_flag", 0) or 0) == 1 else "no"
    parts.append(f"heritage {heritage}")

    sig = row.get("heritage_max_significance")
    if pd.notna(sig):
        parts.append(f"heritage_significance {str(sig).lower()}")

    flood = "yes" if int(row.get("flood_flag", 0) or 0) == 1 else "no"
    parts.append(f"flood {flood}")

    flood_class = row.get("primary_flood_class")
    if pd.notna(flood_class):
        parts.append(f"flood_class {str(flood_class).lower()}")

    parts.append(f"bushfire_risk {int(row.get('bushfire_risk_level', 0) or 0)}")

    within = "yes" if int(row.get("within_800m_catchment", 0) or 0) == 1 else "no"
    parts.append(f"within_800m {within}")

    return " | ".join(parts)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Rows: {len(df)}")

    print("Building derived bands...")
    df["lot_size_band"] = df["lot_size_proxy_sqm"].apply(lot_size_band)
    df["station_distance_band"] = df["distance_to_station_m"].apply(station_distance_band)
    df["constraint_severity_band"] = df.apply(constraint_severity, axis=1)
    df["zoning_band"] = df["primary_zoning_code"].apply(zoning_band)

    print("Computing top strategy...")
    df["top_strategy"] = df[STRATEGY_SCORE_COLS].idxmax(axis=1).str.replace("_score", "", regex=False)
    df["top_strategy_score"] = df[STRATEGY_SCORE_COLS].max(axis=1)

    print("Building retrieval text fields...")
    df["site_summary_text"] = df.apply(build_site_summary_text, axis=1)
    df["candidate_text_debug"] = df.apply(build_candidate_text_debug, axis=1)
    df["candidate_text_clean"] = df.apply(build_candidate_text_clean, axis=1)

    candidate_cols = [
        "RID",
        "address",
        "primary_zoning_code",
        "primary_zoning_class",
        "zoning_band",
        "mixed_zoning_flag",
        "lot_size_proxy_sqm",
        "lot_size_band",
        "heritage_flag",
        "heritage_max_significance",
        "bushfire_flag",
        "bushfire_risk_level",
        "flood_flag",
        "primary_flood_class",
        "distance_to_station_m",
        "within_800m_catchment",
        "station_distance_band",
        "station_access_score",
        "constraint_severity_band",
        "top_strategy",
        "top_strategy_score",
        "site_summary_text",
        "candidate_text_debug",
        "candidate_text_clean",
    ] + STRATEGY_SCORE_COLS

    candidate_sites = df[candidate_cols].copy()

    print("Candidate rows:", len(candidate_sites))
    print("Top strategy distribution:")
    print(candidate_sites["top_strategy"].value_counts(dropna=False).head(20))

    print(f"Writing: {OUTPUT_PATH}")
    candidate_sites.to_parquet(OUTPUT_PATH, index=False)

    print("Verifying output...")
    check = pd.read_parquet(OUTPUT_PATH)
    print("Verified rows:", len(check))
    print(check.head())


if __name__ == "__main__":
    main()