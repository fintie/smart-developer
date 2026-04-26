from __future__ import annotations
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd

from algorithm.src.scoring.scoring import score_dataframe


ROOT = Path(__file__).resolve().parents[3]

# Inputs
PROPERTY_PATH = ROOT / "data" / "processed" / "nsw_property" / "property.parquet"
ZONING_PATH = ROOT / "data" / "processed" / "nsw_zoning" / "land_zoning.parquet"
HERITAGE_PATH = ROOT / "data" / "processed" / "nsw_heritage" / "heritage.parquet"
BUSHFIRE_PATH = ROOT / "data" / "processed" / "nsw_bushfire" / "bushfire.parquet"
FLOOD_PATH = ROOT / "data" / "processed" / "nsw_flood" / "flood.parquet"
STATIONS_PATH = ROOT / "data" / "processed" / "transport" / "rail_metro_stations_raw.parquet"

# Outputs
OUTPUT_DIR = ROOT / "data" / "processed" / "site_features"
SITE_V1_PATH = OUTPUT_DIR / "property_site_features_v1_sample.parquet"
SITE_V1_SCORED_PATH = OUTPUT_DIR / "property_site_features_v1_sample_scored.parquet"

TARGET_DISTANCE_CRS = "EPSG:3857"
PROPERTY_SAMPLE_N = 50_000
RANDOM_SEED = 42


def safe_read_gdf(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet(path)
    if not isinstance(gdf, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


def ensure_same_crs(left: gpd.GeoDataFrame, right: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if left.crs != right.crs:
        left = left.to_crs(right.crs)
    return left, right


def collapse_zoning_group(group: pd.DataFrame) -> pd.Series:
    sym_nonnull = group["SYM_CODE"].dropna()
    class_nonnull = group["LAY_CLASS"].dropna()

    primary_sym = sym_nonnull.value_counts().index[0] if len(sym_nonnull) else pd.NA
    primary_class = class_nonnull.value_counts().index[0] if len(class_nonnull) else pd.NA

    zoning_codes = sorted(sym_nonnull.astype(str).unique().tolist()) if len(sym_nonnull) else []
    zoning_classes = sorted(class_nonnull.astype(str).unique().tolist()) if len(class_nonnull) else []

    return pd.Series(
        {
            "primary_zoning_code": primary_sym,
            "primary_zoning_class": primary_class,
            "zoning_hit_count": int(group["SYM_CODE"].notna().sum()),
            "zoning_code_count": int(sym_nonnull.nunique()),
            "mixed_zoning_flag": int(sym_nonnull.nunique() > 1),
            "zoning_codes": "|".join(zoning_codes),
            "zoning_classes": "|".join(zoning_classes),
        }
    )


def collapse_heritage_group(group: pd.DataFrame) -> pd.Series:
    lay_nonnull = group["LAY_CLASS"].dropna()
    sig_nonnull = group["SIG"].dropna()
    hname_nonnull = group["H_NAME"].dropna()

    heritage_classes = sorted(lay_nonnull.astype(str).unique().tolist()) if len(lay_nonnull) else []
    heritage_names = sorted(hname_nonnull.astype(str).unique().tolist()) if len(hname_nonnull) else []

    sig_order = {"Local": 1, "State": 2, "National": 3}
    max_sig = max(sig_nonnull.astype(str).tolist(), key=lambda x: sig_order.get(x, 0)) if len(sig_nonnull) else pd.NA

    return pd.Series(
        {
            "heritage_flag": int(group["LAY_CLASS"].notna().any()),
            "heritage_hit_count": int(group["LAY_CLASS"].notna().sum()),
            "heritage_class_count": int(lay_nonnull.nunique()),
            "heritage_classes": "|".join(heritage_classes),
            "heritage_names": "|".join(heritage_names),
            "heritage_max_significance": max_sig,
        }
    )


BUSHFIRE_RISK_MAP = {
    "Vegetation Category 1": 3,
    "Vegetation Category 2": 2,
    "Vegetation Category 3": 1,
    "Vegetation Buffer": 1,
}


def collapse_bushfire_group(group: pd.DataFrame) -> pd.Series:
    cat_nonnull = group["d_category"].dropna()
    guide_nonnull = group["d_guidelin"].dropna()

    categories = sorted(cat_nonnull.astype(str).unique().tolist()) if len(cat_nonnull) else []
    guides = sorted(guide_nonnull.astype(str).unique().tolist()) if len(guide_nonnull) else []

    max_risk = max(BUSHFIRE_RISK_MAP.get(str(x), 0) for x in cat_nonnull) if len(cat_nonnull) else 0
    primary_cat = cat_nonnull.value_counts().index[0] if len(cat_nonnull) else pd.NA

    return pd.Series(
        {
            "bushfire_flag": int(group["d_category"].notna().any()),
            "bushfire_hit_count": int(group["d_category"].notna().sum()),
            "bushfire_category_count": int(cat_nonnull.nunique()),
            "bushfire_categories": "|".join(categories),
            "bushfire_guidelines": "|".join(guides),
            "primary_bushfire_category": primary_cat,
            "bushfire_risk_level": int(max_risk),
            "has_bushfire_cat1": int("Vegetation Category 1" in categories),
            "has_bushfire_buffer": int("Vegetation Buffer" in categories),
        }
    )


def collapse_flood_group(group: pd.DataFrame) -> pd.Series:
    lay_nonnull = group["LAY_CLASS"].dropna()
    comment_nonnull = group["COMMENT"].dropna()

    flood_classes = sorted(lay_nonnull.astype(str).unique().tolist()) if len(lay_nonnull) else []
    flood_comments = sorted(comment_nonnull.astype(str).unique().tolist()) if len(comment_nonnull) else []
    primary_class = lay_nonnull.value_counts().index[0] if len(lay_nonnull) else pd.NA

    return pd.Series(
        {
            "flood_flag": int(group["LAY_CLASS"].notna().any()),
            "flood_hit_count": int(group["LAY_CLASS"].notna().sum()),
            "flood_class_count": int(lay_nonnull.nunique()),
            "flood_classes": "|".join(flood_classes),
            "flood_comments": "|".join(flood_comments),
            "primary_flood_class": primary_class,
        }
    )


def collapse_overlay(
    property_gdf: gpd.GeoDataFrame,
    overlay_gdf: gpd.GeoDataFrame,
    overlay_cols: list[str],
    predicate: str,
    collapse_fn,
) -> pd.DataFrame:
    property_gdf, overlay_gdf = ensure_same_crs(property_gdf, overlay_gdf)
    joined = gpd.sjoin(
        property_gdf,
        overlay_gdf[overlay_cols + ["geometry"]],
        how="left",
        predicate=predicate,
    )
    collapsed = (
        joined.groupby("RID", as_index=False)
        .apply(collapse_fn)
        .reset_index(drop=True)
    )
    return collapsed


def add_transport_features(
    site_features: gpd.GeoDataFrame,
    stations_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if not isinstance(site_features, gpd.GeoDataFrame):
        site_features = gpd.GeoDataFrame(site_features, geometry="geometry")
    if site_features.crs is None:
        site_features = site_features.set_crs("EPSG:4326")

    if not isinstance(stations_gdf, gpd.GeoDataFrame):
        stations_gdf = gpd.GeoDataFrame(stations_gdf, geometry="geometry")
    if stations_gdf.crs is None:
        stations_gdf = stations_gdf.set_crs("EPSG:4326")

    site_points = site_features.copy()
    site_points["geometry"] = site_points.geometry.representative_point()
    site_points = gpd.GeoDataFrame(site_points, geometry="geometry", crs=site_features.crs)

    site_points_m = site_points.to_crs(TARGET_DISTANCE_CRS)
    station_points_m = stations_gdf.to_crs(TARGET_DISTANCE_CRS)

    station_keep_cols = [c for c in ["stop_id", "stop_name", "route_type", "location_type", "geometry"] if c in station_points_m.columns]
    station_points_m = station_points_m[station_keep_cols].copy()

    nearest_station = gpd.sjoin_nearest(
        site_points_m,
        station_points_m,
        how="left",
        distance_col="distance_to_station_m",
    )

    nearest_station["within_800m_catchment"] = (nearest_station["distance_to_station_m"] <= 800).astype(int)
    nearest_station["log_distance_to_station_m"] = np.log1p(nearest_station["distance_to_station_m"])
    nearest_station["station_access_score"] = 1 / (1 + nearest_station["distance_to_station_m"])

    transport_features = nearest_station[
        ["RID", "distance_to_station_m", "within_800m_catchment", "log_distance_to_station_m", "station_access_score"]
    ].copy()

    out = site_features.merge(transport_features, on="RID", how="left")
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=site_features.crs)
    return out


def build_site_table() -> gpd.GeoDataFrame:
    print("Reading source layers...")
    property_gdf = safe_read_gdf(PROPERTY_PATH)
    zoning_gdf = safe_read_gdf(ZONING_PATH)
    heritage_gdf = safe_read_gdf(HERITAGE_PATH)
    bushfire_gdf = safe_read_gdf(BUSHFIRE_PATH)
    flood_gdf = safe_read_gdf(FLOOD_PATH)
    stations_gdf = safe_read_gdf(STATIONS_PATH)

    print("Filtering property universe...")
    property_main = property_gdf[property_gdf["propertytype"] == 1].copy()
    print("Property rows after filter:", len(property_main))

    print(f"Sampling {PROPERTY_SAMPLE_N} properties...")
    property_sample = property_main.sample(min(PROPERTY_SAMPLE_N, len(property_main)), random_state=RANDOM_SEED).copy()
    print("Property sample rows:", len(property_sample))

    property_base = property_sample[
        [
            "RID",
            "gurasid",
            "propertytype",
            "valnetpropertystatus",
            "valnetpropertytype",
            "dissolveparcelcount",
            "valnetlotcount",
            "propid",
            "superlot",
            "address",
            "housenumber",
            "urbanity",
            "Shape__Area",
            "Shape__Length",
            "geometry",
        ]
    ].copy()
    property_base["lot_size_proxy_sqm"] = property_base["Shape__Area"]

    print("Collapsing zoning...")
    zoning_features = collapse_overlay(
        property_gdf=property_sample,
        overlay_gdf=zoning_gdf,
        overlay_cols=["OBJECTID", "EPI_NAME", "LGA_NAME", "LAY_CLASS", "SYM_CODE", "PURPOSE", "EPI_TYPE"],
        predicate="intersects",
        collapse_fn=collapse_zoning_group,
    )
    site_features = property_base.merge(zoning_features, on="RID", how="left")

    print("Collapsing heritage...")
    heritage_features = collapse_overlay(
        property_gdf=property_sample,
        overlay_gdf=heritage_gdf,
        overlay_cols=["OBJECTID", "EPI_NAME", "LGA_NAME", "LAY_CLASS", "H_ID", "H_NAME", "SIG", "EPI_TYPE"],
        predicate="intersects",
        collapse_fn=collapse_heritage_group,
    )
    site_features = site_features.merge(heritage_features, on="RID", how="left")

    print("Collapsing bushfire...")
    bushfire_features = collapse_overlay(
        property_gdf=property_sample,
        overlay_gdf=bushfire_gdf,
        overlay_cols=["fid", "d_category", "d_guidelin", "category", "guideline"],
        predicate="intersects",
        collapse_fn=collapse_bushfire_group,
    )
    site_features = site_features.merge(bushfire_features, on="RID", how="left")

    print("Collapsing flood...")
    flood_features = collapse_overlay(
        property_gdf=property_sample,
        overlay_gdf=flood_gdf,
        overlay_cols=["OBJECTID", "EPI_NAME", "LGA_NAME", "LAY_CLASS", "EPI_TYPE", "COMMENT"],
        predicate="intersects",
        collapse_fn=collapse_flood_group,
    )
    site_features = site_features.merge(flood_features, on="RID", how="left")

    print("Filling default values...")
    fill_zero_cols = [
        "heritage_flag",
        "heritage_hit_count",
        "heritage_class_count",
        "bushfire_flag",
        "bushfire_hit_count",
        "bushfire_category_count",
        "bushfire_risk_level",
        "has_bushfire_cat1",
        "has_bushfire_buffer",
        "flood_flag",
        "flood_hit_count",
        "flood_class_count",
    ]
    for col in fill_zero_cols:
        if col in site_features.columns:
            site_features[col] = site_features[col].fillna(0).astype(int)

    site_features = gpd.GeoDataFrame(site_features, geometry="geometry", crs=property_sample.crs)

    print("Adding transport features...")
    site_features = add_transport_features(site_features, stations_gdf)

    return site_features


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    site_features = build_site_table()

    print(f"Writing site features: {SITE_V1_PATH}")
    site_features.to_parquet(SITE_V1_PATH, index=False)

    print("Scoring site features...")
    scored = score_dataframe(pd.DataFrame(site_features))
    scored = gpd.GeoDataFrame(scored, geometry="geometry", crs=site_features.crs)

    print(f"Writing scored site features: {SITE_V1_SCORED_PATH}")
    scored.to_parquet(SITE_V1_SCORED_PATH, index=False)

    print("Done.")
    print("Rows:", len(scored))
    score_cols = [c for c in scored.columns if c.endswith("_score")]
    print(scored[score_cols].describe().T[["mean", "min", "max"]])


if __name__ == "__main__":
    main()