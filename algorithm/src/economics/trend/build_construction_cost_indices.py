from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


DEFAULT_WPI_RAW = Path("data/raw/economics/abs/wpi_raw.csv")
DEFAULT_PPI_RAW = Path("data/raw/economics/abs/ppi_raw.csv")
DEFAULT_OUTPUT = Path("data/processed/economics/trend/construction_cost_indices.parquet")


def parse_quarter(value: object) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()

    try:
        return pd.Period(text.replace("-", ""), freq="Q").to_timestamp()
    except Exception:
        try:
            return pd.Period(text, freq="Q").to_timestamp()
        except Exception:
            return None


def normalise_code(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).upper().strip()


def prepare_abs_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        if col not in {"OBS_VALUE"}:
            out[col] = out[col].map(normalise_code)

    out["quarter"] = out["TIME_PERIOD"].map(parse_quarter)
    out["OBS_VALUE"] = pd.to_numeric(out["OBS_VALUE"], errors="coerce")

    out = out[out["quarter"].notna()]
    out = out[out["OBS_VALUE"].notna()]

    return out


def select_wpi_construction(wpi: pd.DataFrame) -> pd.DataFrame:
    work = prepare_abs_frame(wpi)

    # ABS WPI industry code E is commonly Construction in ANZSIC-style industry coding.
    # Use the cleanest national index-level series first.
    selected = work[
        (work["MEASURE"] == "1")
        & (work["UNIT_MEASURE"] == "IN")
        & (work["INDUSTRY"] == "E")
        & (work["REGION"] == "AUS")
    ].copy()

    if selected.empty:
        print("[WPI warning] No AUS construction index rows. Falling back to all construction index rows.")
        selected = work[
            (work["MEASURE"] == "1")
            & (work["UNIT_MEASURE"] == "IN")
            & (work["INDUSTRY"] == "E")
        ].copy()

    if selected.empty:
        print("[WPI warning] No construction index rows. Falling back to all WPI index rows.")
        selected = work[
            (work["MEASURE"] == "1")
            & (work["UNIT_MEASURE"] == "IN")
        ].copy()

    out = (
        selected.groupby("quarter", as_index=False)
        .agg(wpi_construction_index=("OBS_VALUE", "mean"))
        .sort_values("quarter")
        .reset_index(drop=True)
    )

    out["wpi_construction_qoq_growth"] = out["wpi_construction_index"].pct_change(1)
    out["wpi_construction_yoy_growth"] = out["wpi_construction_index"].pct_change(4)

    print()
    print("WPI selected rows:", len(selected))
    print("WPI selected dimensions:")
    for col in ["MEASURE", "INDEX", "SECTOR", "INDUSTRY", "TSEST", "REGION", "UNIT_MEASURE"]:
        if col in selected.columns:
            print(f"  {col}: {selected[col].value_counts().head(10).to_dict()}")

    return out


def select_ppi_proxy(ppi: pd.DataFrame) -> pd.DataFrame:
    work = prepare_abs_frame(ppi)

    # PPI raw downloaded via /all has numeric INDEX codes without labels.
    # Until we add code-list label lookup, use a conservative national index-level proxy:
    # MEASURE=1, UNIT_MEASURE=IN.
    #
    # Prefer OUTPUT because it directly captures producer output prices.
    selected = work[
        (work["MEASURE"] == "1")
        & (work["UNIT_MEASURE"] == "IN")
        & (work["TYPE"] == "OUTPUT")
    ].copy()

    if selected.empty:
        print("[PPI warning] No OUTPUT index rows. Falling back to all PPI index rows.")
        selected = work[
            (work["MEASURE"] == "1")
            & (work["UNIT_MEASURE"] == "IN")
        ].copy()

    # Robust proxy: use median across available PPI output index series each quarter.
    # This avoids one huge series dominating the mean.
    out = (
        selected.groupby("quarter", as_index=False)
        .agg(ppi_output_proxy_index=("OBS_VALUE", "median"))
        .sort_values("quarter")
        .reset_index(drop=True)
    )

    out["ppi_output_proxy_qoq_growth"] = out["ppi_output_proxy_index"].pct_change(1)
    out["ppi_output_proxy_yoy_growth"] = out["ppi_output_proxy_index"].pct_change(4)

    print()
    print("PPI selected rows:", len(selected))
    print("PPI selected dimensions:")
    for col in ["MEASURE", "TYPE", "UNIT_MEASURE"]:
        if col in selected.columns:
            print(f"  {col}: {selected[col].value_counts().head(10).to_dict()}")

    print("  INDEX sample:", selected["INDEX"].value_counts().head(10).to_dict())

    return out


def rebase_to_100(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return series

    first = float(valid.iloc[0])
    if first == 0:
        return series

    return series / first * 100.0


def build_combined(wpi_df: pd.DataFrame, ppi_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.merge(wpi_df, ppi_df, on="quarter", how="outer").sort_values("quarter")

    out["wpi_construction_index"] = pd.to_numeric(
        out["wpi_construction_index"], errors="coerce"
    ).ffill().bfill()

    out["ppi_output_proxy_index"] = pd.to_numeric(
        out["ppi_output_proxy_index"], errors="coerce"
    ).ffill().bfill()

    out["wpi_construction_index_rebased"] = rebase_to_100(out["wpi_construction_index"])
    out["ppi_output_proxy_index_rebased"] = rebase_to_100(out["ppi_output_proxy_index"])

    # First version:
    # - WPI construction is direct labour-cost proxy.
    # - PPI output proxy is broader producer price pressure proxy.
    out["combined_construction_cost_index"] = (
        0.55 * out["wpi_construction_index_rebased"]
        + 0.45 * out["ppi_output_proxy_index_rebased"]
    )

    out["combined_qoq_growth"] = out["combined_construction_cost_index"].pct_change(1)
    out["combined_yoy_growth"] = out["combined_construction_cost_index"].pct_change(4)

    out["predicted_construction_cost_growth_qoq"] = (
        out["combined_qoq_growth"]
        .rolling(4, min_periods=1)
        .mean()
        .clip(lower=-0.03, upper=0.05)
    )

    out["construction_cost_escalation_multiplier"] = (
        1.0 + out["predicted_construction_cost_growth_qoq"].fillna(0.0)
    )

    out["construction_cost_trend_score"] = (
        50.0
        + out["predicted_construction_cost_growth_qoq"].fillna(0.0) * 1000.0
    ).clip(0, 100)

    def band(x: float) -> str:
        if x >= 0.02:
            return "elevated"
        if x >= 0.005:
            return "moderate"
        if x >= -0.005:
            return "stable"
        return "softening"

    out["construction_cost_trend_band"] = (
        out["predicted_construction_cost_growth_qoq"].fillna(0.0).map(band)
    )

    out["cost_trend_model"] = "wpi_construction_plus_ppi_output_proxy_v1"

    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wpi", default=str(DEFAULT_WPI_RAW))
    parser.add_argument("--ppi", default=str(DEFAULT_PPI_RAW))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    wpi = pd.read_csv(args.wpi)
    ppi = pd.read_csv(args.ppi)

    print(f"Loaded WPI rows: {len(wpi):,}")
    print(f"Loaded PPI rows: {len(ppi):,}")

    wpi_selected = select_wpi_construction(wpi)
    ppi_selected = select_ppi_proxy(ppi)

    print()
    print(f"WPI selected quarters: {len(wpi_selected):,}")
    print(f"PPI selected quarters: {len(ppi_selected):,}")

    combined = build_combined(wpi_selected, ppi_selected)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)

    print()
    print(f"Saved: {output_path}")
    print()
    print(combined.tail(12).to_string(index=False))


if __name__ == "__main__":
    main()