from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


DEFAULT_SALES_PATH = Path("data/interim/nsw_psi/sales.parquet")
DEFAULT_OUTPUT_PATH = Path("data/processed/economics/trend/suburb_monthly_market.parquet")


def normalise_text_series(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales-path", default=str(DEFAULT_SALES_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--min-date", default="2020-01-01")
    args = parser.parse_args()

    sales_path = Path(args.sales_path)
    output_path = Path(args.output)

    df = pd.read_parquet(sales_path)
    print(f"Loaded sales rows: {len(df):,}")

    if "sale_price" not in df.columns:
        raise ValueError("sales data must contain sale_price")

    date_col = None
    for candidate in ["contract_date", "settlement_date"]:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        raise ValueError("sales data must contain contract_date or settlement_date")

    df = df.copy()
    df["sale_price"] = pd.to_numeric(df["sale_price"], errors="coerce")
    df["sale_date"] = pd.to_datetime(df[date_col], errors="coerce")

    df = df[df["sale_price"].notna()]
    df = df[df["sale_date"].notna()]
    df = df[df["sale_price"] > 50_000]
    df = df[df["sale_price"] < 100_000_000]
    df = df[df["sale_date"] <= pd.Timestamp.today().normalize()]
    df = df[df["sale_date"] >= pd.Timestamp(args.min_date)]

    if "suburb" not in df.columns:
        raise ValueError("sales data must contain suburb")

    if "postcode" not in df.columns:
        df["postcode"] = ""

    df["suburb"] = normalise_text_series(df["suburb"])
    df["postcode"] = normalise_text_series(df["postcode"])
    df = df[df["suburb"] != ""]

    df["month"] = df["sale_date"].dt.to_period("M").dt.to_timestamp()

    grouped = (
        df.groupby(["suburb", "postcode", "month"], as_index=False)
        .agg(
            sales_count=("sale_price", "size"),
            median_sale_price=("sale_price", "median"),
            mean_sale_price=("sale_price", "mean"),
            p25_sale_price=("sale_price", lambda x: x.quantile(0.25)),
            p75_sale_price=("sale_price", lambda x: x.quantile(0.75)),
        )
        .sort_values(["suburb", "postcode", "month"])
        .reset_index(drop=True)
    )

    grouped["log_median_sale_price"] = np.log1p(grouped["median_sale_price"])

    pieces = []
    for _, g in grouped.groupby(["suburb", "postcode"], sort=False):
        g = g.sort_values("month").copy()

        g["growth_1m"] = g["log_median_sale_price"].diff(1)
        g["growth_3m"] = g["log_median_sale_price"].diff(3)

        g["rolling_median_3m"] = (
            g["median_sale_price"].rolling(3, min_periods=1).median()
        )
        g["rolling_median_6m"] = (
            g["median_sale_price"].rolling(6, min_periods=1).median()
        )
        g["rolling_sales_count_3m"] = (
            g["sales_count"].rolling(3, min_periods=1).sum()
        )
        g["rolling_sales_count_6m"] = (
            g["sales_count"].rolling(6, min_periods=1).sum()
        )

        g["target_growth_3m"] = g["log_median_sale_price"].shift(-3) - g["log_median_sale_price"]
        g["target_market_trend_multiplier"] = np.exp(g["target_growth_3m"])

        pieces.append(g)

    out = pd.concat(pieces, ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Rows: {len(out):,}")
    print(f"Date range: {out['month'].min()} -> {out['month'].max()}")
    print(f"Suburbs: {out['suburb'].nunique():,}")


if __name__ == "__main__":
    main()