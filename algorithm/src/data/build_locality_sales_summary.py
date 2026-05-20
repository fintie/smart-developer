from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()

    clean = clean.dropna(subset=["suburb", "sale_price"])
    clean = clean[clean["sale_price"] > 0]

    if "contract_date" in clean.columns:
        clean = clean.sort_values("contract_date")

    grouped = (
        clean.groupby(["suburb", "postcode"], dropna=False)
        .agg(
            sales_count=("sale_price", "size"),
            median_sale_price=("sale_price", "median"),
            p25_sale_price=("sale_price", lambda x: x.quantile(0.25)),
            p75_sale_price=("sale_price", lambda x: x.quantile(0.75)),
            min_sale_price=("sale_price", "min"),
            max_sale_price=("sale_price", "max"),
            latest_contract_date=("contract_date", "max"),
        )
        .reset_index()
    )

    def confidence(n: int) -> str:
        if n >= 30:
            return "high"
        if n >= 10:
            return "medium"
        if n >= 3:
            return "low"
        return "very_low"

    grouped["confidence"] = grouped["sales_count"].apply(confidence)
    grouped["source"] = "NSW Bulk Property Sales Information"
    grouped["licence_note"] = "CC BY-NC-ND 4.0; internal prototype only unless commercial usage is verified."

    return grouped.sort_values(["sales_count", "median_sale_price"], ascending=[False, False])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/interim/nsw_psi/sample_sales.parquet")
    parser.add_argument("--output", default="data/processed/cost/locality_sales_summary.parquet")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    summary = build_summary(df)
    summary.to_parquet(output_path, index=False)

    print(f"Input rows: {len(df):,}")
    print(f"Summary rows: {len(summary):,}")
    print(f"Saved to: {output_path}")

    if not summary.empty:
        print()
        print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()