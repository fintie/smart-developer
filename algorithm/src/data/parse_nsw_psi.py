from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


B_COLUMNS = [
    "record_type",
    "district_code",
    "property_id",
    "sale_sequence",
    "file_created_at",
    "property_name",
    "unit_number",
    "house_number",
    "street_name",
    "suburb",
    "postcode",
    "unknown_11",
    "unknown_12",
    "contract_date",
    "settlement_date",
    "sale_price",
    "unknown_16",
    "property_class_code",
    "property_class",
    "unknown_19",
    "unknown_20",
    "unknown_21",
    "unknown_22",
    "dealing_number",
    "unknown_24",
]


def parse_b_rows(path: Path) -> pd.DataFrame:
    rows: list[dict] = []

    with path.open("r", encoding="latin1", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=";")

        for raw in reader:
            if not raw:
                continue

            if raw[0] != "B":
                continue

            # Some rows may have trailing empty fields or slight length variation.
            values = raw[: len(B_COLUMNS)]
            if len(values) < len(B_COLUMNS):
                values = values + [""] * (len(B_COLUMNS) - len(values))

            row = dict(zip(B_COLUMNS, values))
            rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["suburb"] = df["suburb"].astype(str).str.upper().str.strip()
    df["postcode"] = df["postcode"].astype(str).str.strip()
    df["street_name"] = df["street_name"].astype(str).str.upper().str.strip()
    df["property_class"] = df["property_class"].astype(str).str.upper().str.strip()
    df["property_class_code"] = df["property_class_code"].astype(str).str.upper().str.strip()

    df["sale_price"] = pd.to_numeric(df["sale_price"], errors="coerce")

    df["contract_date"] = pd.to_datetime(
        df["contract_date"],
        format="%Y%m%d",
        errors="coerce",
    )
    df["settlement_date"] = pd.to_datetime(
        df["settlement_date"],
        format="%Y%m%d",
        errors="coerce",
    )
    df["file_created_at"] = pd.to_datetime(
        df["file_created_at"],
        format="%Y%m%d %H:%M",
        errors="coerce",
    )

    df["full_address"] = (
        df[["unit_number", "house_number", "street_name", "suburb", "postcode"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/raw/nsw_psi/sample.DAT",
        help="Path to a NSW PSI .DAT file.",
    )
    parser.add_argument(
        "--output",
        default="data/interim/nsw_psi/sample_sales.parquet",
        help="Output parquet path.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = parse_b_rows(input_path)
    df.to_parquet(output_path, index=False)

    print(f"Parsed B rows: {len(df):,}")
    print(f"Saved to: {output_path}")

    if not df.empty:
        print()
        print(df[
            [
                "property_id",
                "full_address",
                "suburb",
                "postcode",
                "contract_date",
                "settlement_date",
                "sale_price",
                "property_class_code",
                "property_class",
                "dealing_number",
            ]
        ].head(20).to_string(index=False))


if __name__ == "__main__":
    main()