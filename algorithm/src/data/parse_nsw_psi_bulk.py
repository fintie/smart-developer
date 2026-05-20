from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from algorithm.src.data.parse_nsw_psi import parse_b_rows


def parse_dat_file(path: Path) -> pd.DataFrame:
    df = parse_b_rows(path)
    if not df.empty:
        df["source_file"] = str(path)
    return df


def parse_zip_file(path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(tmp_dir)

        dat_paths = list(tmp_dir.rglob("*.DAT")) + list(tmp_dir.rglob("*.dat"))

        for dat_path in dat_paths:
            df = parse_b_rows(dat_path)
            if not df.empty:
                df["source_zip"] = str(path)
                df["source_file"] = dat_path.name
                frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def parse_path(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".zip":
        return parse_zip_file(path)

    if suffix == ".dat":
        return parse_dat_file(path)

    return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/raw/nsw_psi",
                        help="Directory containing PSI .zip or .DAT files.")
    parser.add_argument("--output", default="data/interim/nsw_psi/sales.parquet",
                        help="Output parquet path.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        list(input_dir.rglob("*.zip"))
        + list(input_dir.rglob("*.DAT"))
        + list(input_dir.rglob("*.dat"))
    )

    print(f"Found {len(paths):,} input files")

    frames: list[pd.DataFrame] = []

    for i, path in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] Parsing {path}")
        try:
            df = parse_path(path)
        except Exception as exc:
            print(f"  Failed: {exc}")
            continue

        if df.empty:
            print("  No B rows")
            continue

        print(f"  Rows: {len(df):,}")
        frames.append(df)

    if not frames:
        raise RuntimeError("No sales rows parsed.")

    sales = pd.concat(frames, ignore_index=True)

    # Drop exact duplicates if the same file was parsed twice.
    dedupe_cols = [
        "property_id",
        "sale_sequence",
        "contract_date",
        "settlement_date",
        "sale_price",
        "dealing_number",
    ]
    existing_cols = [c for c in dedupe_cols if c in sales.columns]
    if existing_cols:
        before = len(sales)
        sales = sales.drop_duplicates(subset=existing_cols)
        after = len(sales)
        print(f"Dropped duplicates: {before - after:,}")

    sales.to_parquet(output_path, index=False)

    print()
    print(f"Saved to: {output_path}")
    print(f"Rows: {len(sales):,}")

    print()
    print(sales[
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
        ]
    ].head(20).to_string(index=False))


if __name__ == "__main__":
    main()