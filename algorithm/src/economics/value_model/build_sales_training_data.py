from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml
from algorithm.src.economics.common import today_timestamp


DEFAULT_CONFIG_PATH = Path("algorithm/configs/economics/xgb_market_value.yaml")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalise_text_series(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def prepare_base_sales(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "sale_price" not in df.columns:
        raise ValueError("sales data must contain sale_price")

    df["sale_price"] = pd.to_numeric(df["sale_price"], errors="coerce")
    df = df[df["sale_price"].notna()]
    df = df[df["sale_price"] > 50_000]
    df = df[df["sale_price"] < 100_000_000]

    date_col = None
    for candidate in ["contract_date", "settlement_date"]:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        raise ValueError("sales data must contain contract_date or settlement_date")

    df["sale_date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df["sale_date"].notna()]
    df = df[df["sale_date"] <= today_timestamp()]

    for col in ["suburb", "postcode", "property_class_code", "property_class"]:
        if col not in df.columns:
            df[col] = ""

    df["suburb"] = normalise_text_series(df["suburb"])
    df["postcode"] = normalise_text_series(df["postcode"])
    df["property_class_code"] = normalise_text_series(df["property_class_code"])
    df["property_class"] = normalise_text_series(df["property_class"])

    df = df[df["suburb"] != ""]

    df["contract_year"] = df["sale_date"].dt.year.astype(int).astype(str)
    df["contract_month"] = df["sale_date"].dt.month.astype(int).astype(str)
    df["contract_quarter"] = df["sale_date"].dt.quarter.astype(int).astype(str)

    min_month = df["sale_date"].dt.to_period("M").min()
    month_period = df["sale_date"].dt.to_period("M")
    df["months_since_start"] = (month_period - min_month).apply(lambda x: x.n)

    df["log_sale_price"] = np.log1p(df["sale_price"])

    return df.sort_values("sale_date").reset_index(drop=True)


def _add_group_rolling_features(
    df: pd.DataFrame,
    group_col: str,
    prefix: str,
) -> pd.DataFrame:
    """
    Add leakage-safe 365-day rolling statistics by group.

    For each sale row, stats are computed using previous transactions only.
    We sort by group and sale_date, then use rolling('365D') and shift(1)
    within each group to exclude the current transaction.
    """
    work = df[["row_id", group_col, "sale_date", "sale_price"]].copy()
    work = work.sort_values([group_col, "sale_date", "row_id"])

    pieces = []

    for _, g in work.groupby(group_col, sort=False):
        g = g.sort_values(["sale_date", "row_id"]).copy()
        g = g.set_index("sale_date")

        price = g["sale_price"]

        # Use closed='left' so current row is excluded from its own history.
        rolling = price.rolling("365D", closed="left")

        out = pd.DataFrame(
            {
                "row_id": g["row_id"].to_numpy(),
                f"{prefix}_sales_count_12m": rolling.count().to_numpy(),
                f"{prefix}_median_price_12m": rolling.median().to_numpy(),
            }
        )

        if prefix == "suburb":
            out[f"{prefix}_mean_price_12m"] = rolling.mean().to_numpy()
            out[f"{prefix}_p25_price_12m"] = rolling.quantile(0.25).to_numpy()
            out[f"{prefix}_p75_price_12m"] = rolling.quantile(0.75).to_numpy()

        pieces.append(out)

    if not pieces:
        return pd.DataFrame({"row_id": df["row_id"]})

    return pd.concat(pieces, ignore_index=True)


def add_rolling_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fast leakage-safe rolling market features.

    Features:
      suburb_sales_count_12m
      suburb_median_price_12m
      suburb_mean_price_12m
      suburb_p25_price_12m
      suburb_p75_price_12m
      postcode_sales_count_12m
      postcode_median_price_12m

    This replaces the previous O(n^2) row loop.
    """
    df = df.copy()
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
    df = df[df["sale_date"].notna()].copy()

    df = df.sort_values("sale_date").reset_index(drop=True)
    df["row_id"] = np.arange(len(df))

    suburb_features = _add_group_rolling_features(
        df=df,
        group_col="suburb",
        prefix="suburb",
    )

    postcode_features = _add_group_rolling_features(
        df=df,
        group_col="postcode",
        prefix="postcode",
    )

    out = df.merge(suburb_features, on="row_id", how="left")
    out = out.merge(postcode_features, on="row_id", how="left")

    global_median = float(out["sale_price"].median())

    count_cols = [
        "suburb_sales_count_12m",
        "postcode_sales_count_12m",
    ]

    price_cols = [
        "suburb_median_price_12m",
        "suburb_mean_price_12m",
        "suburb_p25_price_12m",
        "suburb_p75_price_12m",
        "postcode_median_price_12m",
    ]

    for col in count_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(float)

    for col in price_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(global_median)

    out = out.drop(columns=["row_id"])

    return out.sort_values("sale_date").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    sales_path = Path(config["data"]["sales_path"])
    output_path = Path(config["data"]["training_output_path"])

    df = pd.read_parquet(sales_path)
    print(f"Loaded sales rows: {len(df):,}")

    base = prepare_base_sales(df)
    print(f"Rows after cleaning: {len(base):,}")

    training = add_rolling_market_features(base)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    training.to_parquet(output_path, index=False)

    print(f"Saved training data: {output_path}")
    print(f"Rows: {len(training):,}")
    print(f"Date range: {training['sale_date'].min()} -> {training['sale_date'].max()}")
    print(f"Median sale price: ${training['sale_price'].median():,.0f}")


if __name__ == "__main__":
    main()