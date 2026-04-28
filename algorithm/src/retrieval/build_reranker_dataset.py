from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "algorithm" / "configs" / "model.yaml"


def deep_update(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: Path, experiment_name: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    defaults = raw["defaults"]
    experiments = raw["experiments"]

    if experiment_name not in experiments:
        raise KeyError(
            f"Unknown experiment '{experiment_name}'. Available: {list(experiments.keys())}"
        )

    return deep_update(defaults, experiments[experiment_name])


def load_queries(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def score_col_for_strategy(strategy: str) -> str:
    return f"{strategy}_score"


def build_label(strategy_score: float, positive_threshold: float = 70.0) -> int:
    return int(float(strategy_score) >= positive_threshold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="Experiment name in model.yaml")
    parser.add_argument("--recall-experiment", default="two_tower_v1",
                        help="First-stage retrieval experiment used to build reranker dataset",)
    parser.add_argument("--eval-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positive-threshold", type=float, default=70.0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config(CONFIG_PATH, args.experiment)

    query_path = ROOT / cfg["data"]["query_path"]
    candidate_path = ROOT / cfg["data"]["candidate_sites_path"]

    retrieval_path = (
        ROOT
        / "algorithm"
        / "artifacts"
        / "models"
        / args.recall_experiment
        / "evaluation"
        / "top50_retrieval.parquet"
    )

    train_out = ROOT / cfg["data"]["reranker_train_path"]
    eval_out = ROOT / cfg["data"]["reranker_eval_path"]

    train_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading queries: {query_path}")
    queries = load_queries(query_path)
    print(f"Queries: {len(queries)}")

    print(f"Reading candidates: {candidate_path}")
    candidates = pd.read_parquet(candidate_path)
    print(f"Candidates: {len(candidates)}")

    print(f"Reading retrieval results: {retrieval_path}")
    retrieval = pd.read_parquet(retrieval_path)
    print(f"Retrieved rows: {len(retrieval)}")

    # Keep only columns we need from candidates in case retrieval output is incomplete in future
    candidate_keep_cols = [
        "RID",
        "address",
        "primary_zoning_code",
        "zoning_band",
        "lot_size_band",
        "constraint_severity_band",
        "station_distance_band",
        "lot_size_proxy_sqm",
        "mixed_zoning_flag",
        "heritage_flag",
        "bushfire_risk_level",
        "flood_flag",
        "within_800m_catchment",
        "top_strategy",
        "top_strategy_score",
        "single_dwelling_rebuild_score",
        "assembly_opportunity_score",
        "granny_flat_score",
        "land_bank_hold_score",
        "townhouse_multi_dwelling_score",
        "low_rise_apartment_score",
        "dual_occupancy_score",
    ]
    candidate_keep_cols = [c for c in candidate_keep_cols if c in candidates.columns]
    candidates_small = candidates[candidate_keep_cols].copy()

    # Merge retrieval output with canonical candidate features
    if "RID" not in retrieval.columns:
        raise KeyError("Expected 'RID' in retrieval results.")

    reranker_df = retrieval.merge(
        candidates_small,
        on="RID",
        how="left",
        suffixes=("", "_cand"),
    )

    if "strategy" not in reranker_df.columns:
        raise KeyError("Expected 'strategy' column in retrieval results.")

    # Build strategy_score + binary label
    strategy_scores = []
    labels = []

    for _, row in reranker_df.iterrows():
        strategy = row["strategy"]
        score_col = score_col_for_strategy(strategy)

        if score_col not in row.index:
            raise KeyError(f"Missing strategy score column '{score_col}' in reranker dataframe.")

        strategy_score = float(row[score_col])
        label = build_label(strategy_score, positive_threshold=args.positive_threshold)

        strategy_scores.append(strategy_score)
        labels.append(label)

    reranker_df["strategy_score"] = strategy_scores
    reranker_df["label"] = labels

    # Optional: keep query text from query table as a clean source of truth
    reranker_df = reranker_df.drop(columns=["query_text"], errors="ignore").merge(
        queries[["query_id", "text"]],
        on="query_id",
        how="left",
    )
    reranker_df = reranker_df.rename(columns={"text": "query_text"})

    # Keep only useful columns
    keep_cols = [
        "query_id",
        "strategy",
        "query_text",
        "RID",
        "address",
        "retrieval_similarity",
        "fusion_score",
        "strategy_score",
        "label",
        "primary_zoning_code",
        "zoning_band",
        "lot_size_band",
        "constraint_severity_band",
        "station_distance_band",
        "lot_size_proxy_sqm",
        "mixed_zoning_flag",
        "heritage_flag",
        "bushfire_risk_level",
        "flood_flag",
        "within_800m_catchment",
        "top_strategy",
        "top_strategy_score",
    ]
    keep_cols = [c for c in keep_cols if c in reranker_df.columns]
    reranker_df = reranker_df[keep_cols].copy()

    print("Label distribution:")
    print(reranker_df["label"].value_counts(dropna=False))

    # Split by query_id to reduce leakage across train/eval
    unique_query_ids = reranker_df["query_id"].dropna().unique().tolist()

    train_qids, eval_qids = train_test_split(
        unique_query_ids,
        test_size=args.eval_size,
        random_state=args.seed,
    )

    train_df = reranker_df[reranker_df["query_id"].isin(train_qids)].copy()
    eval_df = reranker_df[reranker_df["query_id"].isin(eval_qids)].copy()

    print(f"Train rows: {len(train_df)}")
    print(f"Eval rows: {len(eval_df)}")

    print(f"Writing train dataset: {train_out}")
    train_df.to_parquet(train_out, index=False)

    print(f"Writing eval dataset: {eval_out}")
    eval_df.to_parquet(eval_out, index=False)

    print("Done.")


if __name__ == "__main__":
    main()