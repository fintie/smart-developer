from __future__ import annotations
import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler

from algorithm.src.models.dcn_reranker import DCNReranker


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "algorithm" / "configs" / "model.yaml"


def deep_update(base: dict, override: dict) -> dict:
    result = deepcopy(base)
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


def get_device(cfg: dict) -> torch.device:
    device_cfg = cfg["training"]["device"]
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def score_col_for_strategy(strategy: str) -> str:
    return f"{strategy}_score"


def prepare_feature_matrix_from_metadata(
    df: pd.DataFrame,
    preprocessing: dict[str, Any],
) -> np.ndarray:
    work = df.copy()

    numeric_cols = list(preprocessing["numeric_feature_cols"])
    categorical_cols = list(preprocessing["categorical_feature_cols"])
    binary_cols = list(preprocessing["binary_feature_cols"])
    category_levels = preprocessing["category_levels"]

    for col in numeric_cols:
        if col not in work.columns:
            work[col] = 0.0
    numeric_df = work[numeric_cols].copy().fillna(0.0).astype(float)

    scaler = StandardScaler()
    scaler.mean_ = np.array(preprocessing["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.array(preprocessing["scaler_scale"], dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(numeric_cols)

    numeric_arr = scaler.transform(numeric_df)

    for col in binary_cols:
        if col not in work.columns:
            work[col] = 0
    binary_arr = work[binary_cols].copy().fillna(0).astype(float).to_numpy()

    cat_arrays: list[np.ndarray] = []
    for col in categorical_cols:
        if col not in work.columns:
            work[col] = "unknown"

        values = work[col].fillna("unknown").astype(str)
        levels = category_levels[col]
        one_hot = np.zeros((len(work), len(levels)), dtype=np.float32)
        level_to_idx = {level: i for i, level in enumerate(levels)}

        for row_idx, val in enumerate(values):
            if val in level_to_idx:
                one_hot[row_idx, level_to_idx[val]] = 1.0

        cat_arrays.append(one_hot)

    parts = [numeric_arr, binary_arr]
    if cat_arrays:
        parts.extend(cat_arrays)

    X = np.concatenate(parts, axis=1).astype(np.float32)
    return X


@torch.no_grad()
def predict_logits(
    model: DCNReranker,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    outputs = []

    for start in range(0, len(X), batch_size):
        batch = torch.tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
        logits = model(batch)
        outputs.append(logits.cpu().numpy())

    return np.concatenate(outputs)


def evaluate_retrieval_df(
    retrieval_df: pd.DataFrame,
    queries_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    match_rows = []
    score_rows = []

    for _, q in queries_df.iterrows():
        qid = q["query_id"]
        strategy = q["strategy"]
        score_col = score_col_for_strategy(strategy)

        qdf = retrieval_df[retrieval_df["query_id"] == qid].copy()
        top10 = qdf.head(10)
        top20 = qdf.head(20)

        match_rows.append(
            {
                "query_id": qid,
                "strategy": strategy,
                "top10_match_rate": float((top10["top_strategy"] == strategy).mean()),
                "top20_match_rate": float((top20["top_strategy"] == strategy).mean()),
            }
        )

        score_rows.append(
            {
                "query_id": qid,
                "strategy": strategy,
                "top20_mean_score": float(top20[score_col].mean()),
                "top20_median_score": float(top20[score_col].median()),
                "top20_high_score_rate": float((top20[score_col] >= 70).mean()),
            }
        )

    return pd.DataFrame(match_rows), pd.DataFrame(score_rows)


def summarise_metrics(match_df: pd.DataFrame, score_df: pd.DataFrame) -> dict[str, float]:
    return {
        "top10_match_rate_mean": float(match_df["top10_match_rate"].mean()),
        "top20_match_rate_mean": float(match_df["top20_match_rate"].mean()),
        "top20_mean_score_mean": float(score_df["top20_mean_score"].mean()),
        "top20_median_score_mean": float(score_df["top20_median_score"].mean()),
        "top20_high_score_rate_mean": float(score_df["top20_high_score_rate"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="Experiment name in model.yaml")
    parser.add_argument("--recall-experiment", default="two_tower_v1",
                        help="First-stage retrieval experiment used as input to the reranker")
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH, args.experiment)
    device = get_device(cfg)

    query_path = ROOT / cfg["data"]["query_path"]
    reranker_model_dir = ROOT / cfg["output"]["model_dir"]

    retrieval_path = (
        ROOT
        / "algorithm"
        / "artifacts"
        / "models"
        / args.recall_experiment
        / "evaluation"
        / "top50_retrieval.parquet"
    )

    model_path = reranker_model_dir / "model.pt"
    preprocessing_path = reranker_model_dir / "preprocessing.json"
    eval_dir = reranker_model_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading retrieval results: {retrieval_path}")
    retrieval_df = pd.read_parquet(retrieval_path)
    print(f"Rows: {len(retrieval_df)}")

    print(f"Reading queries: {query_path}")
    queries_df = pd.read_json(query_path, lines=True)
    print(f"Queries: {len(queries_df)}")

    print(f"Reading preprocessing metadata: {preprocessing_path}")
    with preprocessing_path.open("r", encoding="utf-8") as f:
        preprocessing = json.load(f)

    print(f"Preparing feature matrix...")
    X = prepare_feature_matrix_from_metadata(retrieval_df, preprocessing)
    print(f"Feature matrix shape: {X.shape}")

    model = DCNReranker(
        input_dim=int(preprocessing["input_dim"]),
        cross_layers=int(cfg["model"]["cross_layers"]),
        deep_hidden_dims=list(cfg["model"]["deep_hidden_dims"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    print(f"Loading model weights: {model_path}")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print("Scoring reranker logits...")
    logits = predict_logits(model, X, device=device, batch_size=512)
    probs = 1.0 / (1.0 + np.exp(-logits))

    reranked_df = retrieval_df.copy()
    reranked_df["dcn_logit"] = logits
    reranked_df["dcn_prob"] = probs

    reranked_parts = []
    for query_id, qdf in reranked_df.groupby("query_id", sort=False):
        qdf = qdf.sort_values("dcn_prob", ascending=False).copy()
        reranked_parts.append(qdf)

    reranked_df = pd.concat(reranked_parts, ignore_index=True)

    match_df, score_df = evaluate_retrieval_df(reranked_df, queries_df)
    summary = summarise_metrics(match_df, score_df)

    print("\nMatch DF:")
    print(match_df)
    print("\nScore DF:")
    print(score_df)
    print("\nSummary:")
    print(summary)

    match_path = eval_dir / "match_metrics.parquet"
    score_path = eval_dir / "score_metrics.parquet"
    reranked_path = eval_dir / "top50_reranked.parquet"
    summary_path = eval_dir / "summary.json"

    match_df.to_parquet(match_path, index=False)
    score_df.to_parquet(score_path, index=False)
    reranked_df.to_parquet(reranked_path, index=False)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved match metrics to: {match_path}")
    print(f"Saved score metrics to: {score_path}")
    print(f"Saved reranked retrieval to: {reranked_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()