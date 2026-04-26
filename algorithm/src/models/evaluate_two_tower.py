from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

from algorithm.src.models.model import TwoTowerModel


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


def load_config(path, experiment_name: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    defaults = raw["defaults"]
    exp_cfg = raw["experiments"][experiment_name]
    return deep_update(defaults, exp_cfg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg: dict) -> torch.device:
    device_cfg = cfg["training"]["device"]
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


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


@torch.no_grad()
def encode_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: TwoTowerModel,
    device: torch.device,
    max_length: int,
    batch_size: int,
    tower: str,
) -> np.ndarray:
    all_embs = []

    for start in tqdm(range(0, len(texts), batch_size), desc=f"Encoding {tower}"):
        batch_texts = texts[start : start + batch_size]

        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        if tower == "query":
            emb = model.encode_query(input_ids=input_ids, attention_mask=attention_mask)
        elif tower == "candidate":
            emb = model.encode_candidate(input_ids=input_ids, attention_mask=attention_mask)
        else:
            raise ValueError(f"Unknown tower: {tower}")

        all_embs.append(emb.cpu().numpy())

    return np.vstack(all_embs)


def retrieve_top_k(
    query_idx: int,
    k: int,
    similarity_matrix: np.ndarray,
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    sims = similarity_matrix[query_idx]
    top_idx = np.argsort(-sims)[:k]

    result = candidates.iloc[top_idx].copy()
    result["similarity"] = sims[top_idx]
    result["query_id"] = queries.iloc[query_idx]["query_id"]
    result["query_text"] = queries.iloc[query_idx]["text"]
    result["strategy"] = queries.iloc[query_idx]["strategy"]
    return result


def top_k_strategy_match_rate(
    query_idx: int,
    k: int,
    similarity_matrix: np.ndarray,
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
) -> float:
    strategy = queries.iloc[query_idx]["strategy"]
    topk = retrieve_top_k(query_idx, k, similarity_matrix, queries, candidates)
    return float((topk["top_strategy"] == strategy).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="Experiment name in model.yaml")
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH, args.experiment)
    print(f"Evaluating experiment: {args.experiment}")
    set_seed(int(cfg["training"]["seed"]))

    candidate_path = ROOT / cfg["data"]["candidate_sites_path"]
    query_path = ROOT / cfg["data"]["query_path"]
    model_dir = ROOT / cfg["output"]["model_dir"]
    model_path = model_dir / "model.pt"

    output_dir = model_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading candidates: {candidate_path}")
    candidates = pd.read_parquet(candidate_path)
    print(f"Candidates: {len(candidates)}")

    print(f"Reading queries: {query_path}")
    queries = load_queries(query_path)
    print(f"Queries: {len(queries)}")

    query_text_col = cfg["data"]["query_text_col"]
    candidate_text_col = cfg["data"]["candidate_text_col"]

    queries[query_text_col] = queries["text"].fillna("").astype(str)
    candidates[candidate_text_col] = candidates[candidate_text_col].fillna("").astype(str)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])

    model = TwoTowerModel(
        encoder_name=cfg["model"]["encoder_name"],
        projection_dim=int(cfg["model"]["projection_dim"]),
        dropout=float(cfg["model"]["dropout"]),
        normalize=bool(cfg["model"]["normalize"]),
        share_tower_weights=bool(cfg["model"]["share_tower_weights"]),
    )

    print(f"Loading model weights: {model_path}")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    device = get_device(cfg)
    model.to(device)
    model.eval()

    query_texts = queries[query_text_col].tolist()
    candidate_texts = candidates[candidate_text_col].tolist()

    query_embeddings = encode_texts(
        texts=query_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=int(cfg["data"]["max_text_length"]),
        batch_size=int(cfg["training"]["batch_size"]),
        tower="query",
    )

    candidate_embeddings = encode_texts(
        texts=candidate_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=int(cfg["data"]["max_text_length"]),
        batch_size=int(cfg["training"]["batch_size"]),
        tower="candidate",
    )

    similarity_matrix = query_embeddings @ candidate_embeddings.T
    print("Similarity matrix shape:", similarity_matrix.shape)

    match_rows = []
    score_check_rows = []
    all_topk = []

    for i in range(len(queries)):
        q = queries.iloc[i]
        strategy = q["strategy"]
        score_col = score_col_for_strategy(strategy)

        top10_match = top_k_strategy_match_rate(i, 10, similarity_matrix, queries, candidates)
        top20_match = top_k_strategy_match_rate(i, 20, similarity_matrix, queries, candidates)

        match_rows.append(
            {
                "query_id": q["query_id"],
                "strategy": strategy,
                "top10_match_rate": top10_match,
                "top20_match_rate": top20_match,
            }
        )

        top20 = retrieve_top_k(i, 20, similarity_matrix, queries, candidates)
        score_check_rows.append(
            {
                "query_id": q["query_id"],
                "strategy": strategy,
                "top20_mean_score": float(top20[score_col].mean()),
                "top20_median_score": float(top20[score_col].median()),
                "top20_high_score_rate": float((top20[score_col] >= 70).mean()),
            }
        )

        top50 = retrieve_top_k(i, 50, similarity_matrix, queries, candidates)
        all_topk.append(top50)

    match_df = pd.DataFrame(match_rows)
    score_check_df = pd.DataFrame(score_check_rows)
    retrieval_df = pd.concat(all_topk, ignore_index=True)

    summary = {
        "top10_match_rate_mean": float(match_df["top10_match_rate"].mean()),
        "top20_match_rate_mean": float(match_df["top20_match_rate"].mean()),
        "top20_mean_score_mean": float(score_check_df["top20_mean_score"].mean()),
        "top20_median_score_mean": float(score_check_df["top20_median_score"].mean()),
        "top20_high_score_rate_mean": float(score_check_df["top20_high_score_rate"].mean()),
    }

    print("\nMatch DF:")
    print(match_df)
    print("\nScore Check DF:")
    print(score_check_df)
    print("\nSummary:")
    print(summary)

    match_path = output_dir / "match_metrics.parquet"
    score_path = output_dir / "score_metrics.parquet"
    retrieval_path = output_dir / "top50_retrieval.parquet"
    summary_path = output_dir / "summary.json"

    match_df.to_parquet(match_path, index=False)
    score_check_df.to_parquet(score_path, index=False)
    retrieval_df.to_parquet(retrieval_path, index=False)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved match metrics to: {match_path}")
    print(f"Saved score metrics to: {score_path}")
    print(f"Saved retrieval results to: {retrieval_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()