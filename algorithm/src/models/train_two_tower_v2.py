from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from algorithm.src.models.datasets import TripletTextDataset
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
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)


def get_device(cfg: dict) -> torch.device:
    device_cfg = cfg["training"]["device"]
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


class TripletMarginRetrievalLoss(nn.Module):
    def __init__(self, margin: float=0.2) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor,
    ) -> torch.Tensor:
        pos_sim = F.cosine_similarity(query_emb, positive_emb, dim=-1)
        neg_sim = F.cosine_similarity(query_emb, negative_emb, dim=-1)
        loss = torch.clamp(self.margin - pos_sim + neg_sim, min=0.0)
        return loss.mean()


def build_triplets(
    pairs_df: pd.DataFrame,
    candidate_text_col: str,
    seed: int = 42,
    prefer_hard_negatives: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed=seed)

    triplet_rows: list[dict] = []

    for query_id, qdf in pairs_df.groupby("query_id"):
        positives = qdf[qdf["label"] == 1].copy()
        hard_negatives = qdf[qdf["pair_type"] == "hard_negative"].copy()
        random_negatives = qdf[qdf["pair_type"] == "random_negative"].copy()

        if positives.empty:
            continue

        if prefer_hard_negatives and not hard_negatives.empty:
            neg_pool = hard_negatives
        elif not random_negatives.empty:
            neg_pool = random_negatives
        elif not hard_negatives.empty:
            neg_pool = hard_negatives
        else:
            continue

        neg_idx = rng.integers(0, len(neg_pool), size=len(positives))

        positives = positives.reset_index(drop=True)
        neg_pool = neg_pool.reset_index(drop=True)

        for i in range(len(positives)):
            p = positives.iloc[i]
            n = neg_pool.iloc[int(neg_idx[i])]

            triplet_rows.append(
                {
                    "query_id": p["query_id"],
                    "strategy": p["strategy"],
                    "query_text": p["query_text"],
                    "positive_rid": p["candidate_rid"],
                    "positive_text": p[candidate_text_col],
                    "positive_score": p["candidate_score"],
                    "negative_rid": n["candidate_rid"],
                    "negative_text": n[candidate_text_col],
                    "negative_score": n["candidate_score"],
                    "negative_pair_type": n["pair_type"],
                }
            )

    return pd.DataFrame(triplet_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="Experiment name in model.yaml")
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH, args.experiment)
    print(f"Running experiment: {args.experiment}")
    set_seed(int(cfg["training"]["seed"]))

    train_pairs_path = ROOT / cfg["data"]["train_pairs_path"]
    output_dir = ROOT / cfg["output"]["model_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading training pairs: {train_pairs_path}")
    pairs_df = pd.read_parquet(train_pairs_path)
    print(f"Pairs rows: {len(pairs_df)}")

    candidate_text_col = cfg["data"]["candidate_text_col"]
    if candidate_text_col not in pairs_df.columns:
        raise KeyError(
            f"Column '{candidate_text_col}' not found in training pairs. "
            f"Available columns: {list(pairs_df.columns)}"
        )

    triplets_df = build_triplets(
        pairs_df=pairs_df,
        candidate_text_col=candidate_text_col,
        seed=int(cfg["training"]["seed"]),
        prefer_hard_negatives=bool(cfg.get("sampling", {}).get("prefer_hard_negatives", True)),
    )

    print(f"Triplets rows: {len(triplets_df)}")
    if triplets_df.empty:
        raise RuntimeError("No triplets were built. Check training pairs and negative pools.")

    triplets_out = output_dir / "train_triplets_snapshot.parquet"
    triplets_df.to_parquet(triplets_out, index=False)
    print(f"Saved triplet snapshot to: {triplets_out}")

    dataset = TripletTextDataset(
        triplets_df=triplets_df,
        tokenizer_name=cfg["model"]["encoder_name"],
        max_length=int(cfg["data"]["max_text_length"]),
    )

    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"]["num_workers"]),
    )

    model = TwoTowerModel(
        encoder_name=cfg["model"]["encoder_name"],
        projection_dim=int(cfg["model"]["projection_dim"]),
        dropout=float(cfg["model"]["dropout"]),
        normalize=bool(cfg["model"]["normalize"]),
        share_tower_weights=bool(cfg["model"]["share_tower_weights"]),
    )

    device = get_device(cfg)
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    total_steps = len(loader) * int(cfg["training"]["epochs"])
    warmup_steps = int(total_steps * float(cfg["training"]["warmup_ratio"]))

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    loss_name = cfg["loss"]["name"]
    if loss_name != "triplet_margin":
        raise ValueError(f"Unsupported loss for v2 trainer: {loss_name}")

    criterion = TripletMarginRetrievalLoss(margin=float(cfg["loss"]["margin"]))

    history = {"train_loss": []}

    model.train()
    for epoch in range(int(cfg["training"]["epochs"])):
        epoch_losses = []

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}")
        for batch in pbar:
            optimizer.zero_grad()

            query_emb = model.encode_query(
                input_ids=batch["query_input_ids"].to(device),
                attention_mask=batch["query_attention_mask"].to(device),
            )

            positive_emb = model.encode_candidate(
                input_ids=batch["positive_input_ids"].to(device),
                attention_mask=batch["positive_attention_mask"].to(device),
            )

            negative_emb = model.encode_candidate(
                input_ids=batch["negative_input_ids"].to(device),
                attention_mask=batch["negative_attention_mask"].to(device),
            )

            loss = criterion(query_emb, positive_emb, negative_emb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(cfg["training"]["gradient_clip_norm"]),
            )

            optimizer.step()
            scheduler.step()

            loss_val = float(loss.item())
            epoch_losses.append(loss_val)
            pbar.set_postfix(loss=loss_val)

        mean_loss = float(np.mean(epoch_losses))
        history["train_loss"].append(mean_loss)
        print(f"Epoch {epoch + 1} mean loss: {mean_loss:.4f}")

    model_path = output_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to: {model_path}")

    metrics_path = ROOT / cfg["output"]["metrics_path"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()