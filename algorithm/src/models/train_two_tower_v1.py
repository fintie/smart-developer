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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from algorithm.src.models.datasets import PositivePairDataset
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


class MultipleNegativesLoss(nn.Module):
    def __init__(self, temperature: float=0.05) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query_emb: torch.Tensor,
        candidate_emb: torch.Tensor
    ) -> torch.Tensor:
        logits = query_emb @ candidate_emb.T
        logits = logits / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="Experiment name in model.yaml")
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH, args.experiment)
    print(f"Running experiment: {args.experiment}")
    set_seed(cfg["training"]["seed"])

    train_pairs_path = ROOT / cfg["data"]["train_pairs_path"]
    output_dir = ROOT / cfg["output"]["model_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading training pairs: {train_pairs_path}")
    df = pd.read_parquet(train_pairs_path)

    # First-pass training: use positive pairs only
    df = df[df[cfg["data"]["label_col"]] == 1].copy()
    df = df.rename(columns={"query_text": cfg["data"]["query_text_col"]})
    print(f"Positive training pairs: {len(df)}")

    dataset = PositivePairDataset(
        df=df,
        tokenizer_name=cfg["model"]["encoder_name"],
        query_text_col=cfg["data"]["query_text_col"],
        candidate_text_col=cfg["data"]["candidate_text_col"],
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

    criterion = MultipleNegativesLoss(temperature=float(cfg["loss"]["temperature"]))

    history = {"train_loss": []}

    model.train()
    for epoch in range(int(cfg["training"]["epochs"])):
        epoch_losses = []

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            optimizer.zero_grad()

            query_emb = model.encode_query(
                input_ids=batch["query_input_ids"],
                attention_mask=batch["query_attention_mask"].to(device),
            )
            candidate_emb = model.encode_candidate(
                input_ids=batch["candidate_input_ids"],
                attention_mask=batch["candidate_attention_mask"].to(device),
            )

            loss = criterion(query_emb, candidate_emb)
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
        print(f"Epoch {epoch+1} mean loss: {mean_loss:.4f}")

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