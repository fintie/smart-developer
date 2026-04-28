from __future__ import annotations
import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from algorithm.src.models.datasets import RerankerDataset
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


def prepare_feature_matrix(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    binary_cols: list[str],
    scaler: StandardScaler | None = None,
    category_levels: dict[str, list[str]] | None = None,
    fit: bool = False,
) -> tuple[np.ndarray, StandardScaler, dict[str, list[str]], list[str]]:
    work = df.copy()

    # Numeric
    for col in numeric_cols:
        if col not in work.columns:
            work[col] = 0.0
    numeric_df = work[numeric_cols].copy().fillna(0.0).astype(float)

    if scaler is None:
        scaler = StandardScaler()
    if fit:
        numeric_arr = scaler.fit_transform(numeric_df)
    else:
        numeric_arr = scaler.transform(numeric_df)

    # Binary
    for col in binary_cols:
        if col not in work.columns:
            work[col] = 0
    binary_df = work[binary_cols].copy().fillna(0).astype(float)
    binary_arr = binary_df.to_numpy()

    # Categorical
    cat_feature_names: list[str] = []
    cat_arrays: list[np.ndarray] = []

    if category_levels is None:
        category_levels = {}

    for col in categorical_cols:
        if col not in work.columns:
            work[col] = "unknown"

        values = work[col].fillna("unknown").astype(str)

        if fit:
            levels = sorted(values.unique().tolist())
            category_levels[col] = levels
        else:
            levels = category_levels[col]

        one_hot = np.zeros((len(work), len(levels)), dtype=np.float32)
        level_to_idx = {level: i for i, level in enumerate(levels)}

        for row_idx, val in enumerate(values):
            if val in level_to_idx:
                one_hot[row_idx, level_to_idx[val]] = 1.0

        cat_arrays.append(one_hot)
        cat_feature_names.extend([f"{col}={level}" for level in levels])

    parts = [numeric_arr, binary_arr]
    feature_names = numeric_cols + binary_cols

    if cat_arrays:
        parts.extend(cat_arrays)
        feature_names.extend(cat_feature_names)

    X = np.concatenate(parts, axis=1).astype(np.float32)
    return X, scaler, category_levels, feature_names


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> dict[str, float]:
    model.eval()

    losses = []
    probs_all = []
    labels_all = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            logits = model(features)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

            losses.append(float(loss.item()))
            probs_all.append(probs.cpu().numpy())
            labels_all.append(labels.cpu().numpy())

    probs = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    preds = (probs >= 0.5).astype(int)

    accuracy = float((preds == labels).mean())
    pos_rate = float(preds.mean())

    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy,
        "pred_positive_rate": pos_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="Experiment name in model.yaml")
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH, args.experiment)
    print(f"Running experiment: {args.experiment}")

    set_seed(int(cfg["training"]["seed"]))
    device = get_device(cfg)

    train_path = ROOT / cfg["data"]["reranker_train_path"]
    eval_path = ROOT / cfg["data"]["reranker_eval_path"]

    output_dir = ROOT / cfg["output"]["model_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading train dataset: {train_path}")
    train_df = pd.read_parquet(train_path)
    print(f"Train rows: {len(train_df)}")

    print(f"Reading eval dataset: {eval_path}")
    eval_df = pd.read_parquet(eval_path)
    print(f"Eval rows: {len(eval_df)}")

    target_col = cfg["data"]["target_col"]
    numeric_cols = list(cfg["data"]["numeric_feature_cols"])
    categorical_cols = list(cfg["data"]["categorical_feature_cols"])
    binary_cols = list(cfg["data"]["binary_feature_cols"])

    X_train, scaler, category_levels, feature_names = prepare_feature_matrix(
        train_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        binary_cols=binary_cols,
        scaler=None,
        category_levels=None,
        fit=True,
    )
    y_train = train_df[target_col].astype(float).to_numpy()

    X_eval, _, _, _ = prepare_feature_matrix(
        eval_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        binary_cols=binary_cols,
        scaler=scaler,
        category_levels=category_levels,
        fit=False,
    )
    y_eval = eval_df[target_col].astype(float).to_numpy()

    print(f"Input feature dim: {X_train.shape[1]}")

    # Override input_dim from actual processed dim
    model_cfg = deepcopy(cfg["model"])
    model_cfg["input_dim"] = int(X_train.shape[1])

    model = DCNReranker(
        input_dim=int(model_cfg["input_dim"]),
        cross_layers=int(model_cfg["cross_layers"]),
        deep_hidden_dims=list(model_cfg["deep_hidden_dims"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)

    train_dataset = RerankerDataset(X_train, y_train)
    eval_dataset = RerankerDataset(X_eval, y_eval)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"]["num_workers"]),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss()

    history: dict[str, list[float]] = {
        "train_loss": [],
        "eval_loss": [],
        "eval_accuracy": [],
        "eval_pred_positive_rate": [],
    }

    best_eval_loss = float("inf")
    best_state = None

    for epoch in range(int(cfg["training"]["epochs"])):
        model.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in pbar:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(cfg["training"]["gradient_clip_norm"]),
            )

            optimizer.step()

            loss_val = float(loss.item())
            epoch_losses.append(loss_val)
            pbar.set_postfix(loss=loss_val)

        train_loss = float(np.mean(epoch_losses))
        eval_metrics = evaluate_model(model, eval_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["eval_loss"].append(eval_metrics["loss"])
        history["eval_accuracy"].append(eval_metrics["accuracy"])
        history["eval_pred_positive_rate"].append(eval_metrics["pred_positive_rate"])

        print(
            f"Epoch {epoch + 1} | "
            f"train_loss={train_loss:.4f} | "
            f"eval_loss={eval_metrics['loss']:.4f} | "
            f"eval_acc={eval_metrics['accuracy']:.4f}"
        )

        if eval_metrics["loss"] < best_eval_loss:
            best_eval_loss = eval_metrics["loss"]
            best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    model_path = output_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to: {model_path}")

    preprocessing_path = output_dir / "preprocessing.json"
    preprocessing_payload = {
        "numeric_feature_cols": numeric_cols,
        "categorical_feature_cols": categorical_cols,
        "binary_feature_cols": binary_cols,
        "feature_names": feature_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "category_levels": category_levels,
        "input_dim": X_train.shape[1],
    }
    with preprocessing_path.open("w", encoding="utf-8") as f:
        json.dump(preprocessing_payload, f, indent=2)
    print(f"Saved preprocessing metadata to: {preprocessing_path}")

    metrics_path = ROOT / cfg["output"]["metrics_path"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()