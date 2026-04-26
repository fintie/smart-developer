from __future__ import annotations
import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer

from algorithm.src.models.model import TwoTowerModel
from algorithm.src.explanation.evidence import build_explanation_payload
from algorithm.src.explanation.pipeline import explain_row


ROOT = Path(__file__).resolve().parents[3]
MODEL_CONFIG_PATH = ROOT / "algorithm" / "configs" / "model.yaml"

DEFAULT_EXPERIMENT = "two_tower_v1"


def deep_update(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


def load_model_config(path: Path, experiment_name: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    defaults = raw["defaults"]
    experiments = raw["experiments"]

    if experiment_name not in experiments:
        raise KeyError(
            f"Unknown experiment '{experiment_name}'. Available: {list(experiments.keys())}"
        )

    return deep_update(defaults, experiments[experiment_name])


def score_col_for_strategy(strategy: str) -> str:
    return f"{strategy}_score"


def normalize_base_address(address: str) -> str:
    if not isinstance(address, str):
        return ""

    s = address.strip().upper()

    # remove prefix of unit/level/strata in front，e.g. 909/3 XXX -> 3 XXX
    s = re.sub(r"^\s*[^/\s]+/\s*", "", s)
    # Compress extra spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def minmax_norm(values: pd.Series) -> pd.Series:
    if values.empty:
        return values
    vmin = values.min()
    vmax = values.max()
    if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - vmin) / (vmax - vmin)


@dataclass
class RetrievalRequest:
    strategy: str
    query_text: str
    top_k: int = 20
    recall_k: int = 200
    alpha: float = 0.5  # retrieval similarity weight
    beta: float = 0.5   # heuristic strategy score weight
    dedupe_by_address: bool = True
    dedupe_address_col: str = "address"
    attach_explanations: bool = False
    explanation_model: str = "llama3.1:8b-instruct-q4_K_M"


class HybridRetriever:
    def __init__(
        self,
        experiment: str = DEFAULT_EXPERIMENT,
        candidate_path: Path | None = None,
        device: str = "auto",
    ) -> None:
        self.cfg = load_model_config(MODEL_CONFIG_PATH, experiment)
        self.experiment = experiment

        if candidate_path is None:
            candidate_path = ROOT / self.cfg["data"]["candidate_sites_path"]

        self.candidate_path = candidate_path
        self.candidate_text_col = self.cfg["data"]["candidate_text_col"]
        self.max_text_length = int(self.cfg["data"]["max_text_length"])

        self.device = self._resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["model"]["encoder_name"])
        self.model = self._load_model()
        self.candidates = self._load_candidates()
        self.candidate_embeddings = self._encode_candidates()

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_model(self) -> TwoTowerModel:
        model = TwoTowerModel(
            encoder_name=self.cfg["model"]["encoder_name"],
            projection_dim=int(self.cfg["model"]["projection_dim"]),
            dropout=float(self.cfg["model"]["dropout"]),
            normalize=bool(self.cfg["model"]["normalize"]),
            share_tower_weights=bool(self.cfg["model"]["share_tower_weights"]),
        )

        model_path = ROOT / self.cfg["output"]["model_dir"] / "model.pt"
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _load_candidates(self) -> pd.DataFrame:
        df = pd.read_parquet(self.candidate_path).copy()
        if self.candidate_text_col not in df.columns:
            raise KeyError(
                f"Candidate text column '{self.candidate_text_col}' not found in candidate table."
            )
        df[self.candidate_text_col] = df[self.candidate_text_col].fillna("").astype(str)
        return df

    @torch.no_grad()
    def _encode_texts(self, texts: list[str], tower: str, batch_size: int = 64) -> np.ndarray:
        outputs: list[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            if tower == "query":
                emb = self.model.encode_query(input_ids=input_ids, attention_mask=attention_mask)
            elif tower == "candidate":
                emb = self.model.encode_candidate(input_ids=input_ids, attention_mask=attention_mask)
            else:
                raise ValueError(f"Unknown tower: {tower}")

            outputs.append(emb.cpu().numpy())

        return np.vstack(outputs)

    def _encode_candidates(self) -> np.ndarray:
        texts = self.candidates[self.candidate_text_col].tolist()
        return self._encode_texts(texts, tower="candidate")

    def _encode_query(self, query_text: str) -> np.ndarray:
        emb = self._encode_texts([query_text], tower="query")
        return emb[0]

    @staticmethod
    def _compact_columns(strategy: str) -> list[str]:
        score_col = score_col_for_strategy(strategy)
        return [
            "RID",
            "address",
            "primary_zoning_code",
            "lot_size_band",
            "constraint_severity_band",
            "station_distance_band",
            "top_strategy",
            "top_strategy_score",
            score_col,
            "retrieval_similarity",
            "fusion_score",
            "explanation",
        ]

    def retrieve(self, request: RetrievalRequest) -> pd.DataFrame:
        strategy = request.strategy
        score_col = score_col_for_strategy(strategy)

        if score_col not in self.candidates.columns:
            raise KeyError(f"Missing score column '{score_col}' in candidate table.")

        query_emb = self._encode_query(request.query_text)
        sims = self.candidate_embeddings @ query_emb

        recall_k = min(request.recall_k, len(self.candidates))
        top_idx = np.argsort(-sims)[:recall_k]

        recalled = self.candidates.iloc[top_idx].copy()
        recalled["retrieval_similarity"] = sims[top_idx]

        recalled["sim_norm"] = minmax_norm(recalled["retrieval_similarity"].astype(float))
        recalled["score_norm"] = minmax_norm(recalled[score_col].astype(float))
        recalled["fusion_score"] = (
            request.alpha * recalled["sim_norm"] + request.beta * recalled["score_norm"]
        )

        reranked = recalled.sort_values("fusion_score", ascending=False).copy()

        if request.dedupe_by_address:
            reranked = dedupe_results_by_address(
                reranked,
                address_col=request.dedupe_address_col,
            )

        reranked = reranked.head(request.top_k).copy()
        reranked["strategy"] = strategy
        reranked["query_text"] = request.query_text
        if request.attach_explanations:
            explanations = []
            for _, row in reranked.iterrows():
                try:
                    explanations.append(
                        explain_row(
                            row=row,
                            strategy=strategy,
                            model=request.explanation_model,
                        )
                    )
                except Exception as e:
                    explanations.append(f"[Explanation generation failed: {e}]")
            reranked["explanation"] = explanations
        else:
            reranked["explanation"] = None

        preferred_cols = [
            "RID",
            "address",
            "strategy",
            "query_text",
            "primary_zoning_code",
            "lot_size_band",
            "constraint_severity_band",
            "station_distance_band",
            "top_strategy",
            "top_strategy_score",
            score_col,
            "retrieval_similarity",
            "fusion_score",
            "explanation",
        ]
        ordered_cols = [c for c in preferred_cols if c in reranked.columns] + [
            c for c in reranked.columns if c not in preferred_cols
        ]
        return reranked[ordered_cols]

    def retrieve_as_dicts(self, request: RetrievalRequest) -> list[dict[str, Any]]:
        df = self.retrieve(request)
        return df.to_dict(orient="records")


def dedupe_results_by_address(
    df: pd.DataFrame,
    address_col: str = "address",
) -> pd.DataFrame:
    if address_col not in df.columns:
        return df

    out = df.copy()
    out[address_col] = out[address_col].fillna("").astype(str).str.strip()
    out["_dedupe_key"] = out[address_col].apply(normalize_base_address)

    missing_mask = out["_dedupe_key"].eq("")
    if missing_mask.any():
        out.loc[missing_mask, "_dedupe_key"] = (
            "__missing_address__" + out.loc[missing_mask].index.astype(str)
        )

    out = out.drop_duplicates(subset=["_dedupe_key"], keep="first").copy()
    out = out.drop(columns=["_dedupe_key"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--query-text", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--recall-k", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--with-explanations", action="store_true")
    parser.add_argument("--explanation-model", default="llama3.1:8b-instruct-q4_K_M")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    retriever = HybridRetriever(experiment=args.experiment)
    request = RetrievalRequest(
        strategy=args.strategy,
        query_text=args.query_text,
        top_k=args.top_k,
        recall_k=args.recall_k,
        alpha=args.alpha,
        beta=args.beta,
        dedupe_by_address=not args.no_dedupe,
        attach_explanations=args.with_explanations,
        explanation_model=args.explanation_model,
    )
    results = retriever.retrieve(request)

    if args.full:
        print(results.head(args.top_k).to_string(index=False))
    else:
        compact_cols = HybridRetriever._compact_columns(args.strategy)
        compact_cols = [c for c in compact_cols if c in results.columns]
        print(results[compact_cols].head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()