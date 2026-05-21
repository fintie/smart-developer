from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from algorithm.src.economics.opportunity.economics_pipeline import EconomicsPipeline
from algorithm.src.inference import __name__ as _inference_package_name  # noqa: F401
from algorithm.src.policy.policy_scorer import PolicyScorer
from algorithm.src.ranking.opportunity_fusion import (
    add_agent_pitch,
    apply_opportunity_fusion,
)
from algorithm.src.retrieval.hybrid_retrieve import HybridRetriever, RetrievalRequest


DEFAULT_RETRIEVAL_MODEL = "two_tower_v1"
DEFAULT_RERANKING_MODEL = "dcn_reranker_v1"
DEFAULT_EXPLANATION_MODEL = "llama3.1:8b-instruct-q4_K_M"


VALID_RANKING_PROFILES = {
    "balanced",
    "policy_upside",
    "budget_sensitive",
    "high_value",
}


@dataclass
class PredictionRequest:
    strategy: str
    query_text: str

    top_k: int = 5
    recall_k: int = 10000

    with_explanations: bool = True

    retrieval_model: str = DEFAULT_RETRIEVAL_MODEL
    use_dcn_reranker: bool = True
    reranking_model: str = DEFAULT_RERANKING_MODEL

    alpha: float = 0.5
    beta: float = 0.5

    dedupe_by_address: bool = True

    locality: str | None = None
    address_contains: str | None = None

    # With locality pre-filtering, fallback should usually stay False for product
    # safety. If a locality has no matching candidate, return empty results rather
    # than unrelated global recommendations.
    location_fallback: bool = False

    ranking_profile: str = "balanced"
    explanation_model: str = DEFAULT_EXPLANATION_MODEL


class SmartDeveloperPredictor:
    """
    Product-facing inference wrapper for Smart Developer.

    Current inference stack:
        locality/address candidate pre-filter
        -> two-tower retrieval
        -> optional DCN reranking
        -> dedupe by base site address
        -> policy scoring + policy RAG evidence
        -> economics scoring
        -> opportunity fusion with ranking profiles
        -> agent-facing pitch generation
    """

    def __init__(
        self,
        default_retrieval_model: str = DEFAULT_RETRIEVAL_MODEL,
        default_reranking_model: str = DEFAULT_RERANKING_MODEL,
        default_explanation_model: str = DEFAULT_EXPLANATION_MODEL,
    ) -> None:
        self.default_retrieval_model = default_retrieval_model
        self.default_reranking_model = default_reranking_model
        self.default_explanation_model = default_explanation_model

        self.policy_scorer = PolicyScorer(
            enable_rag_evidence=True,
            rag_top_k=3,
        )

        self.economics_pipeline = EconomicsPipeline(
            use_ml_market_value=True,
        )

        self._retriever_cache: dict[str, HybridRetriever] = {}

    def _get_retriever(self, experiment: str) -> HybridRetriever:
        if experiment not in self._retriever_cache:
            self._retriever_cache[experiment] = HybridRetriever(experiment=experiment)
        return self._retriever_cache[experiment]

    @staticmethod
    def _normalise_ranking_profile(value: str | None) -> str:
        if value is None:
            return "balanced"

        profile = str(value).strip().lower()

        if profile not in VALID_RANKING_PROFILES:
            return "balanced"

        return profile

    def _build_request(self, request: PredictionRequest) -> RetrievalRequest:
        return RetrievalRequest(
            strategy=request.strategy,
            query_text=request.query_text,
            top_k=request.top_k,
            recall_k=request.recall_k,
            alpha=request.alpha,
            beta=request.beta,
            dedupe_by_address=request.dedupe_by_address,
            locality=request.locality,
            address_contains=request.address_contains,
            location_fallback=request.location_fallback,
            attach_explanations=request.with_explanations,
            explanation_model=request.explanation_model,
            use_dcn_reranker=request.use_dcn_reranker,
            dcn_experiment=request.reranking_model,
        )

    @staticmethod
    def _clean_value(value: Any) -> Any:
        import math

        import numpy as np
        import pandas as pd

        if value is None:
            return None

        if isinstance(value, list):
            return [SmartDeveloperPredictor._clean_value(v) for v in value]

        if isinstance(value, tuple):
            return [SmartDeveloperPredictor._clean_value(v) for v in value]

        if isinstance(value, dict):
            return {
                str(k): SmartDeveloperPredictor._clean_value(v)
                for k, v in value.items()
            }

        if isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, pd.Timestamp):
            if pd.isna(value):
                return None
            return value.isoformat()

        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return value

        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass

        return value

    def _clean_records(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        records = df.to_dict(orient="records")
        cleaned: list[dict[str, Any]] = []

        for row in records:
            cleaned.append(
                {
                    str(k): self._clean_value(v)
                    for k, v in row.items()
                }
            )

        return cleaned

    def predict(self, request: PredictionRequest) -> dict[str, Any]:
        start_time = time.perf_counter()
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        created_at = datetime.now(timezone.utc).isoformat()

        request.ranking_profile = self._normalise_ranking_profile(
            request.ranking_profile
        )

        retriever = self._get_retriever(request.retrieval_model)
        retrieval_request = self._build_request(request)

        results_df = retriever.retrieve(retrieval_request)

        # New locality pre-filter metadata from HybridRetriever.
        location_metadata = dict(results_df.attrs.get("location_metadata", {}))

        if not results_df.empty:
            results_df = self.policy_scorer.score_dataframe(
                results_df,
                strategy=request.strategy,
            )

            results_df = self.economics_pipeline.score_dataframe(
                results_df,
                strategy=request.strategy,
            )

            results_df = apply_opportunity_fusion(
                results_df,
                ranking_profile=request.ranking_profile,
            )

        records = self._clean_records(results_df)
        records = add_agent_pitch(records)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        metadata: dict[str, Any] = {
            "strategy": request.strategy,
            "query_text": request.query_text,
            "top_k": request.top_k,
            "recall_k": request.recall_k,
            "retrieval_model": request.retrieval_model,
            "reranking_model": request.reranking_model,
            "use_dcn_reranker": request.use_dcn_reranker,
            "with_explanations": request.with_explanations,
            "dedupe_by_address": request.dedupe_by_address,
            "locality": request.locality,
            "address_contains": request.address_contains,
            "location_fallback": request.location_fallback,
            "ranking_profile": request.ranking_profile,
            "policy_scoring_enabled": True,
            "policy_rag_enabled": True,
            "economics_enabled": True,
            "ml_market_value_enabled": True,
            "opportunity_fusion_enabled": True,
            "ranking_mode": "policy_economics_aware_agent_opportunity",
            "latency_ms": latency_ms,
            "result_count": len(records),
        }

        metadata.update(location_metadata)

        return {
            "request_id": request_id,
            "created_at": created_at,
            "request": asdict(request),
            "metadata": metadata,
            "result_count": len(records),
            "results": records,
        }

    def predict_from_dict(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = PredictionRequest(
            strategy=str(payload["strategy"]),
            query_text=str(payload["query_text"]),
            top_k=int(payload.get("top_k", 5)),
            recall_k=int(payload.get("recall_k", 10000)),
            with_explanations=bool(payload.get("with_explanations", True)),
            retrieval_model=str(
                payload.get("retrieval_model")
                or payload.get("retrieval_experiment")
                or self.default_retrieval_model
            ),
            use_dcn_reranker=bool(payload.get("use_dcn_reranker", True)),
            reranking_model=str(
                payload.get("reranking_model")
                or payload.get("dcn_experiment")
                or self.default_reranking_model
            ),
            alpha=float(payload.get("alpha", 0.5)),
            beta=float(payload.get("beta", 0.5)),
            dedupe_by_address=bool(payload.get("dedupe_by_address", True)),
            locality=payload.get("locality"),
            address_contains=payload.get("address_contains"),
            location_fallback=bool(payload.get("location_fallback", False)),
            ranking_profile=self._normalise_ranking_profile(
                str(payload.get("ranking_profile", "balanced"))
            ),
            explanation_model=str(
                payload.get("explanation_model")
                or self.default_explanation_model
            ),
        )

        return self.predict(request)


_DEFAULT_PREDICTOR: SmartDeveloperPredictor | None = None


def get_default_predictor() -> SmartDeveloperPredictor:
    global _DEFAULT_PREDICTOR

    if _DEFAULT_PREDICTOR is None:
        _DEFAULT_PREDICTOR = SmartDeveloperPredictor()

    return _DEFAULT_PREDICTOR


def retrieve_sites(
    strategy: str,
    query_text: str,
    top_k: int = 5,
    recall_k: int = 10000,
    with_explanations: bool = True,
    retrieval_model: str = DEFAULT_RETRIEVAL_MODEL,
    use_dcn_reranker: bool = True,
    reranking_model: str = DEFAULT_RERANKING_MODEL,
    alpha: float = 0.5,
    beta: float = 0.5,
    dedupe_by_address: bool = True,
    locality: str | None = None,
    address_contains: str | None = None,
    location_fallback: bool = False,
    ranking_profile: str = "balanced",
    explanation_model: str = DEFAULT_EXPLANATION_MODEL,
) -> dict[str, Any]:
    """
    Convenience function for direct Python usage.
    """
    predictor = get_default_predictor()

    request = PredictionRequest(
        strategy=strategy,
        query_text=query_text,
        top_k=top_k,
        recall_k=recall_k,
        with_explanations=with_explanations,
        retrieval_model=retrieval_model,
        use_dcn_reranker=use_dcn_reranker,
        reranking_model=reranking_model,
        alpha=alpha,
        beta=beta,
        dedupe_by_address=dedupe_by_address,
        locality=locality,
        address_contains=address_contains,
        location_fallback=location_fallback,
        ranking_profile=ranking_profile,
        explanation_model=explanation_model,
    )

    return predictor.predict(request)