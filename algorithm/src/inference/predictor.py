from __future__ import annotations
import uuid
import time
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from typing import Any
import pandas as pd
from algorithm.src.retrieval.hybrid_retrieve import HybridRetriever, RetrievalRequest


DEFAULT_RETRIEVAL_MODEL = "two_tower_v1"
DEFAULT_RERANKING_MODEL = "dcn_reranker_v1"
DEFAULT_EXPLANATION_MODEL = "llama3.1:8b-instruct-q4_K_M"


@dataclass
class PredictionRequest:
    strategy: str
    query_text: str
    top_k: int = 5
    recall_k: int = 200
    with_explanations: bool = True
    retrieval_model: str = DEFAULT_RETRIEVAL_MODEL
    use_dcn_reranker: bool = True
    reranking_model: str = DEFAULT_RERANKING_MODEL
    alpha: float = 0.5
    beta: float = 0.5
    dedupe_by_address: bool = True
    locality: str | None = None
    address_contains: str | None = None
    location_fallback: bool = True
    explanation_model: str = DEFAULT_EXPLANATION_MODEL


class SmartDeveloperPredictor:
    """
    General inference wrapper for the Smart Developer retrieval pipeline.

    Intended usage:
    - import directly from a Python backend
    - wrap behind FastAPI / Flask / Django
    - call from another language through a thin Python bridge

    Current inference stack:
        two_tower recall -> optional DCN rerank -> dedupe -> optional explanation
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

        self._retriever_cache: dict[str, HybridRetriever] = {}

    def _get_retriever(self, experiment: str) -> HybridRetriever:
        if experiment not in self._retriever_cache:
            self._retriever_cache[experiment] = HybridRetriever(experiment=experiment)
        return self._retriever_cache[experiment]

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
    def _clean_records(df: pd.DataFrame) -> list[dict[str, Any]]:
        records = df.to_dict(orient="records")
        cleaned: list[dict[str, Any]] = []

        for record in records:
            item: dict[str, Any] = {}
            for k, v in record.items():
                if pd.isna(v):
                    item[k] = None
                elif hasattr(v, "item"):
                    item[k] = v.item()
                else:
                    item[k] = v

            if "base_site_address" not in item or item.get("base_site_address") is None:
                item["base_site_address"] = item.get("address")

            cleaned.append(item)

        return cleaned

    def predict(self, request: PredictionRequest) -> dict[str, Any]:
        start_time = time.perf_counter()
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        created_at = datetime.now(timezone.utc).isoformat()

        retriever = self._get_retriever(request.retrieval_model)

        retrieval_request = RetrievalRequest(
            strategy=request.strategy,
            query_text=request.query_text,
            top_k=request.top_k,
            recall_k=request.recall_k,
            alpha=request.alpha,
            beta=request.beta,
            dedupe_by_address=request.dedupe_by_address,
            attach_explanations=request.with_explanations,
            explanation_model=request.explanation_model,
            use_dcn_reranker=request.use_dcn_reranker,
            dcn_experiment=request.reranking_model,
            locality=request.locality,
            address_contains=request.address_contains,
        )

        results_df = retriever.retrieve(retrieval_request)
        location_metadata = {
            "location_filter_requested": results_df.attrs.get("location_filter_requested", False),
            "exact_location_match_count": results_df.attrs.get("exact_location_match_count"),
            "location_fallback_used": results_df.attrs.get("location_fallback_used", False),
            "location_warning": results_df.attrs.get("location_warning"),
        }
        records = self._clean_records(results_df)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "request_id": request_id,
            "created_at": created_at,
            "request": asdict(request),
            "metadata": {
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
                "location_filter_requested": location_metadata["location_filter_requested"],
                "exact_location_match_count": location_metadata["exact_location_match_count"],
                "location_fallback_used": location_metadata["location_fallback_used"],
                "location_warning": location_metadata["location_warning"],
                "latency_ms": latency_ms,
                "result_count": len(records),
            },
            "result_count": len(records),
            "results": records,
        }

    def predict_from_dict(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = PredictionRequest(
            strategy=payload["strategy"],
            query_text=payload["query_text"],
            top_k=int(payload.get("top_k", 5)),
            recall_k=int(payload.get("recall_k", 200)),
            with_explanations=bool(payload.get("with_explanations", True)),
            retrieval_model=str(
                payload.get("retrieval_experiment", self.default_retrieval_model)
            ),
            use_dcn_reranker=bool(payload.get("use_dcn_reranker", True)),
            reranking_model=str(payload.get("dcn_experiment", self.default_reranking_model)),
            alpha=float(payload.get("alpha", 0.5)),
            beta=float(payload.get("beta", 0.5)),
            dedupe_by_address=bool(payload.get("dedupe_by_address", True)),
            locality=payload.get("locality"),
            address_contains=payload.get("address_contains"),
            explanation_model=str(
                payload.get("explanation_model", self.default_explanation_model)
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
    recall_k: int = 200,
    with_explanations: bool = True,
    retrieval_model: str = DEFAULT_RETRIEVAL_MODEL,
    use_dcn_reranker: bool = True,
    reranking_model: str = DEFAULT_RERANKING_MODEL,
    alpha: float = 0.5,
    beta: float = 0.5,
    dedupe_by_address: bool = True,
    locality: str | None = None,
    address_contains: str | None = None,
    explanation_model: str = DEFAULT_EXPLANATION_MODEL,
) -> dict[str, Any]:
    """
    Convenience function for direct backend usage.
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
        explanation_model=explanation_model,
    )
    return predictor.predict(request)