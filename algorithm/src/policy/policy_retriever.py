from __future__ import annotations
from pathlib import Path
from typing import Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


DEFAULT_INDEX_DIR = Path("algorithm/artifacts/policy_index/chroma")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _normalise_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _build_policy_query(
    *,
    strategy: str,
    site: dict[str, Any],
    policy_ids: list[str],
) -> str:
    zoning = _normalise_text(site.get("primary_zoning_code"))
    zoning_band = _normalise_text(site.get("zoning_band"))
    lot_size_band = _normalise_text(site.get("lot_size_band"))
    station_band = _normalise_text(site.get("station_distance_band"))
    within_800m = _normalise_text(site.get("within_800m_catchment"))
    constraint = _normalise_text(site.get("constraint_severity_band"))

    matched = ", ".join(policy_ids)

    return (
        f"Policy evidence for property redevelopment recommendation. "
        f"Strategy: {strategy}. "
        f"Matched policy IDs: {matched}. "
        f"Site zoning: {zoning}. "
        f"Zoning band: {zoning_band}. "
        f"Lot size band: {lot_size_band}. "
        f"Station distance band: {station_band}. "
        f"Within 800m catchment: {within_800m}. "
        f"Constraint severity: {constraint}. "
        f"Find NSW planning policy text about housing reform, low and mid-rise housing, "
        f"transport-oriented development, station catchments, Housing SEPP, "
        f"affordable housing bonuses, floor space, height, and redevelopment uplift."
    )


class PolicyRetriever:
    def __init__(
        self,
        index_dir: Path | str = DEFAULT_INDEX_DIR,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.index_dir = Path(index_dir)
        self.embedding_model = embedding_model

        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        self.vectorstore = Chroma(
            persist_directory=str(self.index_dir),
            embedding_function=embeddings,
        )

    def retrieve(
        self,
        *,
        policy_ids: list[str],
        strategy: str,
        site: dict[str, Any],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if not policy_ids:
            return []

        query = _build_policy_query(
            strategy=strategy,
            site=site,
            policy_ids=policy_ids,
        )

        # Chroma filter supports simple equality. Since policy_id is a single metadata value,
        # retrieve per policy_id and merge.
        evidence: list[dict[str, Any]] = []

        per_policy_k = max(1, top_k)

        for policy_id in policy_ids:
            try:
                results = self.vectorstore.similarity_search_with_relevance_scores(
                    query=query,
                    k=per_policy_k,
                    filter={"policy_id": policy_id},
                )
            except Exception:
                results = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=per_policy_k,
                    filter={"policy_id": policy_id},
                )

            for doc, score in results:
                metadata = doc.metadata or {}

                snippet = doc.page_content.strip()
                if len(snippet) > 700:
                    snippet = snippet[:700].rstrip() + "..."

                evidence.append(
                    {
                        "policy_id": metadata.get("policy_id"),
                        "policy_name": metadata.get("policy_name"),
                        "source_url": metadata.get("source_url"),
                        "retrieved_at": metadata.get("retrieved_at"),
                        "chunk_id": metadata.get("chunk_id"),
                        "snippet": snippet,
                        "relevance_score": round(float(score), 4),
                    }
                )

        # Dedupe by policy_id + chunk_id.
        deduped = []
        seen = set()
        for item in evidence:
            key = (item.get("policy_id"), item.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        # Ensure evidence diversity:
        # keep at least one snippet per matched policy when available,
        # then fill remaining slots.
        selected = []
        selected_keys = set()

        for policy_id in policy_ids:
            for item in deduped:
                key = (item.get("policy_id"), item.get("chunk_id"))
                if item.get("policy_id") == policy_id and key not in selected_keys:
                    selected.append(item)
                    selected_keys.add(key)
                    break

        for item in deduped:
            if len(selected) >= top_k:
                break

            key = (item.get("policy_id"), item.get("chunk_id"))
            if key in selected_keys:
                continue

            selected.append(item)
            selected_keys.add(key)

        return selected[:top_k]


def retrieve_policy_evidence_for_site(
    *,
    policy_ids: list[str],
    strategy: str,
    site: dict[str, Any],
    top_k: int = 3,
    index_dir: Path | str = DEFAULT_INDEX_DIR,
) -> list[dict[str, Any]]:
    retriever = PolicyRetriever(index_dir=index_dir)
    return retriever.retrieve(
        policy_ids=policy_ids,
        strategy=strategy,
        site=site,
        top_k=top_k,
    )