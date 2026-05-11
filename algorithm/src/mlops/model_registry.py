from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from sqlalchemy.dialects.postgresql import insert
from algorithm.src.mlops.db import get_session
from algorithm.src.mlops.models import ModelRegistryLog


def _load_json_if_exists(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None

    path = Path(path)
    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


def upsert_model_version(
    *,
    model_version: str,
    model_type: str,
    artifact_path: str,
    preprocessing_path: str | None = None,
    status: str = "candidate",
    metrics_path: str | None = None,
    model_card_path: str | None = None,
    metrics: dict[str, Any] | None = None,
    model_card: dict[str, Any] | None = None,
    notes: str | None = None,
) -> None:
    """
    Insert or update one model version in model_registry.

    status examples:
    - candidate
    - staging
    - production
    - archived
    """
    loaded_metrics = metrics or _load_json_if_exists(metrics_path)
    loaded_model_card = model_card or _load_json_if_exists(model_card_path)

    values = {
        "model_version": model_version,
        "model_type": model_type,
        "artifact_path": artifact_path,
        "preprocessing_path": preprocessing_path,
        "status": status,
        "metrics": loaded_metrics,
        "model_card": loaded_model_card,
        "notes": notes,
    }

    stmt = insert(ModelRegistryLog).values(**values)
    stmt = stmt.on_conflict_do_update(
        index_elements=["model_version"],
        set_={
            "model_type": stmt.excluded.model_type,
            "artifact_path": stmt.excluded.artifact_path,
            "preprocessing_path": stmt.excluded.preprocessing_path,
            "status": stmt.excluded.status,
            "metrics": stmt.excluded.metrics,
            "model_card": stmt.excluded.model_card,
            "notes": stmt.excluded.notes,
        },
    )

    with get_session() as session:
        session.execute(stmt)


def register_current_default_models() -> None:
    """
    Register the current default production retrieval and reranking models.
    """

    upsert_model_version(
        model_version="two_tower_v1",
        model_type="two_tower_retrieval",
        artifact_path="algorithm/artifacts/models/two_tower_v1/model.pt",
        preprocessing_path=None,
        status="production",
        metrics_path="algorithm/artifacts/models/two_tower_v1/metrics.json",
        notes=(
            "Current default retrieval model for intent-to-site recall. "
            "Used as DEFAULT_RETRIEVAL_MODEL."
        ),
    )

    upsert_model_version(
        model_version="dcn_reranker_v1",
        model_type="dcn_reranker",
        artifact_path="algorithm/artifacts/models/dcn_reranker_v1/model.pt",
        preprocessing_path="algorithm/artifacts/models/dcn_reranker_v1/preprocessing.json",
        status="production",
        metrics_path="algorithm/artifacts/models/dcn_reranker_v1/metrics.json",
        notes=(
            "Current default reranking model for second-stage site ranking. "
            "Used as DEFAULT_RERANKING_MODEL."
        ),
    )


if __name__ == "__main__":
    register_current_default_models()
    print("Registered current default Smart Developer models.")