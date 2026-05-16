from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from algorithm.src.mlops.db import get_session
from algorithm.src.mlops.models import ModelRegistryLog


MODEL_CARD_PATHS = [
    Path("algorithm/artifacts/models/two_tower_v1/model_card.json"),
    Path("algorithm/artifacts/models/dcn_reranker_v1/model_card.json"),
]


def load_model_card(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model card not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def upsert_model_card(card: dict[str, Any]) -> None:
    model_version = card["model_version"]

    row = ModelRegistryLog(
        model_version=model_version,
        model_type=card["model_type"],
        artifact_path=card["artifact_path"],
        preprocessing_path=card.get("preprocessing_path"),
        status=card.get("status", "candidate"),
        metrics=card.get("metrics"),
        model_card=card,
        notes="\n".join(card.get("notes", [])) if isinstance(card.get("notes"), list) else card.get("notes"),
    )

    with get_session() as session:
        existing = session.get(ModelRegistryLog, model_version)

        if existing is None:
            session.add(row)
        else:
            existing.model_type = row.model_type
            existing.artifact_path = row.artifact_path
            existing.preprocessing_path = row.preprocessing_path
            existing.status = row.status
            existing.metrics = row.metrics
            existing.model_card = row.model_card
            existing.notes = row.notes


def main() -> None:
    for path in MODEL_CARD_PATHS:
        card = load_model_card(path)
        upsert_model_card(card)
        print(f"Seeded model registry: {card['model_version']}")

    print("Model registry seeding complete.")


if __name__ == "__main__":
    main()