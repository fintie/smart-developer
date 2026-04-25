from __future__ import annotations

import pandas as pd

from .evidence import build_explanation_payload
from .local_generator import generate_with_ollama


def explain_row(
    row: pd.Series,
    strategy: str,
    model: str = "llama3.1:8b-instruct-q4_K_M",
) -> str:
    payload = build_explanation_payload(row, strategy)
    return generate_with_ollama(payload, model=model)