from __future__ import annotations
import re
from typing import Any
import pandas as pd


def normalise_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalise_upper(value: Any) -> str:
    return normalise_text(value).upper()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def money_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    if value <= 0:
        return None
    return round(float(value), 2)


def extract_locality_candidates(address: Any) -> list[str]:
    text = normalise_upper(address)
    if not text:
        return []

    text = re.sub(r"\s+", " ", text).strip()
    tokens = re.findall(r"[A-Z][A-Z\-']+", text)

    candidates = []
    for n in [4, 3, 2, 1]:
        if len(tokens) >= n:
            candidates.append(" ".join(tokens[-n:]))

    return candidates


def band_from_score(score: float) -> str:
    if score >= 75:
        return "very_high"
    if score >= 60:
        return "high"
    if score >= 40:
        return "medium"
    if score > 0:
        return "low"
    return "unknown"


def today_timestamp() -> pd.Timestamp:
    return pd.Timestamp.today().normalize()