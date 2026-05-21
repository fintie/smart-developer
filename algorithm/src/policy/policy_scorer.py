from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pandas as pd
import yaml
from algorithm.src.policy.policy_retriever import PolicyRetriever


DEFAULT_POLICY_RULES_PATH = Path("algorithm/configs/policies/policy_rules.yaml")


@dataclass(frozen=True)
class PolicyMatch:
    rule_id: str
    policy_id: str
    policy_name: str
    points: float
    confidence: str
    explanation_hint: str


def _normalise_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalise_upper(value: Any) -> str:
    return _normalise_str(value).upper()


def _normalise_lower(value: Any) -> str:
    return _normalise_str(value).lower()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = _normalise_lower(value)
    return text in {"1", "true", "yes", "y"}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _strategy_applies(rule: dict[str, Any], strategy: str) -> bool:
    strategies = rule.get("applies_to_strategies") or []
    if not strategies:
        return True
    return strategy in strategies


def _list_contains(value: str, allowed: list[Any], *, upper: bool = False) -> bool:
    if upper:
        allowed_values = {_normalise_upper(x) for x in allowed}
        return _normalise_upper(value) in allowed_values

    allowed_values = {_normalise_lower(x) for x in allowed}
    return _normalise_lower(value) in allowed_values


def _rule_matches_site(rule: dict[str, Any], site: dict[str, Any]) -> bool:
    trigger = rule.get("trigger") or {}

    if "within_800m_catchment" in trigger:
        expected = bool(trigger["within_800m_catchment"])
        actual = _as_bool(site.get("within_800m_catchment"))
        if actual != expected:
            return False

    if "zoning_codes_any" in trigger:
        zoning_code = _normalise_upper(site.get("primary_zoning_code"))
        if not _list_contains(zoning_code, trigger["zoning_codes_any"], upper=True):
            return False

    if "zoning_band_any" in trigger:
        zoning_band = _normalise_lower(site.get("zoning_band"))
        if not _list_contains(zoning_band, trigger["zoning_band_any"]):
            return False

    if "lot_size_band_any" in trigger:
        lot_size_band = _normalise_lower(site.get("lot_size_band"))
        if not _list_contains(lot_size_band, trigger["lot_size_band_any"]):
            return False

    if "constraint_severity_not" in trigger:
        constraint_band = _normalise_lower(site.get("constraint_severity_band"))
        if _list_contains(constraint_band, trigger["constraint_severity_not"]):
            return False

    return True


def _confidence_multiplier(confidence: str) -> float:
    confidence = _normalise_lower(confidence)

    if confidence == "high":
        return 1.0
    if confidence == "medium":
        return 0.85
    if confidence == "low":
        return 0.65
    return 0.5


def _band_from_score(score: float) -> str:
    if score >= 75:
        return "very_high"
    if score >= 55:
        return "high"
    if score >= 30:
        return "medium"
    if score > 0:
        return "low"
    return "none"


def _build_policy_explanation(matches: list[PolicyMatch]) -> str:
    if not matches:
        return (
            "No strong policy upside signal was identified from the available "
            "structured policy rules. This does not rule out site-specific planning "
            "opportunities, but it means the current screening data did not trigger "
            "a major policy-driven uplift signal."
        )

    top_matches = sorted(matches, key=lambda m: m.points, reverse=True)[:3]

    policy_names = []
    for match in top_matches:
        if match.policy_name not in policy_names:
            policy_names.append(match.policy_name)

    hints = [m.explanation_hint.strip() for m in top_matches if m.explanation_hint]

    return (
        "Policy signal detected: "
        + "; ".join(policy_names)
        + ". "
        + " ".join(hints)
        + " This is an indicative policy screening signal only and should be "
          "verified against the NSW Planning Portal, applicable LEP/SEPP controls, "
          "and professional planning advice."
    )


class PolicyScorer:
    def __init__(
        self,
        rules_path: Path | str = DEFAULT_POLICY_RULES_PATH,
        enable_rag_evidence: bool = False,
        policy_retriever: PolicyRetriever | None = None,
        rag_top_k: int = 3,
    ):
        self.rules_path = Path(rules_path)
        config = _load_yaml(self.rules_path)
        self.config = config
        self.rules = config.get("rules", [])

        self.enable_rag_evidence = enable_rag_evidence
        self.policy_retriever = policy_retriever
        self.rag_top_k = rag_top_k

    def score_site(self, site: dict[str, Any], strategy: str) -> dict[str, Any]:
        matches: list[PolicyMatch] = []

        for rule in self.rules:
            if not _strategy_applies(rule, strategy):
                continue

            if not _rule_matches_site(rule, site):
                continue

            score_cfg = rule.get("score") or {}
            raw_points = float(score_cfg.get("policy_upside_points", 0.0))
            confidence = str(score_cfg.get("confidence", "low"))
            points = raw_points * _confidence_multiplier(confidence)

            matches.append(
                PolicyMatch(
                    rule_id=str(rule.get("rule_id")),
                    policy_id=str(rule.get("policy_id")),
                    policy_name=str(rule.get("policy_name")),
                    points=points,
                    confidence=confidence,
                    explanation_hint=str(rule.get("explanation_hint", "")),
                )
            )

        total_score = min(100.0, sum(m.points for m in matches))

        matched_rules = [m.rule_id for m in matches]
        matched_policies = sorted({m.policy_id for m in matches})
        matched_policy_names = sorted({m.policy_name for m in matches})

        policy_evidence: list[dict[str, Any]] = []

        if self.enable_rag_evidence and matched_policies:
            if self.policy_retriever is None:
                self.policy_retriever = PolicyRetriever()

            try:
                policy_evidence = self.policy_retriever.retrieve(
                    policy_ids=matched_policies,
                    strategy=strategy,
                    site=site,
                    top_k=self.rag_top_k,
                )
            except Exception as exc:
                policy_evidence = [
                    {
                        "policy_id": "policy_rag_error",
                        "policy_name": "Policy RAG retrieval failed",
                        "source_url": None,
                        "retrieved_at": None,
                        "chunk_id": None,
                        "snippet": f"Policy evidence retrieval failed: {exc}",
                        "relevance_score": None,
                    }
                ]

        return {
            "policy_upside_score": round(total_score, 2),
            "policy_signal_band": _band_from_score(total_score),
            "policy_matched_rules": matched_rules,
            "policy_matched_policies": matched_policies,
            "policy_matched_policy_names": matched_policy_names,
            "policy_evidence": policy_evidence,
            "policy_evidence_count": len(policy_evidence),
            "policy_explanation": _build_policy_explanation(matches),
        }

    def score_dataframe(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        if df.empty:
            return df

        rows = []
        for record in df.to_dict(orient="records"):
            rows.append(self.score_site(record, strategy))

        policy_df = pd.DataFrame(rows, index=df.index)
        return pd.concat([df.reset_index(drop=True), policy_df.reset_index(drop=True)], axis=1)


def score_policy_for_results(
    results: list[dict[str, Any]],
    strategy: str,
    rules_path: Path | str = DEFAULT_POLICY_RULES_PATH,
) -> list[dict[str, Any]]:
    scorer = PolicyScorer(rules_path)
    out = []

    for item in results:
        enriched = dict(item)
        enriched.update(scorer.score_site(item, strategy))
        out.append(enriched)

    return out