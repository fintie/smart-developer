from __future__ import annotations

import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv

from backend.app.schemas.ai_property import (
    AIExternalSource,
    AIPropertySuggestion,
    AIPropertySummary,
    AIPropertySummaryResponse,
    AIPropertyValueEstimate,
)

load_dotenv()

GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
GOOGLE_GEMINI_MODEL = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-2.5-flash")
GOOGLE_GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
GOOGLE_CUSTOM_SEARCH_URL = "https://customsearch.googleapis.com/customsearch/v1"


class AIPropertySummaryError(RuntimeError):
    pass


def _text(value: Any, fallback: str = "N/A") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _money_label(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "N/A"
    return f"A${number:,.0f}"


def _range_label(site: dict[str, Any]) -> str | None:
    lower = _number(site.get("ml_value_lower_bound"))
    upper = _number(site.get("ml_value_upper_bound"))
    if lower is None or upper is None:
        return None
    return f"A${lower:,.0f} - A${upper:,.0f}"


def _address(site: dict[str, Any]) -> str:
    return _text(site.get("base_site_address") or site.get("address"), "Unknown address")


def _build_context(
    *,
    query_text: str,
    site: dict[str, Any],
    user_requirements: str | None,
) -> dict[str, Any]:
    return {
        "user_query": query_text,
        "user_requirements": user_requirements or query_text,
        "property": {
            "address": _address(site),
            "locality": site.get("locality"),
            "zoning": site.get("primary_zoning_code"),
            "zoning_class": site.get("primary_zoning_class"),
            "lot_size_band": site.get("lot_size_band"),
            "lot_size_proxy_sqm": site.get("lot_size_proxy_sqm"),
            "constraint_severity_band": site.get("constraint_severity_band"),
            "distance_to_station_m": site.get("distance_to_station_m"),
            "heritage_flag": site.get("heritage_flag"),
            "flood_flag": site.get("flood_flag"),
            "bushfire_flag": site.get("bushfire_flag"),
        },
        "scores": {
            "opportunity": site.get("agent_opportunity_score") or site.get("strategy_score"),
            "strategy_fit": site.get("strategy_score"),
            "policy_upside": site.get("policy_upside_score"),
            "policy_signal_band": site.get("policy_signal_band"),
            "value_potential": site.get("value_potential_score"),
            "value_potential_band": site.get("value_potential_band"),
            "cost_efficiency": site.get("cost_efficiency_score"),
            "cost_risk": site.get("cost_risk_score"),
            "cost_band": site.get("cost_band"),
        },
        "value": {
            "ml_estimated_market_value": site.get("ml_estimated_market_value"),
            "trend_adjusted_ml_market_value": site.get("trend_adjusted_ml_market_value"),
            "ml_value_lower_bound": site.get("ml_value_lower_bound"),
            "ml_value_upper_bound": site.get("ml_value_upper_bound"),
            "ml_value_confidence": site.get("ml_value_confidence"),
            "estimated_acquisition_cost": site.get("estimated_acquisition_cost"),
            "estimated_development_cost": site.get("estimated_development_cost"),
            "estimated_total_project_cost": site.get("estimated_total_project_cost"),
        },
        "existing_explanation": site.get("agent_pitch")
        or site.get("cost_value_explanation")
        or site.get("policy_explanation")
        or site.get("fast_explanation"),
    }


def _build_search_query(site: dict[str, Any]) -> str:
    address = _address(site)
    locality = _text(site.get("locality"), "")
    zoning = _text(site.get("primary_zoning_code"), "")
    terms = [address, locality, zoning, "NSW planning property development"]
    return " ".join(term for term in terms if term and term != "N/A")


async def _search_external_property_context(site: dict[str, Any]) -> list[AIExternalSource]:
    if not GOOGLE_GEMINI_API_KEY or not GOOGLE_SEARCH_ENGINE_ID:
        return []

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            response = await client.get(
                GOOGLE_CUSTOM_SEARCH_URL,
                params={
                    "key": GOOGLE_GEMINI_API_KEY,
                    "cx": GOOGLE_SEARCH_ENGINE_ID,
                    "q": _build_search_query(site),
                    "num": 5,
                },
            )
            response.raise_for_status()
    except httpx.HTTPError:
        return []

    payload = response.json()
    items = payload.get("items")
    if not isinstance(items, list):
        return []

    sources: list[AIExternalSource] = []
    for item in items[:5]:
        if not isinstance(item, dict):
            continue
        title = _text(item.get("title"), "")
        link = _text(item.get("link"), "")
        if not title or not link:
            continue
        sources.append(
            AIExternalSource(
                title=title,
                link=link,
                snippet=_text(item.get("snippet"), ""),
            )
        )

    return sources


def generate_fallback_property_summary(
    *,
    query_text: str,
    site: dict[str, Any],
    user_requirements: str | None = None,
) -> AIPropertySummaryResponse:
    address = _address(site)
    market_value = (
        _number(site.get("trend_adjusted_ml_market_value"))
        or _number(site.get("ml_estimated_market_value"))
    )
    value_label = _money_label(market_value)
    range_label = _range_label(site)
    confidence = _text(site.get("ml_value_confidence"), "unknown")
    zoning = _text(site.get("primary_zoning_code"))
    lot = _text(site.get("lot_size_band"))
    constraints = _text(site.get("constraint_severity_band"))
    station_distance = _number(site.get("distance_to_station_m"))
    station_text = (
        f"{station_distance:.0f} m from the nearest station"
        if station_distance is not None
        else "station distance unavailable"
    )
    value_band = _text(site.get("value_potential_band"), "unknown")
    policy_band = _text(site.get("policy_signal_band"), "unknown")
    cost_band = _text(site.get("cost_band"), "unknown")

    summary = AIPropertySummary(
        headline=f"{address} is a candidate site with {zoning} zoning and {value_label} modelled market value.",
        basic_info=[
            f"Address: {address}.",
            f"Planning profile: {zoning} zoning, {lot} lot profile, {constraints} constraints.",
            f"Accessibility: {station_text}.",
            f"Risk flags: heritage={bool(site.get('heritage_flag'))}, flood={bool(site.get('flood_flag'))}, bushfire={bool(site.get('bushfire_flag'))}.",
        ],
        requirement_match=(
            "Based on the current search request, this site is relevant because it "
            f"matches the selected strategy with an opportunity score of "
            f"{_text(site.get('agent_opportunity_score') or site.get('strategy_score'))}."
        ),
        value_estimate=AIPropertyValueEstimate(
            label=value_label,
            amount=market_value,
            confidence=confidence,
            range_label=range_label,
            explanation=(
                "This value is based on the project's ML market value fields and "
                "should be treated as an indicative screen, not a formal valuation."
            ),
        ),
        opportunity_notes=[
            f"Policy signal is {policy_band}.",
            f"Value potential is {value_band}.",
            f"Estimated acquisition cost is {_money_label(site.get('estimated_acquisition_cost'))}.",
        ],
        risk_notes=[
            f"Cost band is {cost_band}.",
            f"Estimated total project cost is {_money_label(site.get('estimated_total_project_cost'))}.",
            "Planning, valuation, legal, and feasibility checks are still required before investment decisions.",
        ],
        disclaimer=(
            "AI summary uses the recommendation result and ML estimate from this app; "
            "it is not financial, legal, planning, or valuation advice."
        ),
    )

    return AIPropertySummaryResponse(
        source="structured_fallback",
        model=None,
        summary=summary,
        ai_suggestion=AIPropertySuggestion(
            headline="AI suggestion",
            suggestion=(
                "Use this recommendation as a screening lead, then verify the site "
                "against official planning controls, comparable sales, and current "
                "market listings before making a decision."
            ),
            external_context_used=False,
            source_notes=[
                "External web search was not used because Google Gemini or Custom Search is not fully configured.",
            ],
            next_steps=[
                "Check the address in NSW Planning Portal or the local council planning viewer.",
                "Compare recent nearby transactions and active listings.",
                "Ask a planner or valuer to verify feasibility and value assumptions.",
            ],
        ),
        external_sources=[],
    )


def _summary_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "headline",
            "basic_info",
            "requirement_match",
            "value_estimate",
            "opportunity_notes",
            "risk_notes",
            "disclaimer",
            "ai_suggestion",
        ],
        "properties": {
            "headline": {"type": "string"},
            "basic_info": {
                "type": "array",
                "items": {"type": "string"},
            },
            "requirement_match": {"type": "string"},
            "value_estimate": {
                "type": "object",
                "required": [
                    "label",
                    "amount",
                    "confidence",
                    "range_label",
                    "explanation",
                ],
                "properties": {
                    "label": {"type": "string"},
                    "amount": {"type": ["number", "null"]},
                    "confidence": {"type": ["string", "null"]},
                    "range_label": {"type": ["string", "null"]},
                    "explanation": {"type": "string"},
                },
            },
            "opportunity_notes": {
                "type": "array",
                "items": {"type": "string"},
            },
            "risk_notes": {
                "type": "array",
                "items": {"type": "string"},
            },
            "disclaimer": {"type": "string"},
            "ai_suggestion": {
                "type": "object",
                "required": [
                    "headline",
                    "suggestion",
                    "external_context_used",
                    "source_notes",
                    "next_steps",
                ],
                "properties": {
                    "headline": {"type": "string"},
                    "suggestion": {"type": "string"},
                    "external_context_used": {"type": "boolean"},
                    "source_notes": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "next_steps": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
    }


def _extract_response_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return ""

    text_parts: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])

    return "\n".join(text_parts).strip()


async def generate_ai_property_summary(
    *,
    query_text: str,
    site: dict[str, Any],
    user_requirements: str | None = None,
) -> AIPropertySummaryResponse:
    if not GOOGLE_GEMINI_API_KEY:
        return generate_fallback_property_summary(
            query_text=query_text,
            site=site,
            user_requirements=user_requirements,
        )

    context = _build_context(
        query_text=query_text,
        site=site,
        user_requirements=user_requirements,
    )
    external_sources = await _search_external_property_context(site)
    external_context = [
        source.model_dump()
        for source in external_sources
    ]

    prompt = (
        "You are a real-estate development assistant. Generate a concise property "
        "summary from the provided structured data. Do not invent prices, zoning, "
        "distances, or constraints. Use the ML estimate only as an indicative value. "
        "Write for a property developer, and reflect the user's requirements. Also "
        "generate an ai_suggestion section. If external search results are provided, "
        "use them only as directional public context, cite them in source_notes, and "
        "do not claim a fact unless it appears in the structured data or source "
        "snippets. If no external search results are provided, explain that the "
        "suggestion is based on the internal recommendation data only.\n\n"
        f"Context JSON:\n{json.dumps(context, ensure_ascii=False)}\n\n"
        f"External search results JSON:\n{json.dumps(external_context, ensure_ascii=False)}"
    )

    request_payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": _summary_json_schema(),
        },
    }

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(
                f"{GOOGLE_GEMINI_API_URL}/{GOOGLE_GEMINI_MODEL}:generateContent",
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": GOOGLE_GEMINI_API_KEY,
                },
                json=request_payload,
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise AIPropertySummaryError(
            f"Google Gemini returned {exc.response.status_code}: {exc.response.text}"
        ) from exc
    except httpx.HTTPError as exc:
        raise AIPropertySummaryError(f"Failed to call Google Gemini: {exc}") from exc

    raw_text = _extract_response_text(response.json())
    if not raw_text:
        raise AIPropertySummaryError("Google Gemini returned an empty summary.")

    try:
        summary_payload = json.loads(raw_text)
        suggestion_payload = summary_payload.pop("ai_suggestion", None)
        summary = AIPropertySummary.model_validate(summary_payload)
        suggestion = AIPropertySuggestion.model_validate(suggestion_payload)
    except (json.JSONDecodeError, ValueError) as exc:
        raise AIPropertySummaryError(
            "Google Gemini returned an invalid summary JSON."
        ) from exc

    return AIPropertySummaryResponse(
        source="google_gemini",
        model=GOOGLE_GEMINI_MODEL,
        summary=summary,
        ai_suggestion=suggestion,
        external_sources=external_sources,
    )
