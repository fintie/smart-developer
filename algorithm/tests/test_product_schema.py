from __future__ import annotations

from algorithm.src.serving.product_schema import (
    PRODUCT_RESULT_FIELDS,
    PRODUCT_RESULT_FIELD_SET,
    filter_product_response,
)


def test_field_set_matches_field_tuple():
    assert PRODUCT_RESULT_FIELD_SET == frozenset(PRODUCT_RESULT_FIELDS)
    assert len(PRODUCT_RESULT_FIELD_SET) == len(PRODUCT_RESULT_FIELDS), (
        "PRODUCT_RESULT_FIELDS contains duplicate field names"
    )


def test_filter_product_response_keeps_only_known_fields():
    response = {
        "request_id": "req-1",
        "results": [
            {
                "RID": 1,
                "address": "10 GEORGE ST",
                "agent_opportunity_score": 4.2,
                "internal_debug_only_field": "should be stripped",
                "another_secret": {"nested": True},
            },
            {
                "RID": 2,
                "address": "11 GEORGE ST",
                "policy_upside_score": 3.1,
            },
        ],
    }

    out = filter_product_response(response)

    assert out["request_id"] == "req-1"
    assert len(out["results"]) == 2
    assert out["results"][0] == {
        "RID": 1,
        "address": "10 GEORGE ST",
        "agent_opportunity_score": 4.2,
    }
    assert "internal_debug_only_field" not in out["results"][0]
    assert "policy_upside_score" in out["results"][1]


def test_filter_product_response_handles_missing_results_key():
    out = filter_product_response({"request_id": "req-2"})
    assert out["results"] == []


def test_required_core_fields_present():
    # Smoke check that the schema didn't accidentally lose key fields the
    # frontend relies on.
    required = {
        "RID",
        "address",
        "agent_opportunity_score",
        "policy_upside_score",
        "estimated_total_project_cost",
        "fast_explanation",
    }
    missing = required - PRODUCT_RESULT_FIELD_SET
    assert not missing, f"Schema is missing required public fields: {missing}"
