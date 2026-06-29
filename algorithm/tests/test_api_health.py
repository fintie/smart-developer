from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("LAZY_LOAD_PREDICTOR", "true")
os.environ.setdefault("SKIP_STARTUP_WARMUP", "true")


@pytest.fixture(scope="module")
def client(monkeypatch_module):
    from algorithm.src.serving import api

    # Avoid loading model artifacts during tests.
    monkeypatch_module.setattr(api, "_load_predictor_in_background", lambda: None)
    with TestClient(api.app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


def test_health_returns_liveness_payload(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["predictor_loaded"] is False
    assert "ranking_profiles" in body


def test_ready_returns_503_until_predictor_loaded(client):
    response = client.get("/ready")
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["status"] == "not_ready"
