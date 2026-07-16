"""Tests for /health, /health/live, /health/ready."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_liveness_always_ok(client: TestClient) -> None:
    response = client.get("/health/live")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"] == {"alive": True}
    assert "request_id" in body and body["request_id"]


def test_aggregate_health_reports_healthy(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["status"] in ("healthy", "degraded")
    assert body["data"]["model_loaded"] is True
    for key in ("runtime", "gpu", "memory", "providers", "artifacts", "environment"):
        assert key in body["data"]


def test_readiness_true_when_registry_initialized(live_client: TestClient) -> None:
    response = live_client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["ready"] is True


def test_readiness_false_when_registry_not_initialized(client: TestClient) -> None:
    # `client` deliberately never runs a real successful startup (see
    # tests/api/conftest.py) -- app.state.registry stays unset, so the
    # registry-state-reading /health/ready route (not dependency-overridable)
    # correctly reports not-ready with a 503.
    response = client.get("/health/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["data"]["ready"] is False
    assert body["data"]["reason"]
