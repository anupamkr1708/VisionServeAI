"""Tests for /info, /ping, /docs-info, and the customized OpenAPI schema."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_ping(client: TestClient) -> None:
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json()["data"]["message"] == "pong"


def test_system_info_reports_registry_state(live_client: TestClient) -> None:
    response = live_client.get("/info")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["registry_initialized"] is True
    assert data["api_version"]
    assert "environment" in data


def test_docs_info(client: TestClient) -> None:
    response = client.get("/docs-info")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["docs_url"] == "/docs"
    assert data["openapi_url"] == "/openapi.json"


def test_openapi_schema_customization(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"]
    assert schema["info"]["contact"]["email"]
    assert schema["info"]["license"]["name"]
    tag_names = {tag["name"] for tag in schema.get("tags", [])}
    assert {"Health", "Prediction", "Explainability", "Metadata", "System"} <= tag_names


def test_every_route_has_a_response_envelope_schema(client: TestClient) -> None:
    """Every operation should declare a typed response (this phase's
    "response_model everywhere" / "no anonymous routes" requirement)."""
    schema = client.get("/openapi.json").json()
    for path, methods in schema["paths"].items():
        for method, operation in methods.items():
            assert "operationId" in operation, f"{method.upper()} {path} has no operation_id"
            assert operation.get("responses", {}).get("200") or operation.get("responses", {}).get("201"), (
                f"{method.upper()} {path} has no documented success response"
            )
