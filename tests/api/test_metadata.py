"""Tests for /model, /runtime, /classes, /version, /artifacts."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_get_model_info(client: TestClient) -> None:
    response = client.get("/model")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["backbone"] == "resnet18"
    assert data["validation"]["architecture_instantiated"] is True
    assert data["validation"]["state_dict_strict_match"] is True


def test_get_runtime_info(client: TestClient) -> None:
    response = client.get("/runtime")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["initialized"] is True
    assert data["runtime_type"] == "pytorch"
    assert data["loaded"] is True


def test_get_classes(client: TestClient) -> None:
    response = client.get("/classes")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["num_classes"] == len(data["class_names"]) == 5
    assert set(data["thresholds"].keys()) == set(data["class_names"])


def test_get_version(client: TestClient) -> None:
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["backbone"] == "resnet18"
    assert data["model_version"].startswith("resnet18-")
    assert data["api_version"]


def test_get_artifacts(client: TestClient) -> None:
    response = client.get("/artifacts")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["total_artifacts"] > 0
    assert isinstance(data["records"], dict)
    assert isinstance(data["critical_missing"], list)
