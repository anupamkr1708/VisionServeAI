"""Tests for /explain/* endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

SIMPLE_METHODS = ["gradcam", "gradcam_plus", "scorecam", "eigencam", "guided_backprop"]


@pytest.mark.parametrize("method", SIMPLE_METHODS)
def test_simple_explain_methods_succeed(client: TestClient, sample_image_path: str, method: str) -> None:
    with open(sample_image_path, "rb") as f:
        response = client.post(f"/explain/{method}", files={"file": ("sample.png", f, "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    data = body["data"]
    assert data["method"] == method
    assert data["success"] is True
    assert data["error"] is None


def test_explain_accepts_target_class_and_sample_id(client: TestClient, sample_image_path: str) -> None:
    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/explain/gradcam",
            files={"file": ("sample.png", f, "image/png")},
            data={"target_class": "Atelectasis", "sample_id": "patient-42"},
        )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["sample_id"] == "patient-42"


def test_explain_accepts_numeric_target_class(client: TestClient, sample_image_path: str) -> None:
    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/explain/gradcam", files={"file": ("sample.png", f, "image/png")}, data={"target_class": "0"},
        )
    assert response.status_code == 200
    assert response.json()["data"]["success"] is True


def test_integrated_gradients_accepts_steps(client: TestClient, sample_image_path: str) -> None:
    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/explain/integrated_gradients", files={"file": ("sample.png", f, "image/png")}, data={"steps": "4"},
        )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["method"] == "integrated_gradients"
    assert data["success"] is True


def test_occlusion_accepts_patch_size_and_stride(client: TestClient, sample_image_path: str) -> None:
    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/explain/occlusion",
            files={"file": ("sample.png", f, "image/png")},
            data={"patch_size": "16", "stride": "16"},
        )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["method"] == "occlusion"
    assert data["success"] is True


def test_explain_rejects_empty_upload(client: TestClient) -> None:
    response = client.post("/explain/gradcam", files={"file": ("empty.png", b"", "image/png")})
    assert response.status_code == 422


def test_explain_rejects_corrupted_image(client: TestClient) -> None:
    """Unlike /predict, /explain/* has no downstream adapter that isolates
    a corrupted image into a structured non-200 result -- this router's
    own decode validation must reject it directly (422), per
    ``backend.utils.validators``'s documented design."""
    response = client.post("/explain/gradcam", files={"file": ("corrupt.png", b"not a real image", "image/png")})
    assert response.status_code == 422
    assert response.json()["success"] is False


def test_explain_unavailable_when_engine_not_injected(client: TestClient, sample_image_path: str) -> None:
    """Simulates a deployment with explainability disabled
    (``ServiceRegistry(enable_explainability=False)``) by overriding
    ``get_initialized_registry`` with a registry-like stub whose
    ``.explainability`` has no engine injected -- one level up from the
    ``get_explainability_service`` override the base `client` fixture
    already installs, so that override is removed first, letting the
    REAL ``get_explainability_service`` (and its own ``is_available()``
    503 check) run against the stub. Demonstrates dependency-override-
    based testing of that 503 branch.
    """
    import logging

    from backend.dependencies.explainability import get_explainability_service
    from backend.dependencies.registry import get_initialized_registry
    from services.explainability_service import ExplainabilityService

    disabled_service = ExplainabilityService(logger=logging.getLogger("visionserve.tests"))
    assert disabled_service.is_available() is False

    class _RegistryStub:
        explainability = disabled_service

    original_override = client.app.dependency_overrides.pop(get_explainability_service)
    client.app.dependency_overrides[get_initialized_registry] = lambda: _RegistryStub()
    try:
        with open(sample_image_path, "rb") as f:
            response = client.post("/explain/gradcam", files={"file": ("sample.png", f, "image/png")})
    finally:
        del client.app.dependency_overrides[get_initialized_registry]
        client.app.dependency_overrides[get_explainability_service] = original_override

    assert response.status_code == 503
