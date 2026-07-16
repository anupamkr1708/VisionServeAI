"""Tests for /predict and /predict/batch."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def test_predict_single_success(client: TestClient, sample_image_path: str) -> None:
    with open(sample_image_path, "rb") as f:
        response = client.post("/predict", files={"file": ("sample.png", f, "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    data = body["data"]
    assert data["success"] is True
    assert data["image_identifier"] == "sample.png"
    assert isinstance(data["predicted_diseases"], list)
    assert isinstance(data["probabilities"], dict)
    assert data["error"] is None


def test_predict_single_with_explicit_identifier(client: TestClient, sample_image_path: str) -> None:
    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/predict", files={"file": ("sample.png", f, "image/png")}, data={"identifier": "custom-id-1"},
        )
    assert response.status_code == 200
    assert response.json()["data"]["image_identifier"] == "custom-id-1"


def test_predict_batch_preserves_order(client: TestClient, sample_image_path: str) -> None:
    payload = Path(sample_image_path).read_bytes()
    files = [
        ("files", ("first.png", payload, "image/png")),
        ("files", ("second.png", payload, "image/png")),
        ("files", ("third.png", payload, "image/png")),
    ]
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 200
    data = response.json()["data"]
    identifiers = [r["image_identifier"] for r in data["results"]]
    assert identifiers == ["first.png", "second.png", "third.png"]
    assert data["summary"]["total"] == 3
    assert data["summary"]["successful"] == 3
    assert data["summary"]["failed"] == 0


def test_predict_batch_with_explicit_identifiers(client: TestClient, sample_image_path: str) -> None:
    payload = Path(sample_image_path).read_bytes()
    files = [("files", ("a.png", payload, "image/png")), ("files", ("b.png", payload, "image/png"))]
    response = client.post("/predict/batch", files=files, data={"identifiers": ["id-a", "id-b"]})
    assert response.status_code == 200
    identifiers = [r["image_identifier"] for r in response.json()["data"]["results"]]
    assert identifiers == ["id-a", "id-b"]


def test_predict_batch_identifier_length_mismatch_is_422(client: TestClient, sample_image_path: str) -> None:
    payload = Path(sample_image_path).read_bytes()
    files = [("files", ("a.png", payload, "image/png"))]
    response = client.post("/predict/batch", files=files, data={"identifiers": ["id-a", "id-b"]})
    assert response.status_code == 422
    assert response.json()["success"] is False


def test_predict_batch_empty_is_422(client: TestClient) -> None:
    response = client.post("/predict/batch", files=[])
    assert response.status_code == 422


def test_predict_rejects_empty_upload(client: TestClient) -> None:
    response = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
    assert response.status_code == 422
    assert response.json()["success"] is False


def test_predict_rejects_unsupported_extension(client: TestClient) -> None:
    response = client.post("/predict", files={"file": ("malware.exe", b"whatever", "application/octet-stream")})
    assert response.status_code == 422


def test_predict_corrupted_image_is_structured_failure_not_500(client: TestClient) -> None:
    """Corrupted-but-plausibly-named uploads are handled by
    ``PredictionService`` itself (per this phase's documented design in
    ``backend.utils.validators``) -- a normal 200 response with
    ``success: false`` and a structured error, never a 500."""
    response = client.post("/predict", files={"file": ("corrupt.png", b"not actually a png", "image/png")})
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["success"] is False
    assert data["error"]["error_type"] == "corrupted_image"
