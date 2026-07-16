"""Tests for centralized exception handling: 404, 422, 500, and the
consistent response envelope every error path returns."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_404_unmatched_route_uses_standard_envelope(client: TestClient) -> None:
    response = client.get("/this/route/does/not/exist")
    assert response.status_code == 404
    body = response.json()
    assert body["success"] is False
    assert body["data"] is None
    assert "request_id" in body
    assert "timestamp" in body


def test_405_method_not_allowed_uses_standard_envelope(client: TestClient) -> None:
    response = client.put("/ping")
    assert response.status_code == 405
    assert response.json()["success"] is False


def test_422_validation_error_uses_standard_envelope(client: TestClient) -> None:
    # Missing the required `file` part entirely -> FastAPI/Pydantic request
    # validation failure (RequestValidationError), not a service-layer error.
    response = client.post("/predict")
    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert "errors" in body["metadata"]


def test_unhandled_exception_returns_generic_500_without_traceback(client: TestClient, sample_image_path: str) -> None:
    """Overrides `get_model_service` to raise a bare `Exception` (the
    catch-all case, distinct from the `RuntimeError`-specific handler),
    demonstrating that (1) the centralized catch-all handler still returns
    the standard envelope and never leaks a stack trace, and (2)
    dependency overrides can be used to deterministically exercise error
    paths that are otherwise hard to trigger with real services.

    Uses a second ``TestClient`` with ``raise_server_exceptions=False`` --
    Starlette's own documented pattern for testing a custom handler
    registered for bare ``Exception``/500: that handler is installed on
    ``ServerErrorMiddleware`` (the outermost middleware), which -- by
    design (see ``ServerErrorMiddleware.__call__``'s own comment, "We
    always continue to raise the exception... allows test clients to
    optionally raise the error within the test case") -- always re-raises
    after generating the response, so a real ASGI server can log it. The
    default ``client`` fixture's ``TestClient`` (``raise_server_exceptions``
    defaults to ``True``) would surface that re-raise as a test failure
    even though a real HTTP client only ever sees the clean response this
    assertion checks.
    """
    from backend.dependencies.runtime import get_model_service

    def _boom():
        raise Exception("boom: something unexpected happened internally")

    client.app.dependency_overrides[get_model_service] = _boom
    try:
        # Deliberately NOT entered as its own `with` block: the outer
        # `client` fixture's TestClient has already driven this same
        # `app`'s lifespan startup; re-entering it here would run startup
        # a second time on the same `app.state`. A plain (unmanaged)
        # TestClient still routes real HTTP calls through the already-
        # running ASGI app just fine.
        lenient_client = TestClient(client.app, raise_server_exceptions=False)
        response = lenient_client.get("/model")
    finally:
        del client.app.dependency_overrides[get_model_service]

    assert response.status_code == 500
    body = response.json()
    assert body["success"] is False
    assert "boom" not in body["message"]
    assert "Traceback" not in response.text


def test_runtime_error_maps_to_500(client: TestClient) -> None:
    from backend.dependencies.runtime import get_runtime_service

    def _boom():
        raise RuntimeError("simulated internal runtime failure")

    client.app.dependency_overrides[get_runtime_service] = _boom
    try:
        response = client.get("/runtime")
    finally:
        del client.app.dependency_overrides[get_runtime_service]

    assert response.status_code == 500
    assert response.json()["success"] is False


def test_value_error_maps_to_400(client: TestClient) -> None:
    from backend.dependencies.prediction import get_prediction_service

    class _BadService:
        def predict_from_bytes(self, *_args, **_kwargs):
            raise ValueError("simulated bad input from the service layer")

    client.app.dependency_overrides[get_prediction_service] = lambda: _BadService()
    try:
        response = client.post("/predict", files={"file": ("sample.png", b"\x89PNG\r\n\x1a\n", "image/png")})
    finally:
        del client.app.dependency_overrides[get_prediction_service]

    assert response.status_code == 400
    assert response.json()["success"] is False


def test_request_id_present_and_consistent_with_header(client: TestClient) -> None:
    response = client.get("/ping")
    body_request_id = response.json()["request_id"]
    header_request_id = response.headers.get("X-Request-ID")
    assert body_request_id == header_request_id
    assert "X-Process-Time-Ms" in response.headers
