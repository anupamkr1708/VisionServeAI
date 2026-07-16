"""Tests for the FastAPI lifespan: startup building a real ServiceRegistry,
shutdown releasing it, and fail-fast vs degraded startup behaviour."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.settings import get_settings


def test_startup_builds_and_initializes_registry(live_client: TestClient) -> None:
    app = live_client.app
    assert app.state.registry is not None
    assert app.state.registry.is_initialized is True
    assert app.state.startup_error is None


def test_shutdown_releases_the_runtime(
    synthetic_fixture: Dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Drives startup AND shutdown explicitly (rather than via the
    `live_client` fixture's own teardown) so shutdown's effect on the
    registry can be asserted directly afterward."""
    monkeypatch.setenv("VISIONSERVE_FAIL_FAST_ON_STARTUP", "true")
    monkeypatch.setenv("VISIONSERVE_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("VISIONSERVE_EXPLAINABILITY_OUTPUT_DIR", str(tmp_path / "explainability"))
    monkeypatch.setenv("VISIONSERVE_WARMUP_ITERATIONS", "1")
    for category, path in synthetic_fixture["artifact_roots"].items():
        if path is not None:
            monkeypatch.setenv(f"VISIONSERVE_ROOT_{category.upper()}", str(path))
        else:
            monkeypatch.delenv(f"VISIONSERVE_ROOT_{category.upper()}", raising=False)
    monkeypatch.delenv("VISIONSERVE_ARTIFACT_ROOT", raising=False)
    get_settings.cache_clear()

    app = create_app()
    with TestClient(app) as test_client:
        registry = app.state.registry
        assert registry.is_initialized is True
        response = test_client.get("/health/ready")
        assert response.json()["data"]["ready"] is True

    # Context exit already ran the lifespan's shutdown half.
    assert registry.is_initialized is False


def test_fail_fast_startup_raises_on_missing_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No artifact roots configured + fail_fast (the default) -> the
    real ``ServiceRegistry.initialize()`` failure propagates out of
    ``TestClient.__enter__`` rather than being swallowed."""
    monkeypatch.setenv("VISIONSERVE_FAIL_FAST_ON_STARTUP", "true")
    monkeypatch.setenv("VISIONSERVE_LOG_DIR", str(tmp_path / "logs"))
    for category in ("sprint03", "sprint04_training", "sprint04_evaluation", "nih_chest_xray"):
        monkeypatch.delenv(f"VISIONSERVE_ROOT_{category.upper()}", raising=False)
    monkeypatch.delenv("VISIONSERVE_ARTIFACT_ROOT", raising=False)
    get_settings.cache_clear()

    app = create_app()
    with pytest.raises(RuntimeError):
        with TestClient(app):
            pass  # pragma: no cover -- startup should raise before this runs


def test_degraded_startup_keeps_process_alive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No artifact roots configured + fail_fast disabled -> startup logs
    the failure and continues; the process stays alive and reports it
    honestly rather than raising."""
    monkeypatch.setenv("VISIONSERVE_FAIL_FAST_ON_STARTUP", "false")
    monkeypatch.setenv("VISIONSERVE_LOG_DIR", str(tmp_path / "logs"))
    for category in ("sprint03", "sprint04_training", "sprint04_evaluation", "nih_chest_xray"):
        monkeypatch.delenv(f"VISIONSERVE_ROOT_{category.upper()}", raising=False)
    monkeypatch.delenv("VISIONSERVE_ARTIFACT_ROOT", raising=False)
    get_settings.cache_clear()

    app = create_app()
    with TestClient(app) as test_client:
        assert app.state.registry is None
        assert app.state.startup_error is not None
        response = test_client.get("/health/live")
        assert response.status_code == 200
        response = test_client.get("/health/ready")
        assert response.status_code == 503
