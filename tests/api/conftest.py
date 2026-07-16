"""
Shared fixtures for ``tests/api``.

Builds on top of the top-level ``tests/conftest.py`` fixtures
(``synthetic_fixture``, ``sample_image_path``, ``initialized_registry``,
``logger``) -- nothing here duplicates artifact/model construction; it only
adds the FastAPI-specific wiring those fixtures need to exercise the
``backend`` package.

Two complementary ``TestClient`` fixtures are provided, deliberately
covering two different (both explicitly requested) testing concerns:

* :func:`client` -- fast, per-test-isolated, uses FastAPI's
  ``dependency_overrides`` to inject the already-built, real (not mocked)
  services from the shared ``initialized_registry`` fixture directly,
  bypassing this test run's ``ServiceRegistry``-construction-via-lifespan
  entirely. Used by the bulk of endpoint tests. Demonstrates this phase's
  explicitly requested "dependency overrides" test coverage. The real
  lifespan still runs underneath it (see ``_degraded_env`` below) and is
  expected to fail softly (no artifacts are configured) -- harmless, since
  every endpoint exercised through this fixture reaches its service via an
  overridden dependency, never via ``app.state`` directly.
* :func:`live_client` -- exercises the REAL environment-variable-driven
  ``lifespan`` startup/shutdown path end to end (see ``backend.lifecycle``),
  pointed at a synthetic artifact tree via ``VISIONSERVE_ROOT_*`` env vars.
  Used by tests that need ``app.state.registry`` to be genuinely populated
  by startup (``/health/ready``, ``/info``, and the startup/shutdown
  lifecycle tests themselves), since those read ``request.app.state``
  directly rather than through an overridable ``Depends()`` callable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.dependencies.explainability import get_explainability_service
from backend.dependencies.prediction import get_prediction_service
from backend.dependencies.runtime import (
    get_artifact_service,
    get_health_service,
    get_model_service,
    get_runtime_service,
)
from backend.settings import get_settings
from services.service_registry import ServiceRegistry

_ARTIFACT_CATEGORIES = ("sprint03", "sprint04_training", "sprint04_evaluation", "nih_chest_xray")


@pytest.fixture(autouse=True)
def _degraded_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Autouse for every test in this package: no artifact env vars are
    configured, and startup is set to fail *softly*
    (``VISIONSERVE_FAIL_FAST_ON_STARTUP=false``) rather than crash. This is
    the right default for :func:`client` (see its docstring); tests using
    :func:`live_client` override these same variables again afterward with
    real values, which wins since fixture setup order runs this one first
    (autouse fixtures resolve before explicitly-requested same-scope
    fixtures)."""
    monkeypatch.setenv("VISIONSERVE_FAIL_FAST_ON_STARTUP", "false")
    for category in _ARTIFACT_CATEGORIES:
        monkeypatch.delenv(f"VISIONSERVE_ROOT_{category.upper()}", raising=False)
    monkeypatch.delenv("VISIONSERVE_ARTIFACT_ROOT", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def client(initialized_registry: ServiceRegistry) -> Iterator[TestClient]:
    """Fast endpoint-test client: real services (from ``initialized_registry``),
    injected via ``dependency_overrides`` rather than a real app startup."""
    app = create_app()

    app.dependency_overrides[get_prediction_service] = lambda: initialized_registry.prediction
    app.dependency_overrides[get_model_service] = lambda: initialized_registry.model
    app.dependency_overrides[get_runtime_service] = lambda: initialized_registry.runtime
    app.dependency_overrides[get_artifact_service] = lambda: initialized_registry.artifact
    app.dependency_overrides[get_explainability_service] = lambda: initialized_registry.explainability
    app.dependency_overrides[get_health_service] = lambda: initialized_registry.health

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture()
def live_client(
    synthetic_fixture: Dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> Iterator[TestClient]:
    """Full real-startup client: environment variables point at a synthetic
    artifact tree, the real ``lifespan`` builds a real ``ServiceRegistry``
    (fail-fast enabled -- the default, and the correct behaviour to test),
    and ``TestClient``'s context manager drives real startup/shutdown.
    """
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
        yield test_client

    get_settings.cache_clear()
