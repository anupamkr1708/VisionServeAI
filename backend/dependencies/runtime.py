"""
Read-only service accessors used by the health and metadata routers:
``ModelService``, ``RuntimeService``, ``ArtifactService``, and
``HealthService``.

Named ``runtime.py`` per this phase's requested ``backend/dependencies/``
layout; it covers every service besides prediction/explainability (which
have their own dedicated modules, since those two carry extra
availability/validation concerns of their own).
"""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from backend.dependencies.registry import get_initialized_registry, get_registry
from services.artifact_service import ArtifactService
from services.health_service import HealthService
from services.model_service import ModelService
from services.runtime_service import RuntimeService
from services.service_registry import ServiceRegistry


def get_health_service(request: Request) -> HealthService:
    """``HealthService`` for the ``/health`` routes.

    Deliberately depends on :func:`get_registry` (NOT
    ``get_initialized_registry``) -- a health check must still respond
    (with an honest ``"degraded"``/``"unhealthy"`` status) while the
    registry is constructed but not yet finished initializing, rather than
    itself returning a bare 503 with no diagnostic detail. Raises only if
    the registry was never constructed at all (see
    ``backend.dependencies.registry.get_registry``).
    """
    registry: ServiceRegistry = get_registry(request)
    if registry.health is not None:
        return registry.health
    # Constructed but ServiceRegistry.initialize() hasn't run far enough to
    # build HealthService yet -- fall back to a clean 503 rather than a
    # confusing AttributeError, so `/health` still fails predictably during
    # the narrow startup window before HealthService exists.
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Health service is not available yet (application is still starting up).",
    )


def get_model_service(
    registry: Annotated[ServiceRegistry, Depends(get_initialized_registry)],
) -> ModelService:
    assert registry.model is not None
    return registry.model


def get_runtime_service(
    registry: Annotated[ServiceRegistry, Depends(get_initialized_registry)],
) -> RuntimeService:
    assert registry.runtime is not None
    return registry.runtime


def get_artifact_service(
    registry: Annotated[ServiceRegistry, Depends(get_initialized_registry)],
) -> ArtifactService:
    assert registry.artifact is not None
    return registry.artifact
