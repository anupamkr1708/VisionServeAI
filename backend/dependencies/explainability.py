"""
``ExplainabilityService`` access for the ``/explain/*`` routes.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, status

from backend.dependencies.registry import get_initialized_registry
from services.explainability_service import ExplainabilityService
from services.service_registry import ServiceRegistry


def get_explainability_service(
    registry: Annotated[ServiceRegistry, Depends(get_initialized_registry)],
) -> ExplainabilityService:
    """Depends on :func:`get_initialized_registry`, then additionally
    checks ``ExplainabilityService.is_available()`` -- a registry can be
    fully initialized with explainability deliberately disabled
    (``ServiceRegistry(..., enable_explainability=False)``, e.g. a
    lightweight deployment that only serves predictions), which is a
    distinct condition from "not initialized yet" and gets its own clear
    503 message rather than being folded into the generic registry check.

    Raises:
        HTTPException: 503, if no ``ExplainabilityEngine`` was ever
            injected into this registry's ``ExplainabilityService``.
    """
    assert registry.explainability is not None
    if not registry.explainability.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Explainability is not available on this deployment "
                "(no ExplainabilityEngine was injected -- see ServiceRegistry(enable_explainability=...))."
            ),
        )
    return registry.explainability
