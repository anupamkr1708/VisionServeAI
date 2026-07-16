"""
``PredictionService`` access for the ``/predict`` routes.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from backend.dependencies.registry import get_initialized_registry
from services.prediction_service import PredictionService
from services.service_registry import ServiceRegistry


def get_prediction_service(
    registry: Annotated[ServiceRegistry, Depends(get_initialized_registry)],
) -> PredictionService:
    """Depends on :func:`get_initialized_registry` -- a 503 is raised
    before this function body ever runs if the model/runtime aren't ready,
    satisfying this phase's "validate ... missing runtime, runtime
    unavailable, service unavailable ... before calling services"
    requirement at the dependency layer rather than inside the router
    body."""
    assert registry.prediction is not None
    return registry.prediction
