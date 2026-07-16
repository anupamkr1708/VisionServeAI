"""
Dependency-injection providers, used via FastAPI ``Depends(...)``
everywhere a router needs a service.

Structural note (documented in the engineering report as this phase's one
unavoidable adjustment from the requested layout): the requested structure
listed both a top-level ``backend/dependencies.py`` *and* a
``backend/dependencies/`` package with submodules -- a filesystem
collision (a name cannot be both a module and a package in the same parent
directory). This package resolves that the only way that preserves both
intents: the modular breakdown (``registry.py`` / ``runtime.py`` /
``prediction.py`` / ``explainability.py``) is kept exactly as requested,
and this ``__init__.py`` re-exports every public dependency provider from
all four, so callers can do ``from backend.dependencies import
get_prediction_service`` -- i.e. import from ``backend.dependencies`` as
one flat namespace -- exactly as they would with a single
``dependencies.py`` module, while the implementation stays split by
concern internally.
"""
from __future__ import annotations

from backend.dependencies.explainability import get_explainability_service
from backend.dependencies.prediction import get_prediction_service
from backend.dependencies.registry import get_initialized_registry, get_registry
from backend.dependencies.runtime import (
    get_artifact_service,
    get_health_service,
    get_model_service,
    get_runtime_service,
)

__all__ = [
    "get_registry",
    "get_initialized_registry",
    "get_health_service",
    "get_model_service",
    "get_runtime_service",
    "get_artifact_service",
    "get_prediction_service",
    "get_explainability_service",
]
