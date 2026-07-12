"""
Application Service Layer: sits between the inference/runtime modules and
future FastAPI endpoints (not implemented here -- see module docstrings
throughout this package for exactly what is and isn't in scope).

Public API re-exported here for convenience::

    from services import ServiceRegistry
    registry = ServiceRegistry(artifact_roots={...}, export_dir=...)
    registry.initialize()
    registry.prediction.predict(image)
"""
from __future__ import annotations

from services.artifact_service import ArtifactService
from services.explainability_service import ExplainabilityService
from services.health_service import HealthService
from services.model_service import ModelService
from services.prediction_service import PredictionService
from services.runtime_service import RuntimeService
from services.service_registry import ServiceRegistry

__all__ = [
    "ArtifactService",
    "ModelService",
    "RuntimeService",
    "PredictionService",
    "ExplainabilityService",
    "HealthService",
    "ServiceRegistry",
]
