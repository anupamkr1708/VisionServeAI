"""
Health response payload schemas, for ``/health``, ``/health/live``, and
``/health/ready``.

``HealthSchema`` mirrors ``services.health_service.HealthService.health()``'s
top-level keys exactly. Its nested sections (``runtime``, ``gpu``,
``memory``, ``providers``, ``artifacts``, ``environment``) stay
``Dict[str, Any]`` rather than being re-declared as parallel Pydantic
models field by field -- those sections are themselves aggregates of
several other frozen services/utilities
(``RuntimeService.runtime_status()``, ``inference.utils.resource_monitor``,
``ArtifactService.artifact_status()``, ``inference.utils.environment.
get_environment_info()``); duplicating their shapes here would silently
drift out of sync the next time any of those change. The *top-level*
contract this API makes (which keys exist, and their meaning) is fully
explicit below.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel


class LivenessSchema(BaseModel):
    """``/health/live``'s ``data`` payload -- process-is-running only, no
    dependency on the service registry."""

    alive: bool = True


class ReadinessSchema(BaseModel):
    """``/health/ready``'s ``data`` payload -- whether the service registry
    is initialized and healthy/degraded enough to accept traffic."""

    ready: bool
    reason: Optional[str] = None


class HealthSchema(BaseModel):
    """Mirrors ``services.health_service.HealthService.health()``'s
    top-level keys."""

    status: str
    model_loaded: bool
    runtime: Dict[str, Any]
    gpu: Dict[str, Any]
    memory: Dict[str, Any]
    providers: Dict[str, Any]
    artifacts: Dict[str, Any]
    environment: Dict[str, Any]
