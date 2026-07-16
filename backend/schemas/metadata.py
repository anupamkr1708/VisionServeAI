"""
Metadata response payload schemas, for ``/model``, ``/runtime``,
``/classes``, ``/version``, and ``/artifacts``.

``ModelInfoSchema`` / ``ModelValidationChecksSchema`` mirror
``inference.model_registry.ModelRegistry`` / ``ModelValidationChecks``
field for field -- small, stable, frozen dataclasses, fully typed here
rather than left as ``Dict[str, Any]``. ``RuntimeInfoSchema`` /
``ArtifactsSchema`` stay ``Dict[str, Any]`` at the leaf level: they wrap
``RuntimeService.runtime_status()`` / ``ArtifactService.discover()`` +
``ArtifactRegistry.to_dict()``, whose per-record/per-runtime shapes are
themselves owned by those (frozen) services -- see
``backend.schemas.health``'s module docstring for the same reasoning
applied there.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelValidationChecksSchema(BaseModel):
    """Mirrors ``inference.model_registry.ModelValidationChecks``."""

    architecture_instantiated: bool
    checkpoint_loaded: bool
    state_dict_strict_match: bool
    parameter_count_expected: Optional[bool] = None
    output_dimension_expected: Optional[bool] = None
    classifier_dimension_expected: Optional[bool] = None
    all_params_correct_device: bool
    all_params_correct_dtype: bool
    eval_mode: bool
    grad_disabled: bool
    errors: List[str] = Field(default_factory=list)


class ModelInfoSchema(BaseModel):
    """``/model``'s ``data`` payload. Mirrors
    ``inference.model_registry.ModelRegistry`` field for field."""

    backbone: str
    num_classes: int
    device: str
    dtype: str
    total_parameters: int
    trainable_parameters: int
    checkpoint_path: str
    checkpoint_sha256: str
    model_signature: Dict[str, Any]
    validation: ModelValidationChecksSchema


class RuntimeInfoSchema(BaseModel):
    """``/runtime``'s ``data`` payload. Mirrors
    ``services.runtime_service.RuntimeService.runtime_status()``'s
    top-level keys; ``metadata`` stays a passthrough dict (see module
    docstring)."""

    initialized: bool
    runtime_type: str
    loaded: Optional[bool] = None
    validated: Optional[bool] = None
    selected_providers: List[str] = Field(default_factory=list)
    available_providers: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    last_validation_errors: List[str] = Field(default_factory=list)


class ClassesSchema(BaseModel):
    """``/classes``'s ``data`` payload -- resolved class ordering and
    per-class thresholds, from ``ModelService.class_names()`` /
    ``ModelService.thresholds()``."""

    num_classes: int
    class_names: List[str]
    thresholds: Dict[str, float]


class VersionSchema(BaseModel):
    """``/version``'s ``data`` payload -- API package version alongside the
    currently-loaded model's own version/fingerprint."""

    api_version: str
    model_version: Optional[str] = None
    model_fingerprint_sha256: Optional[str] = None
    backbone: Optional[str] = None


class ArtifactsSchema(BaseModel):
    """``/artifacts``'s ``data`` payload. Mirrors
    ``services.artifact_service.ArtifactRegistry.to_dict()``'s top-level
    keys; ``records`` stays a passthrough dict of per-artifact records
    (see module docstring)."""

    total_artifacts: int
    critical_missing: List[str]
    duplicate_groups: List[List[str]]
    records: Dict[str, Any]
