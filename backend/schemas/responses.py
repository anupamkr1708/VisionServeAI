"""
Response envelope and shared response-data schemas.

``APIResponse`` is the one envelope every endpoint returns (see the phase
prompt's "Response Format" section). It is generic over its ``data``
payload so each router can declare a precise ``response_model`` (e.g.
``APIResponse[PredictionResultSchema]``) while every endpoint still shares
one consistent top-level shape for clients to parse against.

This file holds the generic envelope plus prediction/system payloads.
Explainability, metadata, and health payloads live in their own sibling
modules (``backend.schemas.explainability`` / ``.metadata`` / ``.health``)
per this phase's requested schema layout.

Nested payload schemas below mirror already-frozen dataclasses field for
field (``inference.postprocessing.PredictionResult``) -- no new fields are
invented, no existing field is renamed or reinterpreted. Aggregate service
outputs that are already loosely-shaped, nested dictionaries by design
(e.g. ``ArtifactService.artifact_status()`` / ``ArtifactRegistry.to_dict()``,
``RuntimeService.runtime_status()``) are typed as ``Dict[str, Any]`` at the
leaf level rather than re-declared as parallel Pydantic models one field at
a time -- doing the latter would mean this file silently drifting out of
sync with those services' own internal dataclasses every time either
changed, which is a worse failure mode than an honestly-untyped-but-always-
current ``Dict[str, Any]``. Top-level shape for each of those is still
explicit and documented per response schema below.
"""
from __future__ import annotations

from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

DataT = TypeVar("DataT")


class APIResponse(BaseModel, Generic[DataT]):
    """Top-level envelope returned by every endpoint in this API."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "success": True,
            "message": "OK",
            "data": {},
            "metadata": {},
            "timestamp": "2026-07-15T12:00:00+00:00",
            "request_id": "3e5f0a5e-2f7e-4a41-9c8e-6b9a2b6f2d31",
        }
    })

    success: bool = Field(..., description="Whether the request completed successfully.")
    message: str = Field(..., description="Human-readable summary of the result.")
    data: Optional[DataT] = Field(default=None, description="Endpoint-specific payload. `null` on failure.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Non-essential auxiliary information.")
    timestamp: str = Field(..., description="UTC ISO-8601 timestamp the response was generated at.")
    request_id: str = Field(..., description="Correlation id for this request (see the `X-Request-ID` header).")


# ======================================================================
# Prediction payloads -- mirror inference.postprocessing.PredictionResult
# ======================================================================


class PredictionErrorSchema(BaseModel):
    """Mirrors ``inference.preprocessing.InferenceError.to_dict()``."""

    error_type: str
    message: str


class PredictionResultSchema(BaseModel):
    """Mirrors ``inference.postprocessing.PredictionResult`` field for
    field."""

    image_identifier: str
    success: bool
    predicted_diseases: List[str] = Field(default_factory=list)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    probabilities: Dict[str, float] = Field(default_factory=dict)
    thresholds_used: Dict[str, float] = Field(default_factory=dict)
    inference_timestamp_utc: Optional[str] = None
    model_fingerprint_sha256: Optional[str] = None
    model_version: Optional[str] = None
    error: Optional[PredictionErrorSchema] = None


class BatchPredictionSummarySchema(BaseModel):
    """Mirrors ``inference.postprocessing.summarize_predictions()``'s
    return shape."""

    total: int
    successful: int
    failed: int
    failure_reasons: List[Optional[str]] = Field(default_factory=list)
    disease_frequency: Dict[str, int] = Field(default_factory=dict)


class BatchPredictionResponseSchema(BaseModel):
    """``/predict/batch``'s ``data`` payload: per-image results (order
    preserved, matching ``PredictionService.predict_batch``'s ordering
    guarantee) plus an aggregate summary."""

    results: List[PredictionResultSchema]
    summary: BatchPredictionSummarySchema


# ======================================================================
# System payloads
# ======================================================================


class PingSchema(BaseModel):
    message: str = "pong"


class SystemInfoSchema(BaseModel):
    """``/info``'s ``data`` payload."""

    app_name: str
    api_version: str
    environment: Dict[str, Any]
    registry_initialized: bool


class DocsInfoSchema(BaseModel):
    """``/docs-info``'s ``data`` payload."""

    docs_url: Optional[str]
    redoc_url: Optional[str]
    openapi_url: Optional[str]
