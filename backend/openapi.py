"""
OpenAPI customization: title, summary, description, contact, license, and
tag descriptions for external API consumers.

FastAPI already generates a fully correct schema from every router's
``response_model``/``summary``/``description``/``operation_id`` -- this
module only adds the top-level metadata FastAPI has no way to infer on its
own (contact/license/tag ordering/long-form description), and caches the
result on ``app.openapi_schema`` exactly as FastAPI's own documented
"Extending OpenAPI" recipe does, so it's computed once, not per-request.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from backend.settings import Settings
from backend.version import API_VERSION

TAGS_METADATA = [
    {"name": "Health", "description": "Liveness/readiness/aggregate health reporting."},
    {"name": "Prediction", "description": "Single-image and batch disease prediction."},
    {"name": "Explainability", "description": "Grad-CAM-family and gradient/perturbation-based explanations."},
    {"name": "Metadata", "description": "Model, runtime, class, version, and artifact metadata."},
    {"name": "System", "description": "Process-level diagnostics and documentation locations."},
]

API_DESCRIPTION = """
VisionServeAI's production REST API: chest X-ray multi-label disease
prediction, model explainability (Grad-CAM and related methods), and
operational metadata/health reporting.

This API is a thin orchestration layer -- every prediction, explainability
computation, and model/runtime fact returned here is produced by
VisionServeAI's frozen, independently validated inference and services
layers. Every response uses one consistent envelope:
`{success, message, data, metadata, timestamp, request_id}`.
""".strip()


def customize_openapi(app: FastAPI, settings: Settings) -> None:
    """Install a custom ``app.openapi()`` override on ``app``. Called once
    from ``backend.app.create_app``."""

    def _build_schema() -> Dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema

        schema = get_openapi(
            title=settings.api_title,
            version=API_VERSION,
            summary="Production inference, explainability, and metadata API for VisionServeAI.",
            description=API_DESCRIPTION,
            routes=app.routes,
            tags=TAGS_METADATA,
        )
        schema["info"]["contact"] = {
            "name": settings.api_contact_name,
            "url": settings.api_contact_url,
            "email": settings.api_contact_email,
        }
        schema["info"]["license"] = {
            "name": settings.api_license_name,
            "url": settings.api_license_url,
        }
        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = _build_schema  # type: ignore[method-assign]
