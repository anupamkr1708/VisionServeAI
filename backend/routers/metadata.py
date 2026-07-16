"""
Metadata endpoints: ``/model``, ``/runtime``, ``/classes``, ``/version``,
``/artifacts``.

Every value returned here comes directly from ``ModelService`` /
``RuntimeService`` / ``ArtifactService`` -- this router performs no
computation of its own beyond envelope construction.
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, Union

from fastapi import APIRouter, Depends, Request

from backend.dependencies.runtime import get_artifact_service, get_model_service, get_runtime_service
from backend.schemas.metadata import ArtifactsSchema, ClassesSchema, ModelInfoSchema, RuntimeInfoSchema, VersionSchema
from backend.schemas.responses import APIResponse
from backend.utils.response import success_response
from backend.version import API_VERSION
from services.artifact_service import ArtifactService
from services.model_service import ModelService
from services.runtime_service import RuntimeService

router = APIRouter(tags=["Metadata"])

_NOT_READY_RESPONSES: Dict[Union[int, str], Dict[str, Any]] = {503: {"description": "Model/runtime not ready."}}


@router.get(
    "/model",
    response_model=APIResponse[ModelInfoSchema],
    summary="Model architecture/checkpoint info",
    description="Reconstructed model's architecture, checkpoint, and validation-check facts.",
    operation_id="get_model_info",
    responses=_NOT_READY_RESPONSES,
)
async def get_model_info(
    request: Request,
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> APIResponse[ModelInfoSchema]:
    return success_response(request, data=model_service.model_info(), message="Model info retrieved.")


@router.get(
    "/runtime",
    response_model=APIResponse[RuntimeInfoSchema],
    summary="Active runtime status",
    description="Currently active inference runtime's type, load/validation state, and execution providers.",
    operation_id="get_runtime_info",
    responses=_NOT_READY_RESPONSES,
)
async def get_runtime_info(
    request: Request,
    runtime_service: Annotated[RuntimeService, Depends(get_runtime_service)],
) -> APIResponse[RuntimeInfoSchema]:
    return success_response(request, data=runtime_service.runtime_status(), message="Runtime status retrieved.")


@router.get(
    "/classes",
    response_model=APIResponse[ClassesSchema],
    summary="Class ordering and thresholds",
    description="Resolved canonical class ordering and per-class decision thresholds.",
    operation_id="get_classes",
    responses=_NOT_READY_RESPONSES,
)
async def get_classes(
    request: Request,
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> APIResponse[ClassesSchema]:
    class_names = model_service.class_names()
    thresholds = model_service.thresholds()
    payload = {"num_classes": len(class_names), "class_names": class_names, "thresholds": thresholds}
    return success_response(request, data=payload, message="Class metadata retrieved.")


@router.get(
    "/version",
    response_model=APIResponse[VersionSchema],
    summary="API and model version",
    description="API package version alongside the currently-loaded model's version and fingerprint.",
    operation_id="get_version",
    responses=_NOT_READY_RESPONSES,
)
async def get_version(
    request: Request,
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> APIResponse[VersionSchema]:
    registry = model_service.model_registry
    payload = {
        "api_version": API_VERSION,
        "model_version": model_service.model_version,
        "model_fingerprint_sha256": registry.checkpoint_sha256,
        "backbone": registry.backbone,
    }
    return success_response(request, data=payload, message="Version info retrieved.")


@router.get(
    "/artifacts",
    response_model=APIResponse[ArtifactsSchema],
    summary="Discovered artifact inventory",
    description="Full per-artifact discovery/validation inventory (checkpoints, registries, exports).",
    operation_id="get_artifacts",
    responses=_NOT_READY_RESPONSES,
)
async def get_artifacts(
    request: Request,
    artifact_service: Annotated[ArtifactService, Depends(get_artifact_service)],
) -> APIResponse[ArtifactsSchema]:
    registry = artifact_service.discover()
    return success_response(request, data=registry.to_dict(), message="Artifact inventory retrieved.")
