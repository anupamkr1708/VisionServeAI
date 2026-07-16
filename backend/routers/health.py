"""
Health endpoints: ``/health``, ``/health/live``, ``/health/ready``.

Every value in every response body here comes straight from
``HealthService.health()`` / ``ServiceRegistry`` state -- this router
performs no health computation of its own, only orchestration and HTTP
status-code selection.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status

from backend.dependencies.runtime import get_health_service
from backend.schemas.health import HealthSchema, LivenessSchema, ReadinessSchema
from backend.schemas.responses import APIResponse
from backend.utils.response import success_response
from services.health_service import HealthService
from services.service_registry import ServiceRegistry

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",
    response_model=APIResponse[HealthSchema],
    summary="Aggregate health report",
    description="Full health snapshot: model, runtime, GPU, memory, providers, artifacts, and environment.",
    operation_id="get_health",
)
async def get_health(
    request: Request,
    health_service: Annotated[HealthService, Depends(get_health_service)],
) -> APIResponse[HealthSchema]:
    report = health_service.health()
    return success_response(request, data=report, message=f"Health status: {report.get('status', 'unknown')}")


@router.get(
    "/live",
    response_model=APIResponse[LivenessSchema],
    summary="Liveness probe",
    description="Process-is-running check with no dependency on model/runtime state. Always 200 once the ASGI app is serving requests.",
    operation_id="get_liveness",
)
async def get_liveness(request: Request) -> APIResponse[LivenessSchema]:
    return success_response(request, data=LivenessSchema(alive=True), message="Process is alive.")


@router.get(
    "/ready",
    response_model=APIResponse[ReadinessSchema],
    summary="Readiness probe",
    description="Whether the service registry has finished initializing and is healthy/degraded enough to accept traffic. Returns HTTP 503 when not ready.",
    operation_id="get_readiness",
    responses={503: {"description": "Not ready to accept traffic."}},
)
async def get_readiness(request: Request, response: Response) -> APIResponse[ReadinessSchema]:
    registry: ServiceRegistry | None = getattr(request.app.state, "registry", None)

    if registry is None or not registry.is_initialized or registry.health is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return success_response(
            request,
            data=ReadinessSchema(ready=False, reason="Service registry has not finished initializing."),
            message="Not ready.",
        )

    report = registry.health.health()
    if report.get("status") == "unhealthy":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return success_response(
            request,
            data=ReadinessSchema(ready=False, reason="Health status is unhealthy."),
            message="Not ready.",
        )

    return success_response(request, data=ReadinessSchema(ready=True), message="Ready.")
