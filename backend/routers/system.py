"""
System endpoints: ``/info``, ``/ping``, ``/docs-info``.

``/info`` reports process-level facts (app name, API version, resolved
runtime/library environment, whether the service registry has finished
initializing) -- it does not require an initialized registry, so it stays
useful as a diagnostic even while the model is still loading or failed to
load (see ``backend.dependencies.registry.get_registry`` vs
``get_initialized_registry``).
"""
from __future__ import annotations

from fastapi import APIRouter, Request

from backend.schemas.responses import APIResponse, DocsInfoSchema, PingSchema, SystemInfoSchema
from backend.settings import get_settings
from backend.utils.response import success_response
from backend.version import API_VERSION
from inference.utils.environment import get_environment_info

router = APIRouter(tags=["System"])


@router.get(
    "/info",
    response_model=APIResponse[SystemInfoSchema],
    summary="Process/environment info",
    description="App name, API version, resolved runtime/library environment, and whether the service registry is initialized.",
    operation_id="get_system_info",
)
async def get_system_info(request: Request) -> APIResponse[SystemInfoSchema]:
    settings = get_settings()
    registry = getattr(request.app.state, "registry", None)
    payload = {
        "app_name": settings.api_title,
        "api_version": API_VERSION,
        "environment": get_environment_info(seed=settings.seed),
        "registry_initialized": bool(registry is not None and registry.is_initialized),
    }
    return success_response(request, data=payload, message="System info retrieved.")


@router.get(
    "/ping",
    response_model=APIResponse[PingSchema],
    summary="Trivial liveness ping",
    description="Always returns `pong` -- for load balancers/uptime checks that just need a fast 200.",
    operation_id="ping",
)
async def ping(request: Request) -> APIResponse[PingSchema]:
    return success_response(request, data=PingSchema(), message="pong")


@router.get(
    "/docs-info",
    response_model=APIResponse[DocsInfoSchema],
    summary="Documentation endpoint locations",
    description="Where this deployment's interactive API docs are served from, if enabled.",
    operation_id="get_docs_info",
)
async def get_docs_info(request: Request) -> APIResponse[DocsInfoSchema]:
    app = request.app
    payload = {"docs_url": app.docs_url, "redoc_url": app.redoc_url, "openapi_url": app.openapi_url}
    return success_response(request, data=payload, message="Docs info retrieved.")
