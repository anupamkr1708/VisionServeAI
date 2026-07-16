"""
Response envelope construction.

Every endpoint in this API returns the same top-level shape (see
``backend.schemas.responses.APIResponse``):
``{success, message, data, metadata, timestamp, request_id}``. This module
is the single place that assembles that envelope, so every router builds
it the same way rather than repeating ``datetime.now()``/request-id lookup
logic per endpoint.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional, TypeVar

from starlette.requests import Request

from backend.schemas.responses import APIResponse

T = TypeVar("T")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_request_id(request: Request) -> str:
    """Read the request id ``backend.middleware.RequestIDMiddleware``
    already generated/propagated for this request (falls back to
    ``"unknown"`` only if called outside that middleware's scope, e.g. in a
    unit test that builds a response without going through the full app)."""
    return getattr(request.state, "request_id", "unknown")


def success_response(
    request: Request,
    data: Any,
    message: str = "OK",
    metadata: Optional[Dict[str, Any]] = None,
) -> APIResponse:
    """Build a successful :class:`APIResponse` envelope."""
    return APIResponse(
        success=True,
        message=message,
        data=data,
        metadata=metadata or {},
        timestamp=_utc_now_iso(),
        request_id=get_request_id(request),
    )


def error_response(
    request: Request,
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> APIResponse:
    """Build a failed :class:`APIResponse` envelope. Used by the centralized
    exception handlers (``backend.exceptions``) -- routers themselves
    should raise ``HTTPException`` (or let a lower-layer exception
    propagate) rather than constructing error envelopes directly, so every
    error path is handled in exactly one place."""
    return APIResponse(
        success=False,
        message=message,
        data=None,
        metadata=metadata or {},
        timestamp=_utc_now_iso(),
        request_id=get_request_id(request),
    )
