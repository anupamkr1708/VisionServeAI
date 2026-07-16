"""
Custom ASGI middleware: request id / correlation id generation, and
structured request logging with timing.

CORS, ``TrustedHost``, and GZip are all standard Starlette middleware with
no custom behaviour needed -- those are registered directly via
``app.add_middleware(...)`` in ``backend.app.create_app`` rather than
wrapped here.
"""
from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from backend.logging import log_request

REQUEST_ID_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Generates a request id for every incoming request (or reuses an
    ``X-Request-ID`` header the caller already supplied, so a request can
    be correlated across service boundaries), stores it on
    ``request.state.request_id`` for every downstream handler/response
    builder (see ``backend.utils.response.get_request_id``), and echoes it
    back as a response header."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Times every request and emits one structured log line
    (``backend.logging.log_request``) per completed request, plus an
    ``X-Process-Time-Ms`` response header."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        request_id = getattr(request.state, "request_id", "unknown")
        log_request(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            elapsed_ms=elapsed_ms,
            client_host=request.client.host if request.client else None,
        )
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response
