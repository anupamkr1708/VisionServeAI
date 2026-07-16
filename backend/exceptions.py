"""
Centralized exception handling.

Every exception handler here returns the same envelope shape
(``backend.schemas.responses.APIResponse`` with ``success=False``) so a
client never has to special-case error responses' top-level shape versus
success responses'. Stack traces are never included in a response body --
unhandled exceptions are logged (with traceback) via
``backend.logging.log_exception`` and reported to the client as a generic,
safe message.

Exception -> HTTP status mapping, and why:

* ``fastapi.exceptions.RequestValidationError`` -> 422 -- FastAPI/Pydantic's
  own request-shape validation (missing/malformed fields).
* ``backend.utils.image.ImageDecodeError`` -> 422 -- malformed/corrupted
  upload content, surfaced from ``backend.utils.validators``.
* ``ValueError`` / ``TypeError`` -> 400 -- caller-supplied input the
  service layer itself rejected as malformed (e.g.
  ``PredictionService.predict_batch``'s length-mismatch ``ValueError``,
  ``PredictionService.predict``'s unsupported-type ``TypeError``,
  ``ExplainabilityEngine.generate``'s unknown-method ``ValueError``) --
  never a server-side fault, so 400 (not 500) is correct.
* ``FileNotFoundError`` -> 500 -- indicates a server-side artifact/config
  problem (a required file the deployment expected to exist is missing at
  runtime), not a missing *HTTP resource*; a bare 404 here would be
  misleading to a client that did nothing wrong. Reported generically,
  never with the underlying file path (that would leak server filesystem
  layout).
* ``RuntimeError`` -> 500 -- anything that reaches here (rather than being
  caught earlier by the ``get_initialized_registry`` dependency, which
  already turns "runtime/model not ready" into a clean 503 before any
  service method is even called) indicates an unexpected internal
  failure, not a transient availability problem.
* ``starlette.exceptions.HTTPException`` (including 404s for unmatched
  routes, 405s for wrong methods, and every explicit ``HTTPException`` a
  dependency/router raises) -- re-wrapped into the standard envelope,
  status code preserved.
* ``Exception`` (catch-all) -> 500 -- last resort; logged with full
  traceback, reported generically.

No dedicated ``PredictionError`` handler: no such exception type exists
anywhere in this repository's frozen ``inference``/``services`` modules
(``PredictionService`` either returns a structured failed
``PredictionResult`` per-image, never raising, or raises the plain
``ValueError``/``TypeError``/``RuntimeError`` already covered above). The
phase prompt names it as one of several exception types to map; this is
recorded here rather than silently ignored -- see the engineering report's
"unavoidable changes / notes" section.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.logging import log_exception
from backend.schemas.responses import APIResponse
from backend.utils.image import ImageDecodeError
from backend.utils.response import error_response


def register_exception_handlers(app: FastAPI) -> None:
    """Register every centralized exception handler on ``app``. Called
    once from ``backend.app.create_app``."""

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        envelope = error_response(
            request,
            message="Request validation failed.",
            metadata={"errors": exc.errors()},
        )
        return _json(422, envelope)

    @app.exception_handler(ImageDecodeError)
    async def _handle_image_decode_error(request: Request, exc: ImageDecodeError) -> JSONResponse:
        envelope = error_response(request, message=str(exc))
        return _json(422, envelope)

    @app.exception_handler(ValueError)
    async def _handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
        envelope = error_response(request, message=str(exc))
        return _json(status.HTTP_400_BAD_REQUEST, envelope)

    @app.exception_handler(TypeError)
    async def _handle_type_error(request: Request, exc: TypeError) -> JSONResponse:
        envelope = error_response(request, message=str(exc))
        return _json(status.HTTP_400_BAD_REQUEST, envelope)

    @app.exception_handler(FileNotFoundError)
    async def _handle_file_not_found(request: Request, exc: FileNotFoundError) -> JSONResponse:
        log_exception(request_id=_rid(request), method=request.method, path=request.url.path, exc=exc)
        envelope = error_response(request, message="A required server-side resource was not found.")
        return _json(status.HTTP_500_INTERNAL_SERVER_ERROR, envelope)

    @app.exception_handler(RuntimeError)
    async def _handle_runtime_error(request: Request, exc: RuntimeError) -> JSONResponse:
        log_exception(request_id=_rid(request), method=request.method, path=request.url.path, exc=exc)
        envelope = error_response(request, message="An internal error occurred while processing the request.")
        return _json(status.HTTP_500_INTERNAL_SERVER_ERROR, envelope)

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        envelope = error_response(request, message=str(exc.detail))
        return _json(exc.status_code, envelope, headers=exc.headers)

    @app.exception_handler(Exception)
    async def _handle_unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        log_exception(request_id=_rid(request), method=request.method, path=request.url.path, exc=exc)
        envelope = error_response(request, message="An unexpected internal error occurred.")
        return _json(status.HTTP_500_INTERNAL_SERVER_ERROR, envelope)


def _rid(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _json(
    status_code: int,
    envelope: "APIResponse[Any]",
    headers: Optional[Mapping[str, str]] = None,
) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=envelope.model_dump(), headers=dict(headers) if headers else None)
