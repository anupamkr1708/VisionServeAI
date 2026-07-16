"""
Structured logging for the FastAPI orchestration layer.

Reuses :func:`inference.utils.logging.build_logger` (the repository's one
canonical console+file logger factory -- see that module's own docstring)
rather than reimplementing formatter/handler setup. This module's only new
contribution is naming/wiring: one shared ``"visionserve.backend"`` logger
for the whole API process, plus a small helper
(:func:`log_request`) that emits the one structured line per request
``backend.middleware.RequestLoggingMiddleware`` needs (request id, route,
status code, elapsed time) in a single, consistent format. No ``print()``
anywhere in this package.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from inference.utils.logging import build_logger

_LOGGER_NAME = "visionserve.backend"

_logger: Optional[logging.Logger] = None


def configure_logging(log_dir: Path, level: str = "INFO") -> logging.Logger:
    """Build (or rebuild) the shared backend logger. Idempotent -- safe to
    call once at app startup; :func:`get_logger` returns whatever this most
    recently configured."""
    global _logger
    _logger = build_logger(name=_LOGGER_NAME, log_dir=log_dir, log_filename="backend_api.log", level=level)
    return _logger


def get_logger() -> logging.Logger:
    """Return the shared backend logger, configuring a sensible default
    (``./logs``) the first time it's called if :func:`configure_logging`
    hasn't run yet (e.g. module import order in a test)."""
    global _logger
    if _logger is None:
        _logger = configure_logging(Path("logs"))
    return _logger


def log_request(
    *,
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    elapsed_ms: float,
    client_host: Optional[str] = None,
) -> None:
    """Emit the one structured access-log line per completed request."""
    get_logger().info(
        "REQUEST id=%s method=%s path=%s status=%d elapsed_ms=%.2f client=%s",
        request_id, method, path, status_code, elapsed_ms, client_host or "-",
    )


def log_exception(*, request_id: str, method: str, path: str, exc: BaseException) -> None:
    """Emit one structured error-log line (with full traceback via
    ``exc_info``) for an unhandled exception -- the traceback is logged
    server-side only; API responses never expose it (see
    ``backend.exceptions``)."""
    get_logger().error(
        "REQUEST_FAILED id=%s method=%s path=%s error_type=%s error=%s",
        request_id, method, path, type(exc).__name__, exc, exc_info=exc,
    )
