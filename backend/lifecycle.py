"""
Application lifespan: constructs the process's one ``ServiceRegistry`` at
startup, tears it down at shutdown.

Uses the modern ``asynccontextmanager``-based FastAPI ``lifespan`` API
(the ``@app.on_event("startup"/"shutdown")`` decorators this phase's
prompt explicitly asks to avoid are deprecated upstream). Building/
destroying the registry is genuinely blocking, CPU/IO-bound work
(checkpoint deserialization, runtime warmup) -- run in a worker thread via
``anyio.to_thread`` so it doesn't block the event loop other requests
(like ``/health/live``) might otherwise be able to answer immediately.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import anyio
from fastapi import FastAPI

from backend.logging import get_logger
from backend.settings import Settings, get_settings
from services.service_registry import ServiceRegistry


def _build_and_initialize_registry(settings: Settings) -> ServiceRegistry:
    """Synchronous, blocking construction — run off the event loop by
    :func:`lifespan` via ``anyio.to_thread.run_sync``."""
    logger = get_logger()
    artifact_roots = settings.resolve_artifact_roots(logger=logger)

    device = None
    if settings.device:
        import torch

        device = torch.device(settings.device)

    registry = ServiceRegistry(
        artifact_roots=artifact_roots,
        export_dir=settings.export_dir,
        device=device,
        runtime_type=settings.runtime_type,
        runtime_path=settings.runtime_path,
        warmup_iterations=settings.warmup_iterations,
        logger=logger,
        log_dir=settings.log_dir,
        enable_explainability=settings.enable_explainability,
        explainability_output_dir=settings.explainability_output_dir,
        seed=settings.seed,
    )
    registry.initialize(warmup=settings.warmup_on_startup, validate_runtime=settings.validate_runtime_on_startup)
    return registry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan: build the one ``ServiceRegistry`` for this process
    on startup (stored on ``app.state.registry`` -- never a module-level
    global, satisfying "no global state" / "only one ServiceRegistry
    instance"), shut it down on exit.

    Startup failure behaviour is controlled by
    ``Settings.fail_fast_on_startup``:

    * ``True`` (default) -- a failed ``ServiceRegistry.initialize()``
      (missing critical artifacts, checkpoint mismatch, runtime validation
      failure, ...) re-raises, which FastAPI/uvicorn surfaces as a hard
      startup failure -- the correct behaviour for a production serving
      container, so an orchestrator sees the crash immediately rather than
      a silently-degraded process serving 503s forever.
    * ``False`` -- the failure is logged and swallowed;
      ``app.state.registry`` stays ``None``, and every endpoint that
      depends on it (see ``backend.dependencies.registry``) reports a
      clean 503 rather than crashing the whole process. Useful for
      environments that want the process alive (e.g. so ``/health/live``
      keeps answering) even while artifacts are being provisioned.
    """
    logger = get_logger()
    settings = get_settings()
    app.state.registry = None
    app.state.startup_error = None
    app.state.started_at = time.time()

    try:
        logger.info("LIFESPAN starting up: building ServiceRegistry...")
        registry = await anyio.to_thread.run_sync(_build_and_initialize_registry, settings)
        app.state.registry = registry
        logger.info("LIFESPAN startup complete: ServiceRegistry initialized.")
    except Exception as exc:  # noqa: BLE001 -- decide fail-fast vs degraded below
        app.state.startup_error = str(exc)
        logger.error("LIFESPAN startup FAILED: %s", exc, exc_info=exc)
        if settings.fail_fast_on_startup:
            raise
        logger.warning("LIFESPAN continuing in degraded mode (fail_fast_on_startup=False).")

    yield

    logger.info("LIFESPAN shutting down...")
    registry = app.state.registry
    if registry is not None:
        await anyio.to_thread.run_sync(registry.shutdown)
    logger.info("LIFESPAN shutdown complete.")
