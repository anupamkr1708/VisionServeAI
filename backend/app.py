"""
FastAPI application factory.

Assembles every piece this phase built -- settings, logging, lifespan,
middleware, exception handlers, routers, OpenAPI customization -- into one
``FastAPI`` instance. Contains no ML/business logic of its own.

Middleware layering (outermost to innermost, i.e. the order each sees an
incoming request): ``RequestIDMiddleware`` -> ``RequestLoggingMiddleware``
-> ``TrustedHostMiddleware`` -> ``CORSMiddleware`` -> ``GZipMiddleware`` ->
routing. Request id is assigned before anything else (so logging/error
responses can always include it); logging wraps everything below it (so
its timing includes compression); host/CORS policy is enforced before
GZip touches the response body; GZip sits innermost so it compresses the
actual response the router produced. Starlette builds its middleware stack
such that the *last* ``add_middleware`` call becomes the *outermost* layer
(see ``Starlette.build_middleware_stack``) -- the call order below is
therefore the reverse of this paragraph's outer-to-inner description.
"""
from __future__ import annotations

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from backend.exceptions import register_exception_handlers
from backend.lifecycle import lifespan
from backend.logging import configure_logging
from backend.middleware import RequestIDMiddleware, RequestLoggingMiddleware
from backend.openapi import TAGS_METADATA, customize_openapi
from backend.routers import explainability, health, metadata, prediction, system
from backend.settings import get_settings
from backend.version import API_VERSION


def create_app() -> FastAPI:
    """Build a fully-wired FastAPI application. Safe to call more than
    once (e.g. once per test) -- each call builds an independent ``app``
    with its own lifespan-managed ``ServiceRegistry``."""
    settings = get_settings()
    configure_logging(settings.log_dir)

    app = FastAPI(
        title=settings.api_title,
        version=API_VERSION,
        lifespan=lifespan,
        openapi_tags=TAGS_METADATA,
    )

    # Innermost -> outermost call order (see module docstring).
    app.add_middleware(GZipMiddleware, minimum_size=settings.gzip_minimum_size)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_hosts)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(prediction.router)
    app.include_router(explainability.router)
    app.include_router(metadata.router)
    app.include_router(system.router)

    customize_openapi(app, settings)

    return app


app = create_app()
