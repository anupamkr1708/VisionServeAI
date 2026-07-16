"""
``ServiceRegistry`` access dependencies.

The registry itself is built once, in ``backend.lifecycle``'s lifespan
context manager, and stored on ``app.state.registry`` -- never a module-
level global (per this phase's "no global state" / "only one
ServiceRegistry instance" requirements). Every other dependency in this
package reaches the registry through :func:`get_registry` /
:func:`get_initialized_registry` rather than importing or constructing one
itself.
"""
from __future__ import annotations

from typing import Optional

from fastapi import HTTPException, Request, status

from services.service_registry import ServiceRegistry


def get_registry(request: Request) -> ServiceRegistry:
    """The process's one :class:`ServiceRegistry`, however far along its
    lifecycle -- may exist but not yet (or no longer) be initialized.

    Raises:
        HTTPException: 503, if no registry was constructed at all (e.g. the
            lifespan startup hook hasn't run, or itself failed and
            ``fail_fast_on_startup`` was disabled -- see
            ``backend.lifecycle``).
    """
    registry: Optional[ServiceRegistry] = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service registry is not available (application failed to start or is still starting up).",
        )
    return registry


def get_initialized_registry(request: Request) -> ServiceRegistry:
    """The process's :class:`ServiceRegistry`, guaranteed fully
    initialized (model loaded, runtime validated). Use this (rather than
    :func:`get_registry`) for any dependency that will call
    ``registry.prediction`` / ``registry.explainability`` / ``registry.model``
    / ``registry.runtime`` -- it turns "not ready yet" into one clean 503
    before any of those attributes (which are ``None`` until
    ``ServiceRegistry.initialize()`` has run) are ever touched.

    Raises:
        HTTPException: 503, if the registry isn't initialized yet.
    """
    registry = get_registry(request)
    if not registry.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service registry has not finished initializing yet.",
        )
    return registry
