"""
Explainability service.

Wraps an ``ExplainabilityEngine`` instance (Sprint 05 **Stage 6**,
"Explainability Runtime") supplied via dependency injection. Per this
phase's explicit instruction ("Wrap ExplainabilityEngine. Do NOT implement
GradCAM. It already exists."), this module implements no explainability
algorithm itself.

Important gap, stated plainly: Stage 6's actual GradCAM-family algorithms
(~630 lines covering ``ActivationsAndGradients``, ``discover_target_layer``,
and ``generate_gradcam`` / ``generate_gradcam_plus`` / ``generate_scorecam``
/ ``generate_eigencam`` / ``generate_guided_backprop`` /
``generate_integrated_gradients`` / ``generate_occlusion``) have **not**
been migrated into ``inference/explainability/`` yet -- that package
currently contains only an empty placeholder ``__init__.py`` from an
earlier repository-foundation phase. Migrating Stage 6's algorithm is a
distinct, substantial migration phase of its own and is out of scope here:
this phase is explicitly "Application Service Layer" / orchestration only,
and "Do NOT implement GradCAM" forecloses writing that algorithm here even
to fill the gap.

This service is therefore forward-wired: it defines the exact public
surface a migrated ``ExplainabilityEngine`` will be called through, via
constructor injection, so that a future migration can drop the real engine
in without touching this file or any of its callers
(``ServiceRegistry``/``PredictionService`` never need to know an engine is
missing). Every method delegates to the injected engine's identically-named
method (Stage 6, lines ~218-231's own documented public API:
``generate_gradcam, generate_gradcam_plus, generate_scorecam,
generate_eigencam, generate_integrated_gradients, generate_guided_backprop,
generate_occlusion, batch_generate, save_visualization, overlay_heatmap,
export_results``) and raises a clear, actionable error -- never a silent
no-op or fabricated result -- if no engine was injected.

Source: sprint05-deployment.ipynb, Stage 6, lines ~218-231 (public API this
wraps, referenced not reimplemented).
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Protocol


class SupportsExplainability(Protocol):
    """Structural type for the Stage 6 ``ExplainabilityEngine`` public API
    this service depends on. Not the real implementation -- a typing
    contract only, so this file has something concrete to type-check
    against ahead of that engine's own migration. Method signatures
    intentionally permissive (``*args, **kwargs``) since Stage 6's actual
    per-method signatures aren't being asserted here without the source
    module to import them from."""

    def generate_gradcam(self, *args: Any, **kwargs: Any) -> Any: ...
    def generate_gradcam_plus(self, *args: Any, **kwargs: Any) -> Any: ...
    def generate_scorecam(self, *args: Any, **kwargs: Any) -> Any: ...
    def generate_eigencam(self, *args: Any, **kwargs: Any) -> Any: ...
    def generate_guided_backprop(self, *args: Any, **kwargs: Any) -> Any: ...
    def generate_integrated_gradients(self, *args: Any, **kwargs: Any) -> Any: ...
    def generate_occlusion(self, *args: Any, **kwargs: Any) -> Any: ...


class ExplainabilityService:
    """Thin delegation layer over an injected ``ExplainabilityEngine``.
    Owns no explainability logic of its own -- see module docstring for why
    the engine itself is not yet available to inject in this repository.
    """

    #: Methods Stage 6's ExplainabilityEngine exposes, this service wraps
    #: 1:1. Kept as a named constant (rather than repeated per-method) so
    #: :meth:`available_methods` and :meth:`_require_engine`'s error
    #: message stay in sync with the delegating methods below by
    #: construction.
    SUPPORTED_METHODS: List[str] = [
        "gradcam", "gradcam_plus", "scorecam", "eigencam",
        "guided_backprop", "integrated_gradients", "occlusion",
    ]

    def __init__(self, logger: logging.Logger, engine: Optional[SupportsExplainability] = None) -> None:
        """
        Args:
            logger: Caller-supplied logger (dependency injection).
            engine: A constructed Stage-6-equivalent ``ExplainabilityEngine``
                instance, once migrated. ``None`` (the default) is fully
                valid construction -- :class:`ServiceRegistry` can wire this
                service up before an engine exists; calling any
                ``generate_*`` method before one is injected raises
                ``RuntimeError`` rather than silently doing nothing.
        """
        self.logger = logger
        self._engine = engine

    def set_engine(self, engine: SupportsExplainability) -> None:
        """Inject (or replace) the wrapped engine after construction --
        e.g. once Stage 6 is migrated and a real engine can be built."""
        self._engine = engine
        self.logger.info("EXPLAINABILITY_SERVICE engine injected: %s", type(engine).__name__)

    def _require_engine(self) -> SupportsExplainability:
        if self._engine is None:
            raise RuntimeError(
                "ExplainabilityService has no ExplainabilityEngine injected. Stage 6's GradCAM-family "
                "algorithms are not yet migrated into inference/explainability/ (see module docstring) -- "
                "inject a compatible engine via the constructor or set_engine() once it exists."
            )
        return self._engine

    def available_methods(self) -> List[str]:
        """Explainability method names this service's public API covers --
        does not indicate whether an engine is currently injected."""
        return list(self.SUPPORTED_METHODS)

    def is_available(self) -> bool:
        """Whether an engine has been injected and methods can actually be
        called right now."""
        return self._engine is not None

    # ------------------------------------------------------------------
    # Delegating public API (1:1 with Stage 6's ExplainabilityEngine)
    # ------------------------------------------------------------------

    def generate_gradcam(self, *args: Any, **kwargs: Any) -> Any:
        return self._require_engine().generate_gradcam(*args, **kwargs)

    def generate_gradcam_plus(self, *args: Any, **kwargs: Any) -> Any:
        return self._require_engine().generate_gradcam_plus(*args, **kwargs)

    def generate_scorecam(self, *args: Any, **kwargs: Any) -> Any:
        return self._require_engine().generate_scorecam(*args, **kwargs)

    def generate_eigencam(self, *args: Any, **kwargs: Any) -> Any:
        return self._require_engine().generate_eigencam(*args, **kwargs)

    def generate_guided_backprop(self, *args: Any, **kwargs: Any) -> Any:
        return self._require_engine().generate_guided_backprop(*args, **kwargs)

    def generate_integrated_gradients(self, *args: Any, **kwargs: Any) -> Any:
        return self._require_engine().generate_integrated_gradients(*args, **kwargs)

    def generate_occlusion(self, *args: Any, **kwargs: Any) -> Any:
        return self._require_engine().generate_occlusion(*args, **kwargs)
