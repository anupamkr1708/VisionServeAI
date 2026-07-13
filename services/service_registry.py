"""
Service registry: dependency container wiring every service in this
package together.

NEW orchestration -- no notebook precedent (the notebook was a linear
sequence of stage functions, each consuming the previous stage's in-memory
globals directly; there was no dependency-injection container). This class
exists purely to assemble the six services above in the correct dependency
order and hand callers one object to reach all of them through, per this
phase's explicit "no singleton globals -- use dependency injection"
requirement: nothing here is a module-level global, every service receives
its dependencies through its own constructor, and a second
``ServiceRegistry()`` in the same process is fully independent of the
first (e.g. for tests, or serving two model versions side by side).

Dependency order (see class docstring for the full graph):

    ArtifactService
        -> ModelService
            -> RuntimeService
                -> PredictionService
                -> HealthService
            -> HealthService
            -> ExplainabilityEngine (inference.explainability.engine) -- wrapped by
                ExplainabilityService, see :meth:`ServiceRegistry.initialize`
        -> HealthService

Now that Stage 6's ``ExplainabilityEngine`` has been migrated
(``inference.explainability.engine``), ``ExplainabilityService`` is no
longer independent of the rest of the graph by necessity -- only by
default injection point. :meth:`initialize` auto-builds a real
``ExplainabilityEngine`` from the already-constructed ``ModelService``
unless an explicit ``explainability_engine`` was passed to the
constructor, or ``enable_explainability=False``.
``ExplainabilityService`` itself (``services.explainability_service``) is
untouched by this: it is still handed an engine through its existing
``engine=`` constructor parameter / ``set_engine()``, exactly as before --
only *what* gets handed to it, by default, has changed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from inference.explainability.engine import ExplainabilityEngine
from inference.utils.environment import set_seed
from inference.utils.logging import build_logger
from services.artifact_service import ArtifactService
from services.explainability_service import ExplainabilityService, SupportsExplainability
from services.health_service import HealthService
from services.model_service import ModelService
from services.prediction_service import PredictionService
from services.runtime_service import RuntimeService


class ServiceRegistry:
    """Dependency container for every service in this package.

    Usage::

        registry = ServiceRegistry(
            artifact_roots={"sprint03": "...", "sprint04_training": "...", ...},
            export_dir=Path("deployment/export"),
        )
        registry.initialize()

        prediction = registry.prediction.predict(image)
        registry.health.health()
        registry.runtime.reload_runtime()
        registry.explainability.generate_gradcam(...)

    Dependency graph (constructed in this order by :meth:`initialize`)::

        ArtifactService (artifact_roots, export_dir, device)
            |
            v
        ModelService (artifact_service, device)  -- load_model() called here
            |
            v
        RuntimeService (model_service)  -- initialize_runtime() called here
            |
            +--> PredictionService (model_service, runtime_service)
            |
            +--> HealthService (model_service, runtime_service, artifact_service)

        ExplainabilityService (logger, engine=explainability_engine)  -- independent,
            wired last; see services.explainability_service for why its
            engine is optional/injectable rather than constructed here.
    """

    def __init__(
        self,
        artifact_roots: Dict[str, Optional[str]],
        export_dir: Path,
        device: Optional[torch.device] = None,
        runtime_type: str = "pytorch",
        runtime_path: Optional[Path] = None,
        warmup_iterations: int = 10,
        explainability_engine: Optional[SupportsExplainability] = None,
        logger: Optional[logging.Logger] = None,
        log_dir: Optional[Path] = None,
        enable_explainability: bool = True,
        explainability_output_dir: Optional[Path] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            artifact_roots: category -> resolved directory (or ``None``),
                forwarded to :class:`ArtifactService`.
            export_dir: TorchScript/ONNX export directory, forwarded to
                :class:`ArtifactService`.
            device: Target device. Defaults to CUDA if available, else CPU.
            runtime_type: Forwarded to :class:`RuntimeService` -- ``
                "pytorch"`` (default), ``"torchscript"``, or ``"onnx"``.
            runtime_path: Forwarded to :class:`RuntimeService` -- required
                for ``"torchscript"``/``"onnx"``.
            warmup_iterations: Forwarded to :class:`RuntimeService`.
            explainability_engine: Optional pre-built engine to inject into
                :class:`ExplainabilityService` directly. Now that Stage 6's
                ``ExplainabilityEngine`` has been migrated
                (``inference.explainability.engine.ExplainabilityEngine``),
                leaving this ``None`` (the default) no longer means "no
                engine" -- :meth:`initialize` builds one automatically from
                the already-constructed :class:`ModelService` (see
                ``explainability_output_dir`` below). Pass an explicit
                engine here only to override that default (e.g. a test
                double, or a differently-configured engine instance).
            logger: Optional pre-built logger, shared by every service.
                Built via :func:`inference.utils.logging.build_logger` if
                not given.
            log_dir: Directory for the auto-built logger's log file, used
                only when ``logger`` is not given. Defaults to ``./logs``.
            enable_explainability: If ``False``, skip auto-constructing an
                ``ExplainabilityEngine`` even when ``explainability_engine``
                is ``None`` -- ``registry.explainability`` is still built,
                just with no engine wrapped (matching this constructor's
                pre-migration default behaviour), for callers who want
                prediction/health only, without the memory/compute cost of
                an explainability engine (e.g. lightweight test wiring).
                Ignored if ``explainability_engine`` is explicitly given.
            explainability_output_dir: Root directory the auto-built
                ``ExplainabilityEngine`` writes heatmaps/overlays/metadata
                under. Defaults to ``artifacts/explainability`` (relative
                to the current working directory) if not given. Unused if
                an explicit ``explainability_engine`` is supplied, or if
                ``enable_explainability=False``.
            seed: Passed to :func:`inference.utils.environment.set_seed`
                during :meth:`initialize`, matching Stage 1's own
                ``set_seed(SEED)`` call at pipeline startup (notebook line
                ~592) -- a gap this cleanup phase closes (see
                ``inference/utils/environment.py``'s module docstring for
                why this matters even for deterministic eval-mode
                inference). Defaults to ``configs.defaults.SEED`` (``42``,
                the notebook's own constant) when not given.
        """
        self.artifact_roots = artifact_roots
        self.export_dir = Path(export_dir)
        self.device = device
        self.runtime_type = runtime_type
        self.runtime_path = runtime_path
        self.warmup_iterations = warmup_iterations
        self.explainability_engine = explainability_engine
        self.logger = logger
        self.log_dir = Path(log_dir) if log_dir is not None else Path("logs")
        self.enable_explainability = enable_explainability
        self.explainability_output_dir = (
            Path(explainability_output_dir) if explainability_output_dir is not None
            else Path("artifacts") / "explainability"
        )
        self.seed = seed

        self.artifact: Optional[ArtifactService] = None
        self.model: Optional[ModelService] = None
        self.runtime: Optional[RuntimeService] = None
        self.prediction: Optional[PredictionService] = None
        self.explainability: Optional[ExplainabilityService] = None
        self.health: Optional[HealthService] = None

        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(self, warmup: bool = True, validate_runtime: bool = True) -> "ServiceRegistry":
        """Build every service in dependency order: artifact discovery ->
        model reconstruction -> runtime initialization -> prediction/health
        wiring -> explainability wiring. Returns ``self`` so it can be
        chained with the constructor (``ServiceRegistry(...).initialize()``).

        Raises:
            RuntimeError: propagated from
                :meth:`ModelService.load_model` or
                :meth:`RuntimeService.initialize_runtime` if either fails --
                this method does not partially succeed silently.
        """
        if self.logger is None:
            self.logger = build_logger(name="visionserve.services", log_dir=self.log_dir)

        set_seed(self.seed) if self.seed is not None else set_seed()
        self.logger.info("SERVICE_REGISTRY seed set (deterministic=True), matching Stage 1 startup behavior.")

        device = self.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.artifact = ArtifactService(
            artifact_roots=self.artifact_roots, export_dir=self.export_dir, device=device, logger=self.logger,
        )
        self.model = ModelService(artifact_service=self.artifact, device=device, logger=self.logger)
        self.model.load_model()

        self.runtime = RuntimeService(
            model_service=self.model, logger=self.logger, runtime_type=self.runtime_type,
            runtime_path=self.runtime_path, warmup_iterations=self.warmup_iterations,
        )
        self.runtime.initialize_runtime(warmup=warmup, validate=validate_runtime)

        self.prediction = PredictionService(model_service=self.model, runtime_service=self.runtime, logger=self.logger)
        self.health = HealthService(
            model_service=self.model, runtime_service=self.runtime, artifact_service=self.artifact, logger=self.logger,
        )

        engine = self.explainability_engine
        if engine is None and self.enable_explainability:
            engine = self._build_default_explainability_engine()
        self.explainability = ExplainabilityService(logger=self.logger, engine=engine)

        self._initialized = True
        self.logger.info("SERVICE_REGISTRY initialized: runtime_type=%s device=%s", self.runtime_type, device)
        return self

    def _build_default_explainability_engine(self) -> ExplainabilityEngine:
        """Construct the real ``ExplainabilityEngine`` from the
        already-loaded :class:`ModelService`, using its raw ``nn.Module``
        directly rather than ``self.runtime``/``RuntimeService`` -- see
        ``ExplainabilityEngine``'s own class docstring for why (gradient-
        based explainability needs an autograd-enabled model with
        hookable layers; the Runtime Abstraction Layer's ``PyTorchRuntime``
        deliberately disables both for serving)."""
        model_registry = self.model.model_registry
        return ExplainabilityEngine(
            model=self.model.model,
            device=self.device,
            class_names=self.model.class_names(),
            preprocessing_config=self.model.preprocessing_config,
            output_dir=self.explainability_output_dir,
            logger=self.logger,
            backbone=model_registry.backbone,
            model_version=self.model.model_version,
            model_fingerprint=model_registry.checkpoint_sha256,
        )

    def shutdown(self) -> None:
        """Shut down the active runtime and mark this registry
        uninitialized. Services remain constructed (so
        :meth:`is_initialized` callers get a clear ``False`` rather than
        the attributes disappearing) but the runtime is released."""
        if self.runtime is not None:
            self.runtime.shutdown_runtime()
        self._initialized = False

    def status(self) -> Dict[str, Any]:
        """Convenience summary -- equivalent to ``self.health.health()``
        but safe to call before :meth:`initialize` (returns
        ``{"initialized": False}`` instead of raising)."""
        if not self._initialized or self.health is None:
            return {"initialized": False}
        return {"initialized": True, **self.health.health()}
