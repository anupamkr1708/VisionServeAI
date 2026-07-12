"""
Runtime lifecycle service.

NEW orchestration -- no notebook precedent. Sprint 05 never had a pluggable
runtime lifecycle: Stage 4's ``InferenceEngine`` was constructed once,
directly, around a raw ``nn.Module``, with no separate
create/load/warmup/validate/reload/shutdown sequence and no choice of
execution backend. That pluggable lifecycle is exactly what
``inference/runtimes/`` (the Runtime Abstraction Layer, migrated in the
immediately prior phase) was built to enable; this service is its intended
consumer -- "a future InferenceEngine should consume this runtime without
changing its own public interface" (``inference/runtimes/base.py``'s own
docstring). ``inference/engine.py`` itself remains untouched (still directly
coupled to a raw model, per this phase's "do not modify existing modules"
rule); this service and :class:`services.prediction_service.PredictionService`
are the new, parallel consumer wired to the pluggable runtime instead.

Every runtime operation this service performs --
``RuntimeFactory.create(...)``, ``runtime.load()``, ``runtime.warmup(...)``,
``runtime.validate(...)`` -- calls the already-migrated, frozen
``inference.runtimes`` API directly. No runtime algorithm (provider
selection, numerical comparison, dynamic-batch verification, graph
fingerprinting) is reimplemented here.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from inference.runtimes import BaseRuntime, RuntimeFactory, RuntimeInfo
from services.model_service import ModelService


class RuntimeService:
    """Owns the lifecycle of exactly one active :class:`BaseRuntime`
    instance: creation (via :class:`RuntimeFactory`), loading, warmup,
    validation, and reload/shutdown. Depends on :class:`ModelService`
    (dependency injection) for the reconstructed model, its fingerprint,
    and its preprocessing config -- never reconstructs a model itself.
    """

    def __init__(
        self,
        model_service: ModelService,
        logger: logging.Logger,
        runtime_type: str = "pytorch",
        runtime_path: Optional[Path] = None,
        warmup_iterations: int = 10,
    ) -> None:
        """
        Args:
            model_service: Already-``load_model()``-ed :class:`ModelService`
                supplying the model/fingerprint/preprocessing config a
                runtime is built around.
            logger: Caller-supplied logger (dependency injection).
            runtime_type: One of ``RuntimeFactory.supported_runtimes()`` --
                ``"pytorch"`` (default), ``"torchscript"``, or ``"onnx"``.
            runtime_path: Required for ``"torchscript"``/``"onnx"`` -- the
                exported artifact path (e.g.
                ``ArtifactService.torchscript_path()`` /
                ``.onnx_path()``). Unused for ``"pytorch"``.
            warmup_iterations: Forwarded to ``BaseRuntime.warmup()``.
        """
        self.model_service = model_service
        self.logger = logger
        self.runtime_type = runtime_type
        self.runtime_path = runtime_path
        self.warmup_iterations = warmup_iterations

        self._runtime: Optional[BaseRuntime] = None
        self._last_validation: Optional[RuntimeInfo] = None

    def _build_runtime_kwargs(self) -> Dict[str, Any]:
        """Assemble constructor kwargs for the configured ``runtime_type``.
        Kwarg *selection* necessarily differs per backend (a
        ``PyTorchRuntime`` needs ``model=``, ``TorchScriptRuntime``/
        ``ONNXRuntime`` need ``model_path=``) -- actual runtime
        *construction* is still fully delegated to :class:`RuntimeFactory`,
        so this is not a duplicate dispatch chain, only kwarg assembly."""
        model_registry = self.model_service.model_registry
        common: Dict[str, Any] = {
            "device": self.model_service.device,
            "model_fingerprint_sha256": model_registry.checkpoint_sha256,
            "logger": self.logger,
        }
        if self.runtime_type == "pytorch":
            return {**common, "model": self.model_service.model}
        if self.runtime_type in ("torchscript", "onnx"):
            if self.runtime_path is None:
                raise RuntimeError(
                    f"runtime_type='{self.runtime_type}' requires 'runtime_path' to be set "
                    f"(e.g. from ArtifactService.torchscript_path() / .onnx_path())."
                )
            return {**common, "model_path": self.runtime_path}
        raise ValueError(
            f"Unknown runtime_type '{self.runtime_type}'. Supported: {RuntimeFactory.supported_runtimes()}"
        )

    def initialize_runtime(self, warmup: bool = True, validate: bool = True) -> BaseRuntime:
        """Create, load, optionally warm up, and optionally validate a
        runtime of the configured type.

        Validation here is executable-only (no cross-runtime numerical
        comparison against a reference) -- ``BaseRuntime.validate()``
        supports this out of the box by leaving ``reference_output=None``.
        Cross-runtime numerical comparison is available (see
        ``inference.runtimes.base.compare_tensors``) but is a power-user
        concern this service does not force onto every initialization; a
        caller wanting it can validate two ``RuntimeService`` instances'
        runtimes against each other directly.

        Raises:
            RuntimeError: if validation is requested and fails.
        """
        kwargs = self._build_runtime_kwargs()
        runtime = RuntimeFactory.create(self.runtime_type, **kwargs)
        runtime.load()
        self.logger.info("RUNTIME_SERVICE runtime_type=%s loaded.", self.runtime_type)

        cfg = self.model_service.preprocessing_config
        input_shape = (1, cfg.channels, cfg.resize_height, cfg.resize_width)

        if warmup:
            runtime.warmup(input_shape=input_shape, iterations=self.warmup_iterations)

        if validate:
            dummy_input = torch.randn(*input_shape, dtype=torch.float32)
            info = runtime.validate(dummy_input)
            self._last_validation = info
            if not info.validated:
                raise RuntimeError(
                    f"Runtime validation FAILED for runtime_type='{self.runtime_type}': {info.validation_errors}"
                )
            self.logger.info("RUNTIME_SERVICE runtime_type=%s validated OK.", self.runtime_type)

        self._runtime = runtime
        return runtime

    def reload_runtime(self, warmup: bool = True, validate: bool = True) -> BaseRuntime:
        """Shut down the current runtime (if any) and initialize a fresh
        one. Use after the underlying model/artifact has changed (e.g.
        :meth:`ModelService.load_model` was called again)."""
        self.shutdown_runtime()
        return self.initialize_runtime(warmup=warmup, validate=validate)

    def get_runtime(self) -> BaseRuntime:
        """Raises:
        RuntimeError: if :meth:`initialize_runtime` hasn't been called yet
            (or the runtime was shut down and not reinitialized).
        """
        if self._runtime is None:
            raise RuntimeError("RuntimeService.initialize_runtime() has not been called yet.")
        return self._runtime

    def shutdown_runtime(self) -> None:
        """Drop the active runtime reference. ``BaseRuntime`` has no
        explicit ``close()``/resource-release method in its public API
        (see ``inference/runtimes/base.py``) -- for the PyTorch/TorchScript/
        ONNX backends this repository currently supports, dropping the
        reference is sufficient for garbage collection to reclaim any
        associated memory (CUDA memory is reclaimed by PyTorch's caching
        allocator on collection, not necessarily returned to the OS
        immediately, matching normal PyTorch behaviour)."""
        if self._runtime is not None:
            self.logger.info("RUNTIME_SERVICE runtime_type=%s shut down.", self.runtime_type)
        self._runtime = None
        self._last_validation = None

    def runtime_status(self) -> Dict[str, Any]:
        """Summary used by :class:`services.health_service.HealthService`."""
        if self._runtime is None:
            return {"initialized": False, "runtime_type": self.runtime_type}
        info = self._runtime.metadata().to_dict()
        return {
            "initialized": True,
            "runtime_type": self.runtime_type,
            "loaded": self._runtime.is_loaded,
            "validated": self._runtime.is_validated,
            "selected_providers": self._runtime.selected_providers,
            "available_providers": self._runtime.available_providers,
            "metadata": info,
            "last_validation_errors": self._last_validation.validation_errors if self._last_validation else [],
        }
