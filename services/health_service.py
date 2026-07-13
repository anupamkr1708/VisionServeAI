"""
Health service.

NEW orchestration -- no single notebook stage exposed a unified health
check the way a long-running service needs one (Sprint 05 was a linear
notebook: each stage ran once and either succeeded or raised). What it does
have, reused here rather than reimplemented, are the per-resource
measurement functions consolidated into
``inference.utils.resource_monitor`` in an earlier phase
(``get_resource_usage`` / ``get_device_resource_usage``, themselves
verbatim ports of Stage 1/5's own resource-logging code) and the
Runtime Abstraction Layer's own provider-reporting properties
(``BaseRuntime.selected_providers`` / ``.available_providers``, Runtime
Abstraction Layer phase). This service's only job is composing those
already-validated primitives, plus :class:`services.model_service.ModelService`
/ :class:`services.runtime_service.RuntimeService` /
:class:`services.artifact_service.ArtifactService` state, into one
process-health snapshot. Pure Python -- no HTTP status codes, no FastAPI
response models; translating this into an HTTP health endpoint is a future
``backend/`` phase's job.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

from inference.utils.environment import get_environment_info
from inference.utils.resource_monitor import get_device_resource_usage, get_resource_usage
from services.artifact_service import ArtifactService
from services.model_service import ModelService
from services.runtime_service import RuntimeService


class HealthService:
    """Read-only system health reporting. Depends on
    :class:`ModelService`, :class:`RuntimeService`, and
    :class:`ArtifactService` (dependency injection) -- performs no loading,
    reconstruction, or prediction of its own."""

    def __init__(
        self,
        model_service: ModelService,
        runtime_service: RuntimeService,
        artifact_service: ArtifactService,
        logger: logging.Logger,
    ) -> None:
        self.model_service = model_service
        self.runtime_service = runtime_service
        self.artifact_service = artifact_service
        self.logger = logger

    def runtime_status(self) -> Dict[str, Any]:
        """Delegates to :meth:`RuntimeService.runtime_status` -- kept as
        its own method here (rather than only inline inside :meth:`health`)
        so a caller can check runtime health in isolation."""
        return self.runtime_service.runtime_status()

    def gpu_status(self) -> Dict[str, Any]:
        """GPU availability and (if a runtime is initialized on a CUDA
        device) current memory usage, via
        :func:`inference.utils.resource_monitor.get_device_resource_usage`."""
        cuda_available = torch.cuda.is_available()
        status: Dict[str, Any] = {
            "cuda_available": cuda_available,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
        }
        device: Optional[torch.device] = getattr(self.model_service, "device", None)
        if device is not None and device.type == "cuda":
            status["device"] = str(device)
            status.update(get_device_resource_usage(device))
        return status

    def memory_status(self) -> Dict[str, Any]:
        """System CPU/RAM/disk snapshot, via
        :func:`inference.utils.resource_monitor.get_resource_usage`."""
        return get_resource_usage(include_disk=True)

    def environment_info(self) -> Dict[str, Any]:
        """Static environment/hardware report -- Python/Torch/torchvision/
        CUDA/cuDNN versions, OS, CPU/RAM totals, and runtime-format
        availability (ONNX/TorchScript). Via
        :func:`inference.utils.environment.get_environment_info`, a Stage 1
        function that had not been reported anywhere in this repository
        before this cleanup pass -- unlike :meth:`gpu_status` /
        :meth:`memory_status` (which are point-in-time *usage* snapshots),
        this is static identity information about the process's
        environment, useful for support/debugging without needing a
        separate ``/version`` endpoint once ``backend/`` lands."""
        return get_environment_info()

    def provider_status(self) -> Dict[str, Any]:
        """Execution-provider report for the active runtime (meaningful for
        the ONNX backend; empty lists for PyTorch/TorchScript, which have
        no provider concept -- see
        ``inference.runtimes.base.BaseRuntime.selected_providers`` /
        ``.available_providers``)."""
        try:
            runtime = self.runtime_service.get_runtime()
        except RuntimeError:
            return {"initialized": False, "selected_providers": [], "available_providers": []}
        return {
            "initialized": True,
            "selected_providers": runtime.selected_providers,
            "available_providers": runtime.available_providers,
        }

    def artifact_status(self) -> Dict[str, Any]:
        """Delegates to :meth:`ArtifactService.artifact_status`."""
        return self.artifact_service.artifact_status()

    def health(self) -> Dict[str, Any]:
        """Aggregate health snapshot combining every status method above
        into one overall ``status`` -- ``"healthy"`` if the model is
        loaded, the runtime is initialized and validated, and no critical
        artifacts are missing; ``"degraded"`` if the runtime is initialized
        but not validated, or model/runtime aren't initialized yet;
        ``"unhealthy"`` if critical artifacts are missing."""
        artifact = self.artifact_status()
        runtime = self.runtime_status()

        model_loaded = self.model_service.is_loaded

        if not artifact.get("healthy", False):
            overall = "unhealthy"
        elif not model_loaded or not runtime.get("initialized", False):
            overall = "degraded"
        elif not runtime.get("validated", False):
            overall = "degraded"
        else:
            overall = "healthy"

        return {
            "status": overall,
            "model_loaded": model_loaded,
            "runtime": runtime,
            "gpu": self.gpu_status(),
            "memory": self.memory_status(),
            "providers": self.provider_status(),
            "artifacts": artifact,
            "environment": self.environment_info(),
        }
