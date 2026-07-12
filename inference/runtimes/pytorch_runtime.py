"""
PyTorch runtime backend.

Wraps an ALREADY-reconstructed, already-validated ``nn.Module`` -- the
output of ``inference.model_loader.reconstruct_model()`` -- behind the
common :class:`~inference.runtimes.base.BaseRuntime` interface. This is
dependency injection, not a reimplementation, of ``model_loader``: per this
phase's explicit instruction ("Reuse existing reconstructed model. Do not
duplicate ModelLoader."), this class never builds an architecture, loads a
checkpoint, or runs any of Stage 2's validation itself -- it takes the
already-reconstructed model as a constructor argument and exposes it behind
``load()`` / ``predict()`` / ``validate()`` / ``metadata()`` / ``warmup()``
so a future ``InferenceEngine`` can depend on the same interface for
PyTorch, TorchScript, and ONNX uniformly.

Source of the behavior being wrapped (not duplicated, only referenced):
  - Stage 2's ``reconstruct_model()`` eval()/no_grad() contract
    (``inference/model_loader.py``, migrated verbatim in a prior phase).
  - Stage 4's inline forward-pass pattern,
    ``with torch.no_grad(): logits = self.model(batch_tensor)``
    (``inference/engine.py``, ``InferenceEngine.predict_batch``, migrated
    verbatim in a prior phase) -- reproduced exactly in :meth:`predict`.

``autocast`` support (``use_amp``) is NEW in this phase -- Stage 1's
``RuntimeConfig.use_amp`` field existed but was explicitly commented
"reserved -- no inference happens here" and was never consumed by any
notebook stage. Consuming it here is additive: it defaults to ``False``
(unchanged numerical behavior, matching the original Stage 4 inference path
exactly) and only activates mixed precision if a caller explicitly opts in
*and* the runtime is on a CUDA device.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from inference.runtimes.base import (
    BaseRuntime,
    DEFAULT_MAX_ABS_ERROR_TOLERANCE,
    DEFAULT_MAX_RELATIVE_ERROR_TOLERANCE,
    RuntimeCapabilities,
    RuntimeInfo,
    RuntimeMetadata,
    compare_tensors,
    tolerance_exceeded,
)


class PyTorchRuntime(BaseRuntime):
    """Serves an already-reconstructed model directly via eager PyTorch."""

    RUNTIME_TYPE = "pytorch"

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        model_fingerprint_sha256: str,
        logger: logging.Logger,
        use_amp: bool = False,
    ) -> None:
        """
        Args:
            model: Already-reconstructed, already-checkpoint-validated
                model -- the ``model`` half of
                ``inference.model_loader.reconstruct_model()``'s return
                value. This runtime does not construct, checkpoint-load,
                or validate it.
            device: Target device. The caller is expected to have already
                placed ``model`` on this device (matching
                ``reconstruct_model``'s own ``model.to(device)`` contract);
                :meth:`load` re-asserts eval()/grad-disabled state
                defensively but does not move the model.
            model_fingerprint_sha256: ``ModelRegistry.checkpoint_sha256``,
                reused (not recomputed) for identity across all three
                runtime backends serving the same checkpoint.
            logger: Caller-supplied logger (dependency injection, matching
                every other module in this package -- this class never
                builds its own logger via ``inference.utils.logging``).
            use_amp: If ``True`` *and* ``device.type == "cuda"``,
                :meth:`predict` wraps the forward pass in
                ``torch.autocast(device_type="cuda")``. Defaults to
                ``False``, reproducing Stage 4's exact (non-autocast)
                numerical behavior.
        """
        super().__init__(device=device, model_fingerprint_sha256=model_fingerprint_sha256, logger=logger)
        self.model = model
        self.use_amp = use_amp

    def load(self) -> None:
        """No-op load in the disk-I/O sense: the model was already
        constructed and checkpoint-validated before being handed to this
        runtime, so there is nothing left to read from disk. Still
        defensively re-asserts eval() mode and grad-disabled state (the
        exact contract ``reconstruct_model()`` already guarantees --
        belt-and-braces, not a behavior change) and flips ``self._loaded``,
        matching every other runtime's ``load()`` contract so callers can
        treat all three runtimes identically."""
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._loaded = True
        self.logger.info("RUNTIME[pytorch] ready (already-reconstructed model, device=%s).", self.device)

    def predict(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass. Reproduces
        ``InferenceEngine.predict_batch``'s exact
        ``with torch.no_grad(): logits = self.model(batch_tensor)`` when
        ``use_amp=False`` (the default) -- identical numerical behavior.

        Raises:
            RuntimeError: if called before :meth:`load`.
        """
        if not self._loaded:
            raise RuntimeError("PyTorchRuntime.predict() called before load().")
        batch = batch.to(self.device)
        if self.use_amp and self.device.type == "cuda":
            with torch.no_grad(), torch.autocast(device_type="cuda"):
                return self.model(batch)
        with torch.no_grad():
            return self.model(batch)

    def validate(
        self,
        dummy_input: torch.Tensor,
        reference_output: Optional[torch.Tensor] = None,
    ) -> RuntimeInfo:
        """Confirm the model executes, and -- if ``reference_output`` is
        given -- that this runtime's own output matches it within
        tolerance. In normal use PyTorch *is* the reference other runtimes
        are checked against (see ``TorchScriptRuntime.validate`` /
        ``ONNXRuntime.validate``), so ``reference_output`` is typically
        left ``None`` here; the parameter exists for interface symmetry
        with the other two runtimes rather than because Stage 3 ever
        compared PyTorch against anything else."""
        errors = []
        executable = False
        output: Optional[torch.Tensor] = None
        try:
            output = self.predict(dummy_input)
            executable = True
        except Exception as exc:  # noqa: BLE001
            errors.append(f"PyTorch model failed to run inference during validation: {exc}")

        if executable and reference_output is not None:
            comparison = compare_tensors(reference_output, output)
            if tolerance_exceeded(comparison, DEFAULT_MAX_ABS_ERROR_TOLERANCE, DEFAULT_MAX_RELATIVE_ERROR_TOLERANCE):
                errors.append(f"Reference vs PyTorch numerical tolerance exceeded: {comparison}")

        self._validated = executable and len(errors) == 0
        for e in errors:
            self.logger.error("RUNTIME[pytorch] VALIDATION FAILURE: %s", e)

        return RuntimeInfo(
            metadata=self.metadata(),
            loaded=self._loaded,
            validated=self._validated,
            selected_providers=self.selected_providers,
            available_providers=self.available_providers,
            validation_errors=errors,
        )

    def metadata(self) -> RuntimeMetadata:
        """PyTorch has no separate exported artifact to fingerprint --
        ``runtime_fingerprint_sha256`` is therefore identical to
        ``model_fingerprint_sha256`` (the checkpoint hash), unlike
        TorchScript/ONNX which fingerprint their own exported file/graph."""
        return RuntimeMetadata(
            runtime_type=self.RUNTIME_TYPE,
            runtime_version=torch.__version__,
            source_path=None,
            device=str(self.device),
            model_fingerprint_sha256=self.model_fingerprint_sha256,
            runtime_fingerprint_sha256=self.model_fingerprint_sha256,
            capabilities=RuntimeCapabilities(
                supports_cpu=True,
                supports_gpu=True,
                supports_dynamic_batch=True,
                supports_amp=True,
            ),
        )
