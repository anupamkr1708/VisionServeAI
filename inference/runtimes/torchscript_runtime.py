"""
TorchScript runtime backend.

Loads a TorchScript module (``model.ts``, produced by Stage 3's export
pipeline -- not yet migrated into this repository, so this runtime is
handed a path to an already-exported artifact, never an ``nn.Module`` it
would export itself) and serves it behind the common
:class:`~inference.runtimes.base.BaseRuntime` interface.

Preserves the following Stage 3 / Stage 5 engineering validations exactly:
  - ``torchscript_readable`` (Stage 3 ``run_stage03_engineering_validation``,
    lines ~508-513): can the saved module be ``torch.jit.load``-ed at all.
    Reproduced here as :meth:`load` raising ``RuntimeError`` on failure --
    fail fast, matching "fail loudly" -- rather than deferring readability
    to a later separate check.
  - ``torchscript_executable`` (Stage 3, lines ~515-517): can the loaded
    module actually run a forward pass. Reproduced in :meth:`validate`.
  - PyTorch-vs-TorchScript numerical equivalence, including the exact
    tolerance thresholds and error-message text (Stage 3
    ``run_numerical_validation``, lines ~388-401, 420-421; duplicated
    again in Stage 5's ``run_numerical_stability``, lines ~562-567,
    613-623). Reproduced in :meth:`validate` via
    ``inference.runtimes.base.compare_tensors`` /
    ``tolerance_exceeded``.
  - The TorchScript export's own graph fingerprint (Stage 3
    ``TorchScriptExportResult.graph_hash``, lines ~272:
    ``hashlib.sha256(str(scripted_module.graph).encode()).hexdigest()``).
    Recomputed identically here, at *load* time rather than export time,
    as ``runtime_fingerprint_sha256`` -- same algorithm, same value for the
    same graph, since ``torch.jit.load`` reconstructs the identical graph
    ``torch.jit.script``/``torch.jit.trace`` produced at export.

Source: sprint05-deployment.ipynb, Stage 3 lines ~230-280, 388-401,
420-421, 508-517; Stage 5 lines ~550-567, 613-623.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import torch

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


class TorchScriptRuntime(BaseRuntime):
    """Serves a ``torch.jit``-exported (``.ts``) module."""

    RUNTIME_TYPE = "torchscript"

    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        model_fingerprint_sha256: str,
        logger: logging.Logger,
        export_method: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_path: Path to the exported ``model.ts`` file (Stage 3's
                ``TorchScriptExportResult.output_path``). Never a
                checkpoint path -- this runtime does not export, it only
                loads an already-exported artifact.
            device: Target device passed to ``torch.jit.load(...,
                map_location=device)``.
            model_fingerprint_sha256: ``ModelRegistry.checkpoint_sha256``
                of the checkpoint this export was produced from, reused
                (not recomputed) for cross-runtime identity.
            logger: Caller-supplied logger (dependency injection).
            export_method: Optional ``"script"`` / ``"trace"`` hint, taken
                from Stage 3's ``TorchScriptExportResult.export_method`` if
                available. Purely informational (surfaced via
                :meth:`metadata`) -- does not change loading or prediction
                behavior, since a loaded TorchScript module runs identically
                regardless of which method produced it.
        """
        super().__init__(device=device, model_fingerprint_sha256=model_fingerprint_sha256, logger=logger)
        self.model_path = Path(model_path)
        self.export_method = export_method
        self._model: Optional[torch.jit.ScriptModule] = None
        self._graph_hash: Optional[str] = None

    def load(self) -> None:
        """Load the TorchScript module. Reproduces Stage 3's
        ``torchscript_readable`` check by construction: if
        ``torch.jit.load`` fails, that failure is surfaced immediately as a
        ``RuntimeError`` (fail fast) rather than being recorded as a
        deferred validation flag, per this phase's "fail loudly" rule.

        Raises:
            RuntimeError: if the module cannot be loaded from
                ``model_path``.
        """
        try:
            self._model = torch.jit.load(str(self.model_path), map_location=self.device)
            self._model.eval()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Saved TorchScript module is not readable: {exc}") from exc

        # Stage 3's TorchScriptExportResult.graph_hash, recomputed at load
        # time from the identical graph structure (see module docstring).
        self._graph_hash = hashlib.sha256(str(self._model.graph).encode()).hexdigest()

        self._loaded = True
        self.logger.info(
            "RUNTIME[torchscript] loaded path=%s device=%s graph_hash=%s",
            self.model_path, self.device, self._graph_hash,
        )

    def predict(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass, reproducing Stage 3/5's exact
        ``with torch.no_grad(): ts_model(batch)`` pattern.

        Raises:
            RuntimeError: if called before :meth:`load`.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("TorchScriptRuntime.predict() called before load().")
        batch = batch.to(self.device)
        with torch.no_grad():
            return self._model(batch)

    def validate(
        self,
        dummy_input: torch.Tensor,
        reference_output: Optional[torch.Tensor] = None,
    ) -> RuntimeInfo:
        """Reproduces Stage 3's ``torchscript_executable`` check plus (when
        ``reference_output`` -- typically the PyTorch runtime's output on
        the same ``dummy_input`` -- is supplied) the PyTorch-vs-TorchScript
        numerical-equivalence check, with Stage 3's exact tolerances and
        error-message wording (``"PyTorch vs TorchScript numerical
        tolerance exceeded: {...}"``, Stage 3 line ~421).

        Never raises on a failed check -- failures are reported in the
        returned :class:`RuntimeInfo`, matching Stage 3's
        aggregate-then-report validation pattern.
        """
        errors = []
        executable = False
        output: Optional[torch.Tensor] = None
        try:
            output = self.predict(dummy_input)
            executable = True
        except Exception as exc:  # noqa: BLE001
            errors.append(f"TorchScript module failed to run inference during numerical validation: {exc}")

        if executable and reference_output is not None:
            comparison = compare_tensors(reference_output, output)
            if tolerance_exceeded(comparison, DEFAULT_MAX_ABS_ERROR_TOLERANCE, DEFAULT_MAX_RELATIVE_ERROR_TOLERANCE):
                errors.append(f"PyTorch vs TorchScript numerical tolerance exceeded: {comparison}")

        self._validated = executable and len(errors) == 0
        for e in errors:
            self.logger.error("RUNTIME[torchscript] VALIDATION FAILURE: %s", e)

        return RuntimeInfo(
            metadata=self.metadata(),
            loaded=self._loaded,
            validated=self._validated,
            selected_providers=self.selected_providers,
            available_providers=self.available_providers,
            validation_errors=errors,
        )

    def metadata(self) -> RuntimeMetadata:
        return RuntimeMetadata(
            runtime_type=self.RUNTIME_TYPE,
            runtime_version=torch.__version__,
            source_path=str(self.model_path),
            device=str(self.device),
            model_fingerprint_sha256=self.model_fingerprint_sha256,
            runtime_fingerprint_sha256=self._graph_hash,
            capabilities=RuntimeCapabilities(
                supports_cpu=True,
                supports_gpu=True,
                # Dynamic batch support for a TorchScript module depends on
                # whether it was scripted (always dynamic-batch-safe) or
                # traced (safe for the CNN classifiers this repository
                # deploys, whose ops are all per-sample-independent, but
                # not guaranteed in general). Reported True either way, as
                # Stage 3's own DYNAMIC_BATCH_TEST_SIZES verification was
                # only ever run against the ONNX export -- TorchScript's
                # dynamic-batch behavior was never independently smoke-
                # tested in the original notebook, so this capability flag
                # is declarative, not a guarantee backed by a Stage-3-style
                # per-batch-size check the way ONNXRuntime's is.
                supports_dynamic_batch=True,
                supports_amp=False,
            ),
        )
