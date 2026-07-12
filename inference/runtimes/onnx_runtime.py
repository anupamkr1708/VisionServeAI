"""
ONNX runtime backend.

Loads an ONNX model (``model.onnx``, produced by Stage 3's export pipeline
-- not yet migrated into this repository, so this runtime is handed a path
to an already-exported artifact, never an ``nn.Module`` it would export
itself) and serves it via ``onnxruntime.InferenceSession`` behind the
common :class:`~inference.runtimes.base.BaseRuntime` interface.

Preserves the following Stage 3 / Stage 5 engineering validations exactly:
  - ``onnx_readable`` / session-load success (Stage 3
    ``run_stage03_engineering_validation``, lines ~523-528; Stage 5
    ``run_numerical_stability``, lines ~569-580).
  - Explicit, non-silent execution-provider selection (Stage 5
    ``run_numerical_stability``, lines ~569-575): CUDAExecutionProvider is
    only requested if the device is CUDA *and* it is actually present in
    ``onnxruntime.get_available_providers()`` -- never assumed, never
    silently retried into on failure. This phase's architectural rule ("If
    CUDAExecutionProvider is unavailable, report it explicitly. Do NOT
    silently switch providers.") is satisfied by adding an explicit
    ``logger.warning`` when a CUDA device was requested but
    CUDAExecutionProvider isn't available -- the original notebook
    silently computed the same CPU-only provider list without logging the
    fallback; this is a additive logging strengthening, not a functional
    behavior change (the selected provider list is identical either way).
  - PyTorch-vs-ONNX numerical equivalence, exact tolerances and
    error-message text (Stage 3 ``run_numerical_validation``, lines
    ~402-423).
  - Dynamic batch verification (Stage 3 ``verify_dynamic_batch_support``,
    lines ~444-461), reproduced verbatim as :meth:`verify_dynamic_batch`
    and folded into :meth:`validate`.
  - Provider discovery for compatibility reporting (Stage 7
    ``onnxruntime_providers()``, lines ~492-497).

Source: sprint05-deployment.ipynb, Stage 3 lines ~283-337, 402-461,
519-532; Stage 5 lines ~569-580, 625-636; Stage 7 lines ~492-497.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import torch

from inference.runtimes.base import (
    BaseRuntime,
    DEFAULT_DYNAMIC_BATCH_TEST_SIZES,
    DEFAULT_MAX_ABS_ERROR_TOLERANCE,
    DEFAULT_MAX_RELATIVE_ERROR_TOLERANCE,
    RuntimeCapabilities,
    RuntimeInfo,
    RuntimeMetadata,
    compare_tensors,
    tolerance_exceeded,
)
from inference.utils.hashing import sha256_of_file


class ONNXRuntime(BaseRuntime):
    """Serves an exported ONNX model via ``onnxruntime.InferenceSession``."""

    RUNTIME_TYPE = "onnx"

    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        model_fingerprint_sha256: str,
        logger: logging.Logger,
        opset_version: Optional[int] = None,
    ) -> None:
        """
        Args:
            model_path: Path to the exported ``model.onnx`` file (Stage 3's
                ``ONNXExportResult.output_path``). Never a checkpoint path.
            device: Target device. Only ``device.type == "cuda"`` is
                inspected (to decide whether to *request*
                CUDAExecutionProvider); ONNX Runtime sessions are not
                otherwise "moved" the way ``nn.Module`` s are.
            model_fingerprint_sha256: ``ModelRegistry.checkpoint_sha256``
                of the checkpoint this export was produced from, reused
                (not recomputed) for cross-runtime identity.
            logger: Caller-supplied logger (dependency injection).
            opset_version: Optional hint from Stage 3's
                ``ONNXExportResult.opset_version``, surfaced via
                :meth:`metadata` for provenance. Purely informational.
        """
        super().__init__(device=device, model_fingerprint_sha256=model_fingerprint_sha256, logger=logger)
        self.model_path = Path(model_path)
        self.opset_version = opset_version
        self._session: Optional[ort.InferenceSession] = None
        self._input_name: Optional[str] = None
        self._selected_providers: List[str] = []
        self._available_providers: List[str] = []
        self._runtime_fingerprint: Optional[str] = None

    def load(self) -> None:
        """Discover available execution providers, select the provider
        list for this device, and load the ``InferenceSession``.

        Provider selection reproduces Stage 5's exact rule (lines
        ~571-575): request ``["CUDAExecutionProvider",
        "CPUExecutionProvider"]`` only if ``device.type == "cuda"`` *and*
        ``"CUDAExecutionProvider"`` is actually in
        ``ort.get_available_providers()``; otherwise request
        ``["CPUExecutionProvider"]``. If a CUDA device was requested but
        CUDA isn't available, that fact is now logged explicitly (see
        module docstring) rather than silently absorbed into the same
        CPU-only provider list the original notebook computed.

        Raises:
            RuntimeError: if the ONNX Runtime session fails to load.
        """
        self._available_providers = ort.get_available_providers()

        if self.device.type == "cuda" and "CUDAExecutionProvider" in self._available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            if self.device.type == "cuda":
                self.logger.warning(
                    "RUNTIME[onnx] CUDA device requested but CUDAExecutionProvider is not in "
                    "available providers %s; falling back to CPUExecutionProvider explicitly.",
                    self._available_providers,
                )
            providers = ["CPUExecutionProvider"]

        try:
            self._session = ort.InferenceSession(str(self.model_path), providers=providers)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"ONNX Runtime session load failed: {exc}") from exc

        self._selected_providers = list(providers)
        self._input_name = self._session.get_inputs()[0].name
        self._runtime_fingerprint = sha256_of_file(self.model_path)
        self._loaded = True

        self.logger.info(
            "RUNTIME[onnx] loaded path=%s selected_providers=%s available_providers=%s fingerprint=%s",
            self.model_path, self._selected_providers, self._available_providers, self._runtime_fingerprint,
        )

    def predict(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass via ``InferenceSession.run``, reproducing Stage
        3/5's exact ``session.run(None, {input_name: batch.cpu().numpy()})``
        pattern, returned as a ``torch.Tensor`` (``torch.from_numpy(...)``)
        so this runtime's ``predict()`` has the identical
        ``torch.Tensor -> torch.Tensor`` signature as the other two
        backends.

        Raises:
            RuntimeError: if called before :meth:`load`.
        """
        if not self._loaded or self._session is None or self._input_name is None:
            raise RuntimeError("ONNXRuntime.predict() called before load().")
        raw_outputs = self._session.run(None, {self._input_name: batch.detach().cpu().numpy()})
        return torch.from_numpy(raw_outputs[0])

    def verify_dynamic_batch(
        self,
        channels: int,
        height: int,
        width: int,
        batch_sizes: Sequence[int] = DEFAULT_DYNAMIC_BATCH_TEST_SIZES,
    ) -> Tuple[bool, List[str]]:
        """Verify the loaded session produces an output whose batch
        dimension matches the input batch dimension, for each size in
        ``batch_sizes``. Verbatim port of Stage 3's
        ``verify_dynamic_batch_support`` (lines ~444-461), adapted to reuse
        this runtime's already-loaded ``self._session`` instead of opening
        a fresh one per call -- same operation, no behavior change, one
        fewer redundant session construction.

        Returns:
            ``(all_passed, errors)``, matching the original function's
            return shape exactly.
        """
        if not self._loaded or self._session is None or self._input_name is None:
            raise RuntimeError("ONNXRuntime.verify_dynamic_batch() called before load().")
        errors: List[str] = []
        try:
            for bs in batch_sizes:
                dummy = np.random.randn(bs, channels, height, width).astype(np.float32)
                outputs = self._session.run(None, {self._input_name: dummy})
                out_batch = outputs[0].shape[0]
                if out_batch != bs:
                    errors.append(f"Dynamic batch mismatch: input batch={bs} produced output batch={out_batch}")
                else:
                    self.logger.info("RUNTIME[onnx] dynamic batch verified for batch_size=%d", bs)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Dynamic batch verification raised an exception: {exc}")
        return len(errors) == 0, errors

    def validate(
        self,
        dummy_input: torch.Tensor,
        reference_output: Optional[torch.Tensor] = None,
        check_dynamic_batch: bool = True,
        dynamic_batch_sizes: Sequence[int] = DEFAULT_DYNAMIC_BATCH_TEST_SIZES,
    ) -> RuntimeInfo:
        """Reproduces Stage 3's ONNX engineering validation: session
        load/executability (via :meth:`load` having already succeeded plus
        a live forward pass here), PyTorch-vs-ONNX numerical equivalence
        (when ``reference_output`` is given) with Stage 3's exact
        tolerances and error-message wording (``"PyTorch vs ONNX numerical
        tolerance exceeded: {...}"``, Stage 3 line ~423), and dynamic-batch
        verification (Stage 3's ``dynamic_axes_correct`` check).

        Never raises on a failed check -- failures are reported in the
        returned :class:`RuntimeInfo`, matching Stage 3's
        aggregate-then-report validation pattern.
        """
        errors: List[str] = []
        executable = False
        output: Optional[torch.Tensor] = None
        try:
            output = self.predict(dummy_input)
            executable = True
        except Exception as exc:  # noqa: BLE001
            errors.append(f"ONNX Runtime load/inference failed: {exc}")

        if executable and reference_output is not None:
            comparison = compare_tensors(reference_output, output)
            if tolerance_exceeded(comparison, DEFAULT_MAX_ABS_ERROR_TOLERANCE, DEFAULT_MAX_RELATIVE_ERROR_TOLERANCE):
                errors.append(f"PyTorch vs ONNX numerical tolerance exceeded: {comparison}")

        if check_dynamic_batch:
            channels, height, width = int(dummy_input.shape[1]), int(dummy_input.shape[2]), int(dummy_input.shape[3])
            _, dynamic_batch_errors = self.verify_dynamic_batch(
                channels=channels, height=height, width=width, batch_sizes=dynamic_batch_sizes,
            )
            errors.extend(dynamic_batch_errors)

        self._validated = executable and len(errors) == 0
        for e in errors:
            self.logger.error("RUNTIME[onnx] VALIDATION FAILURE: %s", e)

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
            runtime_version=ort.__version__,
            source_path=str(self.model_path),
            device=str(self.device),
            model_fingerprint_sha256=self.model_fingerprint_sha256,
            runtime_fingerprint_sha256=self._runtime_fingerprint,
            capabilities=RuntimeCapabilities(
                supports_cpu=True,
                supports_gpu="CUDAExecutionProvider" in (self._available_providers or ort.get_available_providers()),
                # The ONNX export always uses dynamic_axes on the batch
                # dimension (Stage 3's `dynamic_axes = {"input": {0:
                # "batch_size"}, "output": {0: "batch_size"}}`), and Stage
                # 3 explicitly smoke-tests it -- unlike TorchScript's
                # declarative-only flag above, this one is backed by
                # verify_dynamic_batch() whenever validate() has run.
                supports_dynamic_batch=True,
                supports_amp=False,
            ),
        )

    # ------------------------------------------------------------------
    # Provider properties (override BaseRuntime's empty defaults)
    # ------------------------------------------------------------------

    @property
    def selected_providers(self) -> List[str]:
        return list(self._selected_providers)

    @property
    def available_providers(self) -> List[str]:
        return list(self._available_providers)
