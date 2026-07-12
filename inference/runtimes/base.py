"""
Runtime abstraction layer: common interface over the PyTorch / TorchScript /
ONNX backends a reconstructed model can be served from.

Migration note -- this module does not correspond to a single pre-existing
notebook function the way most modules in this repository do. Sprint 05
never had a runtime *abstraction*: Stage 3 ("Production Export Pipeline")
exported TorchScript and ONNX and validated each format inline, one at a
time, with ad hoc loading code; Stage 5 ("Performance & Robustness
Validation") then re-implemented a near-identical load-and-compare routine
for its own cross-runtime numerical-stability check
(``run_numerical_stability``); Stage 7/8 separately re-derived provider
availability for release documentation. Three different stages, three
slightly different copies of "load a TorchScript/ONNX model and run it",
never unified.

This module -- and ``pytorch_runtime.py`` / ``torchscript_runtime.py`` /
``onnx_runtime.py`` alongside it -- is the first time that behavior is
captured behind one common interface (``BaseRuntime``), so a future
``InferenceEngine`` can depend on "a runtime" uniformly instead of on three
separate ad hoc code paths. Every numerical constant, comparison algorithm,
and provider-selection rule below is ported verbatim from wherever it was
observed in the notebook (see each concrete runtime's module docstring for
its specific source mapping) -- nothing here invents new tolerances,
thresholds, or comparison logic.

Source: sprint05-deployment.ipynb, Stage 3 lines ~59-63 (tolerance
constants), ~344-369 (``_compare_tensors``); Stage 5 lines ~550-580
(cross-runtime load + provider discovery pattern).
"""
from __future__ import annotations

import abc
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

# ======================================================================
# CONSTANTS
# ======================================================================

# Verbatim from Stage 3 (sprint05-deployment.ipynb, Stage 3 lines ~62-63):
# MAX_ABS_ERROR_TOLERANCE / MAX_RELATIVE_ERROR_TOLERANCE. Reused, not
# redeclared with different values, by every concrete runtime's validate().
DEFAULT_MAX_ABS_ERROR_TOLERANCE: float = 1e-3
DEFAULT_MAX_RELATIVE_ERROR_TOLERANCE: float = 1e-2

# Stage 3's own dynamic-batch smoke-test sizes (DYNAMIC_BATCH_TEST_SIZES),
# reused by ONNXRuntime.validate() -- see onnx_runtime.py.
DEFAULT_DYNAMIC_BATCH_TEST_SIZES: Tuple[int, ...] = (1, 4)

# NEW in this phase -- no notebook precedent (BenchmarkConfig.warmup_iterations
# was declared but never consumed anywhere in Sprint 05; benchmarking itself
# was explicitly out of scope, `enabled=False`, "reserved for a later
# Sprint05 stage"). This default is chosen to numerically match that
# reserved config value purely for consistency; BaseRuntime.warmup() below
# does not read configs.schema.BenchmarkConfig and has no wiring to it --
# that wiring, if wanted, belongs to a future benchmarking phase.
DEFAULT_WARMUP_ITERATIONS: int = 10


# ======================================================================
# SHARED TENSOR COMPARISON (ported verbatim from Stage 3 / Stage 5)
# ======================================================================


def compare_tensors(reference: torch.Tensor, candidate: torch.Tensor) -> Dict[str, Any]:
    """Compare two output tensors for shape/dtype match and error magnitude.

    Byte-for-byte identical algorithm to Stage 3's ``_compare_tensors``
    (also duplicated verbatim in Stage 5's own numerical-stability check) --
    moved here as the single shared implementation so
    ``TorchScriptRuntime.validate()`` and ``ONNXRuntime.validate()`` call
    one function instead of each carrying their own copy.

    Source: sprint05-deployment.ipynb, Stage 3 lines ~344-369.
    """
    ref = reference.detach().cpu()
    cand = candidate.detach().cpu()
    shape_match = tuple(ref.shape) == tuple(cand.shape)
    dtype_match = ref.dtype == cand.dtype

    if shape_match:
        diff = (ref.float() - cand.float()).abs()
        max_abs_error = diff.max().item()
        mean_abs_error = diff.mean().item()
        denom = ref.float().abs().clamp(min=1e-8)
        max_relative_error = (diff / denom).max().item()
    else:
        max_abs_error = mean_abs_error = max_relative_error = float("inf")

    return {
        "shape_match": shape_match,
        "dtype_match": dtype_match,
        "reference_shape": list(ref.shape),
        "candidate_shape": list(cand.shape),
        "reference_dtype": str(ref.dtype),
        "candidate_dtype": str(cand.dtype),
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "max_relative_error": max_relative_error,
    }


def tolerance_exceeded(
    comparison: Dict[str, Any],
    max_abs_tol: float = DEFAULT_MAX_ABS_ERROR_TOLERANCE,
    max_rel_tol: float = DEFAULT_MAX_RELATIVE_ERROR_TOLERANCE,
) -> bool:
    """``True`` if a ``compare_tensors()`` result fails Stage 3's exact
    pass/fail predicate: shape mismatch, or either error metric over
    tolerance. Extracted from the boolean expression inline in Stage 3's
    ``run_numerical_validation`` (lines ~420-423) so it isn't repeated
    identically in both ``TorchScriptRuntime.validate()`` and
    ``ONNXRuntime.validate()``."""
    return (
        not comparison["shape_match"]
        or comparison["max_abs_error"] > max_abs_tol
        or comparison["max_relative_error"] > max_rel_tol
    )


# ======================================================================
# DATACLASSES
# ======================================================================


@dataclass
class RuntimeCapabilities:
    """What a runtime backend *can* do, independent of whether it has been
    loaded or validated yet. Distinct from :class:`RuntimeInfo`, which is
    what it *actually* resolved to after ``load()``/``validate()`` ran
    (e.g. which ONNX execution provider was actually selected)."""

    supports_cpu: bool
    supports_gpu: bool
    supports_dynamic_batch: bool
    supports_amp: bool  # automatic mixed precision / autocast

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeMetadata:
    """Identity of one runtime instance: which backend it is, what weights
    it serves, and its declared capabilities. Populated by every concrete
    runtime's ``metadata()``; does not itself imply the runtime has been
    loaded (see :class:`RuntimeInfo` for load/validate state)."""

    runtime_type: str  # "pytorch" | "torchscript" | "onnx"
    runtime_version: str  # torch.__version__ / onnxruntime.__version__
    source_path: Optional[str]  # None for pytorch (in-memory model, no artifact file)
    device: str
    model_fingerprint_sha256: str  # ModelRegistry.checkpoint_sha256, reused not recomputed
    runtime_fingerprint_sha256: Optional[str]  # this runtime's own artifact/graph fingerprint, see subclasses
    capabilities: RuntimeCapabilities

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["capabilities"] = self.capabilities.to_dict()
        return d


@dataclass
class RuntimeInfo:
    """Full state of a runtime instance after ``validate()`` has run:
    static :class:`RuntimeMetadata` plus what was actually observed at
    load/validate time (selected execution providers, for ONNX) and the
    outcome of validation. This is the runtime-layer analogue of Stage 3's
    ``run_stage03_engineering_validation`` result dict and
    ``ModelValidationChecks`` -- an aggregate-then-report structure, not an
    exception. Concrete runtimes' ``validate()`` never raise on a failed
    check; they return this with ``validated=False`` and the reasons in
    ``validation_errors``, leaving the fail/continue decision to the
    caller, exactly as ``run_stage03_engineering_validation`` deferred the
    raise to its caller (``run_stage03``)."""

    metadata: RuntimeMetadata
    loaded: bool
    validated: bool
    selected_providers: List[str] = field(default_factory=list)
    available_providers: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "loaded": self.loaded,
            "validated": self.validated,
            "selected_providers": self.selected_providers,
            "available_providers": self.available_providers,
            "validation_errors": self.validation_errors,
        }


# ======================================================================
# BASE RUNTIME
# ======================================================================


class BaseRuntime(abc.ABC):
    """Common interface every runtime backend implements.

    Public API (per this migration phase's requested interface):
        ``load()``, ``predict(batch)``, ``warmup()``, ``validate()``,
        ``metadata()``.

    Every concrete runtime is constructed via dependency injection --
    ``PyTorchRuntime`` takes an already-reconstructed model,
    ``TorchScriptRuntime``/``ONNXRuntime`` take an exported artifact path
    -- never a checkpoint path, never artifact-discovery/registry logic of
    its own. Reconstruction (``inference.model_loader.reconstruct_model``)
    and export (Stage 3's export pipeline, not yet migrated) are both
    upstream concerns this layer consumes, never duplicates.
    """

    #: Set by each concrete subclass; used as the dispatch key in
    #: ``runtime_factory.RuntimeFactory`` and as ``RuntimeMetadata.runtime_type``.
    RUNTIME_TYPE: str = "base"

    def __init__(
        self,
        device: torch.device,
        model_fingerprint_sha256: str,
        logger: logging.Logger,
    ) -> None:
        self.device = device
        self.model_fingerprint_sha256 = model_fingerprint_sha256
        self.logger = logger
        self._loaded: bool = False
        self._validated: bool = False

    # ------------------------------------------------------------------
    # Abstract public API
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def load(self) -> None:
        """Load/prepare this runtime so ``predict()`` can be called.
        Must raise (never return a falsy sentinel) if loading fails --
        "fail loudly", matching every load/reconstruct function elsewhere
        in this repository."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, batch: torch.Tensor) -> torch.Tensor:
        """Run a forward pass on an already-preprocessed batch tensor
        (shape ``(batch_size, channels, height, width)``) and return raw
        model output (logits -- sigmoid activation is a postprocessing
        concern, applied by ``inference.postprocessing.apply_sigmoid``,
        not by this layer). Must raise ``RuntimeError`` if called before
        ``load()``."""
        raise NotImplementedError

    @abc.abstractmethod
    def validate(
        self,
        dummy_input: torch.Tensor,
        reference_output: Optional[torch.Tensor] = None,
    ) -> RuntimeInfo:
        """Run this runtime's engineering validation: confirm it loads and
        executes, and -- if ``reference_output`` is supplied (typically the
        PyTorch runtime's own output on the same ``dummy_input``) -- verify
        numerical equivalence within :data:`DEFAULT_MAX_ABS_ERROR_TOLERANCE`
        / :data:`DEFAULT_MAX_RELATIVE_ERROR_TOLERANCE` via
        :func:`compare_tensors`. Never raises on a failed check; failures
        are reported in the returned :class:`RuntimeInfo`."""
        raise NotImplementedError

    @abc.abstractmethod
    def metadata(self) -> RuntimeMetadata:
        """Return this runtime's static identity/capabilities. Callable
        before ``load()`` (all fields are known from construction args),
        though ``runtime_fingerprint_sha256`` is only populated once the
        runtime artifact has actually been read (i.e. after ``load()`` for
        TorchScript/ONNX; always populated for PyTorch, which has no
        separate artifact to fingerprint)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Concrete, shared public API
    # ------------------------------------------------------------------

    def warmup(
        self,
        input_shape: Tuple[int, int, int, int],
        iterations: int = DEFAULT_WARMUP_ITERATIONS,
    ) -> None:
        """Run ``iterations`` dummy forward passes to warm up lazy
        initialization (CUDA kernel selection, JIT fusion, ONNX Runtime
        session graph optimizations) ahead of latency-sensitive serving.

        NEW in this phase -- see :data:`DEFAULT_WARMUP_ITERATIONS` for why
        no notebook behavior is being changed by adding this. Side-effect
        only (runs and discards outputs); does not alter what ``predict()``
        returns for any given input. Implemented once here, shared by all
        three backends, since ``predict()`` has an identical
        ``torch.Tensor -> torch.Tensor`` signature across all of them --
        no backend-specific override has been needed.

        Raises:
            RuntimeError: if called before ``load()``.
        """
        if not self._loaded:
            raise RuntimeError(f"{type(self).__name__}.warmup() called before load().")
        dummy = torch.randn(*input_shape, dtype=torch.float32)
        for _ in range(iterations):
            self.predict(dummy)
        self.logger.info(
            "RUNTIME[%s] warmup complete (%d iterations, input_shape=%s).",
            self.RUNTIME_TYPE, iterations, tuple(input_shape),
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def is_validated(self) -> bool:
        return self._validated

    @property
    def selected_providers(self) -> List[str]:
        """Execution providers actually selected at load time. Empty for
        runtimes with no provider concept (PyTorch, TorchScript);
        overridden by :class:`inference.runtimes.onnx_runtime.ONNXRuntime`."""
        return []

    @property
    def available_providers(self) -> List[str]:
        """All execution providers available in this environment,
        regardless of what was selected. Empty for runtimes with no
        provider concept; overridden by ``ONNXRuntime``."""
        return []
