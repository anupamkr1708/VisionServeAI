"""
Explainability base types and the shared per-method execution wrapper.

Migrated from Sprint 05 **Stage 6** ("Explainability Runtime") of the
archived notebook. Stage 6 implemented seven explainability algorithms
(GradCAM, GradCAM++, ScoreCAM, EigenCAM, Guided Backprop, Integrated
Gradients, Occlusion) as methods on one large ``ExplainabilityEngine``
class, each going through an identical wrapper
(``ExplainabilityEngine._run_method``, notebook lines ~649-724) that
resolved the prediction target, ran the algorithm's own compute step,
checked the result for NaN/Inf, rendered + saved the heatmap/overlay/raw
attribution, and isolated any failure so one method's exception never
aborted the rest of a batch.

This module extracts that identical wrapper -- byte-for-byte the same
sequence of operations, in the same order, with the same error handling --
into :meth:`BaseExplainer.generate`, so each of the seven algorithm modules
(``gradcam.py``, ``gradcam_plus.py``, ``scorecam.py``, ``eigencam.py``,
``guided_backprop.py``, ``integrated_gradients.py``, ``occlusion.py``)
implements *only* :meth:`BaseExplainer.compute` -- its own attribution math
-- and nothing else. This is the "no shared duplicated logic" architecture
requirement satisfied the same way ``inference/runtimes/base.py`` satisfies
it for the three runtime backends: one shared base class, thin concrete
subclasses.

``ExplainabilityResult`` and ``ExplainabilityMetadata`` are renamed from the
notebook's ``MethodResult`` / ``MethodAvailability`` (clearer names for a
production package with its own ``base.py``), but every field, type, and
default is preserved exactly -- ``to_dict()`` produces byte-identical JSON
to the original dataclasses via the same ``dataclasses.asdict`` call, so
every downstream artifact (``explainability_summary.json``,
``gradcam_metadata.json``, ``method_registry.json``, ...) is unchanged.

Source: sprint05-deployment.ipynb, Stage 6, lines ~127-158 (dataclasses),
~237-397 (constructor / setup / resolve-target / normalize helpers),
~649-724 (``_run_method`` -- the wrapper migrated here verbatim).
"""
from __future__ import annotations

import abc
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from inference.explainability import overlays, visualization
from inference.preprocessing import PreprocessingConfig
from inference.utils.timers import Timer

# ======================================================================
# CONSTANTS -- moved from Stage 6 notebook globals, unchanged values.
# ======================================================================

IG_STEPS: int = 16
OCCLUSION_PATCH_SIZE: int = 32
OCCLUSION_STRIDE: int = 16
SCORECAM_MAX_CHANNELS: int = 32
DEFAULT_OVERLAY_ALPHA: float = 0.45

#: Canonical method name -> implementation module ordering. Preserved
#: verbatim from Stage 6's ``METHOD_NAMES`` (notebook lines ~64-67); every
#: place that needs "all implemented methods, in the original order"
#: (``ExplainabilityEngine.batch_generate`` default, ``export_results``'s
#: ``methods_implemented`` field, ``ExplainabilityService.SUPPORTED_METHODS``)
#: reads from this one list.
METHOD_NAMES: List[str] = [
    "gradcam", "gradcam_plus", "scorecam", "eigencam",
    "guided_backprop", "integrated_gradients", "occlusion",
]


# ======================================================================
# DATACLASSES
# ======================================================================


@dataclass
class ExplainabilityMetadata:
    """Whether one explainability method is available on the current model,
    and why. Renamed from Stage 6's ``MethodAvailability`` -- identical
    fields, identical ``to_dict()`` shape (``{"method", "available",
    "reason"}``), so ``method_registry.json`` is byte-identical."""

    method: str
    available: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExplainabilityResult:
    """Outcome of running one explainability method on one image. Renamed
    from Stage 6's ``MethodResult`` -- every field, default, and
    ``to_dict()`` shape preserved exactly."""

    method: str
    sample_id: str
    success: bool
    execution_time_ms: float = 0.0
    predicted_class: Optional[str] = None
    predicted_class_index: Optional[int] = None
    target_class: Optional[str] = None
    target_class_index: Optional[int] = None
    confidence: Optional[float] = None
    heatmap_path: Optional[str] = None
    overlay_path: Optional[str] = None
    raw_attribution_path: Optional[str] = None
    heatmap_shape: Optional[List[int]] = None
    no_nan: Optional[bool] = None
    no_inf: Optional[bool] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Backward/behavioural-reference aliases -- the notebook's own names, kept
# importable so anything cross-referencing Stage 6 by its original class
# names still resolves to the exact same type.
MethodAvailability = ExplainabilityMetadata
MethodResult = ExplainabilityResult


@dataclass
class ExplainabilityOutputPaths:
    """Resolved output directories for one explainability run, keyed the
    same way as Stage 6's ``make_stage_dirs`` return dict (notebook lines
    ~103-120) -- one subdirectory per method, plus ``overlays``,
    ``metadata``, and ``inputs``."""

    stage_root: Path
    logs: Path
    gradcam: Path
    gradcam_plus: Path
    scorecam: Path
    eigencam: Path
    integrated_gradients: Path
    guided_backprop: Path
    occlusion: Path
    overlays: Path
    metadata: Path
    inputs: Path

    def __getitem__(self, key: str) -> Path:
        return getattr(self, key)

    def as_dict(self) -> Dict[str, Path]:
        return {
            "stage_root": self.stage_root, "logs": self.logs,
            "gradcam": self.gradcam, "gradcam_plus": self.gradcam_plus,
            "scorecam": self.scorecam, "eigencam": self.eigencam,
            "integrated_gradients": self.integrated_gradients,
            "guided_backprop": self.guided_backprop, "occlusion": self.occlusion,
            "overlays": self.overlays, "metadata": self.metadata, "inputs": self.inputs,
        }


def make_output_dirs(output_root: Path) -> ExplainabilityOutputPaths:
    """Create (and return) every explainability output subdirectory.
    Verbatim port of Stage 6's ``make_stage_dirs`` (notebook lines
    ~103-120): same directory names, same "create if missing" behaviour."""
    dirs = ExplainabilityOutputPaths(
        stage_root=output_root,
        logs=output_root / "logs",
        gradcam=output_root / "gradcam",
        gradcam_plus=output_root / "gradcam_plus",
        scorecam=output_root / "scorecam",
        eigencam=output_root / "eigencam",
        integrated_gradients=output_root / "integrated_gradients",
        guided_backprop=output_root / "guided_backprop",
        occlusion=output_root / "occlusion",
        overlays=output_root / "overlays",
        metadata=output_root / "metadata",
        inputs=output_root / "images" / "synthetic_inputs",
    )
    for d in dirs.as_dict().values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ======================================================================
# BASE EXPLAINER
# ======================================================================


class BaseExplainer(abc.ABC):
    """Common interface + shared execution wrapper for every explainability
    algorithm.

    Concrete subclasses (one per algorithm, one file each) implement only
    :meth:`compute` -- the algorithm-specific attribution math -- and
    optionally override :meth:`check_availability` if the method isn't
    unconditionally available (see ``guided_backprop.py``). Everything else
    (target resolution, input-tensor construction, NaN/Inf checking,
    heatmap/overlay/raw-attribution rendering and saving, timing, per-image
    failure isolation) lives here exactly once, ported verbatim from Stage
    6's ``_run_method`` (notebook lines ~649-724).

    Public API (per this phase's requested interface):
        ``initialize()``, ``generate()``, ``validate()``, ``metadata()``.
    """

    #: Set by each concrete subclass -- the key used in ``METHOD_NAMES``,
    #: ``method_availability``, output-directory selection, and every
    #: per-method JSON artifact.
    METHOD_NAME: str = "base"

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        layer_name: str,
        device: torch.device,
        class_names: List[str],
        preprocessing_config: PreprocessingConfig,
        output_dirs: ExplainabilityOutputPaths,
        logger: logging.Logger,
    ) -> None:
        self.model = model
        self.target_layer = target_layer
        self.layer_name = layer_name
        self.device = device
        self.class_names = class_names
        self.preprocessing_config = preprocessing_config
        self.output_dirs = output_dirs
        self.logger = logger

        self._metadata: Optional[ExplainabilityMetadata] = None

    # ------------------------------------------------------------------
    # Common interface
    # ------------------------------------------------------------------

    def initialize(self) -> ExplainabilityMetadata:
        """Resolve (and cache) this method's :class:`ExplainabilityMetadata`
        via :meth:`check_availability`. Idempotent -- safe to call more than
        once; subsequent calls return the cached result rather than
        re-deriving it."""
        if self._metadata is None:
            self._metadata = self.check_availability()
            if not self._metadata.available:
                self.logger.warning(
                    "METHOD %s disabled: %s", self.METHOD_NAME, self._metadata.reason,
                )
        return self._metadata

    def check_availability(self) -> ExplainabilityMetadata:
        """Default availability: unconditionally available, gated only on
        the (already-discovered) target conv layer -- matches Stage 6's
        default branch for every CAM-family/gradient/perturbation method
        except Guided Backprop (notebook lines ~328-329, ~342-347).
        Overridden by :class:`~inference.explainability.guided_backprop.GuidedBackprop`.
        """
        return ExplainabilityMetadata(
            method=self.METHOD_NAME, available=True,
            reason=f"target conv layer '{self.layer_name}' available",
        )

    @property
    def metadata_info(self) -> ExplainabilityMetadata:
        """Public read of this method's resolved availability -- calls
        :meth:`initialize` if it hasn't run yet."""
        return self.initialize()

    def metadata(self) -> Dict[str, Any]:
        """JSON-serializable snapshot of this method's identity and
        availability -- the per-method entry ``method_registry.json``
        aggregates (see ``registry.py``)."""
        return self.metadata_info.to_dict()

    @abc.abstractmethod
    def compute(self, tensor: torch.Tensor, target_idx: int, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Algorithm-specific attribution computation.

        Args:
            tensor: Already-preprocessed ``(C, H, W)`` input tensor (not yet
                batched), on CPU -- moved to ``self.device`` by the
                implementation as needed (mirrors Stage 6, where each
                ``_compute_*`` method did its own ``.to(self.device)``).
            target_idx: Resolved target class index to explain.

        Returns:
            ``(attribution_2d, probs)`` where ``attribution_2d`` is a
            normalized-to-``[0, 1]`` ``(H, W)`` float32 array (via
            :func:`inference.explainability.overlays.normalize_map`) and
            ``probs`` is the per-class sigmoid probability row observed
            during computation.
        """
        raise NotImplementedError

    def generate(
        self,
        image: Image.Image,
        target_class: Optional[Any] = None,
        sample_id: str = "sample",
        **extra_params: Any,
    ) -> ExplainabilityResult:
        """Run this method end-to-end on one image: resolve target -> run
        :meth:`compute` -> validate -> render + save artifacts -> build
        :class:`ExplainabilityResult`. Every failure is caught and isolated
        -- returns a failed result rather than raising, so one bad
        method/image never aborts a batch.

        Verbatim port of Stage 6's ``_run_method`` (notebook lines
        ~649-724).
        """
        availability = self.initialize()
        if not availability.available:
            msg = f"method disabled: {availability.reason}"
            self.logger.warning(
                "METHOD %s skipped sample=%s reason=%s", self.METHOD_NAME, sample_id, msg,
            )
            return ExplainabilityResult(method=self.METHOD_NAME, sample_id=sample_id, success=False, error=msg)

        with Timer() as timer:
            try:
                self.logger.info("START method=%s sample=%s", self.METHOD_NAME, sample_id)
                tensor = self._to_input_tensor(image)

                with torch.no_grad():
                    probe_logits = self.model(tensor.unsqueeze(0).to(self.device))
                    probe_probs = torch.sigmoid(probe_logits)[0]

                pred_idx, pred_name, target_idx, target_name, pred_conf = self._resolve_target(
                    probe_probs, target_class,
                )

                attribution, _ = self.compute(tensor, target_idx, **extra_params)

                no_nan = not bool(np.isnan(attribution).any())
                no_inf = not bool(np.isinf(attribution).any())
                if not no_nan or not no_inf:
                    raise RuntimeError(f"{self.METHOD_NAME} produced NaN/Inf values in the attribution map.")

                base_rgb = self._to_display_rgb(tensor)

                heatmap_path = self.output_dirs[self.METHOD_NAME] / f"{sample_id}_heatmap.png"
                overlay_path = self.output_dirs.overlays / f"{sample_id}_{self.METHOD_NAME}_overlay.png"
                raw_path = self.output_dirs.metadata / f"{sample_id}_{self.METHOD_NAME}_raw.npy"

                visualization.render_method_artifacts(
                    base_rgb_uint8=base_rgb,
                    attribution_2d=attribution,
                    heatmap_path=heatmap_path,
                    overlay_path=overlay_path,
                    raw_path=raw_path,
                    alpha=DEFAULT_OVERLAY_ALPHA,
                )

                result = ExplainabilityResult(
                    method=self.METHOD_NAME,
                    sample_id=sample_id,
                    success=True,
                    execution_time_ms=0.0,  # set below, after the timer closes
                    predicted_class=pred_name,
                    predicted_class_index=pred_idx,
                    target_class=target_name,
                    target_class_index=target_idx,
                    confidence=round(pred_conf, 6),
                    heatmap_path=str(heatmap_path),
                    overlay_path=str(overlay_path),
                    raw_attribution_path=str(raw_path),
                    heatmap_shape=list(attribution.shape),
                    no_nan=no_nan,
                    no_inf=no_inf,
                    error=None,
                )
            except Exception as exc:  # noqa: BLE001 -- isolate failure, never propagate
                self.logger.error("METHOD %s FAILED sample=%s error=%s", self.METHOD_NAME, sample_id, exc)
                result = ExplainabilityResult(
                    method=self.METHOD_NAME, sample_id=sample_id, success=False, error=str(exc),
                )
            finally:
                self.model.zero_grad(set_to_none=True)

        result.execution_time_ms = round(timer.elapsed_ms, 3)
        if result.success:
            self.logger.info(
                "FINISH method=%s sample=%s time_ms=%.2f", self.METHOD_NAME, sample_id, result.execution_time_ms,
            )
        return result

    def validate(self, result: ExplainabilityResult) -> bool:
        """Per-result sanity check: a successful result must carry a 2-D
        heatmap shape and clean NaN/Inf flags. Used by
        ``registry.run_engineering_validation`` as one input among several
        aggregate checks -- not a replacement for it."""
        if not result.success:
            return True  # a failed result is not itself an invalid one
        return (
            result.heatmap_shape is not None
            and len(result.heatmap_shape) == 2
            and bool(result.no_nan)
            and bool(result.no_inf)
        )

    # ------------------------------------------------------------------
    # Shared preprocessing / display / target-resolution helpers.
    # Ported verbatim from Stage 6's ``ExplainabilityEngine`` (notebook
    # lines ~356-397) -- identical for every method, so they live here once
    # instead of being duplicated across seven algorithm modules.
    # ------------------------------------------------------------------

    def _to_input_tensor(self, image: Image.Image) -> torch.Tensor:
        cfg = self.preprocessing_config
        resized = image.convert("RGB").resize((cfg.resize_width, cfg.resize_height), Image.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        mean = torch.tensor(cfg.mean, dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(cfg.std, dtype=torch.float32).view(-1, 1, 1)
        return (tensor - mean) / std

    def _to_display_rgb(self, tensor: torch.Tensor) -> np.ndarray:
        cfg = self.preprocessing_config
        return overlays.denormalize_to_display_rgb(tensor, cfg.mean, cfg.std)

    def _predicted_class(self, probs: torch.Tensor) -> Tuple[int, str, float]:
        idx = int(torch.argmax(probs).item())
        return idx, self.class_names[idx], float(probs[idx].item())

    def _resolve_target(
        self, probe_probs: torch.Tensor, target_class: Optional[Any],
    ) -> Tuple[int, str, int, str, float]:
        pred_idx, pred_name, pred_conf = self._predicted_class(probe_probs)
        if target_class is None:
            return pred_idx, pred_name, pred_idx, pred_name, pred_conf
        if isinstance(target_class, str):
            if target_class not in self.class_names:
                raise ValueError(f"Unknown target_class '{target_class}'.")
            target_idx = self.class_names.index(target_class)
        else:
            target_idx = int(target_class)
        return pred_idx, pred_name, target_idx, self.class_names[target_idx], pred_conf

    @staticmethod
    def normalize_map(x: np.ndarray) -> np.ndarray:
        """Exposed for algorithm modules that need to normalize their raw
        attribution before returning from :meth:`compute` -- delegates to
        :func:`inference.explainability.overlays.normalize_map` (the single
        implementation; not duplicated here)."""
        return overlays.normalize_map(x)
