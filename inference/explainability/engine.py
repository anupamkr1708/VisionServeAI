"""
Explainability engine: high-level orchestration over the seven independent
explainability algorithms.

Migrated from Sprint 05 Stage 6's ``ExplainabilityEngine`` (notebook lines
~218-843) -- the single large class that previously *contained* every
algorithm's math inline. Here it contains none of that: construction wires
up automatic layer discovery (``layer_discovery.py``), builds one instance
of each algorithm (``gradcam.py`` / ``gradcam_plus.py`` / ``scorecam.py`` /
``eigencam.py`` / ``guided_backprop.py`` / ``integrated_gradients.py`` /
``occlusion.py``, each a thin :class:`~inference.explainability.base.BaseExplainer`
subclass), and every public method dispatches to the matching algorithm
instance's :meth:`~inference.explainability.base.BaseExplainer.generate`.
No CAM math, no hook logic, no colorization/blending, and no JSON-building
happens in this file -- it coordinates calls into
``inference.explainability.layer_discovery``,
``inference.explainability.hooks``, ``inference.explainability.registry``,
``inference.explainability.visualization``, and the seven algorithm
modules, which now own that logic.

Two public API surfaces are exposed, both behaviour-preserving:

  1. **Stage 6's own per-method names** -- ``generate_gradcam``,
     ``generate_gradcam_plus``, ``generate_scorecam``, ``generate_eigencam``,
     ``generate_guided_backprop``, ``generate_integrated_gradients``,
     ``generate_occlusion``, ``batch_generate``, ``export_results``,
     ``save_visualization``, ``overlay_heatmap`` -- identical signatures and
     behaviour to the notebook, since ``services.explainability_service.
     ExplainabilityService`` (frozen, not modified by this phase) calls
     these exact names on whatever engine is injected into it.
  2. **This phase's requested high-level API** -- ``generate()``,
     ``generate_batch()``, ``generate_all_methods()``, ``discover_layers()``,
     ``list_supported_methods()`` -- new, additive orchestration built
     entirely out of calls to (1); no new algorithm code.

Source: sprint05-deployment.ipynb, Stage 6, lines ~218-843.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from inference.explainability import colormaps, layer_discovery, registry, visualization
from inference.explainability.base import (
    BaseExplainer,
    ExplainabilityMetadata,
    ExplainabilityOutputPaths,
    ExplainabilityResult,
    METHOD_NAMES,
    make_output_dirs,
)
from inference.explainability.eigencam import EigenCAM
from inference.explainability.gradcam import GradCAM
from inference.explainability.gradcam_plus import GradCAMPlusPlus
from inference.explainability.guided_backprop import GuidedBackprop
from inference.explainability.integrated_gradients import IG_STEPS, IntegratedGradients
from inference.explainability.occlusion import OCCLUSION_PATCH_SIZE, OCCLUSION_STRIDE, Occlusion
from inference.explainability.scorecam import ScoreCAM
from inference.preprocessing import PreprocessingConfig
from inference.utils.io import save_json


class ExplainabilityEngine:
    """Production explainability engine.

    Consumes an already-reconstructed model plus its resolved class
    ordering and preprocessing config -- via dependency injection, never
    reconstructing or rediscovering anything itself, matching every other
    module in this repository. Exposes a clean, reusable, object-oriented
    API (Stage 6's own per-method names, plus this phase's higher-level
    ``generate``/``generate_batch``/``generate_all_methods``/
    ``discover_layers``/``list_supported_methods``) that any deployment
    surface (REST API, FastAPI, Streamlit, Gradio, desktop, future
    microservices) can call directly without rewriting any CAM/gradient/
    perturbation logic.

    Design note -- why this depends on a raw ``nn.Module`` and not
    ``inference.runtimes``/``RuntimeFactory``: every gradient-based method
    here (GradCAM, GradCAM++, Guided Backprop, Integrated Gradients) needs
    an autograd-enabled forward *and* backward pass through the live model,
    with hooks registered directly on its ``nn.Module`` layers. The Runtime
    Abstraction Layer's ``PyTorchRuntime`` (``inference/runtimes/pytorch_runtime.py``)
    always wraps ``predict()`` in ``torch.no_grad()`` and permanently sets
    ``requires_grad_(False)`` on every parameter at ``load()`` -- exactly
    right for serving, but structurally incompatible with computing
    gradients for explainability. Stage 6 itself never routed through any
    runtime/export layer either: it consumed ``RECONSTRUCTED_MODEL``
    (the raw, still-differentiable model) directly. This engine does the
    same -- it is constructed from ``ModelService.model`` directly (see
    ``services.service_registry.ServiceRegistry``'s wiring), not from
    ``RuntimeService``/``RuntimeFactory``/``RuntimeRegistry``, which remain
    exactly as they were for the prediction-serving path. This is a
    faithful behaviour-preservation decision, not a deviation from the
    "use existing RuntimeFactory/RuntimeRegistry" guidance -- those are
    fully reused for prediction serving (``services.prediction_service``),
    just not for gradient-based explainability, which cannot go through
    them without changing the notebook's numerical behaviour.
    """

    #: Method name -> concrete :class:`BaseExplainer` subclass, in Stage
    #: 6's original ``METHOD_NAMES`` order (notebook lines ~64-67).
    ALGORITHM_CLASSES: Dict[str, type] = {
        "gradcam": GradCAM,
        "gradcam_plus": GradCAMPlusPlus,
        "scorecam": ScoreCAM,
        "eigencam": EigenCAM,
        "guided_backprop": GuidedBackprop,
        "integrated_gradients": IntegratedGradients,
        "occlusion": Occlusion,
    }

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_names: List[str],
        preprocessing_config: PreprocessingConfig,
        output_dir: Path,
        logger: logging.Logger,
        backbone: str = "unknown",
        model_version: Optional[str] = None,
        model_fingerprint: Optional[str] = None,
    ) -> None:
        """
        Args:
            model: Already-reconstructed, already-validated model (the raw
                ``nn.Module`` -- see class docstring for why this, and not
                a ``BaseRuntime``, is required here). Set to ``eval()`` on
                construction, matching Stage 6 exactly; NOT grad-disabled
                (unlike ``PyTorchRuntime.load()``), since every
                gradient-based method needs autograd available.
            device: Target device (matches ``inference_engine.device`` in
                the notebook).
            class_names: Resolved canonical class ordering (matches
                ``inference_engine.class_names``).
            preprocessing_config: Resolved preprocessing config (matches
                ``inference_engine.preprocessing_config``).
            output_dir: Root directory explainability artifacts are written
                under (matches the notebook's ``stage_root``, typically
                supplied by an ``ArtifactService``-resolved path in this
                repository rather than the notebook's hardcoded
                ``/kaggle/working/...``).
            logger: Caller-supplied logger (dependency injection, matching
                every other module in this package).
            backbone: ``ModelRegistry.backbone`` -- reported in
                ``explainability_summary.json`` / ``layer_registry.json``
                exactly as Stage 6 did (via ``self.model_registry.backbone``).
            model_version: Reported in ``explainability_summary.json``
                (matches ``getattr(self.engine, "model_version", None)``).
            model_fingerprint: Reported in ``explainability_summary.json``
                (matches ``getattr(self.engine, "model_fingerprint", None)``).
        """
        self.model = model
        self.model.eval()
        self.device = device
        self.class_names = list(class_names)
        self.preprocessing_config = preprocessing_config
        self.logger = logger
        self.backbone = backbone
        self.model_version = model_version
        self.model_fingerprint = model_fingerprint

        self.output_dirs: ExplainabilityOutputPaths = make_output_dirs(Path(output_dir))

        # ---- Automatic layer discovery (verbatim Stage 6 behaviour) ----
        self.layer_name, self.target_layer, self.feature_dim = layer_discovery.discover_target_layer(
            self.model, self.logger,
        )
        self.activation_size: Optional[List[int]] = layer_discovery.probe_activation_size(
            model=self.model,
            target_layer=self.target_layer,
            device=self.device,
            channels=preprocessing_config.channels,
            height=preprocessing_config.resize_height,
            width=preprocessing_config.resize_width,
            layer_name=self.layer_name,
            logger=self.logger,
        )

        # ---- Colorization backend detection (logged once) ----
        self._colormap_backend = colormaps.log_backend_selection(self.logger)

        # ---- Build one instance of each algorithm, sharing the same
        #      discovered layer/model/config (dependency injection, not
        #      each algorithm rediscovering anything itself) ----
        self._algorithms: Dict[str, BaseExplainer] = {
            name: cls(
                model=self.model,
                target_layer=self.target_layer,
                layer_name=self.layer_name,
                device=self.device,
                class_names=self.class_names,
                preprocessing_config=self.preprocessing_config,
                output_dirs=self.output_dirs,
                logger=self.logger,
            )
            for name, cls in self.ALGORITHM_CLASSES.items()
        }

        # ---- Method availability detection (never silently fail) ----
        self.method_availability: Dict[str, ExplainabilityMetadata] = {
            name: algo.initialize() for name, algo in self._algorithms.items()
        }

        self.results: List[ExplainabilityResult] = []

        self._method_dispatch: Dict[str, Callable[..., ExplainabilityResult]] = {
            "gradcam": self.generate_gradcam,
            "gradcam_plus": self.generate_gradcam_plus,
            "scorecam": self.generate_scorecam,
            "eigencam": self.generate_eigencam,
            "guided_backprop": self.generate_guided_backprop,
            "integrated_gradients": self.generate_integrated_gradients,
            "occlusion": self.generate_occlusion,
        }

    # ------------------------------------------------------------------
    # Stage 6's own public API (byte-identical names/signatures --
    # services.explainability_service.ExplainabilityService delegates to
    # exactly these).
    # ------------------------------------------------------------------

    def generate_gradcam(self, image: Image.Image, target_class: Optional[Any] = None, sample_id: str = "sample") -> ExplainabilityResult:
        return self._algorithms["gradcam"].generate(image, target_class=target_class, sample_id=sample_id)

    def generate_gradcam_plus(self, image: Image.Image, target_class: Optional[Any] = None, sample_id: str = "sample") -> ExplainabilityResult:
        return self._algorithms["gradcam_plus"].generate(image, target_class=target_class, sample_id=sample_id)

    def generate_scorecam(self, image: Image.Image, target_class: Optional[Any] = None, sample_id: str = "sample") -> ExplainabilityResult:
        return self._algorithms["scorecam"].generate(image, target_class=target_class, sample_id=sample_id)

    def generate_eigencam(self, image: Image.Image, target_class: Optional[Any] = None, sample_id: str = "sample") -> ExplainabilityResult:
        return self._algorithms["eigencam"].generate(image, target_class=target_class, sample_id=sample_id)

    def generate_guided_backprop(self, image: Image.Image, target_class: Optional[Any] = None, sample_id: str = "sample") -> ExplainabilityResult:
        return self._algorithms["guided_backprop"].generate(image, target_class=target_class, sample_id=sample_id)

    def generate_integrated_gradients(
        self, image: Image.Image, target_class: Optional[Any] = None, sample_id: str = "sample", steps: int = IG_STEPS,
    ) -> ExplainabilityResult:
        return self._algorithms["integrated_gradients"].generate(
            image, target_class=target_class, sample_id=sample_id, steps=steps,
        )

    def generate_occlusion(
        self, image: Image.Image, target_class: Optional[Any] = None, sample_id: str = "sample",
        patch_size: int = OCCLUSION_PATCH_SIZE, stride: int = OCCLUSION_STRIDE,
    ) -> ExplainabilityResult:
        return self._algorithms["occlusion"].generate(
            image, target_class=target_class, sample_id=sample_id, patch_size=patch_size, stride=stride,
        )

    def batch_generate(
        self,
        images: List[Tuple[str, Image.Image]],
        methods: Optional[List[str]] = None,
        target_class: Optional[Any] = None,
    ) -> List[ExplainabilityResult]:
        """Runs the requested (default: all implemented) methods across a
        batch of ``(sample_id, PIL.Image)`` pairs. Every method/image
        combination is isolated -- one failure never stops the rest.

        Verbatim port of Stage 6's ``ExplainabilityEngine.batch_generate``
        (notebook lines ~762-782).
        """
        methods = methods or METHOD_NAMES
        batch_results: List[ExplainabilityResult] = []
        for sample_id, image in images:
            for method_name in methods:
                fn = self._method_dispatch.get(method_name)
                if fn is None:
                    self.logger.warning("BATCH unknown method '%s' requested; skipping.", method_name)
                    continue
                result = fn(image, target_class=target_class, sample_id=sample_id)
                batch_results.append(result)
        self.results.extend(batch_results)
        return batch_results

    def export_results(self, engineering_validation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Writes all required JSON artifacts to the metadata directory and
        returns the top-level ``explainability_summary`` dict.

        Verbatim port of Stage 6's ``ExplainabilityEngine.export_results``
        (notebook lines ~784-843); JSON writing itself delegates to
        ``inference.utils.io.save_json`` (the canonical, already-migrated
        utility -- ``ensure_parent=True`` reproduces Stage 6's own variant
        of that function, per that module's docstring) rather than a
        locally duplicated writer.
        """
        metadata_dir = self.output_dirs.metadata

        summary = registry.build_explainability_summary(
            method_availability=self.method_availability,
            layer_name=self.layer_name,
            layer_type=type(self.target_layer).__name__,
            feature_dim=self.feature_dim,
            activation_size=self.activation_size,
            results=self.results,
            class_names=self.class_names,
            model_version=self.model_version,
            model_fingerprint=self.model_fingerprint,
            backbone=self.backbone,
        )
        save_json(metadata_dir / "explainability_summary.json", summary, ensure_parent=True)

        gradcam_metadata = registry.build_gradcam_metadata(
            layer_name=self.layer_name, feature_dim=self.feature_dim,
            activation_size=self.activation_size, results=self.results,
        )
        save_json(metadata_dir / "gradcam_metadata.json", gradcam_metadata, ensure_parent=True)

        layer_registry = registry.build_layer_registry(
            layer_name=self.layer_name, layer_type=type(self.target_layer).__name__,
            feature_dim=self.feature_dim, activation_size=self.activation_size, backbone=self.backbone,
        )
        save_json(metadata_dir / "layer_registry.json", layer_registry, ensure_parent=True)

        method_registry = registry.build_method_registry(self.method_availability)
        save_json(metadata_dir / "method_registry.json", method_registry, ensure_parent=True)

        performance_summary = registry.build_performance_summary(self.results)
        save_json(metadata_dir / "performance_summary.json", performance_summary, ensure_parent=True)

        if engineering_validation is not None:
            save_json(metadata_dir / "engineering_validation.json", engineering_validation, ensure_parent=True)

        return summary

    @staticmethod
    def save_visualization(array_uint8, path: Path) -> None:
        """Public reusable primitive -- delegates to
        ``inference.explainability.visualization.save_visualization``
        (same name/signature/behaviour as Stage 6's own method)."""
        visualization.save_visualization(array_uint8, path)

    def overlay_heatmap(self, base_rgb_uint8, heatmap_2d, alpha: float = 0.45):
        """Public reusable primitive -- delegates to
        ``inference.explainability.visualization.overlay_heatmap`` (same
        name/signature/behaviour, including the default ``alpha=0.45``, as
        Stage 6's own method)."""
        return visualization.overlay_heatmap(base_rgb_uint8, heatmap_2d, alpha=alpha)

    # ------------------------------------------------------------------
    # This phase's requested high-level orchestration API. Additive --
    # composed entirely out of the methods above; no new algorithm code.
    # ------------------------------------------------------------------

    def generate(
        self,
        method: str,
        image: Image.Image,
        target_class: Optional[Any] = None,
        sample_id: str = "sample",
        **kwargs: Any,
    ) -> ExplainabilityResult:
        """Unified single-method entry point: run ``method`` (one of
        :meth:`list_supported_methods`' keys) on ``image``.

        Raises:
            ValueError: if ``method`` isn't a recognized method name.
        """
        fn = self._method_dispatch.get(method)
        if fn is None:
            raise ValueError(
                f"Unknown explainability method '{method}'. Supported: {list(self._method_dispatch)}"
            )
        return fn(image, target_class=target_class, sample_id=sample_id, **kwargs)

    def generate_batch(
        self,
        images: List[Tuple[str, Image.Image]],
        methods: Optional[List[str]] = None,
        target_class: Optional[Any] = None,
    ) -> List[ExplainabilityResult]:
        """Alias of :meth:`batch_generate` under this phase's requested
        name. Identical behaviour -- ``batch_generate`` is kept as the
        primary implementation (and as the exact name Stage 6 used /
        ``ExplainabilityService`` could delegate to), this is a thin
        wrapper for callers using the new high-level API."""
        return self.batch_generate(images, methods=methods, target_class=target_class)

    def generate_all_methods(
        self, image: Image.Image, target_class: Optional[Any] = None, sample_id: str = "sample",
    ) -> Dict[str, ExplainabilityResult]:
        """Run every implemented method on a single image, keyed by method
        name. Convenience composition over :meth:`generate` -- no new
        algorithm code; equivalent to ``batch_generate([(sample_id, image)])``
        but returns a dict instead of a flat list, and does not append to
        :attr:`results` (so exploratory single-image calls don't pollute a
        later :meth:`export_results` summary; use :meth:`batch_generate` /
        :meth:`generate_batch` when accumulation is wanted)."""
        return {
            method_name: fn(image, target_class=target_class, sample_id=sample_id)
            for method_name, fn in self._method_dispatch.items()
        }

    def discover_layers(self) -> Dict[str, Any]:
        """Report of the auto-discovered target layer -- the same content
        ``export_results`` writes to ``layer_registry.json``, available
        without a full export. Does not re-run discovery (already resolved
        once at construction, matching Stage 6's "discover once, reuse for
        every method" behaviour) -- also lists every candidate
        ``nn.Conv2d`` layer found, for transparency, without changing which
        one was actually selected."""
        info = registry.build_layer_registry(
            layer_name=self.layer_name, layer_type=type(self.target_layer).__name__,
            feature_dim=self.feature_dim, activation_size=self.activation_size, backbone=self.backbone,
        )
        info["candidate_layers"] = layer_discovery.list_conv_candidates(self.model)
        return info

    def list_supported_methods(self) -> Dict[str, Dict[str, Any]]:
        """Snapshot of every method's availability + reason -- the same
        content ``export_results`` writes to ``method_registry.json``,
        available without a full export."""
        return registry.build_method_registry(self.method_availability)

    def run_engineering_validation(self) -> Dict[str, Any]:
        """Aggregate engineering validation over every result accumulated
        so far in :attr:`results`. Verbatim port of Stage 6's
        ``run_stage06_engineering_validation`` (notebook lines ~850-902),
        via ``registry.run_engineering_validation``."""
        expected_size = (self.preprocessing_config.resize_width, self.preprocessing_config.resize_height)
        return registry.run_engineering_validation(
            method_availability=self.method_availability,
            target_layer_found=self.target_layer is not None,
            activation_size=self.activation_size,
            results=self.results,
            cv2_available=colormaps.CV2_AVAILABLE,
            expected_overlay_size=expected_size,
            logger=self.logger,
        )
