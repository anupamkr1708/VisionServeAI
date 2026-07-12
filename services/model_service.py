"""
Model lifecycle service.

Migrated orchestration from Sprint 05 **Stage 4** ("Production Inference
Runtime")'s ``run_stage04`` preamble -- the sequence that resolves a model's
class ordering/thresholds/preprocessing config and reconstructs it, before
any prediction happens. None of the underlying algorithms are duplicated
here: this module is pure coordination over already-migrated, frozen
functions --
:func:`inference.model_loader.reconstruct_model`,
:func:`inference.thresholding.load_threshold_registry`,
:func:`inference.thresholding.resolve_class_names_and_thresholds`,
:func:`inference.preprocessing.resolve_preprocessing_config` -- plus
:class:`services.artifact_service.ArtifactService` for raw artifact
paths/JSON.

One small piece of genuinely new orchestration is included:
:func:`resolve_input_signature`. The frozen ``resolve_preprocessing_config``
needs an ``input_shape`` (channels/height/width), which the original
notebook took from Stage 3's ``EXPORT_REGISTRY.model_signature`` -- Stage 3
(export) hasn't been migrated into this repository yet (out of scope, same
as noted in ``inference/preprocessing.py``'s own docstring). This service
therefore ports Stage 3's ``resolve_input_signature`` instead: since
``MODEL_REGISTRY`` carries no resolution field, it was *already* a logged,
explicit default (torchvision's standard 224x224x3 ImageNet resolution),
never something Stage 3 derived from the export itself. Porting that one
small function here is a faithful, minimal bridge -- not a new assumption --
and is exactly what ``preprocessing.py``'s docstring anticipated: "future
wiring code will extract input_shape ... and pass them here."

Source: sprint05-deployment.ipynb, Stage 4 lines ~745-799 (``run_stage04``
preamble, ported as :meth:`ModelService.load_model`); Stage 2 line ~1083
(``expected_class_names`` derivation, via
``ArtifactService.expected_class_names_for_thresholds``); Stage 3 lines
~210-219 (``resolve_input_signature``, ported verbatim).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from inference.model_loader import reconstruct_model
from inference.model_registry import ModelRegistry
from inference.preprocessing import PreprocessingConfig, resolve_preprocessing_config
from inference.thresholding import ThresholdRegistry, load_threshold_registry, resolve_class_names_and_thresholds
from services.artifact_service import ArtifactService

# Stage 3's defaults (sprint05-deployment.ipynb lines ~55-57) -- ported
# verbatim alongside resolve_input_signature() below, since MODEL_REGISTRY
# never carried a resolution field even before Stage 3 existed.
DEFAULT_INPUT_CHANNELS: int = 3
DEFAULT_INPUT_HEIGHT: int = 224
DEFAULT_INPUT_WIDTH: int = 224


def resolve_input_signature(model_registry: ModelRegistry, logger: logging.Logger) -> Tuple[int, int, int]:
    """Resolve ``(channels, height, width)`` for a reconstructed model.

    Ported verbatim from Stage 3's ``resolve_input_signature`` (lines
    ~210-219): ``ModelRegistry`` carries no resolution field, so the
    documented default matching the torchvision backbone builders in
    ``inference.model_loader.BASE_MODEL_BUILDERS`` is used -- never guessed
    silently, always logged.
    """
    logger.info(
        "MODEL input signature defaulted to (%d, %d, %d) for backbone='%s' "
        "(torchvision standard ImageNet resolution -- ModelRegistry carries no resolution field).",
        DEFAULT_INPUT_CHANNELS, DEFAULT_INPUT_HEIGHT, DEFAULT_INPUT_WIDTH, model_registry.backbone,
    )
    return DEFAULT_INPUT_CHANNELS, DEFAULT_INPUT_HEIGHT, DEFAULT_INPUT_WIDTH


class ModelService:
    """Owns model lifecycle: reconstruction, resolved class ordering /
    thresholds, resolved preprocessing config, and read-only access to all
    of it. Depends on :class:`ArtifactService` (dependency injection) for
    raw artifact paths and JSON -- never discovers or reads a file itself.
    """

    def __init__(self, artifact_service: ArtifactService, device: torch.device, logger: logging.Logger) -> None:
        self.artifact_service = artifact_service
        self.device = device
        self.logger = logger

        self._model: Optional[nn.Module] = None
        self._model_registry: Optional[ModelRegistry] = None
        self._threshold_registry: Optional[ThresholdRegistry] = None
        self._class_names: Optional[List[str]] = None
        self._thresholds: Optional[Dict[str, float]] = None
        self._preprocessing_config: Optional[PreprocessingConfig] = None
        self._training_summary: Optional[Dict[str, Any]] = None
        self._disease_registry: Optional[Dict[str, Any]] = None

    def load_model(self) -> ModelRegistry:
        """Reconstruct the model and resolve everything a prediction
        pipeline needs to run it: class ordering, thresholds, preprocessing
        config. Idempotent-by-recomputation -- calling this again fully
        reloads (see :meth:`services.runtime_service.RuntimeService.reload_runtime`
        for the paired runtime reload).

        Raises:
            RuntimeError: propagated from ``reconstruct_model`` (checkpoint
                or architecture validation failure) or
                ``resolve_class_names_and_thresholds`` (class/threshold
                mismatch) -- both fail fast, matching Stage 4 exactly.
        """
        self._training_summary = self.artifact_service.load_training_summary()
        self._disease_registry = self.artifact_service.load_disease_registry()
        checkpoint_path = self.artifact_service.checkpoint_path()

        model, model_registry = reconstruct_model(
            training_summary=self._training_summary,
            disease_registry=self._disease_registry,
            checkpoint_path=checkpoint_path,
            device=self.device,
            logger=self.logger,
        )
        self.logger.info(
            "MODEL reconstructed: backbone=%s params=%d device=%s",
            model_registry.backbone, model_registry.total_parameters, model_registry.device,
        )

        thresholds_path = self.artifact_service.optimal_thresholds_path()
        expected_class_names = ArtifactService.expected_class_names_for_thresholds(self._disease_registry)
        threshold_registry = load_threshold_registry(thresholds_path, expected_class_names, self.logger)

        class_names, thresholds = resolve_class_names_and_thresholds(
            disease_metadata=self._disease_registry,
            threshold_registry=threshold_registry,
            num_classes=model_registry.num_classes,
            logger=self.logger,
        )

        channels, height, width = resolve_input_signature(model_registry, self.logger)
        preprocessing_config = resolve_preprocessing_config(
            input_shape=(None, channels, height, width),
            training_metadata=self._training_summary,
            logger=self.logger,
        )

        self._model = model
        self._model_registry = model_registry
        self._threshold_registry = threshold_registry
        self._class_names = class_names
        self._thresholds = thresholds
        self._preprocessing_config = preprocessing_config

        return model_registry

    # ------------------------------------------------------------------
    # Read-only accessors -- all raise RuntimeError if load_model() hasn't
    # run yet, rather than returning None/empty and masking a wiring bug.
    # ------------------------------------------------------------------

    def _require_loaded(self) -> None:
        if self._model_registry is None:
            raise RuntimeError("ModelService.load_model() has not been called yet.")

    @property
    def is_loaded(self) -> bool:
        """Whether :meth:`load_model` has been called successfully.
        Public, non-raising status check -- used by
        ``services.health_service.HealthService`` rather than reaching into
        private state."""
        return self._model_registry is not None

    @property
    def model(self) -> nn.Module:
        self._require_loaded()
        assert self._model is not None
        return self._model

    @property
    def model_registry(self) -> ModelRegistry:
        self._require_loaded()
        assert self._model_registry is not None
        return self._model_registry

    @property
    def preprocessing_config(self) -> PreprocessingConfig:
        self._require_loaded()
        assert self._preprocessing_config is not None
        return self._preprocessing_config

    @property
    def model_version(self) -> str:
        """``f"{backbone}-{checkpoint_sha256[:12]}"`` -- ported verbatim
        from ``inference.engine.InferenceEngine.__init__`` (line ~87),
        which computes the identical formula for the same purpose. Kept
        here too (rather than only in the frozen, untouched ``engine.py``)
        since :class:`ModelService` is the natural read-only owner of
        model version metadata for the new Runtime-based prediction path
        (see ``services.prediction_service``)."""
        registry = self.model_registry
        return f"{registry.backbone}-{registry.checkpoint_sha256[:12]}"

    def model_info(self) -> Dict[str, Any]:
        """Architecture/checkpoint/validation facts (``ModelRegistry``)."""
        return self.model_registry.to_dict()

    def class_names(self) -> List[str]:
        self._require_loaded()
        assert self._class_names is not None
        return list(self._class_names)

    def thresholds(self) -> Dict[str, float]:
        self._require_loaded()
        assert self._thresholds is not None
        return dict(self._thresholds)

    def metadata(self) -> Dict[str, Any]:
        """Raw supporting metadata: training summary, disease registry, and
        the resolved preprocessing config -- distinct from
        :meth:`model_info`, which reports on the reconstructed model itself
        rather than the metadata used to reconstruct it."""
        self._require_loaded()
        return {
            "training_summary": self._training_summary,
            "disease_registry": self._disease_registry,
            "preprocessing_config": self.preprocessing_config.to_dict(),
            "threshold_registry": self._threshold_registry.to_dict() if self._threshold_registry else None,
        }
