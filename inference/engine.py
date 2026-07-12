"""
Inference engine: coordinates preprocessing -> model forward pass ->
postprocessing into a single prediction API.

Migrated from Sprint 05 Stage 4's ``InferenceEngine`` class. In the
original notebook, this class *contained* the validation/decode/preprocess
logic and the success/error result-building logic inline. Here it contains
almost none of that -- it coordinates calls into
``inference.preprocessing``, ``inference.thresholding``, and
``inference.postprocessing``, which now own that logic. The sequencing,
values computed, and error conditions are unchanged; only where the code
*lives* has changed.

Public API addition: ``predict(image)`` is a new primary entry point (the
requested public interface for this phase) that delegates to
``predict_image`` for a single path. ``predict_image`` and ``predict_batch``
are both preserved as-is (same signatures, same behaviour, same ordering
guarantees) since batch prediction is real, validated original
functionality that must remain available.

Dependency injection: the constructor takes an already-reconstructed model,
an already-resolved ``ThresholdRegistry``-derived ``(class_names,
thresholds)`` pair, and an already-resolved ``PreprocessingConfig`` -- it
does not build any of these itself (no artifact discovery, no checkpoint
loading, no threshold-file loading). That wiring is the responsibility of
whatever assembles an ``InferenceEngine`` (a future ``deployment/`` phase,
or a test) -- consistent with "no notebook execution-order dependencies"
and keeping this module fully independent of FastAPI and deployment concerns.

Source: sprint05-deployment.ipynb, Stage 4, lines ~359-484.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

import torch
import torch.nn as nn

from inference.model_registry import ModelRegistry
from inference.postprocessing import (
    PredictionResult,
    apply_sigmoid,
    build_error_result,
    build_prediction_result,
)
from inference.preprocessing import (
    PreprocessingConfig,
    build_batch_tensor,
    preprocess_image,
    validate_and_decode_image,
)


class InferenceEngine:
    """Production inference runtime.

    Consumes an already-reconstructed, already-validated model plus its
    resolved class ordering, thresholds, and preprocessing config.
    Performs no discovery, no checkpoint loading, and no rediscovery of any
    kind -- purely: validate -> decode -> preprocess -> forward pass ->
    threshold -> structured prediction.

    Deterministic: model runs in ``eval()`` mode with grad disabled, and
    preprocessing has no stochastic component.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_names: List[str],
        thresholds: dict,
        preprocessing_config: PreprocessingConfig,
        model_registry: ModelRegistry,
        logger: logging.Logger,
    ) -> None:
        self.model = model
        self.model.eval()
        self.device = device
        self.class_names = class_names
        self.thresholds = thresholds
        self.preprocessing_config = preprocessing_config
        self.model_registry = model_registry
        self.model_fingerprint = model_registry.checkpoint_sha256
        self.model_version = f"{model_registry.backbone}-{model_registry.checkpoint_sha256[:12]}"
        self.logger = logger

    def predict(self, image: str) -> PredictionResult:
        """Predict on a single image path. Primary public entry point for
        this phase's requested interface (``engine.predict(image)``);
        delegates to :meth:`predict_image`."""
        return self.predict_image(image)

    def predict_image(self, image_path: str, image_identifier: Optional[str] = None) -> PredictionResult:
        """Predict on a single image path, optionally under a caller-chosen
        identifier (defaults to the path itself)."""
        identifier = image_identifier if image_identifier is not None else str(image_path)
        return self.predict_batch([image_path], [identifier])[0]

    def predict_batch(
        self, image_paths: List[str], image_identifiers: Optional[List[str]] = None,
    ) -> List[PredictionResult]:
        """Predict on a batch of image paths. Preserves the original
        notebook's exact ordering guarantee: results are returned in the
        same order as ``image_paths``, regardless of which images succeed
        or fail preprocessing.

        Raises:
            ValueError: if ``image_paths`` and ``image_identifiers`` have
                mismatched lengths.
            RuntimeError: if the model's output shape doesn't match the
                expected ``(valid_batch_size, num_classes)`` -- see
                :func:`inference.postprocessing.apply_sigmoid`.
        """
        if image_identifiers is None:
            image_identifiers = [str(p) for p in image_paths]
        if len(image_paths) != len(image_identifiers):
            raise ValueError(
                f"image_paths ({len(image_paths)}) and image_identifiers "
                f"({len(image_identifiers)}) length mismatch."
            )
        if len(image_paths) == 0:
            return []

        results: List[Optional[PredictionResult]] = [None] * len(image_paths)
        valid_indices: List[int] = []
        valid_tensors: List[torch.Tensor] = []

        for i, (path, identifier) in enumerate(zip(image_paths, image_identifiers)):
            img, decode_error = validate_and_decode_image(path, self.preprocessing_config.channels)
            if decode_error is not None:
                self.logger.warning(
                    "INFERENCE reject identifier=%s reason=%s: %s",
                    identifier, decode_error.error_type, decode_error.message,
                )
                results[i] = build_error_result(identifier, decode_error)
                continue

            tensor, preprocess_error = preprocess_image(img, self.preprocessing_config)
            if preprocess_error is not None:
                self.logger.warning(
                    "INFERENCE reject identifier=%s reason=%s: %s",
                    identifier, preprocess_error.error_type, preprocess_error.message,
                )
                results[i] = build_error_result(identifier, preprocess_error)
                continue

            valid_indices.append(i)
            valid_tensors.append(tensor)

        if valid_tensors:
            batch_tensor = build_batch_tensor(valid_tensors, self.device)
            with torch.no_grad():
                logits = self.model(batch_tensor)

            expected_shape = (len(valid_tensors), len(self.class_names))
            probabilities = apply_sigmoid(logits, expected_shape)

            timestamp = datetime.now(timezone.utc).isoformat()
            for local_idx, global_idx in enumerate(valid_indices):
                identifier = image_identifiers[global_idx]
                results[global_idx] = build_prediction_result(
                    identifier=identifier,
                    prob_row=probabilities[local_idx],
                    class_names=self.class_names,
                    thresholds=self.thresholds,
                    timestamp=timestamp,
                    model_fingerprint=self.model_fingerprint,
                    model_version=self.model_version,
                )

        return [r for r in results if r is not None]
