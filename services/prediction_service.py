"""
Prediction service.

Coordinates the complete prediction pipeline requested for this phase:
Preprocessing -> RuntimeFactory -> Runtime -> Postprocessing ->
PredictionResult. This is new orchestration -- no algorithm this module
touches is reimplemented; every computation is delegated to already-
migrated, frozen functions:

    inference.preprocessing.validate_and_decode_image / preprocess_image /
        build_batch_tensor
    inference.runtimes (Runtime Abstraction Layer, via services.runtime_service)
    inference.postprocessing.apply_sigmoid / build_prediction_result /
        build_error_result

This is structurally the same sequence ``inference.engine.InferenceEngine``
already runs (Sprint 05 Stage 4, lines ~102-174) -- decode/validate each
image, preprocess the valid ones, stack into one batch tensor, one forward
pass, postprocess each row -- reproduced here because ``engine.py`` calls
the raw model directly (``self.model(batch_tensor)``) and is frozen/
untouched by this phase, whereas this service is explicitly requested to
run the forward pass through the pluggable Runtime Abstraction Layer
instead (``runtime.predict(batch_tensor)``). The *sequence* is intentionally
parallel to ``engine.py``'s proven-correct one; the *leaf-level algorithms*
it calls are the same frozen functions ``engine.py`` calls, imported, never
copied.

``predict_from_bytes`` / ``predict_from_numpy`` / ``predict_from_pil`` are
NEW input adapters -- Stage 4 only ever accepted file paths. Rather than
partially re-implementing ``validate_and_decode_image``'s corruption/
band-count/RGB-conversion checks for in-memory inputs (which would
duplicate that frozen algorithm), each adapter serializes its input to PNG
bytes and writes them to a temporary file, then runs it through the exact
same file-path pipeline every other input type uses. This guarantees
byte-for-byte identical validation behaviour across all five entry points,
at the cost of a small, one-time PNG encode/decode round trip for inputs
that were already decoded in memory -- a deliberate trade-off in favour of
zero duplicated validation logic.
"""
from __future__ import annotations

import io
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from inference.postprocessing import (
    PredictionResult,
    apply_sigmoid,
    build_error_result,
    build_prediction_result,
)
from inference.preprocessing import build_batch_tensor, preprocess_image, validate_and_decode_image
from services.model_service import ModelService
from services.runtime_service import RuntimeService

ImageInput = Union[str, bytes, bytearray, np.ndarray, Image.Image]


class PredictionService:
    """Owns the complete prediction pipeline. Depends on
    :class:`ModelService` (class names / thresholds / preprocessing config
    / model version) and :class:`RuntimeService` (an initialized
    :class:`~inference.runtimes.base.BaseRuntime`) via dependency injection
    -- performs no model reconstruction, no artifact discovery, and no
    runtime lifecycle management itself.
    """

    def __init__(self, model_service: ModelService, runtime_service: RuntimeService, logger: logging.Logger) -> None:
        self.model_service = model_service
        self.runtime_service = runtime_service
        self.logger = logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image: ImageInput, identifier: Optional[str] = None) -> PredictionResult:
        """Predict on a single image of any supported input type,
        dispatching by type to the matching ``predict_from_*``/
        :meth:`predict_image` method.

        Raises:
            TypeError: if ``image`` isn't one of ``str`` (file path),
                ``bytes``/``bytearray``, ``numpy.ndarray``, or
                ``PIL.Image.Image``.
        """
        if isinstance(image, str):
            return self.predict_image(image, identifier)
        if isinstance(image, (bytes, bytearray)):
            return self.predict_from_bytes(bytes(image), identifier)
        if isinstance(image, np.ndarray):
            return self.predict_from_numpy(image, identifier)
        if isinstance(image, Image.Image):
            return self.predict_from_pil(image, identifier)
        raise TypeError(
            f"Unsupported image input type: {type(image)!r}. "
            f"Expected str (file path), bytes, numpy.ndarray, or PIL.Image.Image."
        )

    def predict_image(self, image_path: str, image_identifier: Optional[str] = None) -> PredictionResult:
        """Predict on a single image file path, optionally under a
        caller-chosen identifier (defaults to the path itself). Mirrors
        ``InferenceEngine.predict_image``'s single-item-via-batch pattern
        exactly (Stage 4, lines ~96-100)."""
        identifier = image_identifier if image_identifier is not None else str(image_path)
        return self.predict_batch([image_path], [identifier])[0]

    def predict_batch(
        self, images: List[ImageInput], image_identifiers: Optional[List[str]] = None,
    ) -> List[PredictionResult]:
        """Predict on a batch of images of any supported input type (mixed
        types allowed). Preserves the original notebook's exact ordering
        guarantee: results are returned in the same order as ``images``,
        regardless of which images succeed or fail.

        Raises:
            ValueError: if ``images`` and ``image_identifiers`` have
                mismatched lengths.
            RuntimeError: if no runtime has been initialized (see
                ``RuntimeService.initialize_runtime``), or the runtime's
                output shape doesn't match the expected
                ``(valid_batch_size, num_classes)``.
        """
        if image_identifiers is None:
            image_identifiers = [img if isinstance(img, str) else f"image_{i}" for i, img in enumerate(images)]
        if len(images) != len(image_identifiers):
            raise ValueError(
                f"images ({len(images)}) and image_identifiers ({len(image_identifiers)}) length mismatch."
            )
        if len(images) == 0:
            return []

        with tempfile.TemporaryDirectory(prefix="visionserve_predict_") as tmp_dir:
            paths = [self._materialize_path(img, Path(tmp_dir), i) for i, img in enumerate(images)]
            return self._predict_batch_paths(paths, image_identifiers)

    def predict_from_bytes(self, data: bytes, identifier: Optional[str] = None) -> PredictionResult:
        """Predict on raw encoded image bytes (any format PIL can decode --
        the temp file is always given a ``.png`` suffix for the extension
        allowlist check, but decoding itself is content-based, not
        extension-based, so this works for any actually-decodable format)."""
        identifier = identifier if identifier is not None else "bytes_input"
        with tempfile.TemporaryDirectory(prefix="visionserve_predict_") as tmp_dir:
            path = Path(tmp_dir) / "image.png"
            path.write_bytes(data)
            return self._predict_batch_paths([str(path)], [identifier])[0]

    def predict_from_numpy(self, array: np.ndarray, identifier: Optional[str] = None) -> PredictionResult:
        """Predict on an in-memory ``HxWxC`` (or ``HxW`` grayscale)
        ``uint8``-convertible array."""
        identifier = identifier if identifier is not None else "numpy_input"
        arr = array
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        return self.predict_from_pil(img, identifier)

    def predict_from_pil(self, image: Image.Image, identifier: Optional[str] = None) -> PredictionResult:
        """Predict on an already-decoded ``PIL.Image.Image``."""
        identifier = identifier if identifier is not None else "pil_input"
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return self.predict_from_bytes(buffer.getvalue(), identifier)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _materialize_path(self, image: ImageInput, tmp_dir: Path, index: int) -> str:
        """Resolve any supported input type to a file path for
        :meth:`_predict_batch_paths`, writing non-path inputs into
        ``tmp_dir`` (caller-managed, cleaned up automatically when the
        ``TemporaryDirectory`` context in :meth:`predict_batch` exits)."""
        if isinstance(image, str):
            return image
        if isinstance(image, (bytes, bytearray)):
            path = tmp_dir / f"image_{index}.png"
            path.write_bytes(bytes(image))
            return str(path)
        if isinstance(image, np.ndarray):
            arr = image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(arr)
        elif isinstance(image, Image.Image):
            pil_img = image
        else:
            raise TypeError(
                f"Unsupported image input type at index {index}: {type(image)!r}. "
                f"Expected str (file path), bytes, numpy.ndarray, or PIL.Image.Image."
            )
        path = tmp_dir / f"image_{index}.png"
        pil_img.save(path, format="PNG")
        return str(path)

    def _predict_batch_paths(self, image_paths: List[str], image_identifiers: List[str]) -> List[PredictionResult]:
        """Core pipeline: validate -> decode -> preprocess -> batch ->
        ``RuntimeFactory``-built runtime forward pass -> threshold ->
        structured prediction. Structurally parallel to
        ``InferenceEngine.predict_batch`` (Stage 4, lines ~102-174); see
        module docstring for why it isn't calling that frozen method
        directly."""
        cfg = self.model_service.preprocessing_config

        results: List[Optional[PredictionResult]] = [None] * len(image_paths)
        valid_indices: List[int] = []
        valid_tensors: List[torch.Tensor] = []

        for i, (path, identifier) in enumerate(zip(image_paths, image_identifiers)):
            img, decode_error = validate_and_decode_image(path, cfg.channels)
            if decode_error is not None:
                self.logger.warning(
                    "PREDICTION reject identifier=%s reason=%s: %s",
                    identifier, decode_error.error_type, decode_error.message,
                )
                results[i] = build_error_result(identifier, decode_error)
                continue

            tensor, preprocess_error = preprocess_image(img, cfg)
            if preprocess_error is not None:
                self.logger.warning(
                    "PREDICTION reject identifier=%s reason=%s: %s",
                    identifier, preprocess_error.error_type, preprocess_error.message,
                )
                results[i] = build_error_result(identifier, preprocess_error)
                continue

            valid_indices.append(i)
            valid_tensors.append(tensor)

        if valid_tensors:
            batch_tensor = build_batch_tensor(valid_tensors, self.model_service.device)
            runtime = self.runtime_service.get_runtime()
            logits = runtime.predict(batch_tensor)

            class_names = self.model_service.class_names()
            expected_shape = (len(valid_tensors), len(class_names))
            probabilities = apply_sigmoid(logits, expected_shape)

            timestamp = datetime.now(timezone.utc).isoformat()
            model_registry = self.model_service.model_registry
            thresholds = self.model_service.thresholds()

            for local_idx, global_idx in enumerate(valid_indices):
                identifier = image_identifiers[global_idx]
                results[global_idx] = build_prediction_result(
                    identifier=identifier,
                    prob_row=probabilities[local_idx],
                    class_names=class_names,
                    thresholds=thresholds,
                    timestamp=timestamp,
                    model_fingerprint=model_registry.checkpoint_sha256,
                    model_version=self.model_service.model_version,
                )

        return [r for r in results if r is not None]
