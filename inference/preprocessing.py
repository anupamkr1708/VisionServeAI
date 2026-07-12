"""
Preprocessing pipeline: raw image -> validated, normalized tensor.

Migrated from Sprint 05 Stage 4 ("Production Inference Runtime") of the
archived notebook. This module owns everything upstream of the model: image
validation, decoding, RGB conversion, resize, normalization, tensor and
batch construction. It has no knowledge of models, checkpoints, or
predictions.

Architectural adaptation (behaviour-preserving): the original
``resolve_preprocessing_config()`` took ``metadata_registry`` and
``export_registry`` objects (Stage 2's ``MetadataRegistry`` and Stage 3's
export-signature object) and pulled ``input_shape`` /
``training_metadata`` out of them. Neither Stage 2's MetadataRegistry nor
Stage 3's export pipeline has been migrated into this repository yet (out
of scope for this phase), so this function now takes those two pieces of
data directly as parameters. The resolution *algorithm* -- searching
``training_metadata`` for known mean/std keys, one level of nesting, with
an explicit logged fallback to ImageNet defaults -- is unchanged. When
Stage 2/3 are migrated, the future ``deployment/`` or wiring code will
extract ``input_shape`` and ``training_metadata`` from those registries and
pass them here exactly as it does today with test/caller-supplied values.

Source: sprint05-deployment.ipynb, Stage 4, lines ~53-65, 132-153, 217-352.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

try:
    from PIL import UnidentifiedImageError
except ImportError:  # older Pillow versions raise plain OSError instead
    UnidentifiedImageError = OSError

# ======================================================================
# CONSTANTS -- moved from Stage 4 notebook globals, unchanged values.
# ======================================================================

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

# Standard PIL band counts that convert() to RGB unambiguously.
# L=1 (grayscale), RGB=3, RGBA/CMYK=4. Anything else is rejected before
# attempting conversion -- a defensive check for malformed/non-standard
# inputs; ordinary corrupted files are already caught by decode failure.
ALLOWED_SOURCE_BAND_COUNTS = {1, 3, 4}

FALLBACK_IMAGENET_MEAN = [0.485, 0.456, 0.406]
FALLBACK_IMAGENET_STD = [0.229, 0.224, 0.225]


# ======================================================================
# DATACLASSES
# ======================================================================


@dataclass
class PreprocessingConfig:
    """Resolved image preprocessing parameters for one model."""

    resize_height: int
    resize_width: int
    channels: int
    mean: List[float]
    std: List[float]
    resize_source: str
    mean_source: str
    std_source: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceError:
    """Structured error for any pipeline stage that can fail per-image
    without aborting the rest of a batch. Produced by this module
    (validation/decoding/preprocessing failures); also used by
    ``inference.postprocessing`` to build a failed ``PredictionResult`` --
    imported from here rather than duplicated, to avoid a circular
    dependency (postprocessing depends on preprocessing, not vice versa)."""

    error_type: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


# ======================================================================
# CONFIG RESOLUTION
# ======================================================================


def resolve_preprocessing_config(
    input_shape: Tuple[Optional[int], int, int, int],
    training_metadata: Dict[str, Any],
    logger: logging.Logger,
) -> PreprocessingConfig:
    """Resolve resize dimensions (from ``input_shape``, a
    ``[batch, channels, height, width]``-shaped tuple) and normalization
    mean/std (searched in ``training_metadata``, falling back to logged
    ImageNet defaults if not recorded).

    See the module docstring for why this takes raw ``input_shape`` /
    ``training_metadata`` rather than the original notebook's
    ``export_registry`` / ``metadata_registry`` objects.
    """
    channels, height, width = input_shape[1], input_shape[2], input_shape[3]

    def _search(keys: List[str]) -> Tuple[Optional[str], Any]:
        for key in keys:
            val = training_metadata.get(key)
            if val not in (None, ""):
                return key, val
        for v in training_metadata.values():
            if isinstance(v, dict):
                for key in keys:
                    val = v.get(key)
                    if val not in (None, ""):
                        return key, val
        return None, None

    mean_key, mean_val = _search(["normalization_mean", "mean", "image_mean", "normalize_mean", "pixel_mean"])
    std_key, std_val = _search(["normalization_std", "std", "image_std", "normalize_std", "pixel_std"])

    if mean_val is not None and std_val is not None:
        mean = [float(x) for x in mean_val]
        std = [float(x) for x in std_val]
        mean_source = f"training_summary['{mean_key}']"
        std_source = f"training_summary['{std_key}']"
        logger.info("RUNTIME normalization resolved from training_summary.json: mean=%s std=%s", mean, std)
    else:
        mean = list(FALLBACK_IMAGENET_MEAN)
        std = list(FALLBACK_IMAGENET_STD)
        mean_source = "fallback_imagenet_default (no normalization recorded in training_summary.json)"
        std_source = "fallback_imagenet_default (no normalization recorded in training_summary.json)"
        logger.warning(
            "RUNTIME no explicit normalization mean/std found in training_summary.json; "
            "falling back to torchvision ImageNet defaults mean=%s std=%s. This is an explicit, "
            "logged fallback -- not a silent assumption.", mean, std,
        )

    config = PreprocessingConfig(
        resize_height=height, resize_width=width, channels=channels,
        mean=mean, std=std,
        resize_source="input_shape",
        mean_source=mean_source, std_source=std_source,
    )
    logger.info(
        "RUNTIME preprocessing config: resize=(%d,%d) channels=%d mean_source=%s std_source=%s",
        height, width, channels, mean_source, std_source,
    )
    return config


# ======================================================================
# VALIDATION / DECODING / PREPROCESSING
# ======================================================================


def validate_and_decode_image(
    image_path: str, expected_channels: int,
) -> Tuple[Optional[Image.Image], Optional[InferenceError]]:
    """Decode and validate a single image file. Never raises -- every
    failure mode is returned as a structured InferenceError so batch
    processing can reject gracefully without aborting other images."""
    path = Path(image_path)

    if not path.exists() or not path.is_file():
        return None, InferenceError("missing_file", f"Image file does not exist: {image_path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return None, InferenceError(
            "unsupported_extension", f"Unsupported file extension '{path.suffix}' for {image_path}"
        )

    try:
        size = path.stat().st_size
    except OSError as exc:
        return None, InferenceError("corrupted_image", f"Unable to stat file: {exc}")

    if size == 0:
        return None, InferenceError("zero_byte_image", f"Image file is zero bytes: {image_path}")

    try:
        img = Image.open(path)
        img.verify()               # structural integrity check
        img = Image.open(path)     # verify() invalidates the file handle -- must reopen
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return None, InferenceError("corrupted_image", f"Failed to decode image: {exc}")

    source_band_count = len(img.getbands())
    if source_band_count not in ALLOWED_SOURCE_BAND_COUNTS:
        return None, InferenceError(
            "wrong_channel_count",
            f"Decoded image has {source_band_count} band(s); expected one of {sorted(ALLOWED_SOURCE_BAND_COUNTS)}.",
        )

    try:
        img = img.convert("RGB")
    except Exception as exc:  # noqa: BLE001 -- any conversion failure is a channel/format problem
        return None, InferenceError("wrong_channel_count", f"Failed to convert image to RGB: {exc}")

    if len(img.getbands()) != expected_channels:
        return None, InferenceError(
            "wrong_channel_count",
            f"Converted image has {len(img.getbands())} channel(s), expected {expected_channels}.",
        )

    return img, None


def preprocess_image(
    img: Image.Image, config: PreprocessingConfig,
) -> Tuple[Optional[torch.Tensor], Optional[InferenceError]]:
    """Resize -> normalize -> tensor conversion. Returns ``(tensor, None)``
    on success or ``(None, InferenceError)`` on any dimensional
    inconsistency. Purely deterministic -- no random component."""
    try:
        resized = img.resize((config.resize_width, config.resize_height), Image.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0  # H, W, C

        if array.ndim != 3 or array.shape[2] != config.channels:
            return None, InferenceError(
                "invalid_tensor_dimensions", f"Unexpected array shape after resize: {array.shape}"
            )

        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()  # C, H, W
        mean = torch.tensor(config.mean, dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(config.std, dtype=torch.float32).view(-1, 1, 1)
        tensor = (tensor - mean) / std

        expected_shape = (config.channels, config.resize_height, config.resize_width)
        if tuple(tensor.shape) != expected_shape:
            return None, InferenceError(
                "invalid_tensor_dimensions",
                f"Final tensor shape {tuple(tensor.shape)} does not match expected {expected_shape}.",
            )
        return tensor, None
    except Exception as exc:  # noqa: BLE001
        return None, InferenceError("invalid_tensor_dimensions", f"Preprocessing failed: {exc}")


def build_batch_tensor(tensors: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    """Stack per-image tensors into a batch and move to ``device``.

    Extracted from the notebook's inline
    ``torch.stack(valid_tensors, dim=0).to(self.device)`` inside
    ``InferenceEngine.predict_batch`` -- same operation, named and moved
    here so ``engine.py`` doesn't perform tensor construction itself.
    """
    return torch.stack(tensors, dim=0).to(device)
