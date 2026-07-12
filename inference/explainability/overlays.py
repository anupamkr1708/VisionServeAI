"""
Low-level array utilities for explainability visualization: normalization,
resizing, alpha blending, and uint8 conversion.

Migrated from Sprint 05 Stage 6, where these were private helper methods
inline on ``ExplainabilityEngine`` (``_normalize_map``, the resize branch
inside ``overlay_heatmap``, and the denormalization inside
``_to_display_rgb``). Extracted here as free functions -- the "resize,
blend, normalize, convert_to_uint8" primitives this phase's module layout
calls for -- so ``visualization.py`` composes them instead of inlining
array math, and so no algorithm module needs to reimplement normalization
itself.

Source: sprint05-deployment.ipynb, Stage 6, lines ~365-371 (``_to_display_rgb``),
~391-397 (``_normalize_map``), ~419-430 (``overlay_heatmap``'s resize branch).
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
from PIL import Image


def normalize_map(x: np.ndarray) -> np.ndarray:
    """Min-max normalize an attribution map to ``[0, 1]``, with NaN/Inf
    sanitized to 0 first and a degenerate (near-constant) map collapsed to
    all-zeros rather than dividing by ~0.

    Verbatim port of Stage 6's ``ExplainabilityEngine._normalize_map``
    (notebook lines ~391-397).
    """
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    x_min, x_max = float(x.min()), float(x.max())
    if (x_max - x_min) < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - x_min) / (x_max - x_min)).astype(np.float32)


def resize_to_shape(heatmap_2d: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a ``[0, 1]``-normalized 2-D heatmap to ``(height, width)`` if
    it doesn't already match, via bilinear PIL resampling. No-op (returns
    the input unchanged) if the shape already matches.

    Extracted from the resize branch inline in Stage 6's
    ``overlay_heatmap`` (notebook lines ~422-427) -- identical
    resize-then-rescale-to-``[0, 1]`` behaviour.
    """
    if heatmap_2d.shape == (height, width):
        return heatmap_2d
    resized = Image.fromarray((np.clip(heatmap_2d, 0, 1) * 255).astype(np.uint8)).resize(
        (width, height), Image.BILINEAR,
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


def alpha_blend(colored_rgb_uint8: np.ndarray, base_rgb_uint8: np.ndarray, alpha: float) -> np.ndarray:
    """Alpha-blend a colorized heatmap over a base RGB image:
    ``alpha * colored + (1 - alpha) * base``, clipped to a valid uint8
    image. Verbatim port of the blend line inline in Stage 6's
    ``overlay_heatmap`` (notebook lines ~428-430)."""
    blended = alpha * colored_rgb_uint8.astype(np.float32) + (1 - alpha) * base_rgb_uint8.astype(np.float32)
    return to_uint8(blended, already_scaled=True)


def to_uint8(array: np.ndarray, already_scaled: bool = False) -> np.ndarray:
    """Convert a float array to a valid ``uint8`` image array.

    Args:
        array: Input array. If ``already_scaled`` is ``False`` (default),
            values are assumed to be in ``[0, 1]`` and are scaled by 255
            before clipping; if ``True``, values are assumed to already be
            in the ``[0, 255]`` range (e.g. the output of an alpha blend)
            and are only clipped.
    """
    if not already_scaled:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def denormalize_to_display_rgb(tensor: torch.Tensor, mean: List[float], std: List[float]) -> np.ndarray:
    """Invert preprocessing normalization to recover a displayable ``uint8``
    RGB image from an already-normalized ``(C, H, W)`` tensor.

    Verbatim port of Stage 6's ``ExplainabilityEngine._to_display_rgb``
    (notebook lines ~365-371).
    """
    mean_t = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
    denorm = (tensor.detach().cpu() * std_t + mean_t).clamp(0, 1)
    array = denorm.permute(1, 2, 0).numpy()
    return to_uint8(array, already_scaled=False)
