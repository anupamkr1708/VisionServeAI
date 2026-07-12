"""
Heatmap + overlay rendering and artifact saving.

Migrated from Sprint 05 Stage 6's ``ExplainabilityEngine.overlay_heatmap``
and ``ExplainabilityEngine.save_visualization`` (notebook lines ~419-436),
plus the artifact-saving sequence inline in ``_run_method`` (colorize ->
overlay -> save heatmap PNG -> save overlay PNG -> save raw ``.npy``,
notebook lines ~683-693). This module composes the low-level primitives in
``colormaps.py`` (colorization) and ``overlays.py`` (resize/blend/normalize)
into the two public entry points every algorithm module needs, so none of
them re-implements colorization, blending, or file I/O.

Source: sprint05-deployment.ipynb, Stage 6, lines ~403-436, ~683-693.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from inference.explainability import colormaps, overlays

DEFAULT_ALPHA: float = 0.45


def overlay_heatmap(
    base_rgb_uint8: np.ndarray,
    heatmap_2d: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    colormap: str = colormaps.DEFAULT_COLORMAP,
) -> np.ndarray:
    """Alpha-blend a colorized heatmap onto a base RGB display image,
    resizing the heatmap first if its shape doesn't match the base image.

    Public reusable primitive -- verbatim port of Stage 6's
    ``ExplainabilityEngine.overlay_heatmap`` (notebook lines ~419-430),
    same default ``alpha=0.45``.
    """
    height, width = base_rgb_uint8.shape[:2]
    resized = overlays.resize_to_shape(heatmap_2d, height, width)
    colored = colormaps.colorize(resized, colormap=colormap)
    return overlays.alpha_blend(colored, base_rgb_uint8, alpha)


def save_visualization(array_uint8: np.ndarray, path: Path) -> None:
    """Save an RGB ``uint8`` array to disk as a PNG, creating the parent
    directory if needed. Verbatim port of Stage 6's
    ``ExplainabilityEngine.save_visualization`` (notebook lines ~432-436)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array_uint8).save(path)


def render_method_artifacts(
    base_rgb_uint8: np.ndarray,
    attribution_2d: np.ndarray,
    heatmap_path: Path,
    overlay_path: Path,
    raw_path: Path,
    alpha: float = DEFAULT_ALPHA,
    colormap: str = colormaps.DEFAULT_COLORMAP,
) -> None:
    """Render and save the three artifacts every explainability method
    produces for one (image, method) pair: a colorized heatmap PNG, an
    overlay PNG, and the raw (normalized) attribution as a ``.npy`` array.

    Composes :func:`overlay_heatmap` / :func:`save_visualization` and
    ``numpy.save`` in the exact sequence Stage 6's ``_run_method`` used
    inline (notebook lines ~683-693) -- one call site instead of that
    5-statement block being repeated per algorithm module.
    """
    colorized = colormaps.colorize(attribution_2d, colormap=colormap)
    overlay = overlay_heatmap(base_rgb_uint8, attribution_2d, alpha=alpha, colormap=colormap)

    save_visualization(colorized, heatmap_path)
    save_visualization(overlay, overlay_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(raw_path, attribution_2d.astype(np.float32))
