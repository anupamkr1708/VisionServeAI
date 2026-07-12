"""
Colormap backend detection and heatmap colorization.

Migrated from Sprint 05 Stage 6's colorization backend detection
(``_CV2_AVAILABLE`` / ``_MATPLOTLIB_AVAILABLE`` module-level probes) and its
``ExplainabilityEngine._colorize`` method (notebook lines ~35-47, ~403-417):
a three-tier fallback chain -- OpenCV's ``applyColorMap`` first, Matplotlib's
colormap second, a dependency-free manual jet-style approximation last --
that guarantees a valid RGB colorization no matter which optional imaging
libraries happen to be installed.

Behaviour-preserving extension: the notebook only ever colorized with JET
(hardcoded ``cv2.COLORMAP_JET`` / matplotlib's ``"jet"``). This module keeps
``"jet"`` as the default for every call site (byte-identical output to
Stage 6 unless a caller explicitly asks for something else) while adding
``"hot"`` and ``"viridis"`` as selectable alternatives through the same
cv2 -> matplotlib -> manual fallback chain, per this phase's file-layout
request ("Provide OpenCV colormaps, Matplotlib fallback, JET, HOT,
VIRIDIS"). The manual (no-cv2-no-matplotlib) fallback only has a jet-style
implementation -- exactly as Stage 6 did -- so requesting "hot"/"viridis"
without either library falls back to the jet approximation with a logged
warning, rather than inventing a new manual gradient with no notebook
precedent.

Source: sprint05-deployment.ipynb, Stage 6, lines ~35-47, ~273-279, ~403-417.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as _mpl_cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

DEFAULT_COLORMAP: str = "jet"
SUPPORTED_COLORMAPS = ("jet", "hot", "viridis")

_CV2_COLORMAP_CONSTANTS = {
    "jet": "COLORMAP_JET",
    "hot": "COLORMAP_HOT",
    "viridis": "COLORMAP_VIRIDIS",
}
_MATPLOTLIB_COLORMAP_NAMES = {
    "jet": "jet",
    "hot": "hot",
    "viridis": "viridis",
}


def log_backend_selection(logger: logging.Logger) -> str:
    """Log (once, at engine construction) which colorization backend is
    active -- verbatim port of Stage 6's constructor-time backend log
    (notebook lines ~273-279). Returns the backend name for callers that
    want it (``"cv2"``, ``"matplotlib"``, or ``"manual"``)."""
    if CV2_AVAILABLE:
        logger.info("COLORMAP backend=cv2 (COLORMAP_JET)")
        return "cv2"
    if MATPLOTLIB_AVAILABLE:
        logger.warning("COLORMAP cv2 unavailable; falling back to matplotlib 'jet' colormap.")
        return "matplotlib"
    logger.warning("COLORMAP cv2 and matplotlib unavailable; using manual jet-style fallback.")
    return "manual"


def _manual_jet(h_u8: np.ndarray) -> np.ndarray:
    """Dependency-free 3-stop blue -> green -> red approximation of the jet
    colormap. Verbatim port of Stage 6's manual fallback branch (notebook
    lines ~412-417)."""
    n = h_u8.astype(np.float32) / 255.0
    r = np.clip(1.5 - np.abs(2.0 * n - 1.5) * 2.0, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * n - 1.0) * 2.0, 0.0, 1.0)
    b = np.clip(1.5 - np.abs(2.0 * n - 0.5) * 2.0, 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def colorize(heatmap_2d: np.ndarray, colormap: str = DEFAULT_COLORMAP, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Colorize a ``[0, 1]``-normalized 2-D heatmap into an ``(H, W, 3)``
    ``uint8`` RGB array.

    Args:
        heatmap_2d: Normalized attribution map.
        colormap: One of :data:`SUPPORTED_COLORMAPS`. Defaults to
            ``"jet"`` -- Stage 6's only colormap, preserved as the default
            so every existing call site's output is unchanged.
        logger: Optional logger for a fallback warning when a non-jet
            colormap is requested but neither cv2 nor matplotlib is
            available.

    Returns:
        ``(H, W, 3)`` ``uint8`` RGB array. Byte-identical to Stage 6's
        ``_colorize`` when ``colormap="jet"`` (the default).
    """
    h_u8 = (np.clip(heatmap_2d, 0.0, 1.0) * 255.0).astype(np.uint8)

    if CV2_AVAILABLE:
        const_name = _CV2_COLORMAP_CONSTANTS.get(colormap, _CV2_COLORMAP_CONSTANTS["jet"])
        colored = cv2.applyColorMap(h_u8, getattr(cv2, const_name))
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    if MATPLOTLIB_AVAILABLE:
        cmap_name = _MATPLOTLIB_COLORMAP_NAMES.get(colormap, _MATPLOTLIB_COLORMAP_NAMES["jet"])
        cmap = _mpl_cm.get_cmap(cmap_name)
        colored = cmap(h_u8.astype(np.float32) / 255.0)[..., :3]
        return (colored * 255.0).astype(np.uint8)

    if colormap != "jet" and logger is not None:
        logger.warning(
            "COLORMAP '%s' requested but cv2/matplotlib unavailable; using manual jet-style fallback instead.",
            colormap,
        )
    return _manual_jet(h_u8)
