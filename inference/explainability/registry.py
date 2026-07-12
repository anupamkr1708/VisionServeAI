"""
Explainability registry: supported methods, supported layers, aggregate
metadata, per-method runtime timings, validation status, and capabilities.

Migrated from Sprint 05 Stage 6's ``ExplainabilityEngine.export_results``
(the JSON-building half, notebook lines ~784-843) and
``run_stage06_engineering_validation`` (notebook lines ~850-902). Both were
methods/functions entangled with ``ExplainabilityEngine`` itself in the
notebook; extracted here as free functions operating on already-computed
data (method availability, results, layer facts) so ``engine.py`` composes
them instead of building JSON payloads inline, and so this bookkeeping
doesn't have to be duplicated if a future surface (e.g. the FastAPI health
endpoint) wants the same registry snapshot without a full export.

No behaviour differs: every JSON payload produced here has the exact same
keys, computed the exact same way, as Stage 6's originals.

Source: sprint05-deployment.ipynb, Stage 6, lines ~784-843, ~850-902.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from inference.explainability.base import ExplainabilityMetadata, ExplainabilityResult, METHOD_NAMES


def build_method_registry(method_availability: Dict[str, ExplainabilityMetadata]) -> Dict[str, Any]:
    """``method_registry.json`` payload: every method's availability + reason.
    Verbatim port of Stage 6's inline dict in ``export_results`` (notebook
    line ~825-827)."""
    return {k: v.to_dict() for k, v in method_availability.items()}


def build_layer_registry(
    layer_name: str, layer_type: str, feature_dim: int, activation_size: Optional[List[int]], backbone: str,
) -> Dict[str, Any]:
    """``layer_registry.json`` payload: the single auto-discovered target
    layer's identity. Verbatim port of Stage 6's inline dict in
    ``export_results`` (notebook lines ~816-823)."""
    return {
        "selected_layer": layer_name,
        "layer_type": layer_type,
        "feature_dim": feature_dim,
        "activation_size": activation_size,
        "backbone": backbone,
        "discovery_method": "last_nn.Conv2d_via_named_modules",
    }


def build_gradcam_metadata(
    layer_name: str, feature_dim: int, activation_size: Optional[List[int]], results: List[ExplainabilityResult],
) -> Dict[str, Any]:
    """``gradcam_metadata.json`` payload. Verbatim port of Stage 6's inline
    dict in ``export_results`` (notebook lines ~808-814)."""
    gradcam_results = [r.to_dict() for r in results if r.method == "gradcam"]
    return {
        "layer_name": layer_name,
        "feature_dim": feature_dim,
        "activation_size": activation_size,
        "results": gradcam_results,
    }


def build_performance_summary(results: List[ExplainabilityResult]) -> Dict[str, Any]:
    """``performance_summary.json`` payload: per-method count/mean/min/max
    execution time in milliseconds, over successful results only. Verbatim
    port of Stage 6's inline loop in ``export_results`` (notebook lines
    ~829-838)."""
    perf: Dict[str, Any] = {}
    for method_name in METHOD_NAMES:
        times = [r.execution_time_ms for r in results if r.method == method_name and r.success]
        perf[method_name] = {
            "count": len(times),
            "mean_ms": float(np.mean(times)) if times else None,
            "min_ms": float(np.min(times)) if times else None,
            "max_ms": float(np.max(times)) if times else None,
        }
    return perf


def build_explainability_summary(
    method_availability: Dict[str, ExplainabilityMetadata],
    layer_name: str,
    layer_type: str,
    feature_dim: int,
    activation_size: Optional[List[int]],
    results: List[ExplainabilityResult],
    class_names: List[str],
    model_version: Optional[str],
    model_fingerprint: Optional[str],
    backbone: str,
) -> Dict[str, Any]:
    """``explainability_summary.json`` payload: the top-level summary
    returned by ``export_results``. Verbatim port of Stage 6's inline dict
    (notebook lines ~789-805)."""
    return {
        "methods_implemented": list(METHOD_NAMES),
        "methods_available": {k: v.available for k, v in method_availability.items()},
        "methods_disabled": {k: v.reason for k, v in method_availability.items() if not v.available},
        "layer_name": layer_name,
        "layer_type": layer_type,
        "feature_dim": feature_dim,
        "activation_size": activation_size,
        "total_results": len(results),
        "successful_results": sum(1 for r in results if r.success),
        "failed_results": sum(1 for r in results if not r.success),
        "class_names": class_names,
        "num_classes": len(class_names),
        "model_version": model_version,
        "model_fingerprint_sha256": model_fingerprint,
        "backbone": backbone,
    }


def run_engineering_validation(
    method_availability: Dict[str, ExplainabilityMetadata],
    target_layer_found: bool,
    activation_size: Optional[List[int]],
    results: List[ExplainabilityResult],
    cv2_available: bool,
    expected_overlay_size: Optional[tuple],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Aggregate engineering validation over a batch of results: layer
    discovery, forward/backward hook health, heatmap dimensionality,
    NaN/Inf cleanliness, overlay size/colorization, timing, and per-image
    failure isolation.

    Verbatim port of Stage 6's ``run_stage06_engineering_validation``
    (notebook lines ~850-902), parameterized to take already-known facts
    (layer discovery success, activation size, colorization backend) rather
    than a live ``ExplainabilityEngine`` instance, so it can be called
    without importing ``engine.py`` (avoiding an import cycle).
    """
    checks: Dict[str, bool] = {}
    warnings: List[str] = []

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    checks["layer_found"] = target_layer_found
    checks["forward_hook_works"] = activation_size is not None
    checks["backward_hook_works"] = any(r.success for r in results if r.method in ("gradcam", "gradcam_plus"))
    checks["heatmap_dimensions_correct"] = all(
        r.heatmap_shape is not None and len(r.heatmap_shape) == 2 for r in successful
    )
    checks["no_nan"] = all(r.no_nan for r in successful) if successful else True
    checks["no_inf"] = all(r.no_inf for r in successful) if successful else True

    overlay_size_ok = True
    if successful and expected_overlay_size is not None:
        try:
            sample = successful[0]
            with Image.open(sample.overlay_path) as im:
                overlay_size_ok = im.size == expected_overlay_size
        except Exception as exc:  # noqa: BLE001
            overlay_size_ok = False
            warnings.append(f"Could not verify overlay size: {exc}")
    checks["overlay_size_correct"] = overlay_size_ok

    checks["color_map_valid"] = True  # cv2 -> matplotlib -> manual fallback chain always yields a valid RGB map
    checks["execution_time_measured"] = all(r.execution_time_ms is not None and r.execution_time_ms >= 0 for r in results)
    checks["every_image_saved"] = all(
        r.heatmap_path is not None and Path(r.heatmap_path).exists()
        and r.overlay_path is not None and Path(r.overlay_path).exists()
        for r in successful
    ) if successful else False

    disabled = [k for k, v in method_availability.items() if not v.available]
    checks["failures_isolated"] = True  # by construction: every method call is wrapped in try/except

    if failed:
        warnings.append(f"{len(failed)} method execution(s) failed but were isolated; pipeline continued.")
    if disabled:
        warnings.append(f"Method(s) disabled due to unavailable capability: {disabled}")
    if not cv2_available:
        warnings.append("cv2 unavailable; colorization used matplotlib or manual fallback instead.")

    fatal_errors: List[str] = []
    if not successful:
        fatal_errors.append("No method execution succeeded across any sample.")

    for check_name, ok in checks.items():
        if not ok:
            logger.warning("ENGINEERING CHECK FAILED: %s", check_name)

    passed = len(fatal_errors) == 0 and all(checks.values())
    return {"checks": checks, "warnings": warnings, "fatal_errors": fatal_errors, "passed": passed}
