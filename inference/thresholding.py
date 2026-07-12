"""
Threshold registry: per-class decision thresholds for multi-label prediction.

``ThresholdRegistry`` and ``load_threshold_registry()`` are migrated from
Sprint 05 **Stage 2** ("Deployment Artifact Registry & Model
Reconstruction") of the archived notebook, not Stage 4 -- Stage 4 only
*consumes* an already-built ``ThresholdRegistry``; the loading and
validation logic that builds one lives in Stage 2. Flagging this
discrepancy explicitly since the brief named Stage 4 as the source of
truth: the functionality requested ("ThresholdRegistry, load
optimal_thresholds.json, threshold validation...") is real and validated,
it just physically lives in a different stage of the notebook than
expected.

``resolve_class_names_and_thresholds()`` is migrated from Stage 4 itself.

Architectural adaptation (behaviour-preserving): the original took
``metadata_registry`` and ``model_registry`` objects and pulled
``disease_metadata`` / ``num_classes`` out of them. Stage 2's
``MetadataRegistry`` hasn't been migrated into this repository yet (out of
scope for this phase), so this function takes ``disease_metadata`` and
``num_classes`` directly. The resolution algorithm is unchanged.

This module performs no prediction of any kind -- load, validate, and look
up only.

Source: sprint05-deployment.ipynb, Stage 2 lines ~693-808 (ThresholdRegistry,
load_threshold_registry); Stage 4 lines ~177-214 (resolve_class_names_and_thresholds).
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from inference.utils.io import load_json


@dataclass
class ThresholdRegistry:
    """Per-class deployment decision thresholds, plus any additional
    calibration values (e.g. Version-3 schema's balanced-accuracy and
    Youden's J variants) preserved as metadata rather than discarded."""

    source_path: Optional[str]
    class_count: int
    class_names: List[str]
    thresholds: Dict[str, float]
    threshold_metadata: Dict[str, Dict[str, float]] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_threshold_registry(
    optimal_thresholds_path: Optional[str],
    expected_class_names: List[str],
    logger: logging.Logger,
) -> ThresholdRegistry:
    """Load and validate ``optimal_thresholds.json``, accepting any of 3
    schema variants observed in practice:

    1. **Version 3** (list of per-class records with f1 / balanced_accuracy /
       Youden's J threshold variants) -- deployment threshold is
       ``f1_optimal_threshold``; every other calibration value is preserved
       in ``threshold_metadata``.
    2. **Flat dict** (``{class_name: threshold}``).
    3. **Wrapped dict** (``{"thresholds": [...], "classes": [...]}``).

    Load + validate only -- **never computes** thresholds. Validation
    errors (duplicate class names, missing/extra classes vs.
    ``expected_class_names``, out-of-range values) are collected into
    ``ThresholdRegistry.validation_errors`` and logged as warnings, but do
    **not** raise -- this function always returns a registry; it's the
    caller's decision whether accumulated validation errors are fatal for
    their use case. This is a deliberate original design choice, preserved
    exactly (see the original notebook's own "load + validate only -- NEVER
    compute" comment).

    If ``optimal_thresholds_path`` is ``None`` (not discovered during
    artifact discovery), returns an intentionally empty registry with a
    validation error recorded, rather than raising.

    Raises:
        RuntimeError: only for a genuinely unrecognized/malformed JSON
            schema, or Version-3 records with duplicate class names or
            missing required keys -- these indicate corrupt input, not a
            calibration disagreement, so they still fail fast.
    """
    if optimal_thresholds_path is None:
        logger.warning("THRESHOLDS optimal_thresholds.json not discovered by Stage 1; registry will be empty.")
        return ThresholdRegistry(
            source_path=None, class_count=0, class_names=[], thresholds={}, threshold_metadata={},
            validation_errors=["optimal_thresholds.json not available from Stage 1 discovery."],
        )

    raw = load_json(Path(optimal_thresholds_path))

    threshold_metadata: Dict[str, Dict[str, float]] = {}

    if isinstance(raw, list):
        REQUIRED_V3_KEYS = (
            "class_name",
            "f1_optimal_threshold",
            "f1_optimal_value",
            "balanced_accuracy_optimal_threshold",
            "balanced_accuracy_optimal_value",
            "youden_j_optimal_threshold",
            "youden_j_optimal_value",
        )
        thresholds: Dict[str, float] = {}
        for i, record in enumerate(raw):
            if not isinstance(record, dict):
                raise RuntimeError(
                    f"Unrecognized optimal_thresholds.json schema: record at index {i} is not an object."
                )
            missing_keys = [k for k in REQUIRED_V3_KEYS if k not in record]
            if missing_keys:
                raise RuntimeError(
                    f"optimal_thresholds.json Version 3 record at index {i} is missing required "
                    f"keys {missing_keys}: record={record}"
                )
            name = record["class_name"]
            if name in thresholds:
                raise RuntimeError(
                    f"Duplicate class_name '{name}' found in optimal_thresholds.json Version 3 records."
                )
            thresholds[name] = float(record["f1_optimal_threshold"])
            threshold_metadata[name] = {
                "f1_optimal_threshold": float(record["f1_optimal_threshold"]),
                "f1_optimal_value": float(record["f1_optimal_value"]),
                "balanced_accuracy_optimal_threshold": float(record["balanced_accuracy_optimal_threshold"]),
                "balanced_accuracy_optimal_value": float(record["balanced_accuracy_optimal_value"]),
                "youden_j_optimal_threshold": float(record["youden_j_optimal_threshold"]),
                "youden_j_optimal_value": float(record["youden_j_optimal_value"]),
            }
    elif isinstance(raw, dict) and raw and all(isinstance(v, (int, float)) for v in raw.values()):
        thresholds = {k: float(v) for k, v in raw.items()}
    elif isinstance(raw, dict) and "thresholds" in raw:
        names = raw.get("classes") or raw.get("class_names") or expected_class_names
        thresholds = {name: float(v) for name, v in zip(names, raw["thresholds"])}
    else:
        raise RuntimeError(f"Unrecognized optimal_thresholds.json schema: keys={list(raw)[:10]}")

    errors: List[str] = []
    class_names = list(thresholds.keys())

    if len(class_names) != len(set(class_names)):
        errors.append("Duplicate class names found in threshold registry.")

    if expected_class_names:
        missing = set(expected_class_names) - set(class_names)
        extra = set(class_names) - set(expected_class_names)
        if missing:
            errors.append(f"Thresholds missing classes: {sorted(missing)}")
        if extra:
            errors.append(f"Thresholds contain unexpected classes: {sorted(extra)}")

    for name, value in thresholds.items():
        if not (0.0 <= value <= 1.0):
            errors.append(f"Threshold for '{name}' out of range [0,1]: {value}")

    # Version 3 carries additional threshold-valued fields that must independently
    # satisfy the same [0,1] contract as the deployment threshold.
    for name, meta in threshold_metadata.items():
        for key in ("balanced_accuracy_optimal_threshold", "youden_j_optimal_threshold"):
            value = meta[key]
            if not (0.0 <= value <= 1.0):
                errors.append(f"'{key}' for '{name}' out of range [0,1]: {value}")

    for e in errors:
        logger.warning("THRESHOLD VALIDATION: %s", e)

    return ThresholdRegistry(
        source_path=optimal_thresholds_path,
        class_count=len(class_names),
        class_names=class_names,
        thresholds=thresholds,
        threshold_metadata=threshold_metadata,
        validation_errors=errors,
    )


def resolve_class_names_and_thresholds(
    disease_metadata: Dict[str, Any],
    threshold_registry: ThresholdRegistry,
    num_classes: int,
    logger: logging.Logger,
) -> Tuple[List[str], Dict[str, float]]:
    """Resolve the canonical class ordering from ``disease_metadata`` and
    match it against ``threshold_registry``.

    See the module docstring for why this takes ``disease_metadata`` /
    ``num_classes`` directly rather than the original notebook's
    ``metadata_registry`` / ``model_registry`` objects.

    Raises:
        RuntimeError: if no recognized class-list key is found, the class
            count doesn't match ``num_classes``, or (when the registry is
            non-empty) any resolved class name has no threshold.
    """
    class_names: Optional[List[str]] = None
    source_key = None
    for key in ("classes", "class_names", "diseases", "labels"):
        if isinstance(disease_metadata.get(key), list):
            class_names = disease_metadata[key]
            source_key = key
            break

    if class_names is None:
        raise RuntimeError(
            "Unable to resolve canonical class ordering from disease_metadata "
            "(checked keys: classes, class_names, diseases, labels)."
        )
    if len(class_names) != num_classes:
        raise RuntimeError(
            f"disease_metadata['{source_key}'] has {len(class_names)} classes but num_classes={num_classes}."
        )

    if threshold_registry.class_count == 0:
        logger.warning("RUNTIME THRESHOLD_REGISTRY is empty; defaulting every class threshold to 0.5.")
        thresholds = {name: 0.5 for name in class_names}
    else:
        missing = set(class_names) - set(threshold_registry.thresholds.keys())
        if missing:
            raise RuntimeError(f"THRESHOLD_REGISTRY is missing thresholds for classes: {sorted(missing)}")
        thresholds = {name: threshold_registry.thresholds[name] for name in class_names}

    logger.info(
        "RUNTIME class ordering resolved via disease_metadata['%s'] (%d classes).", source_key, len(class_names)
    )
    return class_names, thresholds


def get_threshold(registry: ThresholdRegistry, class_name: str) -> float:
    """Look up a single class's threshold.

    New convenience function -- the original notebook only ever looked up
    thresholds in bulk (via the dict comprehension inside
    ``resolve_class_names_and_thresholds``); no single-class lookup
    function existed. Added to satisfy "per-class threshold retrieval" as
    its own callable unit; does not change any existing behaviour, purely
    additive.

    Raises:
        KeyError: with a clear message if ``class_name`` isn't in the
            registry (rather than a bare ``KeyError`` on the underlying dict).
    """
    if class_name not in registry.thresholds:
        raise KeyError(
            f"No threshold registered for class '{class_name}'. "
            f"Known classes: {registry.class_names}"
        )
    return registry.thresholds[class_name]
