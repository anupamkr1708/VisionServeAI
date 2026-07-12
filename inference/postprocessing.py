"""
Postprocessing pipeline: raw model logits -> structured PredictionResult.

Migrated from Sprint 05 Stage 4 of the archived notebook. In the original
notebook, sigmoid conversion, threshold application, and result construction
were inlined inside ``InferenceEngine.predict_batch`` /
``_build_success_result`` / ``_build_error_result``. They're extracted here,
unchanged, so ``engine.py`` can coordinate rather than compute.

``InferenceError`` is imported from ``inference.preprocessing`` (produced
there) rather than duplicated -- this module depends on preprocessing's
error type, never the other way around, so there is no import cycle.

This module performs no image preprocessing and no model loading/checkpoint
handling of any kind.

Source: sprint05-deployment.ipynb, Stage 4, lines ~156-170, 435-484.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from inference.preprocessing import InferenceError

__all__ = [
    "PredictionResult",
    "InferenceError",
    "apply_sigmoid",
    "build_prediction_result",
    "build_error_result",
    "top_k_predictions",
    "summarize_predictions",
]


@dataclass
class PredictionResult:
    """Structured result of one image's inference, success or failure.
    ``to_dict()`` gives the exact JSON-serializable shape the original
    notebook produced (via ``dataclasses.asdict``)."""

    image_identifier: str
    success: bool
    predicted_diseases: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    probabilities: Dict[str, float] = field(default_factory=dict)
    thresholds_used: Dict[str, float] = field(default_factory=dict)
    inference_timestamp_utc: Optional[str] = None
    model_fingerprint_sha256: Optional[str] = None
    model_version: Optional[str] = None
    error: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def apply_sigmoid(logits: torch.Tensor, expected_shape: Tuple[int, int]) -> torch.Tensor:
    """Sigmoid-activate raw model logits into per-class probabilities on
    CPU, and verify the resulting shape.

    Extracted verbatim from the notebook's inline
    ``probabilities = torch.sigmoid(logits).detach().cpu()`` plus the
    immediately-following shape assertion inside ``predict_batch``.

    Raises:
        RuntimeError: if the resulting shape doesn't match
            ``expected_shape`` (``(batch_size, num_classes)``) -- indicates
            a model/architecture mismatch, not a per-image data problem, so
            this still fails the whole batch rather than being caught
            per-image like preprocessing errors are.
    """
    probabilities = torch.sigmoid(logits).detach().cpu()
    if tuple(probabilities.shape) != expected_shape:
        raise RuntimeError(
            f"Model output shape {tuple(probabilities.shape)} does not match expected {expected_shape}."
        )
    return probabilities


def build_prediction_result(
    identifier: str,
    prob_row: torch.Tensor,
    class_names: List[str],
    thresholds: Dict[str, float],
    timestamp: str,
    model_fingerprint: Optional[str],
    model_version: Optional[str],
) -> PredictionResult:
    """Build a successful :class:`PredictionResult`: per-class probability
    range check, threshold comparison, confidence/prediction assembly.

    Extracted verbatim from ``InferenceEngine._build_success_result``.

    Raises:
        RuntimeError: if any probability falls outside ``[0, 1]`` --
            indicates a numerical problem upstream (e.g. a non-sigmoid
            output), not a normal per-image failure mode.
    """
    probabilities_dict: Dict[str, float] = {}
    confidence_scores: Dict[str, float] = {}
    predicted_diseases: List[str] = []

    for class_name, prob in zip(class_names, prob_row.tolist()):
        prob = float(prob)
        if not (0.0 <= prob <= 1.0):
            raise RuntimeError(f"Probability out of range [0,1] for class '{class_name}': {prob}")
        probabilities_dict[class_name] = prob
        threshold = thresholds[class_name]
        if prob >= threshold:
            predicted_diseases.append(class_name)
            confidence_scores[class_name] = prob

    return PredictionResult(
        image_identifier=identifier,
        success=True,
        predicted_diseases=predicted_diseases,
        confidence_scores=confidence_scores,
        probabilities=probabilities_dict,
        thresholds_used=dict(thresholds),
        inference_timestamp_utc=timestamp,
        model_fingerprint_sha256=model_fingerprint,
        model_version=model_version,
        error=None,
    )


def build_error_result(identifier: str, error: InferenceError) -> PredictionResult:
    """Build a failed :class:`PredictionResult` from a structured
    :class:`InferenceError`. Extracted verbatim from
    ``InferenceEngine._build_error_result``."""
    return PredictionResult(image_identifier=identifier, success=False, error=error.to_dict())


# ======================================================================
# ADDITIVE (not present in the original notebook -- see migration report)
# ======================================================================


def top_k_predictions(result: PredictionResult, k: int = 5) -> List[Tuple[str, float]]:
    """Return the top ``k`` classes by probability, highest first.

    New function -- no top-k ranking existed in the original Stage 4
    notebook (it reported all classes above threshold, unranked). This is
    a pure, side-effect-free read over an already-built
    ``PredictionResult.probabilities`` -- it does not change how
    ``PredictionResult`` itself is constructed or what any existing field
    contains, so it cannot affect output compatibility with the notebook.
    Returns an empty list for a failed result (``probabilities`` is empty
    by construction in that case).
    """
    ranked = sorted(result.probabilities.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:k]


def summarize_predictions(results: List[PredictionResult]) -> Dict[str, Any]:
    """Aggregate summary over a batch of results (success/failure counts,
    disease frequency). New function, additive for the same reason as
    :func:`top_k_predictions` -- a pure read over already-built results,
    no existing field or computation touched."""
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]
    disease_counts: Dict[str, int] = {}
    for r in successes:
        for disease in r.predicted_diseases:
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
    return {
        "total": len(results),
        "successful": len(successes),
        "failed": len(failures),
        "failure_reasons": [r.error.get("error_type") for r in failures if r.error],
        "disease_frequency": dict(sorted(disease_counts.items(), key=lambda kv: kv[1], reverse=True)),
    }
