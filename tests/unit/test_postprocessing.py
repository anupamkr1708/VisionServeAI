"""Unit tests for inference/postprocessing.py."""
from __future__ import annotations

import pytest
import torch

from inference.postprocessing import (
    PredictionResult,
    apply_sigmoid,
    build_error_result,
    build_prediction_result,
    summarize_predictions,
    top_k_predictions,
)
from inference.preprocessing import InferenceError


def test_apply_sigmoid_shape_and_range():
    logits = torch.tensor([[10.0, -10.0, 0.0]])
    probs = apply_sigmoid(logits, expected_shape=(1, 3))
    assert tuple(probs.shape) == (1, 3)
    assert probs[0, 0] > 0.99  # sigmoid(10) ~ 1
    assert probs[0, 1] < 0.01  # sigmoid(-10) ~ 0
    assert pytest.approx(probs[0, 2].item(), abs=1e-6) == 0.5


def test_apply_sigmoid_raises_on_shape_mismatch():
    logits = torch.tensor([[1.0, 2.0]])
    with pytest.raises(RuntimeError):
        apply_sigmoid(logits, expected_shape=(1, 3))


def test_build_prediction_result_applies_thresholds_exactly():
    class_names = ["A", "B", "C"]
    prob_row = torch.tensor([0.9, 0.3, 0.5])
    thresholds = {"A": 0.5, "B": 0.5, "C": 0.5}
    result = build_prediction_result(
        identifier="img1", prob_row=prob_row, class_names=class_names, thresholds=thresholds,
        timestamp="2026-01-01T00:00:00Z", model_fingerprint="abc123", model_version="v1",
    )
    assert result.success is True
    assert set(result.predicted_diseases) == {"A", "C"}  # C == threshold, counts as positive (>=)
    assert result.probabilities == {"A": pytest.approx(0.9), "B": pytest.approx(0.3), "C": pytest.approx(0.5)}
    assert result.confidence_scores.keys() == {"A", "C"}


def test_build_prediction_result_raises_on_out_of_range_probability():
    with pytest.raises(RuntimeError):
        build_prediction_result(
            identifier="img1", prob_row=torch.tensor([1.5]), class_names=["A"], thresholds={"A": 0.5},
            timestamp="2026-01-01T00:00:00Z", model_fingerprint=None, model_version=None,
        )


def test_build_error_result_marks_failure():
    error = InferenceError("missing_file", "not found")
    result = build_error_result("img1", error)
    assert result.success is False
    assert result.error == {"error_type": "missing_file", "message": "not found"}
    assert result.predicted_diseases == []


def test_top_k_predictions_sorted_descending():
    result = PredictionResult(
        image_identifier="img1", success=True,
        probabilities={"A": 0.2, "B": 0.9, "C": 0.5},
    )
    top2 = top_k_predictions(result, k=2)
    assert top2 == [("B", 0.9), ("C", 0.5)]


def test_top_k_predictions_empty_for_failed_result():
    result = build_error_result("img1", InferenceError("missing_file", "x"))
    assert top_k_predictions(result, k=5) == []


def test_summarize_predictions_counts_successes_and_failures():
    success = build_prediction_result(
        identifier="a", prob_row=torch.tensor([0.9]), class_names=["A"], thresholds={"A": 0.5},
        timestamp="t", model_fingerprint=None, model_version=None,
    )
    failure = build_error_result("b", InferenceError("corrupted_image", "bad"))
    summary = summarize_predictions([success, failure])
    assert summary["total"] == 2
    assert summary["successful"] == 1
    assert summary["failed"] == 1
    assert summary["failure_reasons"] == ["corrupted_image"]
    assert summary["disease_frequency"] == {"A": 1}
