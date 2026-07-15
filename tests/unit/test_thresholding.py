"""Unit tests for inference/thresholding.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference.thresholding import (
    ThresholdRegistry,
    get_threshold,
    load_threshold_registry,
    resolve_class_names_and_thresholds,
)


def test_load_threshold_registry_missing_path_returns_empty_registry(logger):
    registry = load_threshold_registry(None, expected_class_names=["A"], logger=logger)
    assert registry.class_count == 0
    assert registry.validation_errors  # non-empty, records why


def test_load_threshold_registry_flat_dict_schema(tmp_path: Path, logger):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"A": 0.3, "B": 0.7}))
    registry = load_threshold_registry(str(path), expected_class_names=["A", "B"], logger=logger)
    assert registry.thresholds == {"A": 0.3, "B": 0.7}
    assert registry.validation_errors == []


def test_load_threshold_registry_wrapped_dict_schema(tmp_path: Path, logger):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"classes": ["A", "B"], "thresholds": [0.4, 0.6]}))
    registry = load_threshold_registry(str(path), expected_class_names=["A", "B"], logger=logger)
    assert registry.thresholds == {"A": 0.4, "B": 0.6}


def test_load_threshold_registry_version3_schema(tmp_path: Path, logger):
    record = {
        "class_name": "A",
        "f1_optimal_threshold": 0.42,
        "f1_optimal_value": 0.8,
        "balanced_accuracy_optimal_threshold": 0.5,
        "balanced_accuracy_optimal_value": 0.75,
        "youden_j_optimal_threshold": 0.55,
        "youden_j_optimal_value": 0.6,
    }
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps([record]))
    registry = load_threshold_registry(str(path), expected_class_names=["A"], logger=logger)
    assert registry.thresholds == {"A": pytest.approx(0.42)}
    assert registry.threshold_metadata["A"]["youden_j_optimal_threshold"] == pytest.approx(0.55)


def test_load_threshold_registry_flags_missing_and_extra_classes(tmp_path: Path, logger):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"A": 0.5, "C": 0.5}))  # missing B, extra C
    registry = load_threshold_registry(str(path), expected_class_names=["A", "B"], logger=logger)
    joined = " ".join(registry.validation_errors)
    assert "missing classes" in joined
    assert "unexpected classes" in joined


def test_load_threshold_registry_flags_out_of_range_value(tmp_path: Path, logger):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"A": 1.5}))
    registry = load_threshold_registry(str(path), expected_class_names=["A"], logger=logger)
    assert any("out of range" in e for e in registry.validation_errors)


def test_load_threshold_registry_raises_on_unrecognized_schema(tmp_path: Path, logger):
    path = tmp_path / "thresholds.json"
    # A dict whose values aren't all numeric (fails the flat-dict schema
    # check) and has no "thresholds" key (fails the wrapped-dict schema
    # check) falls through to the genuinely-unrecognized branch. NOTE:
    # {"unexpected_key": 123} is NOT such a case -- an all-numeric-valued
    # dict is syntactically a valid flat-dict schema (just with a
    # nonsensical class name), so it's accepted and only flagged via
    # validation_errors, not raised.
    path.write_text(json.dumps({"unexpected_key": "not-a-number"}))
    with pytest.raises(RuntimeError):
        load_threshold_registry(str(path), expected_class_names=["A"], logger=logger)


def test_load_threshold_registry_raises_on_duplicate_v3_class_name(tmp_path: Path, logger):
    record = {
        "class_name": "A", "f1_optimal_threshold": 0.5, "f1_optimal_value": 0.5,
        "balanced_accuracy_optimal_threshold": 0.5, "balanced_accuracy_optimal_value": 0.5,
        "youden_j_optimal_threshold": 0.5, "youden_j_optimal_value": 0.5,
    }
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps([record, record]))
    with pytest.raises(RuntimeError):
        load_threshold_registry(str(path), expected_class_names=["A"], logger=logger)


def test_resolve_class_names_and_thresholds_defaults_to_half_when_empty(logger):
    empty_registry = ThresholdRegistry(source_path=None, class_count=0, class_names=[], thresholds={})
    class_names, thresholds = resolve_class_names_and_thresholds(
        disease_metadata={"classes": ["A", "B"]}, threshold_registry=empty_registry, num_classes=2, logger=logger,
    )
    assert class_names == ["A", "B"]
    assert thresholds == {"A": 0.5, "B": 0.5}


def test_resolve_class_names_and_thresholds_raises_on_class_count_mismatch(logger):
    empty_registry = ThresholdRegistry(source_path=None, class_count=0, class_names=[], thresholds={})
    with pytest.raises(RuntimeError):
        resolve_class_names_and_thresholds(
            disease_metadata={"classes": ["A", "B"]}, threshold_registry=empty_registry, num_classes=3, logger=logger,
        )


def test_resolve_class_names_and_thresholds_raises_on_unresolvable_class_list(logger):
    empty_registry = ThresholdRegistry(source_path=None, class_count=0, class_names=[], thresholds={})
    with pytest.raises(RuntimeError):
        resolve_class_names_and_thresholds(
            disease_metadata={}, threshold_registry=empty_registry, num_classes=2, logger=logger,
        )


def test_get_threshold_looks_up_single_class():
    registry = ThresholdRegistry(source_path=None, class_count=1, class_names=["A"], thresholds={"A": 0.37})
    assert get_threshold(registry, "A") == pytest.approx(0.37)
