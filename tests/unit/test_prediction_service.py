"""Unit tests for services/prediction_service.py.

Uses the fully-initialized ``ServiceRegistry`` fixture rather than
constructing ``PredictionService`` by hand -- it depends on a loaded
``ModelService`` and an initialized ``RuntimeService``, so building those
correctly here would just re-implement ``ServiceRegistry.initialize()``'s
own wiring order. Consistent with the smoke test's "never bypass the
service layer" rule.
"""
from __future__ import annotations

from inference.postprocessing import top_k_predictions


def test_predict_image_returns_successful_result(initialized_registry, sample_image_path):
    result = initialized_registry.prediction.predict_image(sample_image_path)
    assert result.success is True
    assert result.error is None
    class_names = initialized_registry.model.class_names()
    assert set(result.probabilities) == set(class_names)
    assert all(0.0 <= p <= 1.0 for p in result.probabilities.values())


def test_predict_image_default_identifier_is_the_path(initialized_registry, sample_image_path):
    result = initialized_registry.prediction.predict_image(sample_image_path)
    assert result.image_identifier == sample_image_path


def test_predict_image_custom_identifier_is_respected(initialized_registry, sample_image_path):
    result = initialized_registry.prediction.predict_image(sample_image_path, image_identifier="my-custom-id")
    assert result.image_identifier == "my-custom-id"


def test_thresholding_predicted_diseases_matches_probability_vs_threshold(initialized_registry, sample_image_path):
    result = initialized_registry.prediction.predict_image(sample_image_path)
    expected = {
        name for name, prob in result.probabilities.items()
        if prob >= result.thresholds_used[name]
    }
    assert set(result.predicted_diseases) == expected


def test_predict_batch_preserves_input_order(initialized_registry, tmp_path):
    from scripts._synthetic_fixtures import build_sample_image

    path_a = str(build_sample_image(tmp_path / "a.png", seed=10))
    path_b = str(build_sample_image(tmp_path / "b.png", seed=20))
    results = initialized_registry.prediction.predict_batch([path_a, path_b])
    assert [r.image_identifier for r in results] == [path_a, path_b]
    assert all(r.success for r in results)


def test_predict_batch_supports_mixed_input_types(initialized_registry, sample_image_path):
    from PIL import Image

    pil_image = Image.open(sample_image_path).convert("RGB")
    with open(sample_image_path, "rb") as f:
        raw_bytes = f.read()

    results = initialized_registry.prediction.predict_batch([sample_image_path, pil_image, raw_bytes])
    assert len(results) == 3
    assert all(r.success for r in results)
    # Same underlying image (path vs PIL.Image vs raw bytes of that same
    # file) must produce identical probabilities -- input-type normalization
    # shouldn't be lossy.
    assert results[0].probabilities == results[1].probabilities == results[2].probabilities


def test_prediction_is_deterministic_across_repeated_calls(initialized_registry, sample_image_path):
    first = initialized_registry.prediction.predict_image(sample_image_path)
    second = initialized_registry.prediction.predict_image(sample_image_path)
    assert first.probabilities == second.probabilities
    assert first.predicted_diseases == second.predicted_diseases


def test_top_k_predictions_sorted_descending_by_confidence(initialized_registry, sample_image_path):
    result = initialized_registry.prediction.predict_image(sample_image_path)
    class_names = initialized_registry.model.class_names()
    top_k = top_k_predictions(result, k=len(class_names))
    confidences = [conf for _, conf in top_k]
    assert confidences == sorted(confidences, reverse=True)
    assert {name for name, _ in top_k} == set(class_names)


def test_predict_on_corrupt_image_returns_failed_result_not_an_exception(initialized_registry, tmp_path):
    bad_path = tmp_path / "not_an_image.png"
    bad_path.write_bytes(b"this is not a valid png file")
    result = initialized_registry.prediction.predict_image(str(bad_path))
    assert result.success is False
    assert result.error is not None
