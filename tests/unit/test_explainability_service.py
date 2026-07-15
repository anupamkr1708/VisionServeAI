"""Unit tests for services/explainability_service.py."""
from __future__ import annotations

from pathlib import Path

import pytest

ALL_METHODS = [
    "generate_gradcam", "generate_gradcam_plus", "generate_scorecam", "generate_eigencam",
    "generate_guided_backprop", "generate_integrated_gradients", "generate_occlusion",
]


def test_is_available_true_for_a_fully_initialized_registry(initialized_registry):
    assert initialized_registry.explainability.is_available() is True


@pytest.mark.parametrize("method_name", ALL_METHODS)
def test_each_explainability_method_succeeds_and_writes_files(initialized_registry, sample_image_path, method_name):
    from PIL import Image

    image = Image.open(sample_image_path).convert("RGB")
    method = getattr(initialized_registry.explainability, method_name)
    result = method(image, sample_id=f"unit_{method_name}")

    assert result.success is True, result.error
    assert Path(result.heatmap_path).exists()
    assert Path(result.overlay_path).exists()


def test_explainability_service_reports_unavailable_when_disabled(synthetic_fixture, tmp_path):
    from services.service_registry import ServiceRegistry

    registry = ServiceRegistry(
        artifact_roots=synthetic_fixture["artifact_roots"], export_dir=tmp_path / "export",
        runtime_type="pytorch", enable_explainability=False, seed=42,
    )
    registry.initialize()
    try:
        assert registry.explainability.is_available() is False
    finally:
        registry.shutdown()


def test_gradcam_result_is_deterministic_for_the_same_image(initialized_registry, sample_image_path):
    from PIL import Image

    image = Image.open(sample_image_path).convert("RGB")
    first = initialized_registry.explainability.generate_gradcam(image, sample_id="determinism_1")
    second = initialized_registry.explainability.generate_gradcam(image, sample_id="determinism_2")
    assert first.success and second.success
    # Same model, same input, eval mode, no dropout at inference -> the
    # heatmap's underlying values must match even though they're written
    # to two different sample_id-suffixed files.
    import numpy as np
    from PIL import Image as PILImage

    heatmap_1 = np.asarray(PILImage.open(first.heatmap_path))
    heatmap_2 = np.asarray(PILImage.open(second.heatmap_path))
    assert np.array_equal(heatmap_1, heatmap_2)
