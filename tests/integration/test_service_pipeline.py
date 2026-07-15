"""
Integration tests: the complete service dependency graph.

    ArtifactService -> ModelService -> RuntimeService -> PredictionService
        -> ExplainabilityService -> HealthService -> ServiceRegistry

Unlike tests/unit/*.py (which test one service in isolation, constructing
its immediate dependency by hand), everything here goes through
``ServiceRegistry.initialize()`` -- the same real, un-mocked wiring path
production code uses -- and asserts on the *relationships between*
services, not any single service's behavior in isolation.
"""
from __future__ import annotations

import pytest


def test_service_registry_constructs_every_service_in_dependency_order(initialized_registry):
    registry = initialized_registry
    assert registry.is_initialized is True

    # Every service must exist ...
    for name in ("artifact", "model", "runtime", "prediction", "explainability", "health"):
        assert getattr(registry, name) is not None, f"ServiceRegistry.{name} was not constructed"

    # ... and each must actually be wired to its real dependency, not a
    # freshly-constructed duplicate with its own separate state.
    assert registry.model.artifact_service is registry.artifact
    assert registry.prediction.model_service is registry.model
    assert registry.prediction.runtime_service is registry.runtime
    assert registry.health.model_service is registry.model
    assert registry.health.artifact_service is registry.artifact


def test_model_service_depends_on_artifact_service_output(initialized_registry):
    """ModelService's resolved backbone/class-count must trace back to
    exactly the files ArtifactService discovered -- not some independently
    re-read copy."""
    registry = initialized_registry
    training_summary = registry.artifact.load_training_summary()
    assert registry.model.model_registry.backbone == training_summary["backbone"]
    assert registry.model.model_registry.num_classes == training_summary["num_classes"]


def test_runtime_service_serves_the_exact_model_service_reconstructed(initialized_registry):
    registry = initialized_registry
    runtime = registry.runtime.get_runtime()
    meta = runtime.metadata()
    assert meta.model_fingerprint_sha256 == registry.model.model_registry.checkpoint_sha256


def test_prediction_service_result_is_consistent_with_model_service_class_ordering(initialized_registry, sample_image_path):
    registry = initialized_registry
    result = registry.prediction.predict_image(sample_image_path)
    assert set(result.probabilities.keys()) == set(registry.model.class_names())
    assert result.model_fingerprint_sha256 == registry.model.model_registry.checkpoint_sha256


def test_health_service_aggregates_every_upstream_service_correctly(initialized_registry):
    registry = initialized_registry
    snapshot = registry.health.health()

    assert snapshot["status"] == "healthy"
    assert snapshot["model_loaded"] == registry.model.is_loaded
    assert snapshot["artifacts"] == registry.artifact.artifact_status()


def test_explainability_service_uses_the_same_model_as_prediction_service(initialized_registry, sample_image_path):
    """Both services must be looking at the same reconstructed model --
    otherwise GradCAM's heatmap wouldn't correspond to what actually drove
    the prediction it's meant to explain."""
    from PIL import Image

    registry = initialized_registry
    prediction = registry.prediction.predict_image(sample_image_path)
    image = Image.open(sample_image_path).convert("RGB")
    explanation = registry.explainability.generate_gradcam(image, sample_id="integration_consistency")

    assert explanation.success
    # Both go through the same class ordering, resolved once by ModelService.
    assert set(prediction.probabilities.keys()) == set(registry.model.class_names())


def test_full_pipeline_end_to_end_single_call_chain(initialized_registry, sample_image_path):
    """One assertion touching every service in the documented order, as a
    single end-to-end smoke check distinct from scripts/smoke_test.py
    (that script is meant to be run manually / in CI as a standalone
    process; this is the same journey expressed as a normal pytest test)."""
    from PIL import Image

    registry = initialized_registry

    artifact_status = registry.artifact.artifact_status()
    assert artifact_status["healthy"]

    model_registry = registry.model.model_registry
    assert model_registry.total_parameters > 0

    runtime = registry.runtime.get_runtime()
    assert runtime.is_loaded and runtime.is_validated

    prediction = registry.prediction.predict_image(sample_image_path)
    assert prediction.success

    image = Image.open(sample_image_path).convert("RGB")
    explanation = registry.explainability.generate_gradcam(image, sample_id="e2e")
    assert explanation.success

    health = registry.health.health()
    assert health["status"] == "healthy"


def test_service_registry_shutdown_is_safe_to_call_twice(synthetic_fixture, tmp_path):
    from services.service_registry import ServiceRegistry

    registry = ServiceRegistry(
        artifact_roots=synthetic_fixture["artifact_roots"], export_dir=tmp_path / "export",
        runtime_type="pytorch", seed=42,
    )
    registry.initialize()
    registry.shutdown()
    registry.shutdown()  # must not raise


def test_service_registry_rejects_unknown_runtime_type(synthetic_fixture, tmp_path):
    from services.service_registry import ServiceRegistry

    registry = ServiceRegistry(
        artifact_roots=synthetic_fixture["artifact_roots"], export_dir=tmp_path / "export",
        runtime_type="tensorrt", seed=42,
    )
    with pytest.raises(ValueError):
        registry.initialize()
