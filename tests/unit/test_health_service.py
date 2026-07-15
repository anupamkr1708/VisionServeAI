"""Unit tests for services/health_service.py."""
from __future__ import annotations


def test_health_reports_healthy_for_a_fully_initialized_registry(initialized_registry):
    snapshot = initialized_registry.health.health()
    assert snapshot["status"] == "healthy"
    assert snapshot["model_loaded"] is True


def test_health_snapshot_has_all_documented_sections(initialized_registry):
    snapshot = initialized_registry.health.health()
    for key in ("status", "model_loaded", "runtime", "gpu", "memory", "providers", "artifacts", "environment"):
        assert key in snapshot, f"missing '{key}' in health() snapshot"


def test_environment_info_matches_inference_utils_environment(initialized_registry):
    from inference.utils.environment import get_environment_info

    reported = initialized_registry.health.environment_info()
    direct = get_environment_info()
    # random_seed will legitimately differ (initialized_registry was seeded
    # with 42 at ServiceRegistry construction time; direct call uses the
    # module default) -- compare everything else, which must be identical
    # since it's the same process.
    reported.pop("random_seed")
    direct.pop("random_seed")
    assert reported == direct


def test_memory_status_reports_positive_ram_total(initialized_registry):
    snapshot = initialized_registry.health.memory_status()
    assert snapshot["ram_total_gb"] > 0
    assert 0.0 <= snapshot["ram_percent"] <= 100.0


def test_gpu_status_reports_unavailable_on_a_cpu_only_runner(initialized_registry):
    snapshot = initialized_registry.health.gpu_status()
    assert "cuda_available" in snapshot
    if not snapshot["cuda_available"]:
        assert snapshot["device_count"] == 0


def test_artifact_status_matches_artifact_services_own_report(initialized_registry):
    from_health = initialized_registry.health.artifact_status()
    from_artifact_service = initialized_registry.artifact.artifact_status()
    assert from_health == from_artifact_service
