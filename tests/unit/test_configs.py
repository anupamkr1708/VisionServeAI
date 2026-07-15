"""Unit tests for configs/defaults.py and configs/schema.py."""
from __future__ import annotations

from pathlib import Path

from configs import defaults, schema


def test_seed_default_unchanged():
    assert defaults.SEED == 42


def test_kaggle_paths_respect_env_override(monkeypatch):
    monkeypatch.setenv("VISIONSERVE_INPUT_ROOT", "/tmp/custom_input")
    monkeypatch.setenv("VISIONSERVE_OUTPUT_ROOT", "/tmp/custom_output")
    # These constants are resolved at import time -- reimport with a fresh
    # module object to observe the override without mutating global state
    # for every other test.
    import importlib

    reloaded = importlib.reload(defaults)
    try:
        assert reloaded.KAGGLE_INPUT_ROOT == Path("/tmp/custom_input")
        assert reloaded.OUTPUT_ROOT == Path("/tmp/custom_output")
    finally:
        monkeypatch.delenv("VISIONSERVE_INPUT_ROOT", raising=False)
        monkeypatch.delenv("VISIONSERVE_OUTPUT_ROOT", raising=False)
        importlib.reload(defaults)  # restore for subsequent tests


def test_fingerprints_cover_all_four_categories():
    expected_categories = {"sprint03", "sprint04_training", "sprint04_evaluation", "nih_chest_xray"}
    assert set(defaults.FINGERPRINTS.keys()) == expected_categories
    for category, filenames in defaults.FINGERPRINTS.items():
        assert len(filenames) > 0, f"category '{category}' has no fingerprint files"


def test_deployment_config_round_trips_to_dict():
    config = schema.DeploymentConfig(
        device="cpu",
        dtype="float32",
        artifact_roots={"sprint03": "/tmp/sprint03"},
        output_root="/tmp/out",
        logging=schema.LoggingConfig(log_dir="/tmp/logs"),
        benchmark=schema.BenchmarkConfig(),
        export=schema.ExportConfig(),
        api=schema.APIConfig(),
        runtime=schema.RuntimeConfig(),
    )
    as_dict = config.to_dict()
    assert as_dict["device"] == "cpu"
    assert as_dict["artifact_roots"] == {"sprint03": "/tmp/sprint03"}
    assert as_dict["runtime"]["seed"] == defaults.SEED


def test_runtime_config_default_seed_matches_defaults():
    assert schema.RuntimeConfig().seed == defaults.SEED


def test_api_config_reserved_by_default():
    # backend/ isn't built yet -- APIConfig.reserved should still default
    # True, matching the notebook's Stage 1 intent (see schema.py docstring).
    assert schema.APIConfig().reserved is True
