"""
Shared pytest fixtures for the VisionServeAI test suite.

Everything here is built strictly on top of ``scripts._synthetic_fixtures``
(the same fixture builder ``scripts/smoke_test.py --synthetic`` uses) --
no test in this suite depends on the real, proprietary Sprint 04
checkpoint or a real artifact tree being present. Real-artifact
verification is covered separately by ``scripts/validate_artifacts.py``
and ``scripts/smoke_test.py --artifact-root ...``, run manually against
this user's actual files.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pytest


@pytest.fixture()
def logger() -> logging.Logger:
    log = logging.getLogger("visionserve.tests")
    log.setLevel(logging.CRITICAL)  # keep test output clean; assertions carry the signal
    if not log.handlers:
        log.addHandler(logging.NullHandler())
    return log


@pytest.fixture()
def synthetic_fixture(tmp_path: Path) -> Dict[str, Any]:
    """A fresh synthetic artifact tree (flat, ``ArtifactService``-compatible)
    under a per-test temp directory. See
    ``scripts._synthetic_fixtures.build_synthetic_artifact_tree`` for the
    full return-value shape."""
    from scripts._synthetic_fixtures import build_synthetic_artifact_tree

    return build_synthetic_artifact_tree(tmp_path / "artifacts")


@pytest.fixture()
def sample_image_path(tmp_path: Path) -> str:
    from scripts._synthetic_fixtures import build_sample_image

    return str(build_sample_image(tmp_path / "sample.png", seed=1))


@pytest.fixture()
def reconstructed_model(synthetic_fixture: Dict[str, Any], logger: logging.Logger):
    """A real ``(model, ModelRegistry)`` pair -- ``reconstruct_model()``
    run against ``synthetic_fixture``'s checkpoint. Used by runtime-layer
    tests that need an actual ``nn.Module`` without going through
    ``ModelService``/``ServiceRegistry``."""
    import json

    import torch

    from inference.model_loader import reconstruct_model

    training_summary = json.loads(synthetic_fixture["training_summary_path"].read_text())
    disease_registry = json.loads(synthetic_fixture["disease_registry_path"].read_text())
    return reconstruct_model(
        training_summary=training_summary,
        disease_registry=disease_registry,
        checkpoint_path=synthetic_fixture["checkpoint_path"],
        device=torch.device("cpu"),
        logger=logger,
    )


@pytest.fixture()
def initialized_registry(synthetic_fixture: Dict[str, Any], tmp_path: Path):
    """A fully ``.initialize()``-d :class:`~services.service_registry.ServiceRegistry`
    wired to the synthetic fixture (PyTorch runtime only -- see
    ``export_dir`` below, which is deliberately empty, so TorchScript/ONNX
    stay unregistered exactly as they would for a real deployment that
    hasn't run an export step). Shared by service-layer unit tests and the
    integration suite so the same real (not mocked) initialization path is
    exercised everywhere. Calls ``registry.shutdown()`` automatically at
    teardown.
    """
    from services.service_registry import ServiceRegistry

    registry = ServiceRegistry(
        artifact_roots=synthetic_fixture["artifact_roots"],
        export_dir=tmp_path / "export",
        runtime_type="pytorch",
        explainability_output_dir=tmp_path / "explainability",
        seed=42,
    )
    registry.initialize()
    yield registry
    registry.shutdown()
