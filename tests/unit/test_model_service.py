"""Unit tests for services/model_service.py."""
from __future__ import annotations

import pytest
import torch

from services.artifact_service import ArtifactService
from services.model_service import ModelService


@pytest.fixture()
def model_service(synthetic_fixture, tmp_path, logger) -> ModelService:
    artifact_service = ArtifactService(
        artifact_roots=synthetic_fixture["artifact_roots"], export_dir=tmp_path / "export",
        device=torch.device("cpu"), logger=logger,
    )
    return ModelService(artifact_service=artifact_service, device=torch.device("cpu"), logger=logger)


def test_is_loaded_false_before_load_model(model_service):
    assert model_service.is_loaded is False


def test_accessing_model_before_load_raises(model_service):
    with pytest.raises(RuntimeError):
        _ = model_service.model


def test_load_model_populates_all_derived_state(model_service, synthetic_fixture):
    registry = model_service.load_model()
    assert model_service.is_loaded is True
    assert registry.backbone == synthetic_fixture["backbone"]
    assert registry.num_classes == synthetic_fixture["num_classes"]

    assert model_service.class_names() == synthetic_fixture["class_names"]
    assert set(model_service.thresholds().keys()) == set(synthetic_fixture["class_names"])
    assert isinstance(model_service.model, torch.nn.Module)
    assert model_service.model_registry is registry


def test_preprocessing_config_resolves_to_224x224x3(model_service):
    model_service.load_model()
    cfg = model_service.preprocessing_config
    assert (cfg.channels, cfg.resize_height, cfg.resize_width) == (3, 224, 224)


def test_model_is_in_eval_mode_and_grad_disabled(model_service):
    model_service.load_model()
    assert model_service.model.training is False
    assert all(not p.requires_grad for p in model_service.model.parameters())


def test_model_info_and_metadata_are_json_serializable_dicts(model_service):
    import json

    model_service.load_model()
    json.dumps(model_service.model_info())  # raises if not serializable
    json.dumps(model_service.metadata())


def test_load_model_is_idempotent_by_recomputation(model_service):
    """ModelService.load_model()'s own docstring: 'Idempotent-by-recomputation
    -- calling this again fully reloads.' So the second call is a fresh
    ModelRegistry object, not the cached first one -- but it must be
    content-equal, since nothing about the underlying checkpoint changed
    between calls."""
    first = model_service.load_model()
    second = model_service.load_model()
    assert first == second
    assert first is not second
