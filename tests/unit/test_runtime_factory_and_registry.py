"""Unit tests for inference/runtimes/runtime_factory.py and runtime_registry.py."""
from __future__ import annotations

import pytest
import torch

from inference.runtimes.pytorch_runtime import PyTorchRuntime
from inference.runtimes.runtime_factory import RuntimeFactory
from inference.runtimes.runtime_registry import (
    VALIDATION_STATE_NOT_VALIDATED,
    VALIDATION_STATE_PASSED,
    RuntimeRegistry,
)


def test_supported_runtimes_lists_all_three_backends():
    assert RuntimeFactory.supported_runtimes() == ["onnx", "pytorch", "torchscript"]


def test_create_pytorch_runtime_via_factory(reconstructed_model, logger):
    model, registry = reconstructed_model
    runtime = RuntimeFactory.create(
        "pytorch", model=model, device=torch.device("cpu"),
        model_fingerprint_sha256=registry.checkpoint_sha256, logger=logger,
    )
    assert isinstance(runtime, PyTorchRuntime)
    assert runtime.RUNTIME_TYPE == "pytorch"
    assert not runtime.is_loaded  # create() must not implicitly load()


def test_create_unknown_runtime_type_raises_value_error(logger):
    with pytest.raises(ValueError, match="Unknown runtime_type"):
        RuntimeFactory.create("tensorrt", logger=logger)


def test_create_forwards_bad_kwargs_as_plain_type_error(logger):
    # RuntimeFactory deliberately does not add its own validation layer --
    # a missing/unexpected constructor argument should surface as the same
    # TypeError the concrete runtime class itself would raise.
    with pytest.raises(TypeError):
        RuntimeFactory.create("pytorch", logger=logger)  # missing model/device/fingerprint


def _load_and_validate_pytorch_runtime(reconstructed_model, logger):
    model, registry = reconstructed_model
    runtime = RuntimeFactory.create(
        "pytorch", model=model, device=torch.device("cpu"),
        model_fingerprint_sha256=registry.checkpoint_sha256, logger=logger,
    )
    runtime.load()
    dummy = torch.randn(1, 3, 224, 224)
    info = runtime.validate(dummy)
    return runtime, info


def test_runtime_registry_register_and_get_round_trip(reconstructed_model, logger):
    runtime, info = _load_and_validate_pytorch_runtime(reconstructed_model, logger)
    registry = RuntimeRegistry()

    entry = registry.register(runtime, validation_info=info)
    assert entry.runtime_type == "pytorch"
    assert entry.validation_state == VALIDATION_STATE_PASSED
    assert entry.model_fingerprint_sha256 == runtime.metadata().model_fingerprint_sha256

    fetched = registry.get("pytorch")
    assert fetched is entry
    assert registry.all() == {"pytorch": entry}


def test_runtime_registry_records_not_validated_when_validation_info_omitted(reconstructed_model, logger):
    model, model_registry = reconstructed_model
    runtime = RuntimeFactory.create(
        "pytorch", model=model, device=torch.device("cpu"),
        model_fingerprint_sha256=model_registry.checkpoint_sha256, logger=logger,
    )
    runtime.load()
    registry = RuntimeRegistry()

    entry = registry.register(runtime)  # no validation_info passed
    assert entry.validation_state == VALIDATION_STATE_NOT_VALIDATED
    assert entry.validation_errors == []


def test_runtime_registry_get_raises_clear_key_error_for_unregistered_type():
    registry = RuntimeRegistry()
    with pytest.raises(KeyError, match="No runtime registered under 'onnx'"):
        registry.get("onnx")


def test_runtime_registry_overwrites_entry_for_same_runtime_type(reconstructed_model, logger):
    runtime, info = _load_and_validate_pytorch_runtime(reconstructed_model, logger)
    registry = RuntimeRegistry()
    registry.register(runtime, validation_info=info)
    assert len(registry.all()) == 1

    registry.register(runtime, validation_info=info)  # register again
    assert len(registry.all()) == 1  # still one entry, not two
