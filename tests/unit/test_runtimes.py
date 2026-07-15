"""Unit tests for the three concrete BaseRuntime implementations:
PyTorchRuntime, TorchScriptRuntime, ONNXRuntime."""
from __future__ import annotations

import pytest
import torch

from inference.runtimes.onnx_runtime import ONNXRuntime
from inference.runtimes.pytorch_runtime import PyTorchRuntime
from inference.runtimes.torchscript_runtime import TorchScriptRuntime

INPUT_SHAPE = (1, 3, 224, 224)


# ----------------------------------------------------------------------
# PyTorchRuntime
# ----------------------------------------------------------------------


def test_pytorch_runtime_warmup_before_load_raises(reconstructed_model, logger):
    model, registry = reconstructed_model
    runtime = PyTorchRuntime(
        model=model, device=torch.device("cpu"),
        model_fingerprint_sha256=registry.checkpoint_sha256, logger=logger,
    )
    assert not runtime.is_loaded
    with pytest.raises(RuntimeError):
        runtime.warmup(INPUT_SHAPE)


def test_pytorch_runtime_load_predict_validate(reconstructed_model, logger):
    model, registry = reconstructed_model
    runtime = PyTorchRuntime(
        model=model, device=torch.device("cpu"),
        model_fingerprint_sha256=registry.checkpoint_sha256, logger=logger,
    )
    runtime.load()
    assert runtime.is_loaded

    dummy = torch.randn(*INPUT_SHAPE)
    output = runtime.predict(dummy)
    assert tuple(output.shape) == (1, registry.num_classes)

    info = runtime.validate(dummy)
    assert info.validated is True
    assert info.validation_errors == []
    assert runtime.is_validated

    meta = runtime.metadata()
    assert meta.runtime_type == "pytorch"
    assert meta.model_fingerprint_sha256 == registry.checkpoint_sha256


def test_pytorch_runtime_is_deterministic_across_repeated_predict_calls(reconstructed_model, logger):
    model, registry = reconstructed_model
    runtime = PyTorchRuntime(
        model=model, device=torch.device("cpu"),
        model_fingerprint_sha256=registry.checkpoint_sha256, logger=logger,
    )
    runtime.load()
    dummy = torch.randn(*INPUT_SHAPE)
    out1 = runtime.predict(dummy)
    out2 = runtime.predict(dummy)
    assert torch.equal(out1, out2)


def test_pytorch_runtime_warmup_does_not_change_predict_output(reconstructed_model, logger):
    """warmup() is documented as side-effect-only (base.py) -- verify that
    holds: predicting the same input before and after warmup must be
    identical."""
    model, registry = reconstructed_model
    runtime = PyTorchRuntime(
        model=model, device=torch.device("cpu"),
        model_fingerprint_sha256=registry.checkpoint_sha256, logger=logger,
    )
    runtime.load()
    dummy = torch.randn(*INPUT_SHAPE)
    before = runtime.predict(dummy)
    runtime.warmup(INPUT_SHAPE, iterations=3)
    after = runtime.predict(dummy)
    assert torch.equal(before, after)


# ----------------------------------------------------------------------
# TorchScriptRuntime
# ----------------------------------------------------------------------


def test_torchscript_runtime_matches_pytorch_reference(reconstructed_model, logger, tmp_path):
    from scripts._synthetic_fixtures import export_torchscript

    model, registry = reconstructed_model
    ts_path = export_torchscript(model, tmp_path / "model.ts", INPUT_SHAPE)

    pytorch_runtime = PyTorchRuntime(
        model=model, device=torch.device("cpu"),
        model_fingerprint_sha256=registry.checkpoint_sha256, logger=logger,
    )
    pytorch_runtime.load()

    ts_runtime = TorchScriptRuntime(
        model_path=ts_path, device=torch.device("cpu"),
        model_fingerprint_sha256=registry.checkpoint_sha256, logger=logger,
    )
    ts_runtime.load()
    assert ts_runtime.is_loaded

    dummy = torch.randn(*INPUT_SHAPE)
    reference_output = pytorch_runtime.predict(dummy)
    info = ts_runtime.validate(dummy, reference_output=reference_output)
    assert info.validated is True, info.validation_errors
    assert ts_runtime.is_validated

    meta = ts_runtime.metadata()
    assert meta.runtime_type == "torchscript"
    assert meta.runtime_fingerprint_sha256 is not None  # populated once the artifact is read


def test_torchscript_runtime_metadata_available_before_load(reconstructed_model, logger, tmp_path):
    from scripts._synthetic_fixtures import export_torchscript

    model, registry = reconstructed_model
    ts_path = export_torchscript(model, tmp_path / "model.ts", INPUT_SHAPE)
    runtime = TorchScriptRuntime(
        model_path=ts_path, device=torch.device("cpu"),
        model_fingerprint_sha256=registry.checkpoint_sha256, logger=logger,
    )
    meta = runtime.metadata()  # before load()
    assert meta.runtime_type == "torchscript"
    assert not runtime.is_loaded


# ----------------------------------------------------------------------
# ONNXRuntime
# ----------------------------------------------------------------------


def test_onnx_runtime_matches_pytorch_reference(reconstructed_model, logger, tmp_path):
    from scripts._synthetic_fixtures import export_onnx

    model, registry = reconstructed_model
    onnx_path = export_onnx(model, tmp_path / "model.onnx", INPUT_SHAPE)

    pytorch_runtime = PyTorchRuntime(
        model=model, device=torch.device("cpu"),
        model_fingerprint_sha256=registry.checkpoint_sha256, logger=logger,
    )
    pytorch_runtime.load()

    onnx_runtime = ONNXRuntime(
        model_path=onnx_path, device=torch.device("cpu"),
        model_fingerprint_sha256=registry.checkpoint_sha256, logger=logger,
    )
    onnx_runtime.load()
    assert onnx_runtime.is_loaded

    dummy = torch.randn(*INPUT_SHAPE)
    reference_output = pytorch_runtime.predict(dummy)
    info = onnx_runtime.validate(dummy, reference_output=reference_output)
    assert info.validated is True, info.validation_errors
    assert onnx_runtime.is_validated

    meta = onnx_runtime.metadata()
    assert meta.runtime_type == "onnx"


def test_onnx_runtime_predict_output_shape(reconstructed_model, logger, tmp_path):
    from scripts._synthetic_fixtures import export_onnx

    model, registry = reconstructed_model
    onnx_path = export_onnx(model, tmp_path / "model.onnx", INPUT_SHAPE)
    runtime = ONNXRuntime(
        model_path=onnx_path, device=torch.device("cpu"),
        model_fingerprint_sha256=registry.checkpoint_sha256, logger=logger,
    )
    runtime.load()
    output = runtime.predict(torch.randn(*INPUT_SHAPE))
    assert tuple(output.shape) == (1, registry.num_classes)
