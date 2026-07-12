"""Model runtime backends (PyTorch, TorchScript, ONNX Runtime) behind a
common interface -- see ``inference.runtimes.base.BaseRuntime``.

Public API re-exported here for convenience::

    from inference.runtimes import RuntimeFactory
    runtime = RuntimeFactory.create("pytorch", model=..., device=..., ...)
"""
from __future__ import annotations

from inference.runtimes.base import (
    BaseRuntime,
    RuntimeCapabilities,
    RuntimeInfo,
    RuntimeMetadata,
)
from inference.runtimes.onnx_runtime import ONNXRuntime
from inference.runtimes.pytorch_runtime import PyTorchRuntime
from inference.runtimes.runtime_factory import RuntimeFactory
from inference.runtimes.runtime_registry import RuntimeRegistry, RuntimeRegistryEntry
from inference.runtimes.torchscript_runtime import TorchScriptRuntime

__all__ = [
    "BaseRuntime",
    "RuntimeCapabilities",
    "RuntimeInfo",
    "RuntimeMetadata",
    "PyTorchRuntime",
    "TorchScriptRuntime",
    "ONNXRuntime",
    "RuntimeFactory",
    "RuntimeRegistry",
    "RuntimeRegistryEntry",
]
