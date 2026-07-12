"""
Runtime factory: constructs the correct :class:`BaseRuntime` subclass for a
declared runtime type.

NEW in this phase -- no notebook precedent (Stage 3 exported to TorchScript
and ONNX inline in one linear script; there was never a point where "which
runtime backend to construct" was a runtime decision to be dispatched).
This exists purely to satisfy the architectural rule "no if/else chains
elsewhere [outside RuntimeFactory]" for runtime selection: any future code
that needs "a runtime" (an ``InferenceEngine`` wiring step, a benchmarking
script, a deployment health check) calls
``RuntimeFactory.create(runtime_type, **kwargs)`` instead of writing its own
``if runtime_type == "pytorch": ... elif ...`` chain.
"""
from __future__ import annotations

from typing import Callable, Dict, List

from inference.runtimes.base import BaseRuntime
from inference.runtimes.onnx_runtime import ONNXRuntime
from inference.runtimes.pytorch_runtime import PyTorchRuntime
from inference.runtimes.torchscript_runtime import TorchScriptRuntime


class RuntimeFactory:
    """Builds a :class:`BaseRuntime` instance by declared ``runtime_type``,
    via a dispatch table rather than a conditional chain.

    Each concrete runtime's own ``__init__`` still enforces its own
    required constructor arguments -- this factory does not reinterpret,
    validate, or default any of them. Missing/unexpected keyword arguments
    surface as the same ``TypeError`` they would from constructing the
    runtime class directly, matching "fail loudly": this factory adds
    dispatch, not a new validation layer.

    Usage::

        runtime = RuntimeFactory.create(
            "pytorch", model=reconstructed_model, device=device,
            model_fingerprint_sha256=model_registry.checkpoint_sha256, logger=logger,
        )
        runtime.load()
        predictions = runtime.predict(batch)
    """

    _BUILDERS: Dict[str, Callable[..., BaseRuntime]] = {
        PyTorchRuntime.RUNTIME_TYPE: PyTorchRuntime,
        TorchScriptRuntime.RUNTIME_TYPE: TorchScriptRuntime,
        ONNXRuntime.RUNTIME_TYPE: ONNXRuntime,
    }

    @classmethod
    def create(cls, runtime_type: str, **kwargs: object) -> BaseRuntime:
        """Construct a runtime of the given type.

        Args:
            runtime_type: One of :meth:`supported_runtimes` --
                ``"pytorch"``, ``"torchscript"``, or ``"onnx"``.
            **kwargs: Forwarded verbatim to the selected runtime class's
                constructor (see ``PyTorchRuntime`` / ``TorchScriptRuntime``
                / ``ONNXRuntime`` for their respective required arguments).

        Raises:
            ValueError: if ``runtime_type`` is not a known runtime.
        """
        try:
            builder = cls._BUILDERS[runtime_type]
        except KeyError:
            raise ValueError(
                f"Unknown runtime_type '{runtime_type}'. Supported runtimes: {cls.supported_runtimes()}"
            ) from None
        return builder(**kwargs)

    @classmethod
    def supported_runtimes(cls) -> List[str]:
        """The set of ``runtime_type`` strings :meth:`create` accepts."""
        return sorted(cls._BUILDERS)
