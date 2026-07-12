"""
Forward/backward hook utilities: activation capture, gradient capture, and
guaranteed cleanup.

Migrated from Sprint 05 Stage 6, where hook registration/removal appeared
in three places: the reusable ``ActivationsAndGradients`` class (notebook
lines ~164-186, used by GradCAM/GradCAM++), an inline forward-only hook
repeated in ``_compute_scorecam``/``_compute_eigencam``/
``_probe_activation_size`` (notebook lines ~300-323, ~480-554), and the
inline ReLU-gradient-clamping hook set in ``_compute_guided_backprop``
(notebook lines ~556-589). All three are consolidated here as the single
implementation each algorithm module imports -- "own forward hooks,
backward hooks, activation capture, gradient capture, cleanup; no
duplicated hook logic", per this phase's explicit requirement.

Every hook utility here guarantees removal via ``try/finally`` (the class's
own ``remove()`` for :class:`ActivationsAndGradients`, or a context
manager's ``__exit__`` for the two function-based helpers) -- matching
Stage 6's own discipline of never leaving a hook registered past the
computation it was needed for.

Source: sprint05-deployment.ipynb, Stage 6, lines ~164-186 (``ActivationsAndGradients``),
~300-323 (``_probe_activation_size``'s inline hook), ~480-554 (ScoreCAM/EigenCAM's
inline hook), ~556-589 (Guided Backprop's ReLU hooks).
"""
from __future__ import annotations

import contextlib
from typing import Dict, Iterator, List, Optional

import torch
import torch.nn as nn


class ActivationsAndGradients:
    """Registers forward+backward hooks on a single target layer and
    captures its activations and gradients for the last forward/backward
    pass. Always removed via :meth:`remove` in a ``finally`` block by
    callers.

    Verbatim port of Stage 6's ``ActivationsAndGradients`` (notebook lines
    ~164-186), used by GradCAM and GradCAM++ (the two methods that need
    both activations *and* their gradient with respect to the target
    class's logit).
    """

    def __init__(self, target_layer: nn.Module) -> None:
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        try:
            self._bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)
        except AttributeError:
            self._bwd_handle = target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module: nn.Module, inputs: object, output: torch.Tensor) -> None:
        self.activations = output

    def _backward_hook(self, module: nn.Module, grad_input: object, grad_output: object) -> None:
        self.gradients = grad_output[0]  # type: ignore[index]

    def remove(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()


@contextlib.contextmanager
def forward_activation_capture(target_layer: nn.Module) -> Iterator[Dict[str, torch.Tensor]]:
    """Context manager capturing a single target layer's forward-pass
    activation into the yielded dict under key ``"a"`` -- the hook is
    always removed on exit, even if the caller's block raises.

    Consolidates the identical inline forward-hook pattern Stage 6 repeated
    three times: ``_probe_activation_size``, ``_compute_scorecam``, and
    ``_compute_eigencam`` (notebook lines ~300-323, ~480-554) -- no
    gradient capture needed, since ScoreCAM/EigenCAM only read forward
    activations (ScoreCAM re-runs the forward pass per perturbation mask;
    EigenCAM decomposes the raw activation via SVD).

    Usage::

        with forward_activation_capture(target_layer) as holder:
            with torch.no_grad():
                model(dummy_input)
            activation = holder["a"]
    """
    holder: Dict[str, torch.Tensor] = {}

    def _hook(module: nn.Module, inputs: object, output: torch.Tensor) -> None:
        holder["a"] = output

    handle = target_layer.register_forward_hook(_hook)
    try:
        yield holder
    finally:
        handle.remove()


@contextlib.contextmanager
def guided_backprop_relu_hooks(model: nn.Module) -> Iterator[None]:
    """Context manager implementing Guided Backprop's gradient-clamping
    hook set: registers a backward hook on every ``nn.ReLU`` module that
    zeroes out negative gradients flowing back through it, and temporarily
    disables ``inplace=True`` on every such ReLU (``register_full_backward_hook``
    conflicts with in-place ReLUs -- a view+inplace autograd error).
    Everything is restored on exit -- hooks removed, ``inplace`` flags put
    back exactly as found -- so the frozen reconstructed model is left
    byte-identical to how :meth:`__enter__` found it, regardless of whether
    the caller's block raised.

    Verbatim port of Stage 6's ``_compute_guided_backprop`` hook setup/teardown
    (notebook lines ~556-576, ~585-589).

    Usage::

        with guided_backprop_relu_hooks(model):
            logits = model(inp)
            logits[0, target_idx].backward()
    """
    relu_modules: List[nn.Module] = [m for m in model.modules() if isinstance(m, nn.ReLU)]
    original_inplace = [m.inplace for m in relu_modules]  # type: ignore[attr-defined]
    handles = []

    def _clamp_hook(module: nn.Module, grad_input: object, grad_output: object):
        g = grad_input[0]  # type: ignore[index]
        if g is None:
            return grad_input
        return (torch.clamp(g, min=0.0).clone(),)

    for m in relu_modules:
        m.inplace = False  # type: ignore[attr-defined]
        try:
            handles.append(m.register_full_backward_hook(_clamp_hook))
        except AttributeError:
            handles.append(m.register_backward_hook(_clamp_hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()
        for m, orig in zip(relu_modules, original_inplace):
            m.inplace = orig  # type: ignore[attr-defined]


def count_relu_modules(model: nn.Module) -> int:
    """Number of ``nn.ReLU`` modules in ``model`` -- used by
    ``guided_backprop.GuidedBackprop.check_availability`` to decide whether
    the method is available at all (Stage 6, notebook lines ~331-340)."""
    return sum(1 for mod in model.modules() if isinstance(mod, nn.ReLU))
