"""
Automatic target-layer discovery for CAM-family explainability methods.

Migrated verbatim from Sprint 05 Stage 6's ``discover_target_layer``
(notebook lines ~189-211) and the activation-size probe inline in
``ExplainabilityEngine.__init__``/``_probe_activation_size`` (notebook
lines ~268-271, ~300-323).

No hard-coded layer names: discovery works by traversing every named
module and selecting the *last* ``nn.Conv2d`` encountered, which is
architecture-agnostic by construction -- it requires no knowledge of
DenseNet's ``features.denseblock4.denselayer16.conv2``, ResNet's
``layer4.2.conv3``, EfficientNet's ``features.8.0``, or VGG's
``features.28`` naming conventions specifically. This is exactly how Stage
6 supported all four backbones without a single per-architecture branch,
and is preserved unchanged here.

Source: sprint05-deployment.ipynb, Stage 6, lines ~189-211, ~268-271, ~300-323.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from inference.explainability.hooks import forward_activation_capture


def discover_target_layer(model: nn.Module, logger: logging.Logger) -> Tuple[str, nn.Module, int]:
    """Automatic layer discovery: selects the LAST ``nn.Conv2d`` module
    found via ``named_modules()`` traversal. Works identically across
    DenseNet121, ResNet, EfficientNet, VGG, and any future
    torchvision-style CNN backbone without ever hardcoding a layer name.

    Verbatim port of Stage 6's ``discover_target_layer`` (notebook lines
    ~189-211).

    Raises:
        RuntimeError: if no ``nn.Conv2d`` layer exists anywhere in the
            model -- CAM-based methods require at least one convolutional
            layer.
    """
    last_name: Optional[str] = None
    last_module: Optional[nn.Module] = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_name, last_module = name, module

    if last_module is None:
        raise RuntimeError(
            "Automatic layer discovery failed: no nn.Conv2d layer found anywhere in the "
            "model. CAM-based methods require at least one convolutional layer."
        )

    feature_dim = last_module.out_channels
    logger.info(
        "LAYER DISCOVERY selected target layer='%s' type=%s feature_dim=%d",
        last_name, type(last_module).__name__, feature_dim,
    )
    return last_name, last_module, feature_dim


def list_conv_candidates(model: nn.Module) -> List[str]:
    """List every ``nn.Conv2d`` layer name found in traversal order --
    informational only (does not change which layer is *selected*, see
    :func:`discover_target_layer`). New in this phase: exposes the full
    candidate set a caller could inspect, e.g. for
    ``ExplainabilityEngine.discover_layers()``'s registry snapshot, without
    altering the single-last-Conv2d selection rule Stage 6 always used."""
    return [name for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]


def probe_activation_size(
    model: nn.Module,
    target_layer: nn.Module,
    device: torch.device,
    channels: int,
    height: int,
    width: int,
    layer_name: str,
    logger: logging.Logger,
) -> List[int]:
    """Run one dummy forward pass to confirm the target layer's forward
    hook actually fires, and record its activation shape ``[C, H, W]``.

    Verbatim port of Stage 6's ``ExplainabilityEngine._probe_activation_size``
    (notebook lines ~300-323).

    Raises:
        RuntimeError: if the forward hook never fires (activation never
            captured) -- indicates the target layer is unreachable from the
            model's forward path.
    """
    dummy = torch.zeros(1, channels, height, width, device=device)
    with forward_activation_capture(target_layer) as holder:
        with torch.no_grad():
            model(dummy)
        if "a" not in holder:
            raise RuntimeError(f"Forward hook on layer '{layer_name}' failed to capture an activation.")
        shape = list(holder["a"].shape[1:])  # [C, H, W]

    logger.info(
        "LAYER DISCOVERY activation size for layer='%s': %s (forward hook verified)",
        layer_name, shape,
    )
    return shape
