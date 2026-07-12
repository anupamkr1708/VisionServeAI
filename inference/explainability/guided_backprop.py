"""
Guided Backpropagation: ReLU-gradient-clamped input saliency.

Migrated verbatim from Sprint 05 Stage 6's
``ExplainabilityEngine._compute_guided_backprop`` (notebook lines ~556-589)
and its availability check (notebook lines ~331-340). Contains only this
algorithm's own computation -- see ``gradcam.py``'s module docstring for
why the shared run wrapper isn't duplicated here.

This is the one method whose availability is conditional: Stage 6 disabled
it (with a logged reason, never a silent skip) if the model has no
``nn.ReLU`` modules at all, since it has nothing to hook. Every other
method defaults to unconditionally available in ``base.BaseExplainer``;
this is the sole override.

Source: sprint05-deployment.ipynb, Stage 6, lines ~331-340, ~556-589.
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch

from inference.explainability.base import BaseExplainer, ExplainabilityMetadata
from inference.explainability.hooks import count_relu_modules, guided_backprop_relu_hooks


class GuidedBackprop(BaseExplainer):
    """Guided Backpropagation (Springenberg et al., 2015): standard
    input-gradient saliency, except backward passes through every
    ``nn.ReLU`` clamp negative gradients to zero (in addition to the
    ReLU's own forward-pass zeroing of negative activations), suppressing
    noisy negative-gradient contributions for a sharper saliency map."""

    METHOD_NAME = "guided_backprop"

    def check_availability(self) -> ExplainabilityMetadata:
        """Available only if the model contains at least one ``nn.ReLU``
        module (the hook target this method needs). Verbatim port of Stage
        6's guided-backprop availability branch (notebook lines ~331-340)."""
        relu_count = count_relu_modules(self.model)
        if relu_count > 0:
            return ExplainabilityMetadata(
                method=self.METHOD_NAME, available=True,
                reason=f"{relu_count} nn.ReLU module(s) found for hook-based gradient override",
            )
        return ExplainabilityMetadata(
            method=self.METHOD_NAME, available=False,
            reason="no nn.ReLU modules found in model; method disabled",
        )

    def compute(self, tensor: torch.Tensor, target_idx: int, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        with guided_backprop_relu_hooks(self.model):
            inp = tensor.unsqueeze(0).to(self.device).clone().requires_grad_(True)
            logits = self.model(inp)
            probs = torch.sigmoid(logits)[0]
            self.model.zero_grad(set_to_none=True)
            logits[0, target_idx].backward()
            grad = inp.grad[0].detach().cpu().numpy()
            saliency = np.abs(grad).max(axis=0)
            return self.normalize_map(saliency), probs.detach().cpu().numpy()
