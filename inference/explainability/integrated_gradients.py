"""
Integrated Gradients: path-integrated input attribution.

Migrated verbatim from Sprint 05 Stage 6's
``ExplainabilityEngine._compute_integrated_gradients`` (notebook lines
~591-610). Contains only this algorithm's own computation -- see
``gradcam.py``'s module docstring for why the shared run wrapper isn't
duplicated here.

Source: sprint05-deployment.ipynb, Stage 6, lines ~591-610.
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch

from inference.explainability.base import BaseExplainer

#: Verbatim from Stage 6 (notebook line ~59) -- number of interpolation
#: steps along the straight-line path from the zero baseline to the input.
IG_STEPS: int = 16


class IntegratedGradients(BaseExplainer):
    """Integrated Gradients (Sundararajan et al., 2017): attributes the
    target class logit to each input pixel by averaging gradients along a
    straight-line path from an all-zero baseline to the actual input, then
    multiplying by ``(input - baseline)`` -- a pure input-gradient method
    with no dependence on any particular layer (always available,
    regardless of architecture)."""

    METHOD_NAME = "integrated_gradients"

    def compute(
        self, tensor: torch.Tensor, target_idx: int, steps: int = IG_STEPS, **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        input_tensor = tensor.to(self.device)
        baseline = torch.zeros_like(input_tensor)
        grads = []
        for i in range(steps + 1):
            alpha = float(i) / steps
            scaled = (baseline + alpha * (input_tensor - baseline)).unsqueeze(0).clone().requires_grad_(True)
            logits = self.model(scaled)
            self.model.zero_grad(set_to_none=True)
            logits[0, target_idx].backward()
            grads.append(scaled.grad[0].detach())
        grads_stack = torch.stack(grads, dim=0)
        avg_grads = (grads_stack[:-1] + grads_stack[1:]) / 2.0
        avg_grad = avg_grads.mean(dim=0)
        attribution = ((input_tensor - baseline) * avg_grad).sum(dim=0).cpu().numpy()
        with torch.no_grad():
            probs = torch.sigmoid(self.model(input_tensor.unsqueeze(0)))[0].cpu().numpy()
        return self.normalize_map(np.abs(attribution)), probs
