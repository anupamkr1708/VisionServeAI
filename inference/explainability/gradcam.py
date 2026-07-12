"""
GradCAM: gradient-weighted class activation mapping.

Migrated verbatim from Sprint 05 Stage 6's
``ExplainabilityEngine._compute_gradcam`` (notebook lines ~442-456).
Contains only this algorithm's own computation -- the run wrapper (target
resolution, NaN/Inf checking, rendering, saving, failure isolation) lives
once in ``base.BaseExplainer.generate``, not duplicated here.

Source: sprint05-deployment.ipynb, Stage 6, lines ~442-456.
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch

from inference.explainability.base import BaseExplainer
from inference.explainability.hooks import ActivationsAndGradients


class GradCAM(BaseExplainer):
    """Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017).

    Weights the target layer's activation channels by the global-average-
    pooled gradient of the target class logit with respect to each channel,
    then ReLUs the weighted sum -- the original GradCAM formulation.
    """

    METHOD_NAME = "gradcam"

    def compute(self, tensor: torch.Tensor, target_idx: int, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        hooks = ActivationsAndGradients(self.target_layer)
        try:
            inp = tensor.unsqueeze(0).to(self.device).clone().requires_grad_(True)
            logits = self.model(inp)
            probs = torch.sigmoid(logits)[0]
            self.model.zero_grad(set_to_none=True)
            logits[0, target_idx].backward()
            activations = hooks.activations[0]
            gradients = hooks.gradients[0]
            weights = gradients.mean(dim=(1, 2))
            cam = torch.relu((weights.view(-1, 1, 1) * activations).sum(dim=0))
            return self.normalize_map(cam.detach().cpu().numpy()), probs.detach().cpu().numpy()
        finally:
            hooks.remove()
