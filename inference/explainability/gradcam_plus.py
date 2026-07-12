"""
GradCAM++: improved pixel-wise weighting of GradCAM.

Migrated verbatim from Sprint 05 Stage 6's
``ExplainabilityEngine._compute_gradcam_plus`` (notebook lines ~458-478).
Contains only this algorithm's own computation -- see ``gradcam.py``'s
module docstring for why the shared run wrapper isn't duplicated here.

Source: sprint05-deployment.ipynb, Stage 6, lines ~458-478.
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch

from inference.explainability.base import BaseExplainer
from inference.explainability.hooks import ActivationsAndGradients


class GradCAMPlusPlus(BaseExplainer):
    """GradCAM++ (Chattopadhay et al., 2018): replaces GradCAM's uniform
    global-average-pooled gradient weighting with a per-pixel weighting
    (``alpha``) derived from second- and third-order gradients, better
    localizing multiple instances of the same class."""

    METHOD_NAME = "gradcam_plus"

    def compute(self, tensor: torch.Tensor, target_idx: int, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        hooks = ActivationsAndGradients(self.target_layer)
        try:
            inp = tensor.unsqueeze(0).to(self.device).clone().requires_grad_(True)
            logits = self.model(inp)
            probs = torch.sigmoid(logits)[0]
            self.model.zero_grad(set_to_none=True)
            logits[0, target_idx].backward()
            activations = hooks.activations[0]
            grads = hooks.gradients[0]
            grads_sq = grads.pow(2)
            grads_cube = grads.pow(3)
            sum_act_grad3 = (activations * grads_cube).sum(dim=(1, 2), keepdim=True)
            denom = 2.0 * grads_sq + sum_act_grad3
            denom = torch.where(denom.abs() > 1e-8, denom, torch.ones_like(denom))
            alpha = grads_sq / denom
            weights = (alpha * torch.relu(grads)).sum(dim=(1, 2))
            cam = torch.relu((weights.view(-1, 1, 1) * activations).sum(dim=0))
            return self.normalize_map(cam.detach().cpu().numpy()), probs.detach().cpu().numpy()
        finally:
            hooks.remove()
