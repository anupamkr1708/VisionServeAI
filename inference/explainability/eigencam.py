"""
EigenCAM: principal-component-based class activation mapping.

Migrated verbatim from Sprint 05 Stage 6's
``ExplainabilityEngine._compute_eigencam`` (notebook lines ~532-554).
Contains only this algorithm's own computation -- see ``gradcam.py``'s
module docstring for why the shared run wrapper isn't duplicated here.

Source: sprint05-deployment.ipynb, Stage 6, lines ~532-554.
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch

from inference.explainability.base import BaseExplainer
from inference.explainability.hooks import forward_activation_capture


class EigenCAM(BaseExplainer):
    """EigenCAM (Muhammad & Yeasin, 2020): class-agnostic CAM variant that
    takes the first principal component of the target layer's activations
    (via SVD) as the saliency map -- no gradients, no class-conditioned
    forward passes, just a projection of the activation's dominant
    direction of variance."""

    METHOD_NAME = "eigencam"

    def compute(self, tensor: torch.Tensor, target_idx: int, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        with forward_activation_capture(self.target_layer) as holder:
            with torch.no_grad():
                inp = tensor.unsqueeze(0).to(self.device)
                logits = self.model(inp)
                probs = torch.sigmoid(logits)[0].cpu().numpy()
                activations = holder["a"][0]  # C, H, W
                c, h, w = activations.shape
                flat = activations.reshape(c, h * w).cpu().numpy()
                flat_centered = flat - flat.mean(axis=0, keepdims=True)
                _, _, vt = np.linalg.svd(flat_centered, full_matrices=False)
                principal = vt[0].reshape(h, w)
                if principal.sum() < 0:
                    principal = -principal
                return self.normalize_map(principal), probs
