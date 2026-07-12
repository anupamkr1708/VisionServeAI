"""
Occlusion sensitivity: perturbation-based attribution via patch masking.

Migrated verbatim from Sprint 05 Stage 6's
``ExplainabilityEngine._compute_occlusion`` (notebook lines ~612-642).
Contains only this algorithm's own computation -- see ``gradcam.py``'s
module docstring for why the shared run wrapper isn't duplicated here.

Source: sprint05-deployment.ipynb, Stage 6, lines ~612-642.
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch

from inference.explainability.base import BaseExplainer

#: Verbatim from Stage 6 (notebook lines ~60-61) -- sliding-window patch
#: size and stride for the occlusion sweep.
OCCLUSION_PATCH_SIZE: int = 32
OCCLUSION_STRIDE: int = 16


class Occlusion(BaseExplainer):
    """Occlusion sensitivity (Zeiler & Fergus, 2014): slides a zeroed patch
    across the input in a grid, measuring how much the target class'
    predicted probability drops when each region is occluded. A pure
    forward-pass perturbation method -- no gradients, no hooks, always
    available regardless of architecture."""

    METHOD_NAME = "occlusion"

    def compute(
        self,
        tensor: torch.Tensor,
        target_idx: int,
        patch_size: int = OCCLUSION_PATCH_SIZE,
        stride: int = OCCLUSION_STRIDE,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            input_tensor = tensor.unsqueeze(0).to(self.device)
            base_logits = self.model(input_tensor)
            base_prob = torch.sigmoid(base_logits)[0, target_idx].item()
            probs = torch.sigmoid(base_logits)[0].cpu().numpy()

            _, h, w = tensor.shape
            heatmap = np.zeros((h, w), dtype=np.float32)
            counts = np.zeros((h, w), dtype=np.float32)

            for y in range(0, h, stride):
                for x in range(0, w, stride):
                    y2, x2 = min(y + patch_size, h), min(x + patch_size, w)
                    occluded = input_tensor.clone()
                    occluded[:, :, y:y2, x:x2] = 0.0
                    out = self.model(occluded)
                    prob = torch.sigmoid(out)[0, target_idx].item()
                    drop = base_prob - prob
                    heatmap[y:y2, x:x2] += drop
                    counts[y:y2, x:x2] += 1.0

            counts[counts == 0] = 1.0
            heatmap = heatmap / counts
            return self.normalize_map(heatmap), probs
