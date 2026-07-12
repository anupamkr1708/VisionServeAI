"""
ScoreCAM: gradient-free, perturbation-weighted class activation mapping.

Migrated verbatim from Sprint 05 Stage 6's
``ExplainabilityEngine._compute_scorecam`` (notebook lines ~480-530).
Contains only this algorithm's own computation -- see ``gradcam.py``'s
module docstring for why the shared run wrapper isn't duplicated here.

Source: sprint05-deployment.ipynb, Stage 6, lines ~480-530.
"""
from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from inference.explainability.base import BaseExplainer
from inference.explainability.hooks import forward_activation_capture

#: Verbatim from Stage 6 (notebook line ~62) -- caps the number of
#: activation channels ScoreCAM evaluates per image for performance; capped
#: by activation magnitude when the layer has more channels than this.
SCORECAM_MAX_CHANNELS: int = 32


class ScoreCAM(BaseExplainer):
    """ScoreCAM (Wang et al., 2020): gradient-free CAM variant that weighs
    each activation channel by the model's own confidence score on the
    input masked by that (upsampled, normalized) channel, avoiding
    gradient-saturation artifacts that can affect gradient-based CAM
    methods."""

    METHOD_NAME = "scorecam"

    def compute(self, tensor: torch.Tensor, target_idx: int, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        with forward_activation_capture(self.target_layer) as holder:
            with torch.no_grad():
                inp = tensor.unsqueeze(0).to(self.device)
                base_logits = self.model(inp)
                base_probs = torch.sigmoid(base_logits)[0].cpu().numpy()
                activations = holder["a"][0]  # C, H, W
                in_h, in_w = inp.shape[2], inp.shape[3]

                num_channels = activations.shape[0]
                if num_channels > SCORECAM_MAX_CHANNELS:
                    magnitude = activations.abs().mean(dim=(1, 2))
                    top_idx = torch.topk(magnitude, SCORECAM_MAX_CHANNELS).indices.tolist()
                    self.logger.info(
                        "SCORECAM capped %d -> %d channels by activation magnitude for performance.",
                        num_channels, SCORECAM_MAX_CHANNELS,
                    )
                else:
                    top_idx = list(range(num_channels))

                scores: List[float] = []
                masks: List[torch.Tensor] = []
                for c in top_idx:
                    am = activations[c:c + 1].unsqueeze(0)
                    am_up = F.interpolate(am, size=(in_h, in_w), mode="bilinear", align_corners=False)[0, 0]
                    amin, amax = am_up.min(), am_up.max()
                    if float(amax - amin) < 1e-8:
                        continue
                    norm_mask = (am_up - amin) / (amax - amin)
                    masked_input = inp * norm_mask.unsqueeze(0).unsqueeze(0)
                    out = self.model(masked_input)
                    scores.append(torch.sigmoid(out)[0, target_idx].item())
                    masks.append(norm_mask)

                if not masks:
                    raise RuntimeError("ScoreCAM produced no valid channel masks (degenerate activations).")

                weights = torch.softmax(torch.tensor(scores, device=self.device), dim=0)
                cam = torch.zeros(in_h, in_w, device=self.device)
                for w, m in zip(weights, masks):
                    cam += w * m
                cam = torch.relu(cam).cpu().numpy()
                return self.normalize_map(cam), base_probs
