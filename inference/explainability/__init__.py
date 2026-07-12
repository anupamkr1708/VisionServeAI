"""
Explainability runtime: GradCAM-family, gradient-based, and
perturbation-based explainability methods for the reconstructed chest
X-ray classifier.

Migrated from Sprint 05 Stage 6 ("Explainability Runtime") of the archived
notebook. See ``engine.py`` for the high-level orchestration entry point
(:class:`ExplainabilityEngine`) and each of ``gradcam.py`` / ``gradcam_plus.py``
/ ``scorecam.py`` / ``eigencam.py`` / ``guided_backprop.py`` /
``integrated_gradients.py`` / ``occlusion.py`` for one independent algorithm
each.
"""
from inference.explainability.base import (
    BaseExplainer,
    ExplainabilityMetadata,
    ExplainabilityOutputPaths,
    ExplainabilityResult,
    MethodAvailability,
    MethodResult,
    METHOD_NAMES,
    make_output_dirs,
)
from inference.explainability.eigencam import EigenCAM
from inference.explainability.engine import ExplainabilityEngine
from inference.explainability.gradcam import GradCAM
from inference.explainability.gradcam_plus import GradCAMPlusPlus
from inference.explainability.guided_backprop import GuidedBackprop
from inference.explainability.integrated_gradients import IntegratedGradients
from inference.explainability.layer_discovery import discover_target_layer, list_conv_candidates
from inference.explainability.occlusion import Occlusion
from inference.explainability.scorecam import ScoreCAM

__all__ = [
    "ExplainabilityEngine",
    "BaseExplainer",
    "ExplainabilityMetadata",
    "ExplainabilityResult",
    "ExplainabilityOutputPaths",
    "MethodAvailability",
    "MethodResult",
    "METHOD_NAMES",
    "make_output_dirs",
    "GradCAM",
    "GradCAMPlusPlus",
    "ScoreCAM",
    "EigenCAM",
    "GuidedBackprop",
    "IntegratedGradients",
    "Occlusion",
    "discover_target_layer",
    "list_conv_candidates",
]
