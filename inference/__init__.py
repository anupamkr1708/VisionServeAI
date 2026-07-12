"""VisionServeAI inference runtime -- model loading, preprocessing, postprocessing, thresholding, prediction, explainability. No FastAPI code lives here."""
from inference.model_loader import (
    WrappedClassifier,
    build_model,
    compute_model_fingerprint,
    extract_state_dict,
    load_checkpoint,
    reconstruct_model,
    resolve_backbone_name,
    resolve_num_classes,
    resolve_dropout,
    validate_checkpoint,
    verify_classifier_dimensions,
    get_model_signature,
)
from inference.model_registry import ModelRegistry, ModelValidationChecks

__all__ = [
    "WrappedClassifier",
    "build_model",
    "reconstruct_model",
    "load_checkpoint",
    "extract_state_dict",
    "validate_checkpoint",
    "verify_classifier_dimensions",
    "compute_model_fingerprint",
    "get_model_signature",
    "resolve_backbone_name",
    "resolve_num_classes",
    "resolve_dropout",
    "ModelRegistry",
    "ModelValidationChecks",
]

