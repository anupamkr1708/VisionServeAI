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

# Inference Pipeline phase (preprocessing / postprocessing / thresholding /
# InferenceEngine) -- added when that phase landed, but never re-exported
# here at the time; ``inference.engine.InferenceEngine`` in particular was
# consequently unreachable from anywhere else in the package (confirmed by
# repository-wide static-import audit -- see audit report). Purely additive:
# no existing symbol above is changed.
from inference.engine import InferenceEngine
from inference.postprocessing import (
    PredictionResult,
    apply_sigmoid,
    build_error_result,
    build_prediction_result,
    summarize_predictions,
    top_k_predictions,
)
from inference.preprocessing import (
    InferenceError,
    PreprocessingConfig,
    build_batch_tensor,
    preprocess_image,
    resolve_preprocessing_config,
    validate_and_decode_image,
)
from inference.thresholding import (
    ThresholdRegistry,
    get_threshold,
    load_threshold_registry,
    resolve_class_names_and_thresholds,
)

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
    "InferenceEngine",
    "PredictionResult",
    "apply_sigmoid",
    "build_error_result",
    "build_prediction_result",
    "summarize_predictions",
    "top_k_predictions",
    "InferenceError",
    "PreprocessingConfig",
    "build_batch_tensor",
    "preprocess_image",
    "resolve_preprocessing_config",
    "validate_and_decode_image",
    "ThresholdRegistry",
    "get_threshold",
    "load_threshold_registry",
    "resolve_class_names_and_thresholds",
]

