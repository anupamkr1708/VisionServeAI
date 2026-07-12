"""
Model registry structures.

Moved verbatim from Sprint 05 Stage 2 ("Deployment Artifact Registry & Model
Reconstruction") of the archived notebook -- no fields, types, or defaults
changed. Split out from ``inference/model_loader.py`` into their own module
since they are reusable registry structures other modules (later
deployment/, backend/ phases) will import independently of the
reconstruction logic itself.

Note: this is distinct from the notebook's ``ArtifactRegistry`` /
``ThresholdRegistry`` / ``MetadataRegistry`` / ``ConfigRegistry`` /
``DeploymentRegistry`` dataclasses, which describe artifact discovery and
deployment packaging concerns -- those belong to a future deployment/ phase
and are intentionally not touched here.

Source: sprint05-deployment.ipynb, Stage 2, lines ~453-481.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelValidationChecks:
    """Aggregated pass/fail flags produced while reconstructing a model from
    a checkpoint. Mirrors the original notebook's aggregate-then-raise
    pattern: most fields are set progressively during reconstruction, and
    any accumulated ``errors`` are raised together at the end rather than
    failing on the first one (checkpoint loading and strict state_dict
    match are the two exceptions that still fail fast, matching the
    original notebook exactly)."""

    architecture_instantiated: bool = False
    checkpoint_loaded: bool = False
    state_dict_strict_match: bool = False
    parameter_count_expected: Optional[bool] = None
    output_dimension_expected: Optional[bool] = None
    classifier_dimension_expected: Optional[bool] = None
    all_params_correct_device: bool = False
    all_params_correct_dtype: bool = False
    eval_mode: bool = False
    grad_disabled: bool = False
    errors: List[str] = field(default_factory=list)


@dataclass
class ModelRegistry:
    """Result of a successful :func:`inference.model_loader.reconstruct_model`
    call: the resolved architecture, checkpoint provenance (path + SHA-256
    fingerprint), the model signature, and the validation checks that were
    run against it."""

    backbone: str
    num_classes: int
    device: str
    dtype: str
    total_parameters: int
    trainable_parameters: int
    checkpoint_path: str
    checkpoint_sha256: str
    model_signature: Dict[str, Any]
    validation: ModelValidationChecks

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
