"""
Model loader.

Migrated from Sprint 05 Stage 2 ("Deployment Artifact Registry & Model
Reconstruction") of the archived notebook. This module covers only the
model-reconstruction slice of that stage -- ``WrappedClassifier``, backbone
resolution, checkpoint loading/validation, and ``reconstruct_model()``.
The artifact-discovery/registry logic (``ArtifactRecord``, ``ArtifactRegistry``,
schema validation, etc.) and the threshold/metadata/config/deployment
registries from the same stage belong to a future ``deployment/`` phase and
are intentionally not included here.

Decomposition note: the original notebook's ``reconstruct_model()`` was one
~110-line function that did checkpoint fingerprinting, loading, strict
validation, and dimension/parameter checks all inline. Per this migration
phase's requested public interface, that logic has been split into
individually named, independently callable functions
(``compute_model_fingerprint``, ``load_checkpoint``, ``validate_checkpoint``,
``verify_classifier_dimensions``, ``get_model_signature``) that
``reconstruct_model()`` now calls in the exact same order the original
inline code executed them, with identical checks, identical error messages,
and identical fail-fast-vs-aggregate behaviour (checkpoint load and strict
state_dict match still raise immediately; parameter-count, dimension,
device, and dtype checks are still aggregated into ``checks.errors`` and
raised together at the end). No validation was removed, relaxed, or
reordered.

Two checks from the original function -- the parameter-count-vs-training-summary
check and the device/dtype consistency checks -- were *not* promoted to their
own named public functions, since they weren't part of the requested public
interface and have no other caller yet. They remain inline in
``reconstruct_model()``, byte-for-byte identical to the original logic. See
the migration report in the accompanying chat response for the full
breakdown.

Source: sprint05-deployment.ipynb, Stage 2, lines ~78-690.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision

from inference.model_registry import ModelRegistry, ModelValidationChecks
from inference.utils.hashing import sha256_of_file

# ======================================================================
# ARCHITECTURE: WrappedClassifier + known backbone factories
# ======================================================================
# Known backbone factories for EXACT architecture reconstruction. Extend
# only when a new backbone is explicitly named in training_summary.json --
# never guess an architecture that isn't named there. (Original notebook
# comment, preserved.)


class WrappedClassifier(nn.Module):
    """Matches the Sprint04 training convention observed in the checkpoint:
    ``self.backbone = <torchvision base model>``, with the base model's
    native classification head replaced by
    ``nn.Sequential(Dropout(p), Linear(...))``. This is inferred directly
    from checkpoint key names (``backbone.*``, head index 1), not assumed --
    do not change without re-checking checkpoint keys."""

    def __init__(
        self,
        base_model: nn.Module,
        head_attr: str,
        in_features: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = base_model
        head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
        setattr(self.backbone, head_attr, head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def _base_densenet121(_num_classes: int) -> Tuple[nn.Module, str, int]:
    m = torchvision.models.densenet121(weights=None)
    return m, "classifier", m.classifier.in_features


def _base_resnet50(_num_classes: int) -> Tuple[nn.Module, str, int]:
    m = torchvision.models.resnet50(weights=None)
    return m, "fc", m.fc.in_features


def _base_resnet18(_num_classes: int) -> Tuple[nn.Module, str, int]:
    m = torchvision.models.resnet18(weights=None)
    return m, "fc", m.fc.in_features


def _base_efficientnet_b0(_num_classes: int) -> Tuple[nn.Module, str, int]:
    m = torchvision.models.efficientnet_b0(weights=None)
    return m, "classifier", m.classifier[-1].in_features


def _base_vgg16(_num_classes: int) -> Tuple[nn.Module, str, int]:
    m = torchvision.models.vgg16(weights=None)
    return m, "classifier", m.classifier[-1].in_features


BASE_MODEL_BUILDERS = {
    "densenet121": _base_densenet121,
    "resnet50": _base_resnet50,
    "resnet18": _base_resnet18,
    "efficientnet_b0": _base_efficientnet_b0,
    "vgg16": _base_vgg16,
}


def build_model(backbone_name: str, num_classes: int, dropout: float) -> Tuple[nn.Module, str]:
    """Instantiate a :class:`WrappedClassifier` for the given backbone.

    Returns:
        ``(wrapped_model, full_dotted_path_to_classifier_head)``, e.g.
        ``(model, "backbone.classifier")``.
    """
    base_model, head_attr, in_features = BASE_MODEL_BUILDERS[backbone_name](num_classes)
    model = WrappedClassifier(base_model, head_attr, in_features, num_classes, dropout)
    return model, f"backbone.{head_attr}"


def _get_nested_attr(obj: Any, dotted_path: str) -> Any:
    """Resolve a dotted attribute path, e.g. ``"backbone.classifier"``."""
    current = obj
    for part in dotted_path.split("."):
        current = getattr(current, part)
    return current


def _final_out_features(module: nn.Module) -> int:
    """Return the output feature count of a classifier head module (a bare
    ``nn.Linear`` or an ``nn.Sequential`` ending in one)."""
    if isinstance(module, nn.Linear):
        return module.out_features
    if isinstance(module, nn.Sequential):
        for layer in reversed(module):
            if isinstance(layer, nn.Linear):
                return layer.out_features
    raise RuntimeError(f"Unable to determine output features from classifier module: {module}")


# ======================================================================
# BACKBONE RESOLUTION (from training_summary.json / disease_registry.json)
# ======================================================================


def _search_keys(d: Dict[str, Any], keys: List[str]) -> Tuple[Optional[str], Optional[Any]]:
    """Look for the first matching key at top level, then one level of
    nesting. Returns ``(key_found, value)`` or ``(None, None)``."""
    for key in keys:
        val = d.get(key)
        if val not in (None, ""):
            return key, val
    for v in d.values():
        if isinstance(v, dict):
            for key in keys:
                val = v.get(key)
                if val not in (None, ""):
                    return key, val
    return None, None


def resolve_backbone_name(training_summary: Dict[str, Any], logger: logging.Logger) -> str:
    """Resolve the backbone architecture name from ``training_summary.json``,
    salvaging a known backbone id as a substring match if the recorded value
    is a descriptive string (e.g. ``"densenet121-chestxray14"``).

    Raises:
        RuntimeError: if no recognized backbone key is found, or the
            resolved name doesn't match (or contain) a supported backbone.
    """
    candidate_keys = ["backbone", "architecture", "model_name", "arch", "model_architecture"]
    found_key, value = _search_keys(training_summary, candidate_keys)
    if value is None:
        raise RuntimeError(
            f"training_summary.json does not contain any recognized backbone key {candidate_keys} "
            f"(checked top level and one level of nesting)."
        )
    name = str(value).strip().lower()
    logger.info("MODEL backbone resolved via key '%s' = '%s'", found_key, name)

    if name not in BASE_MODEL_BUILDERS:
        for known in BASE_MODEL_BUILDERS:
            if known in name:
                logger.warning(
                    "MODEL backbone value '%s' not an exact match; matched known backbone '%s' as substring.",
                    name, known,
                )
                return known
        raise RuntimeError(
            f"Backbone '{name}' from training_summary.json is not a supported/known architecture. "
            f"Supported: {list(BASE_MODEL_BUILDERS.keys())}"
        )
    return name


def resolve_num_classes(
    training_summary: Dict[str, Any], disease_registry: Dict[str, Any], logger: logging.Logger,
) -> int:
    """Resolve ``num_classes`` from ``training_summary.json``, falling back
    to the length of the class list in ``disease_registry.json``.

    Raises:
        RuntimeError: if neither source yields a usable value.
    """
    candidate_keys = ["num_classes", "n_classes", "output_dim", "num_labels"]
    found_key, value = _search_keys(training_summary, candidate_keys)
    if value is not None:
        n = int(value)
        logger.info("MODEL num_classes resolved via training_summary key '%s' = %d", found_key, n)
        return n
    for key in ("classes", "class_names", "diseases", "labels"):
        if isinstance(disease_registry.get(key), list):
            n = len(disease_registry[key])
            logger.info("MODEL num_classes resolved from disease_registry['%s'] length = %d", key, n)
            return n
    raise RuntimeError("Unable to resolve num_classes from training_summary.json or disease_registry.json.")


def resolve_dropout(training_summary: Dict[str, Any], logger: logging.Logger) -> float:
    """Resolve the dropout probability from ``training_summary.json``,
    defaulting to 0.0 (logged) if not present. Dropout has no learnable
    parameters and the model always runs in ``eval()`` mode, so a missing
    value has no effect on checkpoint compatibility -- it's only logged for
    traceability, matching the original notebook's reasoning exactly."""
    candidate_keys = ["dropout", "dropout_rate", "dropout_p"]
    found_key, value = _search_keys(training_summary, candidate_keys)
    if value is not None:
        p = float(value)
        logger.info("MODEL dropout resolved via training_summary key '%s' = %.3f", found_key, p)
        return p
    logger.warning(
        "MODEL dropout not found in training_summary.json; defaulting to 0.0. "
        "This has no effect on checkpoint compatibility since Dropout has no learnable "
        "parameters and the model is run in eval() mode, but is logged for traceability."
    )
    return 0.0


# ======================================================================
# CHECKPOINT LOADING
# ======================================================================


def _torch_load(path: Path, device: torch.device) -> Any:
    """``torch.load`` wrapper tolerant of ``weights_only`` kwarg availability
    across torch versions. This loads our OWN previously-produced
    checkpoint -- not an untrusted third-party file. (Original notebook
    comment, preserved.)"""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Any:
    """Load a checkpoint file, wrapping any failure in a ``RuntimeError``
    with the checkpoint path for context.

    Raises:
        RuntimeError: if ``torch.load`` fails for any reason.
    """
    try:
        return _torch_load(checkpoint_path, device)
    except Exception as exc:
        raise RuntimeError(f"Failed to torch.load checkpoint at {checkpoint_path}: {exc}") from exc


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """Unwrap known checkpoint container conventions only -- no silent key
    surgery.

    Raises:
        RuntimeError: if no valid state_dict can be located.
    """
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise RuntimeError("Unable to locate a valid state_dict inside the checkpoint object.")


def compute_model_fingerprint(checkpoint_path: Path, logger: logging.Logger) -> str:
    """Compute the checkpoint's SHA-256 fingerprint and log it.

    Thin, purpose-named wrapper around the canonical
    :func:`inference.utils.hashing.sha256_of_file` -- kept as its own
    function (rather than calling ``sha256_of_file`` directly from
    :func:`reconstruct_model`) so the model-loading call site reads as
    "fingerprint the checkpoint" and so other callers needing the same
    fingerprint don't have to know it happens to be implemented via a
    generic file-hashing utility.
    """
    checkpoint_sha256 = sha256_of_file(checkpoint_path)
    logger.info("MODEL checkpoint sha256=%s path=%s", checkpoint_sha256, checkpoint_path)
    return checkpoint_sha256


def validate_checkpoint(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    backbone_name: str,
    checks: ModelValidationChecks,
    logger: logging.Logger,
) -> None:
    """Strictly load ``state_dict`` into ``model`` (in place) and verify an
    exact key match. Mutates ``checks.checkpoint_loaded`` and
    ``checks.state_dict_strict_match`` on success.

    Raises:
        RuntimeError: if there are any missing or unexpected keys. No
            silent fixes are permitted, matching the original notebook
            exactly.
    """
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint is NOT compatible with the reconstructed architecture "
            f"(backbone='{backbone_name}'). missing_keys={missing_keys} "
            f"unexpected_keys={unexpected_keys}. No silent fixes permitted."
        )
    checks.checkpoint_loaded = True
    checks.state_dict_strict_match = True
    logger.info("MODEL checkpoint loaded with STRICT key match (0 missing, 0 unexpected).")


def verify_classifier_dimensions(
    model: nn.Module,
    classifier_path: str,
    num_classes: int,
    checks: ModelValidationChecks,
) -> int:
    """Verify the classifier head's output dimension matches
    ``num_classes``, via attribute inspection only (NOT a forward pass).

    On mismatch, appends to ``checks.errors`` rather than raising
    immediately -- matches the original notebook's aggregate-then-raise
    pattern, where this check is one of several batched together and raised
    as a single ``RuntimeError`` at the end of :func:`reconstruct_model`.

    Returns:
        The resolved output feature count (used later to build the model
        signature).
    """
    out_features = _final_out_features(_get_nested_attr(model, classifier_path))
    checks.classifier_dimension_expected = out_features == num_classes
    checks.output_dimension_expected = out_features == num_classes
    if not checks.classifier_dimension_expected:
        checks.errors.append(
            f"Classifier output dimension {out_features} != expected num_classes {num_classes}"
        )
    return out_features


def get_model_signature(
    backbone_name: str,
    num_classes: int,
    dropout: float,
    classifier_path: str,
    output_features: int,
    total_parameters: int,
    trainable_parameters: int,
    device: torch.device,
    dtype: str,
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """Build the model signature dict recorded on :class:`ModelRegistry`.

    Pure function, no side effects. ``state_dict_key_hash`` hashes the
    sorted, pipe-joined state_dict key names (not file bytes) -- this is
    deliberately a separate hash from the checkpoint file's own SHA-256
    fingerprint (see :func:`compute_model_fingerprint`), so it uses
    ``hashlib`` directly rather than the file-hashing utility.
    """
    return {
        "backbone": backbone_name,
        "num_classes": num_classes,
        "dropout": dropout,
        "classifier_path": classifier_path,
        "output_features": output_features,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "device": str(device),
        "dtype": dtype,
        "state_dict_key_count": len(state_dict),
        "state_dict_key_hash": hashlib.sha256("|".join(sorted(state_dict.keys())).encode()).hexdigest(),
    }


# ======================================================================
# ORCHESTRATION
# ======================================================================


def reconstruct_model(
    training_summary: Dict[str, Any],
    disease_registry: Dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[nn.Module, ModelRegistry]:
    """Reconstruct the production model from a frozen checkpoint, with full
    engineering validation.

    Orchestrates, in the exact order the original monolithic notebook
    function executed them: backbone/num_classes/dropout resolution ->
    architecture instantiation -> checkpoint fingerprinting -> checkpoint
    loading -> state_dict extraction -> strict state_dict validation
    (fail-fast) -> device placement + eval mode + grad-disable -> parameter
    count check (aggregated) -> classifier dimension check (aggregated) ->
    device/dtype consistency checks (aggregated) -> raise if any aggregated
    errors -> build model signature -> build ModelRegistry.

    Raises:
        RuntimeError: if checkpoint loading fails, the state_dict doesn't
            strictly match the reconstructed architecture, or any
            aggregated validation check fails (parameter count, classifier
            dimension, device placement, dtype).

    Returns:
        ``(model, registry)`` -- the reconstructed, validated, eval-mode
        model and its :class:`ModelRegistry` record.
    """
    checks = ModelValidationChecks()

    backbone_name = resolve_backbone_name(training_summary, logger)
    num_classes = resolve_num_classes(training_summary, disease_registry, logger)
    dropout = resolve_dropout(training_summary, logger)

    model, classifier_path = build_model(backbone_name, num_classes, dropout)
    checks.architecture_instantiated = True
    logger.info(
        "MODEL architecture instantiated: backbone=%s num_classes=%d dropout=%.3f classifier_path=%s",
        backbone_name, num_classes, dropout, classifier_path,
    )

    checkpoint_sha256 = compute_model_fingerprint(checkpoint_path, logger)
    checkpoint = load_checkpoint(checkpoint_path, device)
    state_dict = extract_state_dict(checkpoint)

    validate_checkpoint(model, state_dict, backbone_name, checks, logger)

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    checks.eval_mode = not model.training
    checks.grad_disabled = all(not p.requires_grad for p in model.parameters())

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Parameter-count-vs-training-summary check. Kept inline (not promoted
    # to its own named function) -- not part of the requested public
    # interface for this phase and has no other caller yet. Identical logic
    # to the original.
    expected_total = training_summary.get("total_parameters") or training_summary.get("num_parameters")
    if expected_total:
        checks.parameter_count_expected = int(expected_total) == total_params
        if not checks.parameter_count_expected:
            checks.errors.append(
                f"Parameter count mismatch: reconstructed={total_params} expected={expected_total}"
            )
    else:
        logger.warning("MODEL training_summary.json has no recorded parameter count; skipping exact-match check.")

    out_features = verify_classifier_dimensions(model, classifier_path, num_classes, checks)

    # Device/dtype consistency checks. Kept inline for the same reason as
    # the parameter-count check above. Identical logic to the original.
    devices = {p.device.type for p in model.parameters()}
    checks.all_params_correct_device = devices == {device.type}
    if not checks.all_params_correct_device:
        checks.errors.append(f"Parameters found on unexpected devices: {devices}")

    dtypes = {p.dtype for p in model.parameters()}
    checks.all_params_correct_dtype = dtypes == {torch.float32}
    if not checks.all_params_correct_dtype:
        checks.errors.append(f"Parameters found with unexpected dtypes: {dtypes}")

    if checks.errors:
        raise RuntimeError("Model reconstruction validation FAILED: " + "; ".join(checks.errors))

    model_signature = get_model_signature(
        backbone_name=backbone_name,
        num_classes=num_classes,
        dropout=dropout,
        classifier_path=classifier_path,
        output_features=out_features,
        total_parameters=total_params,
        trainable_parameters=trainable_params,
        device=device,
        dtype="float32",
        state_dict=state_dict,
    )

    registry = ModelRegistry(
        backbone=backbone_name,
        num_classes=num_classes,
        device=str(device),
        dtype="float32",
        total_parameters=total_params,
        trainable_parameters=trainable_params,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=checkpoint_sha256,
        model_signature=model_signature,
        validation=checks,
    )
    return model, registry
