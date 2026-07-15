"""
Synthetic artifact fixture builder -- verification infrastructure only.

Builds a tiny, real (not mocked) checkpoint + training_summary.json +
disease_registry.json + optimal_thresholds.json tree, laid out exactly the
way ``services.artifact_service.ArtifactService`` expects to find one
(flat ``root/filename`` per category -- see that module's own docstring).
This lets ``scripts/smoke_test.py`` (in ``--synthetic`` mode) and the
``tests/`` suite exercise the *entire real pipeline* -- actual
``reconstruct_model()``, actual ``PyTorchRuntime``/``TorchScriptRuntime``/
``ONNXRuntime``, actual ``ExplainabilityEngine`` -- end to end, without
depending on the real (proprietary, multi-hundred-MB) Sprint 04 checkpoint
being present.

This is NOT a stand-in for testing against your real artifacts. A
synthetic ``resnet18`` with random weights proves the *pipeline* is wired
correctly (every service boundary, every runtime backend, every
explainability algorithm actually executes against a real
``nn.Module``/checkpoint pair) -- it says nothing about your actual model's
accuracy or your actual checkpoint's compatibility. Run
``scripts/smoke_test.py`` against your real ``artifact_roots`` (e.g. via
``scripts/resolve_artifact_roots.py``) to verify that.

Nothing here is migrated from the notebook -- it deliberately uses the
smallest supported backbone (``resnet18``, ``inference.model_loader.
BASE_MODEL_BUILDERS``) and a small ``num_classes`` for speed, not because
Sprint 05 used either.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from inference.model_loader import build_model

DEFAULT_BACKBONE = "resnet18"
DEFAULT_CLASS_NAMES: List[str] = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
]
DEFAULT_INPUT_SIZE = (224, 224)  # (height, width) -- matches
                                 # services.model_service.resolve_input_signature's
                                 # default, so the synthetic fixture exercises the
                                 # same preprocessing path real artifacts would.


def build_synthetic_artifact_tree(
    root: Path,
    backbone: str = DEFAULT_BACKBONE,
    class_names: List[str] = None,
    dropout: float = 0.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Build a complete, flat, ``ArtifactService``-compatible artifact tree
    under ``root`` with a real (randomly-initialized, not trained)
    checkpoint.

    Returns a dict with:
        ``artifact_roots``: ready to pass into ``ServiceRegistry``/
            ``ArtifactService``.
        ``class_names``, ``num_classes``, ``backbone``: the values used to
            build the checkpoint, for tests to assert against.
        ``checkpoint_path``, ``training_summary_path``,
        ``disease_registry_path``, ``thresholds_path``: individual file
            paths, for tests that want to inspect them directly.
    """
    class_names = list(class_names) if class_names is not None else list(DEFAULT_CLASS_NAMES)
    num_classes = len(class_names)

    sprint03_dir = root / "sprint03"
    training_dir = root / "sprint04_training"
    evaluation_dir = root / "sprint04_evaluation"
    for d in (sprint03_dir, training_dir, evaluation_dir):
        d.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    model, _classifier_path = build_model(backbone, num_classes, dropout)
    model.eval()

    checkpoint_path = training_dir / "best_model.pt"
    # Raw state_dict, no wrapper -- one of the three container conventions
    # inference.model_loader.extract_state_dict() accepts ("all values are
    # tensors" branch).
    torch.save(model.state_dict(), checkpoint_path)

    training_summary = {
        "backbone": backbone,
        "num_classes": num_classes,
        "dropout": dropout,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        # No normalization_mean/std recorded on purpose -- exercises
        # inference.preprocessing.resolve_preprocessing_config's documented
        # logged ImageNet fallback path, exactly as it would for any real
        # training_summary.json that doesn't record normalization stats.
    }
    training_summary_path = training_dir / "training_summary.json"
    training_summary_path.write_text(json.dumps(training_summary, indent=2))

    disease_registry = {"classes": class_names}
    disease_registry_path = sprint03_dir / "disease_registry.json"
    disease_registry_path.write_text(json.dumps(disease_registry, indent=2))

    # Flat-dict schema variant (schema 2 of the 3 load_threshold_registry
    # accepts) -- simplest one that still exercises real validation.
    thresholds = {name: 0.5 for name in class_names}
    thresholds_path = evaluation_dir / "optimal_thresholds.json"
    thresholds_path.write_text(json.dumps(thresholds, indent=2))

    evaluation_summary = {"classes": class_names, "metrics": {}}
    evaluation_summary_path = evaluation_dir / "evaluation_summary.json"
    evaluation_summary_path.write_text(json.dumps(evaluation_summary, indent=2))

    artifact_roots: Dict[str, Any] = {
        "sprint03": str(sprint03_dir),
        "sprint04_training": str(training_dir),
        "sprint04_evaluation": str(evaluation_dir),
        "nih_chest_xray": None,  # never consumed by the serving path -- see
                                 # ArtifactService.SERVING_CATEGORIES
    }

    return {
        "artifact_roots": artifact_roots,
        "class_names": class_names,
        "num_classes": num_classes,
        "backbone": backbone,
        "checkpoint_path": checkpoint_path,
        "training_summary_path": training_summary_path,
        "disease_registry_path": disease_registry_path,
        "thresholds_path": thresholds_path,
    }


def build_sample_image(path: Path, size: Tuple[int, int] = DEFAULT_INPUT_SIZE, seed: int = 0) -> Path:
    """Write a small synthetic RGB PNG (deterministic pseudo-random noise,
    not a real chest X-ray) to ``path`` and return it. Used wherever a
    predictable, non-copyrighted image input is needed to exercise
    preprocessing/prediction/explainability."""
    rng = np.random.default_rng(seed)
    height, width = size
    array = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)
    return path


def export_torchscript(model: torch.nn.Module, output_path: Path, input_shape: Tuple[int, int, int, int]) -> Path:
    """Export ``model`` to TorchScript at ``output_path`` via
    ``torch.jit.trace`` -- so ``TorchScriptRuntime`` can be exercised
    end-to-end against the same synthetic model
    ``build_synthetic_artifact_tree`` reconstructs from its checkpoint.
    Standalone verification tooling; not a port of Stage 3 (export
    generation remains explicitly out of scope for the production
    repository -- see ``inference/preprocessing.py``'s module docstring)."""
    model.eval()
    dummy = torch.randn(*input_shape, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)
    traced.save(str(output_path))
    return output_path


def export_onnx(model: torch.nn.Module, output_path: Path, input_shape: Tuple[int, int, int, int], opset_version: int = 18) -> Path:
    """Export ``model`` to ONNX at ``output_path`` via ``torch.onnx.export``
    -- so ``ONNXRuntime`` can be exercised end-to-end. See
    :func:`export_torchscript`'s docstring for the same out-of-scope
    caveat."""
    model.eval()
    dummy = torch.randn(*input_shape, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, str(output_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset_version,
    )
    return output_path
