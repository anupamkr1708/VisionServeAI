"""Unit tests for inference/model_loader.py."""
from __future__ import annotations

import json

import pytest
import torch

from inference.model_loader import (
    BASE_MODEL_BUILDERS,
    build_model,
    compute_model_fingerprint,
    extract_state_dict,
    reconstruct_model,
    resolve_dropout,
    resolve_num_classes,
)


def test_build_model_produces_correct_output_shape():
    model, classifier_path = build_model("resnet18", num_classes=7, dropout=0.1)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 224, 224))
    assert tuple(output.shape) == (2, 7)
    assert classifier_path == "backbone.fc"


def test_build_model_supports_every_known_backbone_except_vgg16():
    # See test_vgg16_backbone_has_a_known_notebook_inherited_defect below --
    # vgg16 is excluded here on purpose, not an oversight.
    for backbone_name in BASE_MODEL_BUILDERS:
        if backbone_name == "vgg16":
            continue
        model, _ = build_model(backbone_name, num_classes=3, dropout=0.0)
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 224, 224))
        assert tuple(output.shape) == (1, 3), f"backbone={backbone_name}"


def test_vgg16_backbone_has_a_known_notebook_inherited_defect():
    """Documents a genuine, pre-existing defect discovered while writing
    this test suite -- NOT introduced by the migration, and NOT fixed here
    (Model Loader is a frozen phase; fixing this is a product decision for
    a human to make, not a silent migration-cleanup change).

    ``_base_vgg16`` (``inference/model_loader.py``, and byte-identical in
    the notebook's Stage 2, ``sprint05-deployment.ipynb`` lines ~114-116)
    reads ``in_features`` from ``m.classifier[-1].in_features`` (4096 --
    VGG16's *last* classifier layer). ``WrappedClassifier`` then replaces
    the *entire* ``classifier`` attribute (VGG16's real classifier is a
    3-layer MLP expecting a 25088-dim flattened input, not 4096) with a
    single ``Linear(4096, num_classes)``. The result crashes on its very
    first forward pass with a shape mismatch (25088 vs 4096) -- this isn't
    a corner case, it's guaranteed for every input.

    ``efficientnet_b0`` uses the identical `classifier[-1].in_features`
    pattern safely, because EfficientNet's classifier genuinely is a
    single-Linear head over a pooled feature vector -- VGG16's is not.

    Net effect: if ``training_summary.json``'s resolved backbone is ever
    ``"vgg16"``, model reconstruction will build successfully (no
    checkpoint-compatibility error -- this defect is architectural, not a
    key-matching one) but the very first prediction will raise. If the real
    Sprint 04 checkpoint never actually used a VGG16 backbone, this is
    inert; if it might, this needs a human decision (fix ``_base_vgg16`` to
    replace only ``classifier[6]``, not all of ``classifier``) before this
    repository can be trusted to serve a VGG16-backboned model. Flagged
    here rather than fixed silently.
    """
    model, _ = build_model("vgg16", num_classes=3, dropout=0.0)
    model.eval()
    with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
        with torch.no_grad():
            model(torch.randn(1, 3, 224, 224))


def test_resolve_num_classes_prefers_training_summary(logger):
    n = resolve_num_classes({"num_classes": 5}, {"classes": ["a", "b"]}, logger)
    assert n == 5


def test_resolve_num_classes_falls_back_to_disease_registry_length(logger):
    n = resolve_num_classes({}, {"classes": ["a", "b", "c"]}, logger)
    assert n == 3


def test_resolve_num_classes_raises_if_unresolvable(logger):
    with pytest.raises(RuntimeError):
        resolve_num_classes({}, {}, logger)


def test_resolve_dropout_defaults_to_zero_when_absent(logger):
    assert resolve_dropout({}, logger) == 0.0


def test_resolve_dropout_reads_recorded_value(logger):
    assert resolve_dropout({"dropout": 0.35}, logger) == 0.35


def test_extract_state_dict_accepts_raw_dict():
    model, _ = build_model("resnet18", num_classes=2, dropout=0.0)
    extracted = extract_state_dict(model.state_dict())
    assert set(extracted.keys()) == set(model.state_dict().keys())


def test_extract_state_dict_accepts_wrapped_state_dict_key():
    model, _ = build_model("resnet18", num_classes=2, dropout=0.0)
    wrapped = {"state_dict": model.state_dict(), "epoch": 12}
    extracted = extract_state_dict(wrapped)
    assert set(extracted.keys()) == set(model.state_dict().keys())


def test_extract_state_dict_accepts_nn_module_directly():
    model, _ = build_model("resnet18", num_classes=2, dropout=0.0)
    extracted = extract_state_dict(model)
    assert set(extracted.keys()) == set(model.state_dict().keys())


def test_extract_state_dict_raises_on_unrecognized_container():
    with pytest.raises(RuntimeError):
        extract_state_dict({"totally_unrelated_key": "not a state dict"})


def test_compute_model_fingerprint_is_deterministic_and_matches_file_content(tmp_path, logger):
    path = tmp_path / "ckpt.pt"
    path.write_bytes(b"identical content")
    fp1 = compute_model_fingerprint(path, logger)
    fp2 = compute_model_fingerprint(path, logger)
    assert fp1 == fp2
    assert len(fp1) == 64  # sha256 hex digest


def test_reconstruct_model_round_trips_synthetic_checkpoint(synthetic_fixture, logger):

    training_summary = json.loads(synthetic_fixture["training_summary_path"].read_text())
    disease_registry = json.loads(synthetic_fixture["disease_registry_path"].read_text())

    model, registry = reconstruct_model(
        training_summary=training_summary,
        disease_registry=disease_registry,
        checkpoint_path=synthetic_fixture["checkpoint_path"],
        device=torch.device("cpu"),
        logger=logger,
    )

    assert registry.backbone == synthetic_fixture["backbone"]
    assert registry.num_classes == synthetic_fixture["num_classes"]
    assert registry.total_parameters > 0
    # reconstruct_model deliberately disables every parameter's
    # requires_grad (this is a frozen inference-only model -- see
    # ModelValidationChecks.grad_disabled), so trainable_parameters is
    # correctly 0, not a leftover from training.
    assert registry.trainable_parameters == 0
    assert registry.validation.grad_disabled is True
    assert registry.device == "cpu"
    assert len(registry.checkpoint_sha256) == 64
    assert not registry.validation.errors

    # The reconstructed model must actually be usable: eval-mode forward
    # pass produces the expected (batch, num_classes) shape.
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 224, 224))
    assert tuple(output.shape) == (1, synthetic_fixture["num_classes"])


def test_reconstruct_model_raises_on_class_count_mismatch(synthetic_fixture, logger):


    training_summary = json.loads(synthetic_fixture["training_summary_path"].read_text())
    disease_registry = json.loads(synthetic_fixture["disease_registry_path"].read_text())
    training_summary["num_classes"] = synthetic_fixture["num_classes"] + 1  # deliberately wrong

    with pytest.raises(RuntimeError):
        reconstruct_model(
            training_summary=training_summary,
            disease_registry=disease_registry,
            checkpoint_path=synthetic_fixture["checkpoint_path"],
            device=torch.device("cpu"),
            logger=logger,
        )
