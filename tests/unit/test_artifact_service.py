"""Unit tests for services/artifact_service.py."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from services.artifact_service import ArtifactService, resolve_artifact_file, select_canonical_candidate


def _build_service(synthetic_fixture, tmp_path, logger) -> ArtifactService:
    return ArtifactService(
        artifact_roots=synthetic_fixture["artifact_roots"],
        export_dir=tmp_path / "export",
        device=torch.device("cpu"),
        logger=logger,
    )


def _touch(path: Path, content: str = "{}") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _build_nested_artifact_tree(root: Path) -> dict:
    """A tree shaped like the real local export -- each category's files
    live several directories deep (``registry/``, ``checkpoints/``,
    ``stage05_metrics_engine/``, ...), plus duplicate ``best_model.pt``
    copies under scratch/smoke-test subdirectories, exactly the layout
    ``discover_category_inventory``'s old flat ``root / filename`` lookup
    could not see."""
    sprint03 = root / "sprint03"
    sprint04 = root / "sprint04"
    evaluation = root / "sprint04_evaluation"

    _touch(sprint03 / "registry" / "disease_registry.json", json.dumps({"classes": ["A", "B"]}))
    _touch(sprint03 / "statistics" / "dataset_statistics.json")
    _touch(sprint03 / "manifests" / "train_manifest.csv", "a,b\n1,2\n")

    canonical_checkpoint = sprint04 / "checkpoints" / "best_model.pt"
    canonical_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"stub": True}, canonical_checkpoint)
    for scratch_dir in ("_s31_smoke", "_s36_run_a", "_s36_run_apm"):
        scratch_checkpoint = sprint04 / "checkpoints" / scratch_dir / "best_model.pt"
        scratch_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"stub": True}, scratch_checkpoint)
    _touch(sprint04 / "training_summary.json", json.dumps({"backbone": "resnet18", "num_classes": 2}))
    _touch(sprint04 / "checkpoints" / "checkpoint_summary.json")

    _touch(evaluation / "stage05_metrics_engine" / "evaluation_summary.json", json.dumps({"classes": ["A", "B"]}))
    _touch(evaluation / "stage06_threshold_calibration_engine" / "optimal_thresholds.json", json.dumps({"A": 0.5, "B": 0.5}))
    _touch(evaluation / "logs" / "stage05_metrics_engine.log", "not an artifact")

    return {
        "artifact_roots": {
            "sprint03": str(sprint03),
            "sprint04_training": str(sprint04),
            "sprint04_evaluation": str(evaluation),
            "nih_chest_xray": None,
        },
        "canonical_checkpoint": canonical_checkpoint,
    }


def test_discover_finds_files_nested_under_category_subdirectories(tmp_path, logger):
    # Reproduces the real local layout (e.g. artifacts/sprint03/registry/
    # disease_registry.json rather than artifacts/sprint03/disease_registry.json)
    # that the old flat root/filename lookup could not find.
    tree = _build_nested_artifact_tree(tmp_path / "artifacts")
    service = ArtifactService(
        artifact_roots=tree["artifact_roots"], export_dir=tmp_path / "export",
        device=torch.device("cpu"), logger=logger,
    )
    status = service.artifact_status()
    assert status["healthy"] is True
    assert status["serving_critical_missing"] == []
    assert service.load_disease_registry()["classes"] == ["A", "B"]


def test_duplicate_checkpoints_resolve_to_the_canonical_copy(tmp_path, logger):
    # best_model.pt exists under checkpoints/ AND under several smoke-test /
    # scratch-run subdirectories -- the canonical (non-scratch) copy must
    # win, and it must never be silently chosen (verified via caplog).
    tree = _build_nested_artifact_tree(tmp_path / "artifacts")
    service = ArtifactService(
        artifact_roots=tree["artifact_roots"], export_dir=tmp_path / "export",
        device=torch.device("cpu"), logger=logger,
    )
    assert service.checkpoint_path() == tree["canonical_checkpoint"]


def test_resolve_artifact_file_logs_selection_when_duplicates_exist(tmp_path):
    tree = _build_nested_artifact_tree(tmp_path / "artifacts")
    log = logging.getLogger("visionserve.tests.duplicate_logging")
    log.setLevel(logging.INFO)
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    log.addHandler(_Capture())

    selected = resolve_artifact_file(
        tree["artifact_roots"]["sprint04_training"], "best_model.pt", "sprint04_training", log,
    )
    assert Path(selected) == tree["canonical_checkpoint"]
    assert any("Found 4 copies of best_model.pt" in msg and "Selected" in msg and "Ignored" in msg for msg in records)


def test_select_canonical_candidate_prefers_registry_over_archive_dir(tmp_path):
    root = tmp_path / "sprint03"
    canonical = _touch(root / "registry" / "disease_registry.json")
    archived = _touch(root / "archive" / "disease_registry.json")

    selected, ignored = select_canonical_candidate(root, [canonical, archived], "sprint03")
    assert selected == canonical
    assert ignored == [archived]


def test_get_path_returns_none_when_filename_not_found_anywhere_under_root(tmp_path, logger):
    tree = _build_nested_artifact_tree(tmp_path / "artifacts")
    service = ArtifactService(
        artifact_roots=tree["artifact_roots"], export_dir=tmp_path / "export",
        device=torch.device("cpu"), logger=logger,
    )
    assert service.get_path("sprint03", "this_file_does_not_exist.json") is None


def test_discover_finds_all_serving_critical_files(synthetic_fixture, tmp_path, logger):
    service = _build_service(synthetic_fixture, tmp_path, logger)
    registry = service.discover()
    assert registry.total_artifacts > 0
    status = service.artifact_status()
    assert status["healthy"] is True
    assert status["serving_critical_missing"] == []


def test_discover_is_cached_until_force_true(synthetic_fixture, tmp_path, logger):
    service = _build_service(synthetic_fixture, tmp_path, logger)
    first = service.discover()
    second = service.discover()
    assert first is second  # cached, same object

    third = service.discover(force=True)
    assert third is not first  # re-discovered


def test_load_training_summary_and_disease_registry_round_trip(synthetic_fixture, tmp_path, logger):
    service = _build_service(synthetic_fixture, tmp_path, logger)
    training_summary = service.load_training_summary()
    disease_registry = service.load_disease_registry()

    assert training_summary["backbone"] == synthetic_fixture["backbone"]
    assert training_summary["num_classes"] == synthetic_fixture["num_classes"]
    assert disease_registry["classes"] == synthetic_fixture["class_names"]


def test_checkpoint_path_resolves_to_the_real_file(synthetic_fixture, tmp_path, logger):
    service = _build_service(synthetic_fixture, tmp_path, logger)
    assert service.checkpoint_path() == synthetic_fixture["checkpoint_path"]


def test_torchscript_and_onnx_paths_are_none_when_not_exported(synthetic_fixture, tmp_path, logger):
    # No export step has run for this fixture (export_dir is empty) --
    # ArtifactService must report None, not raise, matching how a real
    # deployment without Stage 3 export should behave.
    service = _build_service(synthetic_fixture, tmp_path, logger)
    assert service.torchscript_path() is None
    assert service.onnx_path() is None


def test_torchscript_path_resolves_once_exported(synthetic_fixture, tmp_path, logger, reconstructed_model):
    from scripts._synthetic_fixtures import export_torchscript

    export_dir = tmp_path / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    model, _ = reconstructed_model
    export_torchscript(model, export_dir / "model.ts", (1, 3, 224, 224))

    service = ArtifactService(
        artifact_roots=synthetic_fixture["artifact_roots"], export_dir=export_dir,
        device=torch.device("cpu"), logger=logger,
    )
    assert service.torchscript_path() == export_dir / "model.ts"


def test_get_path_returns_none_for_unknown_filename(synthetic_fixture, tmp_path, logger):
    service = _build_service(synthetic_fixture, tmp_path, logger)
    assert service.get_path("sprint03", "this_file_does_not_exist.json") is None


def test_serving_critical_missing_excludes_non_serving_categories(synthetic_fixture, tmp_path, logger):
    # The synthetic fixture deliberately leaves "nih_chest_xray" unresolved
    # (artifact_roots["nih_chest_xray"] is None) -- it's a training/eval
    # dataset category, never needed to serve a prediction. Confirms
    # ArtifactService distinguishes "critical for the full research
    # lineage" from "critical to serve".
    service = _build_service(synthetic_fixture, tmp_path, logger)
    status = service.artifact_status()
    assert status["healthy"] is True
    registry = service.discover()
    assert len(registry.critical_missing) >= 1  # BBox_List_2017.csv is genuinely missing
    assert status["serving_critical_missing"] == []  # but that doesn't block serving