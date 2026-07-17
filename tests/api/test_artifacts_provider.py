"""Tests for backend.artifacts_provider -- cache-hit/cache-miss logic and
error handling, with huggingface_hub.snapshot_download mocked out (no real
network access; this test suite must never hit huggingface.co)."""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.artifacts_provider import ArtifactProvisioningError, ensure_artifacts_available

_LOGGER = logging.getLogger("visionserve.tests")


def _write_required_files(root: Path) -> None:
    (root / "sprint04" / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "sprint04" / "checkpoints" / "best_model.pt").write_bytes(b"fake-checkpoint")
    (root / "sprint03" / "registry").mkdir(parents=True, exist_ok=True)
    (root / "sprint03" / "registry" / "disease_registry.json").write_text("{}")
    (root / "sprint04" / "training_summary.json").write_text("{}")


def test_downloads_on_cache_miss(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"

    def _fake_snapshot_download(*, repo_id, revision, local_dir, token):
        _write_required_files(Path(local_dir))
        return str(local_dir)

    with patch("huggingface_hub.snapshot_download", side_effect=_fake_snapshot_download) as mock_dl:
        result = ensure_artifacts_available(
            repo_id="jibral1857/VisionServeAI", cache_dir=cache_dir, revision="main", token="hf_fake", logger=_LOGGER,
        )

    mock_dl.assert_called_once()
    assert result == cache_dir
    assert (cache_dir / ".hf_snapshot_complete.json").exists()


def test_skips_download_on_cache_hit(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_required_files(cache_dir)
    (cache_dir / ".hf_snapshot_complete.json").write_text('{"repo_id": "jibral1857/VisionServeAI", "revision": "main"}')

    with patch("huggingface_hub.snapshot_download") as mock_dl:
        result = ensure_artifacts_available(
            repo_id="jibral1857/VisionServeAI", cache_dir=cache_dir, revision="main", logger=_LOGGER,
        )

    mock_dl.assert_not_called()
    assert result == cache_dir


def test_redownloads_when_marker_repo_mismatches(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_required_files(cache_dir)
    (cache_dir / ".hf_snapshot_complete.json").write_text('{"repo_id": "someone-else/other-model", "revision": "main"}')

    def _fake_snapshot_download(*, repo_id, revision, local_dir, token):
        _write_required_files(Path(local_dir))
        return str(local_dir)

    with patch("huggingface_hub.snapshot_download", side_effect=_fake_snapshot_download) as mock_dl:
        ensure_artifacts_available(repo_id="jibral1857/VisionServeAI", cache_dir=cache_dir, logger=_LOGGER)

    mock_dl.assert_called_once()


def test_redownloads_when_marker_present_but_files_missing(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # Marker present, but required files are NOT (e.g. a volume got partially cleared).
    (cache_dir / ".hf_snapshot_complete.json").write_text('{"repo_id": "jibral1857/VisionServeAI", "revision": "main"}')

    def _fake_snapshot_download(*, repo_id, revision, local_dir, token):
        _write_required_files(Path(local_dir))
        return str(local_dir)

    with patch("huggingface_hub.snapshot_download", side_effect=_fake_snapshot_download) as mock_dl:
        ensure_artifacts_available(repo_id="jibral1857/VisionServeAI", cache_dir=cache_dir, logger=_LOGGER)

    mock_dl.assert_called_once()


def test_raises_typed_error_when_download_fails(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"

    with patch("huggingface_hub.snapshot_download", side_effect=RuntimeError("network is down")):
        with pytest.raises(ArtifactProvisioningError, match="network is down"):
            ensure_artifacts_available(repo_id="jibral1857/VisionServeAI", cache_dir=cache_dir, logger=_LOGGER)


def test_raises_typed_error_when_snapshot_missing_required_files(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"

    def _incomplete_snapshot_download(*, repo_id, revision, local_dir, token):
        # Simulates a misconfigured HF_REPO_ID/HF_REVISION: download
        # "succeeds" but the repo doesn't actually contain what we need.
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        return str(local_dir)

    with patch("huggingface_hub.snapshot_download", side_effect=_incomplete_snapshot_download):
        with pytest.raises(ArtifactProvisioningError, match="missing required file"):
            ensure_artifacts_available(repo_id="jibral1857/VisionServeAI", cache_dir=cache_dir, logger=_LOGGER)


def test_settings_hf_repo_id_defaults_to_none_and_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirms the whole HF path is a genuine no-op when HF_REPO_ID isn't
    set -- existing VISIONSERVE_ARTIFACT_ROOT-based local/Docker Compose
    workflows are completely unaffected."""
    from backend.settings import get_settings

    monkeypatch.delenv("HF_REPO_ID", raising=False)
    monkeypatch.delenv("VISIONSERVE_ARTIFACT_ROOT", raising=False)
    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert settings.hf_repo_id is None
        resolved = settings.resolve_artifact_roots()
        assert all(value is None for value in resolved.values())
    finally:
        get_settings.cache_clear()
