"""
Cloud-native artifact provisioning: downloads and caches production
artifacts from a HuggingFace Hub model repository, so a deployment
platform (Railway, etc.) never needs artifacts manually uploaded or
bind-mounted.

Activated only when ``HF_REPO_ID`` is configured (see
``backend.settings.Settings.hf_repo_id``) -- a no-op otherwise, so every
existing local/Docker Compose workflow that relies on
``VISIONSERVE_ARTIFACT_ROOT`` / ``VISIONSERVE_ROOT_*`` bind-mounted
directories is completely unaffected.

Flow (explicit and logged at each step, matching this project's existing
convention of never silently falling back -- see e.g. the frozen runtime
layer's own logged ImageNet-normalization fallback):

    1. Check for a completion marker in ``cache_dir`` written by a
       previous successful download of this exact ``repo_id``+``revision``,
       AND re-verify the required files are still actually present (a
       volume could have been partially cleared since).
    2. Cache hit -> log it, skip the network entirely, return immediately.
    3. Cache miss -> call ``huggingface_hub.snapshot_download()``, which
       replicates the repo's directory structure verbatim into
       ``cache_dir``. Because that structure (``sprint03/``, ``sprint04/``,
       ``sprint04_evaluation/``, ``Sprint05_Complete/``) is exactly what
       ``scripts.resolve_artifact_roots``'s fingerprint-based discovery
       already expects from a *local* directory, the returned path is fed
       straight into that same, unmodified discovery function by
       ``backend.settings.Settings.resolve_artifact_roots`` -- no new
       category-mapping logic exists here or anywhere else.
    4. Verify the files this deployment cannot run without are present
       after download -- fails loudly with a typed error rather than
       continuing into a ``ServiceRegistry.initialize()`` failure that
       would surface several layers deeper with a less obvious cause.
    5. Write the completion marker, return ``cache_dir``.

Failure here is a normal ``RuntimeError`` subclass, so it flows through
the *existing* ``backend.lifecycle`` fail-fast/degraded-startup toggle
(``VISIONSERVE_FAIL_FAST_ON_STARTUP``) exactly like any other startup
failure already does -- no new toggle was added.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

#: Marker file recording which repo_id/revision this cache directory holds.
_MARKER_FILENAME = ".hf_snapshot_complete.json"

#: Files this deployment cannot function without, checked immediately after
#: download. NOT a reimplementation of ArtifactService's own (far more
#: thorough) discovery/validation -- ServiceRegistry.initialize() ->
#: ArtifactService still performs that in full. This is only a fast,
#: HF-download-specific sanity check so a bad/incomplete/misconfigured
#: snapshot fails clearly at the provisioning step instead of surfacing as
#: a confusing error several layers deeper.
_REQUIRED_FILENAMES = ("best_model.pt", "disease_registry.json", "training_summary.json")


class ArtifactProvisioningError(RuntimeError):
    """Raised when artifacts cannot be obtained or verified from HuggingFace Hub."""


def ensure_artifacts_available(
    *,
    repo_id: str,
    cache_dir: Path,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Ensure ``cache_dir`` holds a complete local copy of ``repo_id`` (at
    ``revision``), downloading it from HuggingFace Hub if needed.

    Returns:
        ``cache_dir`` -- callers feed this straight into
        ``scripts.resolve_artifact_roots.resolve_artifact_roots()`` exactly
        as they would a bind-mounted host directory.

    Raises:
        ArtifactProvisioningError: if the download fails or the resulting
            directory is missing required files.
    """
    log = logger or logging.getLogger("visionserve.backend")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker_path = cache_dir / _MARKER_FILENAME

    if _is_cache_valid(marker_path, repo_id=repo_id, revision=revision, cache_dir=cache_dir, log=log):
        log.info(
            "ARTIFACT_PROVISIONING cache hit: repo_id=%s revision=%s dir=%s -- skipping download.",
            repo_id, revision or "main", cache_dir,
        )
        return cache_dir

    log.info(
        "ARTIFACT_PROVISIONING cache miss: downloading repo_id=%s revision=%s to %s ...",
        repo_id, revision or "main", cache_dir,
    )
    _download_snapshot(repo_id=repo_id, cache_dir=cache_dir, revision=revision, token=token, log=log)
    _verify_required_files(cache_dir)
    _write_marker(marker_path, repo_id=repo_id, revision=revision)
    log.info("ARTIFACT_PROVISIONING download complete and verified: dir=%s", cache_dir)
    return cache_dir


def _is_cache_valid(
    marker_path: Path, *, repo_id: str, revision: Optional[str], cache_dir: Path, log: logging.Logger,
) -> bool:
    if not marker_path.exists():
        return False
    try:
        marker = json.loads(marker_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("ARTIFACT_PROVISIONING marker unreadable (%s) -- treating as cache miss.", exc)
        return False

    if marker.get("repo_id") != repo_id or marker.get("revision") != (revision or "main"):
        log.info("ARTIFACT_PROVISIONING marker is for a different repo_id/revision -- treating as cache miss.")
        return False

    try:
        _verify_required_files(cache_dir)
    except ArtifactProvisioningError:
        log.warning("ARTIFACT_PROVISIONING marker present but required files are missing -- treating as cache miss.")
        return False
    return True


def _download_snapshot(
    *, repo_id: str, cache_dir: Path, revision: Optional[str], token: Optional[str], log: logging.Logger,
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ArtifactProvisioningError(
            "HF_REPO_ID is set but the 'huggingface_hub' package is not installed "
            "(add it to requirements.txt -- see backend/artifacts_provider.py's module docstring)."
        ) from exc

    try:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(cache_dir),
            token=token,
        )
    except Exception as exc:  # noqa: BLE001 -- any HF Hub failure becomes one typed, clear error
        raise ArtifactProvisioningError(
            f"Failed to download artifacts from HuggingFace Hub "
            f"repo_id={repo_id!r} revision={revision or 'main'!r}: {exc}"
        ) from exc


def _verify_required_files(cache_dir: Path) -> None:
    missing = [name for name in _REQUIRED_FILENAMES if not any(cache_dir.rglob(name))]
    if missing:
        raise ArtifactProvisioningError(
            f"Downloaded snapshot at {cache_dir} is missing required file(s): {missing}. "
            "Check HF_REPO_ID / HF_REVISION and the repository's actual contents."
        )


def _write_marker(marker_path: Path, *, repo_id: str, revision: Optional[str]) -> None:
    marker_path.write_text(json.dumps({"repo_id": repo_id, "revision": revision or "main"}))
