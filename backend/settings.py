"""
Backend settings: environment-driven configuration for the FastAPI
orchestration layer only.

This module owns exactly one concern -- resolving process configuration
(artifact locations, runtime backend choice, CORS/host policy, upload
limits, logging directory) from environment variables into one typed,
immutable object. It contains no ML logic: every value here is either
forwarded verbatim into ``services.service_registry.ServiceRegistry``
(already-frozen, already-validated construction arguments) or used purely
for HTTP-layer concerns (CORS origins, trusted hosts, upload size caps).

Deliberately NOT a ``pydantic-settings`` ``BaseSettings`` subclass: that
package is not in ``requirements.txt``, and adding a new runtime dependency
for a plain-``os.environ`` read would be a scope violation of "do not modify
requirements.txt unless absolutely unavoidable". A frozen dataclass with a
small set of typed parsing helpers gives the same "read once, typed,
immutable" behaviour without a new dependency.

Every ``VISIONSERVE_*`` variable has a safe, explicit default so the app can
boot in a local/dev/test environment with no environment configured at all
(against whatever ``artifact_roots`` resolution finds, or nothing -- see
:meth:`Settings.resolve_artifact_roots`, which returns an all-``None`` map
rather than raising when nothing is configured; ``ServiceRegistry.initialize()``
is what actually fails fast on missing critical artifacts, exactly as it
already does today).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from configs.defaults import OUTPUT_ROOT

#: Categories ``services.artifact_service.ArtifactService`` knows about --
#: mirrors ``services.artifact_service.KNOWN_CATEGORIES`` (not imported
#: directly to avoid reaching into that module's private constants from a
#: settings file; the four names are a stable, documented contract).
ARTIFACT_CATEGORIES: List[str] = [
    "sprint03", "sprint04_training", "sprint04_evaluation", "nih_chest_xray",
]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_list(name: str, default: List[str]) -> List[str]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_optional_path(name: str) -> Optional[Path]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return Path(raw)


@dataclass(frozen=True)
class Settings:
    """Immutable, process-wide backend configuration. Build via
    :func:`get_settings` (cached) rather than constructing directly, so the
    whole process shares one resolved configuration."""

    # ---- Artifact / model wiring (forwarded to ServiceRegistry) --------
    artifact_root: Optional[Path]              # single root -> auto-resolved via scripts.resolve_artifact_roots
    artifact_roots_override: Dict[str, Optional[str]]  # per-category VISIONSERVE_ROOT_<CATEGORY> overrides
    export_dir: Path
    runtime_type: str
    runtime_path: Optional[Path]
    warmup_iterations: int
    warmup_on_startup: bool
    validate_runtime_on_startup: bool
    enable_explainability: bool
    explainability_output_dir: Optional[Path]
    log_dir: Path
    seed: Optional[int]
    device: Optional[str]

    # ---- Startup behaviour ---------------------------------------------
    fail_fast_on_startup: bool

    # ---- HTTP-layer concerns --------------------------------------------
    cors_allow_origins: List[str]
    trusted_hosts: List[str]
    gzip_minimum_size: int
    max_upload_size_bytes: int
    max_batch_size: int

    # ---- API metadata (OpenAPI) -----------------------------------------
    api_title: str
    api_contact_name: str
    api_contact_url: str
    api_contact_email: str
    api_license_name: str
    api_license_url: str

    def resolve_artifact_roots(self, logger: Optional[logging.Logger] = None) -> Dict[str, Optional[str]]:
        """Resolve ``category -> directory`` for :class:`ServiceRegistry`.

        Resolution order:
          1. Explicit per-category overrides (``VISIONSERVE_ROOT_<CATEGORY>``)
             always win for that category.
          2. If ``artifact_root`` (``VISIONSERVE_ARTIFACT_ROOT``) is set,
             every category not already overridden is resolved from it via
             :func:`scripts.resolve_artifact_roots.resolve_artifact_roots` --
             the exact library entry point that module's own docstring
             documents for feeding ``ServiceRegistry`` directly.
          3. Anything still unresolved is ``None`` (matches
             ``ArtifactService``'s own tolerant-of-missing-category
             contract; ``ServiceRegistry.initialize()`` -> ``ModelService.
             load_model()`` -> ``ArtifactService`` is what fails fast on
             critical files actually being absent, not this function).
        """
        resolved: Dict[str, Optional[str]] = {category: None for category in ARTIFACT_CATEGORIES}

        if self.artifact_root is not None:
            from scripts.resolve_artifact_roots import resolve_artifact_roots as _resolve

            resolved.update(_resolve(str(self.artifact_root), logger=logger))

        for category, path in self.artifact_roots_override.items():
            if path is not None:
                resolved[category] = path

        return resolved


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build (and cache) the process-wide :class:`Settings`. Cached so every
    caller in the same process shares one resolved configuration -- and so
    tests can force a rebuild via ``get_settings.cache_clear()`` after
    changing environment variables."""
    artifact_roots_override = {
        category: os.environ.get(f"VISIONSERVE_ROOT_{category.upper()}")
        for category in ARTIFACT_CATEGORIES
    }

    export_dir = _env_optional_path("VISIONSERVE_EXPORT_DIR") or (OUTPUT_ROOT / "exports")
    explainability_output_dir = _env_optional_path("VISIONSERVE_EXPLAINABILITY_OUTPUT_DIR")

    return Settings(
        artifact_root=_env_optional_path("VISIONSERVE_ARTIFACT_ROOT"),
        artifact_roots_override=artifact_roots_override,
        export_dir=export_dir,
        runtime_type=os.environ.get("VISIONSERVE_RUNTIME_TYPE", "pytorch"),
        runtime_path=_env_optional_path("VISIONSERVE_RUNTIME_PATH"),
        warmup_iterations=_env_int("VISIONSERVE_WARMUP_ITERATIONS", 10),
        warmup_on_startup=_env_bool("VISIONSERVE_WARMUP_ON_STARTUP", True),
        validate_runtime_on_startup=_env_bool("VISIONSERVE_VALIDATE_RUNTIME_ON_STARTUP", True),
        enable_explainability=_env_bool("VISIONSERVE_ENABLE_EXPLAINABILITY", True),
        explainability_output_dir=explainability_output_dir,
        log_dir=_env_optional_path("VISIONSERVE_LOG_DIR") or Path("logs"),
        seed=(_env_int("VISIONSERVE_SEED", -1) if os.environ.get("VISIONSERVE_SEED") else None),
        device=os.environ.get("VISIONSERVE_DEVICE"),
        fail_fast_on_startup=_env_bool("VISIONSERVE_FAIL_FAST_ON_STARTUP", True),
        cors_allow_origins=_env_list("VISIONSERVE_CORS_ORIGINS", ["*"]),
        trusted_hosts=_env_list("VISIONSERVE_TRUSTED_HOSTS", ["*"]),
        gzip_minimum_size=_env_int("VISIONSERVE_GZIP_MIN_SIZE", 1024),
        max_upload_size_bytes=_env_int("VISIONSERVE_MAX_UPLOAD_SIZE_BYTES", 15 * 1024 * 1024),
        max_batch_size=_env_int("VISIONSERVE_MAX_BATCH_SIZE", 16),
        api_title=os.environ.get("VISIONSERVE_API_TITLE", "VisionServeAI API"),
        api_contact_name=os.environ.get("VISIONSERVE_API_CONTACT_NAME", "VisionServeAI Engineering"),
        api_contact_url=os.environ.get("VISIONSERVE_API_CONTACT_URL", "https://example.com/visionserveai"),
        api_contact_email=os.environ.get("VISIONSERVE_API_CONTACT_EMAIL", "support@example.com"),
        api_license_name=os.environ.get("VISIONSERVE_API_LICENSE_NAME", "Proprietary"),
        api_license_url=os.environ.get("VISIONSERVE_API_LICENSE_URL", "https://example.com/license"),
    )
