"""
Version metadata for the FastAPI orchestration layer.

Distinct from *model* version (``services.model_service.ModelService.
model_version``, a function of the loaded checkpoint's backbone + SHA-256
fingerprint) -- this module reports the *API package's* version, read from
installed package metadata (``pyproject.toml``'s ``[project] version``) so
it never drifts out of sync with what was actually packaged/deployed.
"""
from __future__ import annotations

from importlib import metadata as _importlib_metadata

#: Distribution name declared in pyproject.toml's [project] table.
_DISTRIBUTION_NAME = "visionserve-api"

#: Fallback used only if the package isn't installed in a way that exposes
#: metadata (e.g. running straight out of a source checkout without
#: ``pip install -e .``) -- never silently guessed as anything other than
#: an obviously-a-fallback value.
_FALLBACK_VERSION = "0.0.0+unknown"


def get_api_version() -> str:
    """Installed ``visionserve-api`` distribution version, or an explicit
    fallback if metadata isn't available."""
    try:
        return _importlib_metadata.version(_DISTRIBUTION_NAME)
    except _importlib_metadata.PackageNotFoundError:
        return _FALLBACK_VERSION


API_VERSION: str = get_api_version()
