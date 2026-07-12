"""
Runtime registry: a store of :class:`RuntimeRegistryEntry` records -- one
per loaded (and optionally validated) runtime backend -- keyed by runtime
type.

NEW in this phase -- no single notebook stage carried a structure exactly
like this. It exists to answer the same question Stage 3's
``ExportRegistry`` and Stage 7's ``resolve_supported_runtimes`` /
``build_compatibility_report`` each answered separately and differently
("which runtimes exist, what are their capabilities, which providers do
they use, did they validate cleanly") with one common record shape,
populated directly from whichever concrete :class:`BaseRuntime` instances a
caller has actually constructed via :class:`RuntimeFactory` in this
process -- not by re-deriving the answer from files on disk the way Stage
7's compatibility report did.

This is a plain in-memory registry (mirrors ``ModelRegistry`` /
``ThresholdRegistry`` elsewhere in this package: a dataclass-based record
plus, here, a small keyed store on top, since -- unlike those single-result
registries -- more than one runtime can coexist at once, exactly as Stage 5
loaded PyTorch, TorchScript, and ONNX side by side for its cross-runtime
numerical-stability comparison).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from inference.runtimes.base import BaseRuntime, RuntimeInfo

VALIDATION_STATE_NOT_VALIDATED = "not_validated"
VALIDATION_STATE_PASSED = "passed"
VALIDATION_STATE_FAILED = "failed"


@dataclass
class RuntimeRegistryEntry:
    """One runtime backend's recorded identity, capabilities, providers,
    and validation outcome."""

    runtime_type: str
    runtime_version: str
    model_fingerprint_sha256: str
    runtime_fingerprint_sha256: Optional[str]
    device: str
    capabilities: Dict[str, Any]
    supported_providers: List[str]
    selected_providers: List[str]
    validation_state: str  # one of the VALIDATION_STATE_* constants above
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RuntimeRegistry:
    """Keyed store of :class:`RuntimeRegistryEntry` records, one per
    ``runtime_type`` currently registered in this process."""

    def __init__(self) -> None:
        self._entries: Dict[str, RuntimeRegistryEntry] = {}

    def register(self, runtime: BaseRuntime, validation_info: Optional[RuntimeInfo] = None) -> RuntimeRegistryEntry:
        """Record (or overwrite) the entry for ``runtime``.

        Args:
            runtime: A runtime that has at least had :meth:`~BaseRuntime.load`
                called (calling this before ``load()`` is not an error, but
                the entry will reflect an unloaded/not-yet-validated state).
            validation_info: The :class:`RuntimeInfo` returned by
                ``runtime.validate(...)``, if validation has been run.
                When omitted, the entry is recorded with
                ``validation_state="not_validated"`` rather than guessed at.

        Returns:
            The :class:`RuntimeRegistryEntry` just stored.
        """
        metadata = runtime.metadata()

        if validation_info is not None:
            validation_state = VALIDATION_STATE_PASSED if validation_info.validated else VALIDATION_STATE_FAILED
            validation_errors = list(validation_info.validation_errors)
        else:
            validation_state = VALIDATION_STATE_NOT_VALIDATED
            validation_errors = []

        entry = RuntimeRegistryEntry(
            runtime_type=metadata.runtime_type,
            runtime_version=metadata.runtime_version,
            model_fingerprint_sha256=metadata.model_fingerprint_sha256,
            runtime_fingerprint_sha256=metadata.runtime_fingerprint_sha256,
            device=metadata.device,
            capabilities=metadata.capabilities.to_dict(),
            supported_providers=list(runtime.available_providers),
            selected_providers=list(runtime.selected_providers),
            validation_state=validation_state,
            validation_errors=validation_errors,
        )
        self._entries[entry.runtime_type] = entry
        return entry

    def get(self, runtime_type: str) -> RuntimeRegistryEntry:
        """Look up a registered entry by runtime type.

        Raises:
            KeyError: with a clear message if ``runtime_type`` was never
                registered, rather than a bare ``KeyError`` on the
                underlying dict (matching
                ``inference.thresholding.get_threshold``'s convention).
        """
        try:
            return self._entries[runtime_type]
        except KeyError:
            raise KeyError(
                f"No runtime registered under '{runtime_type}'. Registered: {sorted(self._entries)}"
            ) from None

    def all(self) -> Dict[str, RuntimeRegistryEntry]:
        """All registered entries, keyed by runtime type."""
        return dict(self._entries)

    def to_dict(self) -> Dict[str, Any]:
        return {runtime_type: entry.to_dict() for runtime_type, entry in self._entries.items()}
