"""
Artifact discovery and loading service.

Migrated from Sprint 05 **Stage 2** ("Deployment Artifact Registry & Model
Reconstruction") of the archived notebook -- specifically the artifact
*discovery/validation* half of that stage: ``ArtifactRecord``,
``ArtifactRegistry``, ``_classify_file_type``, ``validate_json_schema``,
``validate_artifact_file``, ``detect_duplicate_artifacts``,
``build_artifact_registry``, ``get_artifact_path``. Stage 2's model-
*reconstruction* half was already migrated into ``inference/model_loader.py``
in a prior phase and is not touched here; this module never reconstructs a
model, never touches thresholds beyond locating their file, and never
predicts.

Adaptation (behaviour-preserving for the ported algorithm; new for its entry
point): Stage 2's ``build_artifact_registry()`` consumed an "inventory" dict
that Stage 1 ("Deployment Environment & Artifact Discovery") had already
produced by recursively scanning Kaggle dataset mounts for files matching
``configs.defaults.FINGERPRINTS``. Stage 1 was explicitly flagged
out-of-scope in every prior phase's migration notes -- its scan targets
Kaggle-specific paths (see ``configs/defaults.py``'s own docstring: "a
candidate follow-up once the deployment/artifact_discovery.py phase is
migrated"). This service is that follow-up, written for a production
directory layout instead of a Kaggle notebook: given a resolved
``artifact_roots`` mapping (``configs.schema.DeploymentConfig.artifact_roots``,
category -> directory), :func:`discover_inventory` builds the identical
"inventory" shape Stage 1 used to produce -- ``{category: {"critical":
{filename: path_or_None}, "optional": {...}}}`` -- by recursively
searching each root (``Path(root).rglob(filename)``, see
:func:`resolve_artifact_file`) for each of
``configs.defaults.CRITICAL_FILES`` / ``OPTIONAL_FILES``, since real local
artifact roots nest their files under subdirectories that vary per
category and run rather than sitting flat inside the root. This is NEW
orchestration (a directory search, not present verbatim anywhere in the
notebook), but it feeds Stage 2's ``build_artifact_registry`` the exact
input shape that function already expects, so every downstream
validation/classification/hashing/duplicate-detection check runs
byte-for-byte as originally validated.

Checkpoint torch-readability is verified by calling the already-migrated
:func:`inference.model_loader.load_checkpoint` (wrapped in try/except)
rather than duplicating Stage 2's own ``_torch_load`` helper. File hashing
uses the already-migrated :func:`inference.utils.hashing.sha256_of_file`
rather than duplicating Stage 2's byte-identical copy of the same function.

Source: sprint05-deployment.ipynb, Stage 2, lines ~218-440 (ArtifactRecord /
ArtifactRegistry / discovery+validation algorithm, ported verbatim);
Stage 2 lines ~843-857 (``build_metadata_registry``'s category/filename
mapping -- ``training_metadata`` <- sprint04_training/training_summary.json,
``disease_metadata`` <- sprint03/disease_registry.json,
``deployment_metadata`` <- sprint04_evaluation/deployment_readiness.json --
reused here as the exact filenames :class:`ArtifactService`'s convenience
loaders read); Stage 2 line ~1083 (``expected_class_names`` derivation,
reused verbatim as a one-line helper).
"""
from __future__ import annotations

import csv
import fnmatch
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from configs.defaults import CRITICAL_FILES, OPTIONAL_FILES
from inference.model_loader import load_checkpoint
from inference.utils.hashing import sha256_of_file
from inference.utils.io import load_json

# ======================================================================
# CONSTANTS -- Stage 2's frozen category/filename contract (verbatim).
# ======================================================================

CATEGORY_SPRINT03 = "sprint03"
CATEGORY_TRAINING = "sprint04_training"
CATEGORY_EVALUATION = "sprint04_evaluation"
CATEGORY_NIH = "nih_chest_xray"

# The canonical category list this service discovers -- identical to
# configs.defaults.CRITICAL_FILES / OPTIONAL_FILES keys.
KNOWN_CATEGORIES: Tuple[str, ...] = (CATEGORY_SPRINT03, CATEGORY_TRAINING, CATEGORY_EVALUATION, CATEGORY_NIH)

# Loose structural contracts used ONLY for informational schema validation --
# "present" = ANY of these keys found. Ported verbatim from Stage 2.
EXPECTED_JSON_KEYS: Dict[str, List[str]] = {
    "training_summary.json": ["backbone", "architecture", "model_name", "arch", "model_architecture"],
    "disease_registry.json": ["classes", "class_names", "diseases", "labels"],
    "evaluation_summary.json": ["classes", "class_names", "diseases", "labels", "metrics"],
    "optimal_thresholds.json": ["thresholds", "classes", "class_names"],
}


# ======================================================================
# DATACLASSES (ported verbatim from Stage 2)
# ======================================================================


@dataclass
class ArtifactRecord:
    """Discovery/validation record for one artifact file."""

    artifact_id: str
    category: str
    filename: str
    path: Optional[str]
    is_critical: bool
    exists: bool = False
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    file_type: Optional[str] = None  # json | csv | torch | unknown
    json_readable: Optional[bool] = None
    csv_readable: Optional[bool] = None
    torch_readable: Optional[bool] = None
    schema_valid: Optional[bool] = None
    expected_keys_present: Optional[bool] = None
    schema_warnings: List[str] = field(default_factory=list)
    modified_time: Optional[float] = None
    duplicate_of: Optional[str] = None
    errors: List[str] = field(default_factory=list)


@dataclass
class ArtifactRegistry:
    """Aggregate discovery/validation result over every known artifact."""

    records: Dict[str, ArtifactRecord]
    total_artifacts: int
    critical_missing: List[str]
    duplicate_groups: List[List[str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_artifacts": self.total_artifacts,
            "critical_missing": self.critical_missing,
            "duplicate_groups": self.duplicate_groups,
            "records": {k: asdict(v) for k, v in self.records.items()},
        }


# ======================================================================
# DISCOVERY (NEW -- production directory scan; see module docstring)
#
# ``root / filename`` (a flat, one-level lookup) was the original shape
# here. Real local artifact exports are not flat -- the same category root
# routinely nests its files several directories deep, under subdirectory
# names that vary run to run (e.g. ``sprint03/registry/disease_registry.json``,
# ``sprint04/checkpoints/_s36_run_a/best_model.pt``). Discovery below is now
# a recursive search (``Path(root).rglob(filename)``) followed by canonical
# selection whenever a filename resolves to more than one real file on disk.
# ======================================================================

# Directory-name path segments (case-insensitive; matched with ``fnmatch``,
# so a trailing ``*`` acts as a wildcard) that mark a candidate as this
# category's authoritative location. Used only to RANK candidates when a
# filename has more than one match under a root -- a single match is always
# accepted as-is, preferred or not.
PREFERRED_SUBDIRS: Dict[str, Tuple[str, ...]] = {
    CATEGORY_SPRINT03: ("registry", "statistics", "manifests", "config"),
    CATEGORY_TRAINING: ("checkpoints", "metrics", "models", "optimizer", "scheduler"),
    CATEGORY_EVALUATION: ("stage05_metrics_engine", "stage06_threshold_calibration_engine", "stage08_report_generator", "summary"),
    CATEGORY_NIH: (),
}

# Directory-name path segments that mark a candidate as NOT canonical --
# archived/backup copies, ad-hoc smoke-test or scratch runs, or log/table
# exports that happen to share a filename with the real artifact. A
# candidate under one of these is never dropped outright (it's still
# reported in "Ignored"), only ranked last.
AVOID_SUBDIRS: Dict[str, Tuple[str, ...]] = {
    CATEGORY_SPRINT03: ("archive", "archieve", "backup", "old"),
    CATEGORY_TRAINING: (
        "_s31_smoke", "_s32_smoke", "_s33_smoke", "_s34_smoke", "_s35_smoke",
        "_s36_run_*", "backup", "resume",
    ),
    CATEGORY_EVALUATION: ("logs", "tables", "figures"),
    CATEGORY_NIH: (),
}


def _dir_matches(part: str, patterns: Tuple[str, ...]) -> bool:
    part = part.lower()
    return any(fnmatch.fnmatch(part, pattern.lower()) for pattern in patterns)


def select_canonical_candidate(root: Path, candidates: List[Path], category: str) -> Tuple[Path, List[Path]]:
    """Rank multiple real files matching the same filename and pick exactly
    one canonical path. Returns ``(selected, everything_else)`` -- the
    "everything else" list is what callers log as "Ignored", never
    silently dropped.

    Ranking, in order:
      1. Any path segment (between ``root`` and the file) matching this
         category's :data:`AVOID_SUBDIRS` sorts last.
      2. Among the rest, any path segment matching :data:`PREFERRED_SUBDIRS`
         sorts first.
      3. Shallower paths (fewer intervening directories) sort first.
      4. More recently modified files sort first.
      5. Path string, as a final deterministic tiebreak.
    """
    preferred = PREFERRED_SUBDIRS.get(category, ())
    avoided = AVOID_SUBDIRS.get(category, ())

    def sort_key(path: Path):
        rel_parts = path.relative_to(root).parts[:-1]  # directory components only, not the filename
        is_avoided = any(_dir_matches(part, avoided) for part in rel_parts)
        is_preferred = any(_dir_matches(part, preferred) for part in rel_parts)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (1 if is_avoided else 0, 0 if is_preferred else 1, len(rel_parts), -mtime, str(path))

    ranked = sorted(candidates, key=sort_key)
    return ranked[0], ranked[1:]


def resolve_artifact_file(
    root: Optional[str], filename: str, category: str, logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    """Recursively locate ``filename`` under ``root`` and return exactly one
    canonical path (or ``None`` if it doesn't exist anywhere under
    ``root``).

    Replaces the old flat ``root / filename`` check with
    ``Path(root).rglob(filename)``: every real file matching ``filename``
    anywhere under ``root`` is a candidate. With zero candidates this
    returns ``None`` exactly as the flat check did for a missing file. With
    exactly one candidate it's returned outright. With more than one,
    :func:`select_canonical_candidate` picks the winner and every other
    match is logged (never silently discarded) via ``logger``.
    """
    if root is None:
        return None
    root_path = Path(root)
    if not root_path.exists():
        return None

    candidates = sorted(p for p in root_path.rglob(filename) if p.is_file())
    if not candidates:
        return None
    if len(candidates) == 1:
        return str(candidates[0])

    selected, ignored = select_canonical_candidate(root_path, candidates, category)
    if logger is not None:
        logger.info(
            "ARTIFACT_DISCOVERY[%s] Found %d copies of %s. Selected: %s. Ignored: %s",
            category, len(candidates), filename, selected, [str(p) for p in ignored],
        )
    return str(selected)


def discover_category_inventory(
    root: Optional[str],
    critical_files: List[str],
    optional_files: List[str],
    category: str,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Dict[str, Optional[str]]]:
    """Resolve each known filename for one category against ``root``, via
    recursive, canonical-preferring search (:func:`resolve_artifact_file`).
    Returns the same ``{"critical": {...}, "optional": {...}}`` shape Stage
    1 used to hand Stage 2, with every path ``None`` (not omitted) when
    ``root`` is ``None`` or the file isn't found anywhere under it -- so
    downstream :func:`build_artifact_registry` sees "not discovered"
    exactly as it already expects to.
    """
    def _resolve(files: List[str]) -> Dict[str, Optional[str]]:
        return {filename: resolve_artifact_file(root, filename, category, logger) for filename in files}

    return {"critical": _resolve(critical_files), "optional": _resolve(optional_files)}


def discover_inventory(artifact_roots: Dict[str, Optional[str]], logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Build the full "inventory" dict across every known category.

    ``artifact_roots`` is ``configs.schema.DeploymentConfig.artifact_roots``
    (or an equivalent mapping) -- category name -> resolved directory (or
    ``None`` if that category's source isn't available in this
    environment). Unresolved categories still appear in the returned
    inventory (with every path ``None``), matching Stage 1's behaviour of
    always recording every known category rather than omitting missing ones.
    """
    return {
        category: discover_category_inventory(
            artifact_roots.get(category), CRITICAL_FILES[category], OPTIONAL_FILES[category], category, logger,
        )
        for category in KNOWN_CATEGORIES
    }


# ======================================================================
# VALIDATION / CLASSIFICATION (ported verbatim from Stage 2)
# ======================================================================


def _classify_file_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix in (".pt", ".pth"):
        return "torch"
    return "unknown"


def validate_json_schema(filename: str, data: Any) -> Tuple[Optional[bool], Optional[bool], List[str], List[str]]:
    """Structural check only. Returns ``(schema_valid, expected_keys_present,
    hard_errors, soft_warnings)``. ``hard_errors`` -> real corruption (bad
    top-level type), fails the artifact. ``soft_warnings`` -> the assumed
    key contract didn't match (informational only -- upstream stages are
    free to nest fields however they like)."""
    hard_errors: List[str] = []
    soft_warnings: List[str] = []

    if not isinstance(data, (dict, list)):
        hard_errors.append(f"'{filename}' does not contain a JSON object/array at the top level.")
        return False, False, hard_errors, soft_warnings

    expected = EXPECTED_JSON_KEYS.get(filename)
    if expected is None:
        return None, None, hard_errors, soft_warnings

    if isinstance(data, dict):
        present = any(k in data for k in expected)
        if not present:
            present = any(isinstance(v, dict) and any(k in v for k in expected) for v in data.values())
        if not present:
            soft_warnings.append(
                f"'{filename}' has none of the plausible keys {expected} at top level or one level "
                f"of nesting. Informational only -- verify manually if this file is meant to carry class info."
            )
        return True, present, hard_errors, soft_warnings

    return True, None, hard_errors, soft_warnings


def validate_artifact_file(
    artifact_id: str,
    category: str,
    filename: str,
    path_str: Optional[str],
    is_critical: bool,
    device: torch.device,
    logger: logging.Logger,
) -> ArtifactRecord:
    """Validate one discovered (or missing) artifact file: existence, size,
    SHA-256, and format-appropriate readability (JSON parse / CSV header /
    torch.load). Never raises -- every failure mode is recorded on the
    returned :class:`ArtifactRecord`."""
    record = ArtifactRecord(
        artifact_id=artifact_id, category=category, filename=filename, path=path_str, is_critical=is_critical,
    )

    if path_str is None:
        if is_critical:
            record.errors.append("Critical artifact missing (not discovered).")
        return record

    path = Path(path_str)
    record.exists = path.exists() and path.is_file()
    if not record.exists:
        record.errors.append(f"Path recorded during discovery no longer exists on disk: {path_str}")
        return record

    stat = path.stat()
    record.size_bytes = stat.st_size
    record.modified_time = stat.st_mtime
    record.file_type = _classify_file_type(filename)

    if record.size_bytes == 0:
        record.errors.append("File is empty (0 bytes).")

    try:
        record.sha256 = sha256_of_file(path)
    except OSError as exc:
        record.errors.append(f"Failed to hash file: {exc}")

    if record.file_type == "json":
        try:
            data = json.loads(path.read_text())
            record.json_readable = True
            schema_valid, keys_present, hard_errors, soft_warnings = validate_json_schema(filename, data)
            record.schema_valid = schema_valid
            record.expected_keys_present = keys_present
            record.errors.extend(hard_errors)
            record.schema_warnings.extend(soft_warnings)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            record.json_readable = False
            record.errors.append(f"JSON parse failure: {exc}")

    elif record.file_type == "csv":
        try:
            with path.open("r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
            record.csv_readable = header is not None
            if header is None:
                record.errors.append("CSV file has no header/rows.")
        except (OSError, csv.Error) as exc:
            record.csv_readable = False
            record.errors.append(f"CSV parse failure: {exc}")

    elif record.file_type == "torch":
        # Reuses inference.model_loader.load_checkpoint (frozen) rather than
        # duplicating Stage 2's own torch.load-tolerant wrapper.
        try:
            load_checkpoint(path, device)
            record.torch_readable = True
        except RuntimeError as exc:
            record.torch_readable = False
            record.errors.append(f"Torch checkpoint unreadable: {exc}")

    return record


def detect_duplicate_artifacts(records: Dict[str, ArtifactRecord]) -> List[List[str]]:
    by_hash: Dict[str, List[str]] = {}
    for artifact_id, rec in records.items():
        if rec.sha256:
            by_hash.setdefault(rec.sha256, []).append(artifact_id)
    groups = [ids for ids in by_hash.values() if len(ids) > 1]
    for ids in groups:
        for artifact_id in ids:
            records[artifact_id].duplicate_of = ",".join(i for i in ids if i != artifact_id)
    return groups


def build_artifact_registry(inventory: Dict[str, Any], device: torch.device, logger: logging.Logger) -> ArtifactRegistry:
    """Validate every artifact named in ``inventory`` (the shape produced by
    :func:`discover_inventory`) and aggregate the results, including
    cross-artifact duplicate (identical SHA-256) detection."""
    records: Dict[str, ArtifactRecord] = {}

    for category, data in inventory.items():
        for group_name in ("critical", "optional"):
            group = data.get(group_name, {}) or {}
            is_critical = group_name == "critical"
            for filename, path_str in group.items():
                artifact_id = f"{category}::{filename}"
                record = validate_artifact_file(artifact_id, category, filename, path_str, is_critical, device, logger)
                records[artifact_id] = record
                status = "OK" if record.exists and not record.errors else "ISSUE"
                logger.info(
                    "ARTIFACT[%s] id=%s critical=%s exists=%s type=%s status=%s",
                    category, artifact_id, is_critical, record.exists, record.file_type, status,
                )
                for err in record.errors:
                    logger.warning("ARTIFACT[%s] id=%s issue: %s", category, artifact_id, err)
                for warn in record.schema_warnings:
                    logger.warning("ARTIFACT[%s] id=%s schema note: %s", category, artifact_id, warn)

    duplicate_groups = detect_duplicate_artifacts(records)
    for group in duplicate_groups:
        logger.warning("DUPLICATE ARTIFACTS detected (identical SHA256): %s", group)

    critical_missing = [aid for aid, rec in records.items() if rec.is_critical and (not rec.exists or rec.errors)]

    return ArtifactRegistry(
        records=records, total_artifacts=len(records), critical_missing=critical_missing,
        duplicate_groups=duplicate_groups,
    )


def get_artifact_path(inventory: Dict[str, Any], category: str, filename: str) -> Optional[str]:
    data = inventory.get(category, {})
    for group in ("critical", "optional"):
        path = data.get(group, {}).get(filename)
        if path:
            return path
    return None


# ======================================================================
# SERVICE
# ======================================================================


class ArtifactService:
    """Central artifact discovery and loading.

    Owns exactly one concern: given resolved directory roots, find every
    known artifact file, validate it (Stage 2's algorithm, verbatim), and
    provide typed, read-only access to both raw file paths and parsed JSON
    content. Performs no model reconstruction, no threshold cross-validation
    against class names, and no prediction -- those are
    :class:`services.model_service.ModelService` /
    :class:`services.prediction_service.PredictionService`'s jobs,
    consuming what this service discovers.
    """

    def __init__(
        self,
        artifact_roots: Dict[str, Optional[str]],
        export_dir: Path,
        device: torch.device,
        logger: logging.Logger,
    ) -> None:
        """
        Args:
            artifact_roots: category -> resolved directory (or ``None``),
                matching ``configs.schema.DeploymentConfig.artifact_roots``.
            export_dir: Directory TorchScript/ONNX exports and (if present)
                a model card are read from -- matches
                ``configs.schema.ExportConfig.export_dir``.
            device: Passed through to checkpoint torch-readability checks.
            logger: Caller-supplied logger (dependency injection, matching
                every other module in this repository).
        """
        self.artifact_roots = artifact_roots
        self.export_dir = Path(export_dir)
        self.device = device
        self.logger = logger
        self._inventory: Optional[Dict[str, Any]] = None
        self._registry: Optional[ArtifactRegistry] = None

    def discover(self, force: bool = False) -> ArtifactRegistry:
        """Discover and validate every known artifact. Cached after the
        first call; pass ``force=True`` to re-discover (e.g. after artifacts
        change on disk)."""
        if self._registry is not None and not force:
            return self._registry
        self._inventory = discover_inventory(self.artifact_roots, self.logger)
        self._registry = build_artifact_registry(self._inventory, self.device, self.logger)
        self.logger.info(
            "ARTIFACTS discovered: total=%d critical_missing=%d duplicate_groups=%d",
            self._registry.total_artifacts, len(self._registry.critical_missing), len(self._registry.duplicate_groups),
        )
        return self._registry

    def get_path(self, category: str, filename: str) -> Optional[str]:
        """Look up one artifact's resolved path. Triggers :meth:`discover`
        if it hasn't run yet."""
        self.discover()
        assert self._inventory is not None
        return get_artifact_path(self._inventory, category, filename)

    # ---- Typed convenience loaders (Stage 2 category/filename mapping) --

    def load_training_summary(self) -> Dict[str, Any]:
        path = self.get_path(CATEGORY_TRAINING, "training_summary.json")
        if path is None:
            self.logger.warning("ARTIFACTS training_summary.json not available.")
            return {}
        return load_json(Path(path))

    def load_disease_registry(self) -> Dict[str, Any]:
        path = self.get_path(CATEGORY_SPRINT03, "disease_registry.json")
        if path is None:
            self.logger.warning("ARTIFACTS disease_registry.json not available.")
            return {}
        return load_json(Path(path))

    def load_evaluation_summary(self) -> Optional[Dict[str, Any]]:
        path = self.get_path(CATEGORY_EVALUATION, "evaluation_summary.json")
        return load_json(Path(path)) if path else None

    def load_deployment_metadata(self) -> Optional[Dict[str, Any]]:
        """``deployment_readiness.json`` under the evaluation category --
        the same file Stage 2's ``build_metadata_registry`` reads into
        ``MetadataRegistry.deployment_metadata`` (lines ~843-857)."""
        path = self.get_path(CATEGORY_EVALUATION, "deployment_readiness.json")
        return load_json(Path(path)) if path else None

    def checkpoint_path(self) -> Path:
        """Path to ``best_model.pt``.

        Raises:
            RuntimeError: if it wasn't discovered -- matches Stage 2's own
                fail-fast ("best_model.pt was not discovered by Stage 1 --
                cannot reconstruct model.", line ~1067).
        """
        path = self.get_path(CATEGORY_TRAINING, "best_model.pt")
        if path is None:
            raise RuntimeError("best_model.pt was not discovered -- cannot reconstruct model.")
        return Path(path)

    def optimal_thresholds_path(self) -> Optional[str]:
        return self.get_path(CATEGORY_EVALUATION, "optimal_thresholds.json")

    def torchscript_path(self) -> Optional[Path]:
        """Path to an exported ``model.ts``, if present under
        ``export_dir``. ``None`` if the export pipeline (Stage 3, not yet
        migrated) hasn't produced one."""
        candidate = self.export_dir / "model.ts"
        return candidate if candidate.exists() else None

    def onnx_path(self) -> Optional[Path]:
        """Path to an exported ``model.onnx``, if present under
        ``export_dir``. ``None`` if the export pipeline (Stage 3, not yet
        migrated) hasn't produced one."""
        candidate = self.export_dir / "model.onnx"
        return candidate if candidate.exists() else None

    def load_model_card(self) -> Optional[str]:
        """Best-effort read of a ``MODEL_CARD.md`` alongside ``export_dir``,
        if one exists. Model-card *generation* is Stage 7/8 (release
        engineering) content, explicitly out of scope for this phase and
        not yet migrated -- this only loads one if it already exists on
        disk. Returns ``None`` (not an error) when absent."""
        candidate = self.export_dir.parent / "MODEL_CARD.md"
        if candidate.exists():
            return candidate.read_text()
        return None

    @staticmethod
    def expected_class_names_for_thresholds(disease_registry: Dict[str, Any]) -> List[str]:
        """``expected_class_names`` for
        ``inference.thresholding.load_threshold_registry``. Ported verbatim
        (a one-line expression, not a separate function in the original --
        promoted to a named, reusable unit here) from Stage 2's own call
        site: ``disease_metadata_dict.get("classes") or
        disease_metadata_dict.get("class_names") or []`` (line ~1083)."""
        return disease_registry.get("classes") or disease_registry.get("class_names") or []

    #: Categories actually consumed by the serving path (ModelService ->
    #: reconstruct_model / load_threshold_registry /
    #: resolve_class_names_and_thresholds -- see services/model_service.py).
    #: Deliberately excludes CATEGORY_NIH: the raw NIH Chest X-ray dataset
    #: CSVs are dataset-lineage/audit artifacts from Sprint 03/04 data
    #: engineering, never read by anything in this repository's inference
    #: or service layer (Stage 4/5/6's own "STRICT SCOPE" comments already
    #: establish this -- "no dataset traversal", Stages 1-5 consumed only
    #: as in-memory objects). Stage 2's ArtifactRegistry itself is NOT
    #: changed by this -- discover() still validates and reports on
    #: nih_chest_xray for full audit fidelity; only THIS service's own new
    #: serving-health determination (below) is scoped to exclude it.
    SERVING_CATEGORIES: Tuple[str, ...] = (CATEGORY_SPRINT03, CATEGORY_TRAINING, CATEGORY_EVALUATION)

    def serving_critical_missing(self) -> List[str]:
        """``ArtifactRegistry.critical_missing``, filtered to
        :attr:`SERVING_CATEGORIES` -- i.e. artifacts actually required to
        reconstruct and run the model, excluding dataset-lineage-only
        categories like ``nih_chest_xray``."""
        registry = self.discover()
        return [aid for aid in registry.critical_missing if aid.split("::", 1)[0] in self.SERVING_CATEGORIES]

    def artifact_status(self) -> Dict[str, Any]:
        """Summary used by :class:`services.health_service.HealthService`.
        ``healthy`` reflects :meth:`serving_critical_missing` (what the
        serving path actually needs), not the full audit-scope
        ``critical_missing`` (which also covers dataset-lineage artifacts
        no service in this package ever reads) -- see
        :attr:`SERVING_CATEGORIES`. Full per-record detail is available via
        :meth:`discover` / ``ArtifactRegistry.to_dict()``."""
        registry = self.discover()
        serving_missing = self.serving_critical_missing()
        return {
            "total_artifacts": registry.total_artifacts,
            "critical_missing": registry.critical_missing,
            "serving_critical_missing": serving_missing,
            "duplicate_groups": registry.duplicate_groups,
            "healthy": len(serving_missing) == 0,
        }