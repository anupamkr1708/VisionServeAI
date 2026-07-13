"""
Resolve ``configs.schema.DeploymentConfig.artifact_roots`` for a local
(non-Kaggle) artifact directory tree.

Why this exists
----------------
``services.artifact_service.ArtifactService`` takes an already-resolved
``artifact_roots`` mapping (``{"sprint03": "<dir>", "sprint04_training":
"<dir>", "sprint04_evaluation": "<dir>", "nih_chest_xray": "<dir>"}``) and
does a *flat* ``root / filename`` lookup for each known file -- deliberately
not a recursive scan (see that module's own docstring: Stage 1's Kaggle
dataset-mount scan was explicitly left out of scope for that phase).

That is fine once you already know the exact leaf directory each category's
files live in. In practice, local artifact exports are rarely flat -- e.g.
a directory tree like::

    artifacts/
      sprint03/registry/disease_registry.json
      sprint04/checkpoints/_s36_run_a/best_model.pt
      sprint04_evaluation/stage06_threshold_calibration_engine/optimal_thresholds.json

has each category's real files several directories deep, under
subdirectories that vary run to run. Nothing in this repository resolved
that gap before this cleanup pass -- ``ArtifactService`` was never wired to
anything that could find the flat directories it expects, and
``configs.defaults.MAX_SCAN_DEPTH`` (ported from Stage 1) sat completely
unused.

This script is that missing piece, WITHOUT reintroducing Stage 1's Kaggle
mount-scan into ``ArtifactService`` itself (which stays frozen, flat, and
Kaggle-agnostic). It generalizes Stage 1's own
``discover_kaggle_datasets`` / ``classify_datasets`` / ``_find_files`` /
``_common_ancestor`` algorithm -- ported here, not reimplemented -- to scan
any local root's immediate subdirectories (not only ``/kaggle/input/*``),
fingerprint-match each one against ``configs.defaults.FINGERPRINTS``, and
resolve each category to the deepest directory common to all of its located
fingerprint files. No dataset slug, username, or Kaggle path is ever
hardcoded -- purely content-based, exactly like the original.

Usage
-----
As a library, feeding :class:`services.service_registry.ServiceRegistry`
directly::

    from scripts.resolve_artifact_roots import resolve_artifact_roots
    roots = resolve_artifact_roots("/mnt/g/DEV/projects/VisionServeAI/artifacts")
    registry = ServiceRegistry(artifact_roots=roots, export_dir=Path("deployment/export"))
    registry.initialize()

As a CLI, to inspect what would be resolved before wiring it up::

    python -m scripts.resolve_artifact_roots /path/to/artifacts

Source: sprint05-deployment.ipynb, Stage 1, lines ~283-433
(``_bounded_walk``, ``_find_files``, ``_common_ancestor``,
``discover_kaggle_datasets``, ``classify_datasets`` -- algorithm ported,
generalized to any root rather than only ``KAGGLE_INPUT_ROOT``).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from configs.defaults import FINGERPRINTS, MAX_SCAN_DEPTH


def _bounded_walk(root: Path, max_depth: int):
    """``os.walk`` bounded to ``max_depth`` (root itself = depth 0).
    Verbatim port of Stage 1's ``_bounded_walk``."""
    root_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).parts) - root_depth
        if depth >= max_depth:
            dirnames[:] = []
        yield Path(dirpath), filenames


def _find_files(dataset_root: Path, filenames: List[str], max_depth: int = MAX_SCAN_DEPTH) -> Dict[str, Optional[str]]:
    """Search (bounded-depth) for each filename inside ``dataset_root``.
    Verbatim port of Stage 1's ``_find_files``."""
    found: Dict[str, Optional[str]] = {name: None for name in filenames}
    remaining = set(filenames)
    if not remaining:
        return found
    for dirpath, files in _bounded_walk(dataset_root, max_depth):
        for fname in list(remaining):
            if fname in files:
                found[fname] = str(dirpath / fname)
                remaining.discard(fname)
        if not remaining:
            break
    return found


def _common_ancestor(paths: List[str]) -> Optional[Path]:
    """Deepest directory common to every given file path. Verbatim port of
    Stage 1's ``_common_ancestor``."""
    resolved = [Path(p) for p in paths if p]
    if not resolved:
        return None
    parent_parts = [p.parent.parts for p in resolved]
    common: List[str] = []
    for parts_at_depth in zip(*parent_parts):
        if len(set(parts_at_depth)) == 1:
            common.append(parts_at_depth[0])
        else:
            break
    return Path(*common) if common else None


def discover_local_datasets(root: Path, logger: logging.Logger) -> Dict[str, Path]:
    """List every immediate subdirectory of ``root``.

    Generalization of Stage 1's ``discover_kaggle_datasets``: the original
    only ever listed ``KAGGLE_INPUT_ROOT.iterdir()``. Any local artifact
    root (e.g. this user's ``artifacts/`` directory, whose immediate
    children are ``sprint03/``, ``sprint04/``, ``sprint04_evaluation/``,
    ``Sprint05_Complete/``, ...) has the identical shape, so the same
    "treat each immediate subdirectory as one candidate dataset" strategy
    applies unchanged.
    """
    if not root.exists():
        raise RuntimeError(f"Artifact root not found: {root}")
    roots = {p.name: p for p in root.iterdir() if p.is_dir()}
    if not roots:
        raise RuntimeError(f"No subdirectories found under {root}")
    logger.info("DISCOVERY found %d candidate dataset dir(s) under %s", len(roots), root)
    for name in roots:
        logger.info("DISCOVERY   - %s", name)
    return roots


def classify_datasets(dataset_roots: Dict[str, Path], logger: logging.Logger) -> Dict[str, Dict[str, Any]]:
    """Identify which artifact root satisfies each required category, using
    content fingerprints. Verbatim port of Stage 1's ``classify_datasets``
    (operates over whatever ``dataset_roots`` it's given -- Kaggle mounts in
    the original, local subdirectories here; the algorithm itself doesn't
    know or care which)."""
    candidates: Dict[str, List[Dict[str, Any]]] = {category: [] for category in FINGERPRINTS}

    for dataset_name, dataset_path in dataset_roots.items():
        for category, fingerprint_files in FINGERPRINTS.items():
            hits = _find_files(dataset_path, fingerprint_files)
            found_paths = [v for v in hits.values() if v is not None]
            score = len(found_paths)
            if score == 0:
                continue
            artifact_root = _common_ancestor(found_paths)
            if artifact_root is None:
                continue
            candidates[category].append({
                "matched_dataset": dataset_name,
                "root": str(artifact_root),
                "score": score,
            })

    classification: Dict[str, Dict[str, Any]] = {}
    for category, cand_list in candidates.items():
        if not cand_list:
            classification[category] = {"matched_dataset": None, "root": None, "score": 0, "candidates": []}
            logger.warning("DISCOVERY category='%s' -> NOT FOUND", category)
            continue

        unique_by_root: Dict[str, Dict[str, Any]] = {}
        for c in cand_list:
            existing = unique_by_root.get(c["root"])
            if existing is None or c["score"] > existing["score"]:
                unique_by_root[c["root"]] = c
        deduped = list(unique_by_root.values())

        best_score = max(c["score"] for c in deduped)
        top_candidates = [c for c in deduped if c["score"] == best_score]
        best = top_candidates[0]

        classification[category] = {
            "matched_dataset": best["matched_dataset"],
            "root": best["root"],
            "score": best["score"],
            "candidates": deduped,
        }

        if len(top_candidates) > 1:
            logger.warning(
                "DISCOVERY category='%s' -> AMBIGUOUS: %d equally valid roots %s",
                category, len(top_candidates), [c["root"] for c in top_candidates],
            )
        else:
            logger.info(
                "DISCOVERY category='%s' -> dataset='%s' root='%s' (fingerprint score %d/%d)",
                category, best["matched_dataset"], best["root"], best["score"], len(FINGERPRINTS[category]),
            )

    return classification


def resolve_artifact_roots(
    root: str, logger: Optional[logging.Logger] = None, max_depth: int = MAX_SCAN_DEPTH,
) -> Dict[str, Optional[str]]:
    """Resolve a ``configs.schema.DeploymentConfig.artifact_roots``-shaped
    mapping (``category -> directory | None``) for a local artifact tree
    rooted at ``root``.

    This is the single public entry point most callers want -- pass the
    result straight into ``ServiceRegistry(artifact_roots=..., ...)``.

    Args:
        root: Path to the top-level artifact directory (e.g. this user's
            ``.../VisionServeAI/artifacts``). Its *immediate subdirectories*
            are each treated as one candidate dataset, mirroring
            ``FINGERPRINTS``' original Kaggle-per-dataset-mount shape.
        logger: Optional logger; a basic one is built if not given.
        max_depth: Bounded-walk depth per candidate subdirectory. Defaults
            to ``configs.defaults.MAX_SCAN_DEPTH`` (``6``), Stage 1's own
            value, unchanged.

    Returns:
        ``{"sprint03": <dir or None>, "sprint04_training": <dir or None>,
        "sprint04_evaluation": <dir or None>, "nih_chest_xray": <dir or
        None>}`` -- ``None`` for any category no fingerprint match was found
        for (surfaced as a warning, not an exception -- the caller's own
        ``ArtifactService.discover()`` already reports missing critical
        files clearly; this function's job is best-effort path resolution,
        not validation).
    """
    if logger is None:
        logger = logging.getLogger("visionserve.resolve_artifact_roots")
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    dataset_roots = discover_local_datasets(Path(root), logger)
    classification = classify_datasets(dataset_roots, logger)
    return {category: info["root"] for category, info in classification.items()}


def _main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        print("Usage: python -m scripts.resolve_artifact_roots <artifact_root_dir>", file=sys.stderr)
        return 2

    logger = logging.getLogger("visionserve.resolve_artifact_roots")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    try:
        roots = resolve_artifact_roots(argv[0], logger=logger)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(roots, indent=2))

    missing = [category for category, path in roots.items() if path is None]
    if missing:
        print(f"\nWARNING: no fingerprint match found for: {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
