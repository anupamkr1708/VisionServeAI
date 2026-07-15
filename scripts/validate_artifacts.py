"""
Artifact validation for VisionServeAI.

A focused, lightweight pre-flight check on a real (or synthetic)
``artifact_roots`` tree -- runs only ``services.artifact_service.ArtifactService``
(discovery + per-file validation: existence, type classification, JSON
schema, duplicate detection, checkpoint torch-readability) without
reconstructing the model, loading a runtime, or running a prediction.

This sits between ``scripts/resolve_artifact_roots.py`` (path resolution
only -- no validation) and ``scripts/smoke_test.py`` (full pipeline,
including model reconstruction and inference -- much slower, and fails on
anything, not just missing/malformed artifacts). Use this script first
after resolving ``artifact_roots`` to get a fast, focused answer to "are my
artifact files present and well-formed", before paying the cost of loading
the model at all.

Usage
-----
    # Resolve a local directory tree first, then validate the result:
    python -m scripts.validate_artifacts --artifact-root /path/to/artifacts

    # Or validate an already-resolved artifact_roots mapping:
    python -m scripts.validate_artifacts --artifact-roots-json roots.json

    # Also check for TorchScript/ONNX export artifacts:
    python -m scripts.validate_artifacts --artifact-root /path/to/artifacts \\
        --export-dir /path/to/exports

Exit code is 0 if no *serving-critical* artifact is missing (see
``ArtifactService.SERVING_CATEGORIES`` / ``serving_critical_missing()``),
1 otherwise. Non-critical missing files and export artifacts are reported
but never fail the run.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from services.artifact_service import ArtifactService


def _banner(title: str) -> None:
    print("=" * 56)
    print()
    print(title)
    print()
    print("=" * 56)


def validate(
    artifact_roots: Dict[str, Optional[str]],
    export_dir: Path,
    logger: logging.Logger,
) -> bool:
    """Runs discovery + validation and prints a report. Returns True if the
    artifact tree is healthy for serving (no serving-critical file
    missing)."""
    service = ArtifactService(artifact_roots=artifact_roots, export_dir=export_dir, device=torch.device("cpu"), logger=logger)

    _banner("VisionServe AI\n\nArtifact Validation")

    print("Resolved artifact_roots:")
    for category, root in artifact_roots.items():
        print(f"  {category:20s} -> {root if root else '(not resolved)'}")
    print()

    try:
        registry = service.discover()
    except Exception as exc:  # noqa: BLE001
        print(f"✗ Artifact discovery failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return False

    print(f"Discovered {registry.total_artifacts} known artifact file(s).\n")

    by_category: Dict[str, List[Any]] = {}
    for artifact_id, record in registry.records.items():
        category = artifact_id.split("::", 1)[0]
        by_category.setdefault(category, []).append(record)

    for category, records in by_category.items():
        print(f"[{category}]")
        for record in sorted(records, key=lambda r: r.filename):
            status = "OK" if record.exists else ("MISSING (critical)" if record.is_critical else "missing (optional)")
            marker = "✓" if record.exists else ("✗" if record.is_critical else "–")
            print(f"  {marker} {record.filename:35s} {status}")
        print()

    if registry.duplicate_groups:
        print("! Duplicate artifacts detected (same content, multiple locations):")
        for group in registry.duplicate_groups:
            print(f"    {group}")
        print()

    status = service.artifact_status()
    print(f"Total artifacts known:            {status['total_artifacts']}")
    print(f"Critical missing (full audit):     {status['critical_missing'] or 'none'}")
    print(f"Critical missing (serving path):   {status['serving_critical_missing'] or 'none'}")
    print()

    export_ts = export_dir / "model.ts"
    export_onnx = export_dir / "model.onnx"
    print(f"Export directory: {export_dir}")
    print(f"  TorchScript (model.ts):  {'found' if export_ts.exists() else 'not found (optional -- export is a separate, not-yet-migrated stage)'}")
    print(f"  ONNX (model.onnx):       {'found' if export_onnx.exists() else 'not found (optional -- export is a separate, not-yet-migrated stage)'}")
    print()

    healthy = status["healthy"]
    _banner("Artifact Validation " + ("PASSED" if healthy else "FAILED"))
    return healthy


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--artifact-root", type=str, help="Local artifact directory tree, resolved via scripts.resolve_artifact_roots.")
    mode.add_argument("--artifact-roots-json", type=str, help="Path to a JSON file containing an already-resolved artifact_roots mapping.")
    parser.add_argument("--export-dir", type=str, default=None, help="Directory to check for model.ts/model.onnx (optional).")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(asctime)s | %(levelname)-8s | %(message)s")
    logger = logging.getLogger("visionserve.validate_artifacts")

    if args.artifact_root:
        from scripts.resolve_artifact_roots import resolve_artifact_roots
        artifact_roots = resolve_artifact_roots(args.artifact_root, logger=logger)
    else:
        with open(args.artifact_roots_json) as f:
            artifact_roots = json.load(f)

    export_dir = Path(args.export_dir) if args.export_dir else Path("deployment") / "export"

    healthy = validate(artifact_roots, export_dir, logger)
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
