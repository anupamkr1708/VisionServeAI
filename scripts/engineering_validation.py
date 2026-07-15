"""
Aggregate engineering validation for VisionServeAI.

This is deliberately a thin *orchestrator*, not a fifth reimplementation of
checks the other scripts already do well -- it calls, in order:

    1. scripts.validate_repository.run()      -- structure/imports/packaging
    2. scripts.verify_environment              -- environment sanity
    3. scripts.validate_artifacts.validate()    -- artifact discovery/checksums
    4. scripts.smoke_test.run_smoke_test()      -- the full live service pipeline
    5. pytest (tests/unit + tests/integration)  -- the behavioral test suite

and prints one final PASS/FAIL summary. Nothing here duplicates another
script's logic (see each check's ``label`` -- every one delegates).

Usage::

    python -m scripts.engineering_validation --synthetic
    python -m scripts.engineering_validation --artifact-root /path/to/artifacts --export-dir /path/to/exports

Exit code is 0 only if every stage passes.
"""
from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent


def _banner(title: str) -> None:
    print("=" * 56)
    print()
    print(title)
    print()
    print("=" * 56)


def check_repository_structure() -> Tuple[bool, str]:
    from scripts.validate_repository import run as run_repository_validation

    ok, _lines = run_repository_validation()
    return ok, "Repository integrity (imports, packaging, exports)"


def check_environment() -> Tuple[bool, str]:
    from scripts.verify_environment import run as run_environment_check

    ok, _results = run_environment_check()
    return ok, "Environment (Python/Torch/torchvision/ONNX availability)"


def check_artifacts(artifact_roots, export_dir: Path, logger: logging.Logger) -> Tuple[bool, str]:
    from scripts.validate_artifacts import validate as run_artifact_validation

    ok = run_artifact_validation(artifact_roots, export_dir, logger)
    return ok, "Artifact validation (discovery, checksums, thresholds, metadata)"


def check_smoke_test(artifact_roots_json_path: Path, export_dir: Path,
                      runtime_types: List[str], seed: int) -> Tuple[bool, str]:
    """Runs ``python -m scripts.smoke_test`` as a subprocess -- both for the
    same memory-isolation reason as :func:`check_pytest_suite`, and so this
    exercises the actual documented CLI entrypoint a person would run by
    hand, not a hand-rolled call to its internals that could drift from it.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.smoke_test",
            "--artifact-roots-json", str(artifact_roots_json_path),
            "--export-dir", str(export_dir),
            "--runtimes", ",".join(runtime_types),
            "--seed", str(seed),
        ],
        cwd=str(REPO_ROOT),
    )
    return result.returncode == 0, "Full service pipeline (Artifact->Model->Runtime->Prediction->Explainability->Health->Registry)"


def check_pytest_suite() -> Tuple[bool, str]:
    """Runs as a separate subprocess, not in-process via ``pytest.main()``.

    Found by actually running this end to end: by the time this stage
    runs, the smoke-test stage just built/exported multiple models and
    ran all seven explainability algorithms in *this same process* --
    accumulating enough memory that then running the entire pytest suite
    (which itself repeatedly builds/traces/exports models) in-process too
    caused a fatal low-level abort on this machine's limited RAM. A
    subprocess gets a clean address space, exactly like a separate CI
    stage would, and its failure can't take this orchestrator down with
    it.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(REPO_ROOT / "tests" / "unit"), str(REPO_ROOT / "tests" / "integration")],
        cwd=str(REPO_ROOT),
    )
    return result.returncode == 0, "Behavioral test suite (tests/unit + tests/integration)"


def run(artifact_roots, export_dir: Path, artifact_roots_json_path: Path,
        runtime_types: List[str], logger: logging.Logger, seed: int) -> bool:
    _banner("VisionServe AI\n\nEngineering Validation")

    results: List[Tuple[bool, str]] = []

    for label, fn in [
        ("repository structure", check_repository_structure),
        ("environment", check_environment),
    ]:
        print(f"\n--- {label} ---")
        ok, description = fn()
        results.append((ok, description))
        print(f"{'✓' if ok else '✗'} {description}")

    print("\n--- artifacts ---")
    ok, description = check_artifacts(artifact_roots, export_dir, logger)
    results.append((ok, description))
    print(f"{'✓' if ok else '✗'} {description}")

    print("\n--- smoke test ---")
    ok, description = check_smoke_test(artifact_roots_json_path, export_dir, runtime_types, seed)
    results.append((ok, description))
    print(f"{'✓' if ok else '✗'} {description}")

    print("\n--- pytest suite ---")
    ok, description = check_pytest_suite()
    results.append((ok, description))
    print(f"{'✓' if ok else '✗'} {description}")

    overall = all(ok for ok, _ in results)

    print()
    print("=" * 56)
    print("Summary")
    print("=" * 56)
    for ok, description in results:
        print(f"{'✓' if ok else '✗'} {description}")
    print()
    _banner("Engineering Validation " + ("PASSED" if overall else "FAILED"))
    return overall


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true")
    mode.add_argument("--artifact-root", type=str)
    mode.add_argument("--artifact-roots-json", type=str)
    parser.add_argument("--export-dir", type=str, default=None)
    parser.add_argument("--runtimes", type=str, default="pytorch,torchscript,onnx")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)-8s | %(message)s")
    logger = logging.getLogger("visionserve.engineering_validation")
    runtime_types = [t.strip() for t in args.runtimes.split(",") if t.strip()]

    with tempfile.TemporaryDirectory(prefix="visionserve_engval_") as tmp_dir:
        tmp_path = Path(tmp_dir)

        if args.synthetic:
            from scripts._synthetic_fixtures import (
                build_synthetic_artifact_tree, export_onnx, export_torchscript,
            )
            from inference.model_loader import reconstruct_model
            import torch

            fixture = build_synthetic_artifact_tree(tmp_path / "artifacts", seed=args.seed)
            artifact_roots = fixture["artifact_roots"]
            export_dir = tmp_path / "export"
            export_dir.mkdir(parents=True, exist_ok=True)

            training_summary = json.loads(fixture["training_summary_path"].read_text())
            disease_registry = json.loads(fixture["disease_registry_path"].read_text())
            model, _mr = reconstruct_model(
                training_summary=training_summary, disease_registry=disease_registry,
                checkpoint_path=fixture["checkpoint_path"], device=torch.device("cpu"), logger=logger,
            )
            if "torchscript" in runtime_types:
                export_torchscript(model, export_dir / "model.ts", (1, 3, 224, 224))
            if "onnx" in runtime_types:
                export_onnx(model, export_dir / "model.onnx", (1, 3, 224, 224))

        elif args.artifact_root:
            from scripts.resolve_artifact_roots import resolve_artifact_roots

            artifact_roots = resolve_artifact_roots(args.artifact_root, logger=logger)
            export_dir = Path(args.export_dir) if args.export_dir else Path("deployment") / "export"

        else:
            with open(args.artifact_roots_json) as f:
                artifact_roots = json.load(f)
            export_dir = Path(args.export_dir) if args.export_dir else Path("deployment") / "export"

        artifact_roots_json_path = tmp_path / "artifact_roots.json"
        artifact_roots_json_path.write_text(json.dumps(artifact_roots))

        passed = run(artifact_roots, export_dir, artifact_roots_json_path, runtime_types, logger, args.seed)

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
