"""
Environment verification for VisionServeAI.

Standalone pre-flight check -- run before anything else (before
``resolve_artifact_roots``, before ``smoke_test``) to confirm the machine
itself can run this repository at all: Python version, every package the
serving path imports at runtime, and hardware/runtime capabilities
(CUDA, ONNX, TorchScript). Needs no artifacts and no repository checkout
beyond this file's own imports.

Deliberately layered underneath ``scripts/smoke_test.py``: that script's
first checklist line (``check_environment``) already calls
``inference.utils.environment.get_environment_info()`` and asserts a
handful of required keys are present -- this script is the same
environment report plus a much stricter package-level and version-floor
gate, runnable on its own without needing any artifact tree, export
directory, or model to already exist. If this script fails, the smoke
test's environment step will fail too, before anything more expensive is
attempted.

Usage
-----
    python -m scripts.verify_environment
    python -m scripts.verify_environment -v   # print full environment report

Exit code is 0 if every required check passes, 1 otherwise. Optional
packages (e.g. ``onnx``/``onnxruntime`` when only the PyTorch runtime is
needed) produce a warning line, not a failure.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

MIN_PYTHON = (3, 10)

# (import name, distribution name, minimum version or None, required)
# Mirrors requirements.txt's "Foundation" + "Inference runtime" sections --
# the packages the serving path (not the not-yet-built backend/, not dev
# tooling like ruff/mypy) actually imports at runtime.
REQUIRED_PACKAGES: List[Tuple[str, str, Optional[str], bool]] = [
    ("psutil", "psutil", "5.9", True),
    ("torch", "torch", "2.2", True),
    ("torchvision", "torchvision", "0.17", True),
    ("PIL", "Pillow", "10.0", True),
    ("numpy", "numpy", None, True),
    ("onnx", "onnx", "1.16", False),
    ("onnxruntime", "onnxruntime", "1.17", False),
    ("yaml", "PyYAML", "6.0", False),
]


@dataclass
class CheckResult:
    label: str
    passed: bool
    detail: str = ""
    warning: bool = False


def _version_tuple(v: str) -> Tuple[int, ...]:
    parts = []
    for chunk in v.split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def check_python_version() -> CheckResult:
    actual = sys.version_info[:2]
    ok = actual >= MIN_PYTHON
    return CheckResult(
        "Python version",
        ok,
        f"{'.'.join(map(str, actual))} (need >= {'.'.join(map(str, MIN_PYTHON))})",
    )


def check_package(import_name: str, dist_name: str, min_version: Optional[str], required: bool) -> CheckResult:
    label = f"Package '{dist_name}'"
    try:
        module = importlib.import_module(import_name)
    except ImportError as exc:
        if required:
            return CheckResult(label, False, f"not importable: {exc}")
        return CheckResult(label, True, "not installed (optional)", warning=True)

    version = getattr(module, "__version__", None)
    if version is None:
        try:
            from importlib import metadata as importlib_metadata
            version = importlib_metadata.version(dist_name)
        except Exception:  # noqa: BLE001
            version = "unknown"

    if min_version is not None and version != "unknown":
        if _version_tuple(version) < _version_tuple(min_version):
            detail = f"{version} (need >= {min_version})"
            if required:
                return CheckResult(label, False, detail)
            return CheckResult(label, True, detail, warning=True)

    return CheckResult(label, True, str(version))


def check_runtime_capabilities() -> Tuple[List[CheckResult], dict]:
    """Informational, never fails on its own -- CUDA/GPU absence is normal
    on a CPU-only deployment box; this just surfaces it clearly."""
    from inference.utils.environment import get_environment_info

    info = get_environment_info()
    results = [
        CheckResult("Device", True, info["device"]),
        CheckResult("CUDA available", True, str(info["gpu_available"]), warning=not info["gpu_available"]),
        CheckResult("TorchScript available", info["torchscript_available"], str(info["torchscript_available"])),
        CheckResult("ONNX export available", True, str(info["onnx_available"]), warning=not info["onnx_available"]),
        CheckResult(
            "RAM", info["ram_total_gb"] > 0, f"{info['ram_total_gb']} GB total, {info['cpu_cores']} logical cores"
        ),
    ]
    return results, info


def run(verbose: bool = False) -> Tuple[bool, List[CheckResult]]:
    results: List[CheckResult] = [check_python_version()]
    for import_name, dist_name, min_version, required in REQUIRED_PACKAGES:
        results.append(check_package(import_name, dist_name, min_version, required))

    runtime_results, env_info = check_runtime_capabilities()
    results.extend(runtime_results)

    print("=" * 56)
    print()
    print("VisionServe AI\n\nEnvironment Verification")
    print()
    print("=" * 56)
    ok = True
    for r in results:
        if not r.passed:
            ok = False
            print(f"✗ {r.label}: {r.detail}", file=sys.stderr)
        elif r.warning:
            print(f"! {r.label}: {r.detail} (warning)")
        else:
            print(f"✓ {r.label}: {r.detail}")

    if verbose:
        print()
        print("Full environment report:")
        print(json.dumps(env_info, indent=2))

    print()
    print("Environment Verification " + ("PASSED" if ok else "FAILED"))
    return ok, results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true", help="Print the full environment report as JSON.")
    args = parser.parse_args(argv)
    ok, _ = run(verbose=args.verbose)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
