"""
Repository structural validation for VisionServeAI.

Automates the structural half of what the manual repository audit checked
by hand (packaging config, importability of every module, no orphaned
public classes, tests/ scaffolding matches ``pyproject.toml``) so it can be
re-run after every change instead of re-audited by hand. This is a
*structural* check only -- it proves the repository imports cleanly and is
wired consistently with its own packaging/test configuration; it proves
nothing about runtime correctness against real artifacts (that's
``scripts/validate_artifacts.py`` and ``scripts/smoke_test.py``).

Run from the repository root::

    python -m scripts.validate_repository

Exit code is 0 if every check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
import tomllib
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# Packages pyproject.toml declares as part of the installable distribution
# (`[tool.setuptools.packages.find].include`) -- these must import cleanly
# under their own name with no import-time errors.
PACKAGED_TOP_LEVEL = ["backend", "inference", "deployment", "configs", "services"]

# Not packaged (pyproject.toml explicitly excludes them) but still expected
# to exist and import cleanly when run from the repo root, per each
# module's own "run as `python -m scripts.<name>`" convention.
UNPACKAGED_TOP_LEVEL = ["scripts", "tests"]

EXPECTED_TEST_DIRS = ["tests/unit", "tests/integration", "tests/api"]


def _pass(label: str, detail: str = "") -> Tuple[bool, str]:
    return True, f"✓ {label}" + (f": {detail}" if detail else "")


def _fail(label: str, detail: str) -> Tuple[bool, str]:
    return False, f"✗ {label}: {detail}"


def check_pyproject() -> Tuple[bool, str]:
    path = REPO_ROOT / "pyproject.toml"
    if not path.exists():
        return _fail("pyproject.toml", "not found")
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except Exception as exc:  # noqa: BLE001
        return _fail("pyproject.toml", f"failed to parse: {exc}")

    include = data.get("tool", {}).get("setuptools", {}).get("packages", {}).get("find", {}).get("include", [])
    declared = {name.rstrip("*") for name in include}
    expected = set(PACKAGED_TOP_LEVEL)
    if declared != expected:
        return _fail(
            "pyproject.toml packages.include",
            f"declared={sorted(declared)} expected={sorted(expected)}",
        )
    return _pass("pyproject.toml", f"packages.include matches {sorted(expected)}")


def check_package_importable(top_level: str) -> Tuple[bool, str]:
    """Import every module under ``top_level`` (not just the package
    itself), so a broken submodule several directories deep can't hide
    behind a package ``__init__.py`` that happens to import cleanly on its
    own."""
    pkg_dir = REPO_ROOT / top_level
    if not pkg_dir.is_dir():
        return _fail(f"'{top_level}/' importable", f"directory not found: {pkg_dir}")

    try:
        package = importlib.import_module(top_level)
    except Exception as exc:  # noqa: BLE001
        return _fail(f"'{top_level}' importable", f"{type(exc).__name__}: {exc}")

    if not hasattr(package, "__path__"):
        return _pass(f"'{top_level}' importable", "single module")

    failures: List[str] = []
    count = 0
    for _finder, name, _is_pkg in pkgutil.walk_packages(package.__path__, prefix=f"{top_level}."):
        count += 1
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")

    if failures:
        return _fail(f"'{top_level}' submodules importable", "; ".join(failures))
    return _pass(f"'{top_level}' importable", f"{count} submodule(s), all import cleanly")


def check_inference_engine_exported() -> Tuple[bool, str]:
    """Regression guard for the orphan-module fix noted in the repository
    audit: ``inference.engine.InferenceEngine`` must be reachable as
    ``from inference import InferenceEngine``, not just importable via its
    own submodule path."""
    try:
        import inference

        if not hasattr(inference, "InferenceEngine"):
            return _fail("InferenceEngine exported from inference/__init__.py", "attribute not found")
        if "InferenceEngine" not in getattr(inference, "__all__", []):
            return _fail("InferenceEngine exported from inference/__init__.py", "not listed in __all__")
    except Exception as exc:  # noqa: BLE001
        return _fail("InferenceEngine exported from inference/__init__.py", f"{type(exc).__name__}: {exc}")
    return _pass("InferenceEngine exported from inference/__init__.py")


def check_scripts_excluded_from_package() -> Tuple[bool, str]:
    path = REPO_ROOT / "pyproject.toml"
    with path.open("rb") as f:
        data = tomllib.load(f)
    exclude = data.get("tool", {}).get("setuptools", {}).get("packages", {}).get("find", {}).get("exclude", [])
    excluded = {name.rstrip("*") for name in exclude}
    if "scripts" not in excluded or "tests" not in excluded:
        return _fail("scripts/ and tests/ excluded from package", f"exclude={sorted(excluded)}")
    return _pass("scripts/ and tests/ excluded from package")


def check_test_dirs() -> Tuple[bool, str]:
    missing = [d for d in EXPECTED_TEST_DIRS if not (REPO_ROOT / d).is_dir()]
    if missing:
        return _fail("test directories present", f"missing: {missing}")
    testpaths = None
    try:
        with (REPO_ROOT / "pyproject.toml").open("rb") as f:
            data = tomllib.load(f)
        testpaths = data.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("testpaths")
    except Exception:  # noqa: BLE001
        pass
    if testpaths != ["tests"]:
        return _fail("pytest testpaths configured", f"[tool.pytest.ini_options].testpaths={testpaths}")
    return _pass("test directories present", f"{EXPECTED_TEST_DIRS}, pytest testpaths=['tests']")


def check_requirements_importable() -> Tuple[bool, str]:
    """Cheap cross-check against ``scripts.verify_environment`` -- confirms
    ``requirements.txt`` exists and is non-empty, without duplicating that
    script's actual package/version verification (run
    ``scripts.verify_environment`` for that)."""
    path = REPO_ROOT / "requirements.txt"
    if not path.exists():
        return _fail("requirements.txt", "not found")
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.strip().startswith("#")]
    if not lines:
        return _fail("requirements.txt", "no dependencies listed")
    return _pass("requirements.txt", f"{len(lines)} dependencies listed")


def run() -> Tuple[bool, List[str]]:
    checks = [
        check_pyproject,
        check_scripts_excluded_from_package,
        *[lambda t=t: check_package_importable(t) for t in PACKAGED_TOP_LEVEL + UNPACKAGED_TOP_LEVEL],
        check_inference_engine_exported,
        check_test_dirs,
        check_requirements_importable,
    ]

    print("=" * 56)
    print()
    print("VisionServe AI\n\nRepository Validation")
    print()
    print("=" * 56)

    ok = True
    lines: List[str] = []
    for check in checks:
        passed, line = check()
        ok = ok and passed
        lines.append(line)
        print(line, file=(sys.stdout if passed else sys.stderr))

    print()
    print("Repository Validation " + ("PASSED" if ok else "FAILED"))
    return ok, lines


def main(argv: Optional[List[str]] = None) -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    # Ensure the repo root is importable regardless of CWD when invoked as `python -m scripts.validate_repository`.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    ok, _ = run()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
