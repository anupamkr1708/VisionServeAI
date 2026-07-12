"""
Canonical JSON I/O utilities.

``save_json`` had 3 near-identical copies across the notebook; diffing found
one genuine behavioural difference: the explainability stage's copy (Stage
6) creates the parent directory before writing, the other 7 stages' copies
do not. Preserved via ``ensure_parent`` rather than silently picking one
behaviour -- default (False) matches stages 1-5, 7, 8; pass True to
reproduce Stage 6.

``load_json`` had 2 copies differing only in return-type annotation
(``Dict[str, Any]`` vs ``Any``); behaviourally identical, single function.

Source: sprint05-deployment.ipynb, stages 1-8.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(path: Path, data: Any, ensure_parent: bool = False) -> None:
    """Write ``data`` to ``path`` as indented JSON (non-serializable values
    stringified via ``default=str``, matching every original copy).

    Args:
        path: Destination file path.
        data: JSON-serializable data (or containing values that fall back
            to ``str()``).
        ensure_parent: If True, creates ``path.parent`` first -- reproduces
            the Stage 6 (explainability) variant. Stages 1-5, 7, 8 assumed
            the directory already existed and did not create it; default
            False preserves that.
    """
    if ensure_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    else:
        path.write_text(json.dumps(data, indent=2, default=str))


def load_json(path: Path) -> Any:
    """Read and parse a JSON file."""
    return json.loads(path.read_text())
