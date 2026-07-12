"""
Canonical file-hashing utility.

``sha256_of_file`` was byte-identical across all 4 occurrences in the
notebook (stages 2, 3, 7, 8) -- a true duplicate, ported verbatim with no
parameterization needed.

Source: sprint05-deployment.ipynb, stages 2, 3, 7, 8 (identical).
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 hex digest of a file, reading in chunks so large
    model artifacts don't need to be loaded into memory at once."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
