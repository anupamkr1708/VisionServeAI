"""
Canonical logger factory.

Consolidated from 8 copies of ``build_logger()`` (one per Sprint 05 notebook
stage). Diffed all 8 before consolidating: the *only* variation between them
was the hardcoded logger name (``"sprint05.stageNN"``) and log file name
(``"stageNN_xxx.log"``) -- every other line (formatter, handler setup,
propagate=False) was byte-identical. Both are now parameters instead of
hardcoded, so this one function reproduces all 8 original call sites exactly
when given their original name/filename.

Source: sprint05-deployment.ipynb, stages 1-8 (identical logic in each).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def build_logger(
    name: str,
    log_dir: Path,
    log_filename: Optional[str] = None,
    level: str = "INFO",
) -> logging.Logger:
    """Create a console + file logger (no prints), matching every Sprint 05
    stage logger exactly.

    Args:
        name: Logger name, e.g. ``"visionserve.artifact_registry"``. The
            original notebook used ``"sprint05.stageNN"`` per stage.
        log_dir: Directory the log file is written into (created if missing).
        log_filename: File name within ``log_dir``. Defaults to
            ``f"{name.replace('.', '_')}.log"`` if not given -- pass the
            original stage's filename explicitly (e.g.
            ``"stage01_environment.log"``) to reproduce it exactly.
        level: Log level name. The original notebook always hardcoded
            ``logging.INFO``; defaulting here to ``"INFO"`` preserves that,
            while allowing config-driven overrides going forward.

    Returns:
        A configured ``logging.Logger`` with a file handler and a stdout
        stream handler, propagation disabled, and any previous handlers on
        that logger name cleared (safe to call more than once per process).
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    if log_filename is None:
        log_filename = f"{name.replace('.', '_')}.log"

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_dir / log_filename, mode="w")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger
