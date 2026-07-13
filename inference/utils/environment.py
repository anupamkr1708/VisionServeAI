"""
Reproducibility seeding and static environment/hardware reporting.

Migrated verbatim from Sprint 05 Stage 1 ("Deployment Environment & Artifact
Discovery") of the archived notebook -- ``set_seed()`` and
``get_environment_info()``. Neither had been carried into any phase of this
repository before this cleanup pass (confirmed by a repository-wide search:
no ``manual_seed`` / ``set_seed`` call, and no ``torch_version`` /
``cuda_version`` / equivalent field, existed anywhere outside
``configs.schema.RuntimeConfig``'s and this cleanup's own new
``HealthService.environment_info()``). This is filling a gap left by an
earlier phase, not redesigning anything -- both functions are reproduced
exactly, including field names, so any consumer of the original notebook's
``ENVIRONMENT_INFO`` dict shape can consume this one unchanged.

``set_seed()`` matters for this repository even though inference itself is
purely deterministic (eval-mode model, no dropout, no stochastic
preprocessing -- see ``inference.preprocessing.preprocess_image``'s own
docstring): Sprint 05's own pipeline called ``set_seed(SEED)`` once at
Stage 1 startup, before generating any of its synthetic validation/edge-case
fixtures (Stage 4/5/6 sample generation). Anything in this repository that
later reproduces that fixture generation depends on the same seed call
having happened first, exactly as it did in the notebook.

Source: sprint05-deployment.ipynb, Stage 1, lines ~225-276.
"""
from __future__ import annotations

import os
import platform
import random
import sys
from typing import Any, Dict, Optional

import psutil
import torch
import torchvision

from configs.defaults import SEED

def set_seed(seed: int = SEED) -> None:
    """Seed every RNG Sprint 05 seeded at Stage 1 startup: ``random``,
    ``PYTHONHASHSEED``, and Torch (CPU + all CUDA devices, if available).
    Verbatim port -- identical calls, identical order."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _onnx_available() -> bool:
    """Verbatim port of Stage 1's own ``_onnx_available()`` helper."""
    try:
        import onnx  # noqa: F401
        return True
    except ImportError:
        return False


def get_environment_info(seed: Optional[int] = None) -> Dict[str, Any]:
    """Collect static environment / hardware information.

    Verbatim port of Stage 1's ``get_environment_info(seed)`` -- same keys,
    same values, same fallback for a missing GPU. ``seed`` defaults to
    ``configs.defaults.SEED`` (the notebook always called this
    immediately after ``set_seed(SEED)`` with the same constant; made
    optional here only so a caller can report a different seed if one was
    explicitly overridden upstream).
    """
    if seed is None:
        seed = SEED

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    gpu_mem_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        if cuda_available else None
    )
    # NOTE: KAGGLE_INPUT_ROOT now honors VISIONSERVE_INPUT_ROOT (see
    # configs/defaults.py) -- this flag is purely informational (did this
    # process happen to start inside a Kaggle container / with a Kaggle
    # input mount present), not a behavioral branch anywhere in this
    # repository's serving path.
    from configs.defaults import KAGGLE_INPUT_ROOT  # local import: avoid import-time coupling
    is_kaggle = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or KAGGLE_INPUT_ROOT.exists()

    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "gpu_available": cuda_available,
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_mem_gb,
        "cpu_cores": psutil.cpu_count(logical=True),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "os": f"{platform.system()} {platform.release()}",
        "platform": platform.platform(),
        "is_kaggle_environment": is_kaggle,
        "random_seed": seed,
        "device": "cuda" if cuda_available else "cpu",
        "mixed_precision_available": cuda_available and hasattr(torch.cuda, "amp"),
        "torch_compile_available": hasattr(torch, "compile"),
        "onnx_available": _onnx_available(),
        "torchscript_available": hasattr(torch, "jit"),
    }
