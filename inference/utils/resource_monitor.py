"""
Canonical resource-monitoring utilities.

Consolidated from per-stage copies of ``get_resource_usage()`` /
``log_resources()`` found across the Sprint 05 notebook. Diffing all
occurrences (see docs/MIGRATION_NOTES.md once written) found these were
*not* all identical -- three genuinely different variants existed:

1. Stage 1's own copy: CPU/RAM/disk + GPU (GB, via global
   ``torch.cuda.is_available()`` check).
2. A variant shared verbatim by stages 2, 3, 4, 6, 7, 8: same as (1) but
   without disk stats.
3. Stage 5's variant: takes an explicit ``device: torch.device`` argument,
   reports process RSS, and uses MB (not GB) for GPU figures, checking
   ``device.type == "cuda"`` instead of the global availability flag.

Rather than silently collapsing these into one lossy function, variants 1-2
are preserved as ``get_resource_usage(include_disk=...)`` and variant 3 is
kept as a separate function, ``get_device_resource_usage()``, since its
shape (units, extra field, device-scoping) is different enough that forcing
it through the same parameterized function would risk subtly changing
output for callers that need the device-scoped behaviour.

Source: sprint05-deployment.ipynb, stages 1-8.
"""
from __future__ import annotations

import logging
import shutil
from typing import Any, Dict

import psutil
import torch


def get_resource_usage(include_disk: bool = True) -> Dict[str, Any]:
    """Snapshot CPU / RAM / (optionally disk) / GPU-memory usage.

    Args:
        include_disk: If True (default), reproduces Stage 1's exact output
            (includes ``disk_percent`` / ``disk_free_gb``). If False,
            reproduces the variant shared by stages 2, 3, 4, 6, 7, 8.
    """
    vm = psutil.virtual_memory()
    usage: Dict[str, Any] = {
        "cpu_percent": psutil.cpu_percent(interval=0.2),
        "ram_percent": vm.percent,
        "ram_used_gb": round(vm.used / (1024**3), 2),
        "ram_total_gb": round(vm.total / (1024**3), 2),
    }
    if include_disk:
        disk = shutil.disk_usage("/")
        usage["disk_percent"] = round(disk.used / disk.total * 100, 2)
        usage["disk_free_gb"] = round(disk.free / (1024**3), 2)
    if torch.cuda.is_available():
        usage["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated() / (1024**3), 3)
        usage["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved() / (1024**3), 3)
    return usage


def log_resources(logger: logging.Logger, tag: str, include_disk: bool = True) -> Dict[str, Any]:
    """Log and return a resource-usage snapshot.

    ``include_disk`` mirrors :func:`get_resource_usage` and also switches
    the log message format to match: Stage 1 logged disk info, the shared
    5-stage variant did not.
    """
    usage = get_resource_usage(include_disk=include_disk)
    if include_disk:
        logger.info(
            "RESOURCES [%s] cpu=%.1f%% ram=%.1f%%(%.1fGB/%.1fGB) disk=%.1f%% free=%.1fGB",
            tag, usage["cpu_percent"], usage["ram_percent"],
            usage["ram_used_gb"], usage["ram_total_gb"],
            usage["disk_percent"], usage["disk_free_gb"],
        )
    else:
        logger.info(
            "RESOURCES [%s] cpu=%.1f%% ram=%.1f%%(%.1fGB/%.1fGB)",
            tag, usage["cpu_percent"], usage["ram_percent"],
            usage["ram_used_gb"], usage["ram_total_gb"],
        )
    return usage


def get_device_resource_usage(device: torch.device) -> Dict[str, Any]:
    """Device-scoped resource snapshot -- exact port of Stage 5's variant.

    Differs from :func:`get_resource_usage` by design: adds
    ``process_rss_mb``, reports GPU figures in MB (not GB), and scopes the
    GPU check to the given ``device`` rather than global CUDA availability.
    Used by the benchmarking/robustness validation migrated in a later phase.
    """
    vm = psutil.virtual_memory()
    usage: Dict[str, Any] = {
        "cpu_percent": psutil.cpu_percent(interval=0.2),
        "ram_percent": vm.percent,
        "ram_used_gb": round(vm.used / (1024**3), 2),
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "process_rss_mb": round(psutil.Process().memory_info().rss / (1024**2), 2),
    }
    if device.type == "cuda":
        usage["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated(device) / (1024**2), 2)
        usage["gpu_memory_reserved_mb"] = round(torch.cuda.memory_reserved(device) / (1024**2), 2)
    return usage


def log_device_resources(logger: logging.Logger, tag: str, device: torch.device) -> Dict[str, Any]:
    """Log and return a device-scoped resource snapshot -- exact port of
    Stage 5's ``log_resources(logger, tag, device)``."""
    usage = get_device_resource_usage(device)
    logger.info(
        "RESOURCES [%s] cpu=%.1f%% ram=%.1f%% rss=%.1fMB",
        tag, usage["cpu_percent"], usage["ram_percent"], usage["process_rss_mb"],
    )
    return usage
