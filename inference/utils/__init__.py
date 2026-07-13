"""
Canonical, de-duplicated low-level utilities (logging, JSON I/O, hashing,
timers, resource monitoring) shared across inference/, deployment/,
backend/, and scripts/.

Re-exports the public API of each submodule so callers can do
``from inference.utils import build_logger, save_json, Timer`` etc.
"""
from inference.utils.environment import get_environment_info, set_seed
from inference.utils.hashing import sha256_of_file
from inference.utils.io import load_json, save_json
from inference.utils.logging import build_logger
from inference.utils.resource_monitor import (
    get_device_resource_usage,
    get_resource_usage,
    log_device_resources,
    log_resources,
)
from inference.utils.timers import Timer

__all__ = [
    "build_logger",
    "save_json",
    "load_json",
    "sha256_of_file",
    "Timer",
    "get_resource_usage",
    "log_resources",
    "get_device_resource_usage",
    "log_device_resources",
    "set_seed",
    "get_environment_info",
]
