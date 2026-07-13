"""
Default constant values for VisionServeAI.

Migrated verbatim from Sprint 05 Stage 1 of the archived notebook. These were
previously module-level globals re-declared (or silently assumed) across
multiple notebook stages; this is now their single source of truth, imported
wherever needed instead of redefined.

No default *values* were changed -- ``KAGGLE_INPUT_ROOT`` and ``OUTPUT_ROOT``
still default to the exact Kaggle-specific paths the original notebook used.
What was added in this cleanup phase (task: "Artifact compatibility audit --
do not hardcode Kaggle paths, support local deployment") is an optional
environment-variable override for each, resolved once at import time:
``VISIONSERVE_INPUT_ROOT`` / ``VISIONSERVE_OUTPUT_ROOT``. When neither is
set, behavior is byte-for-byte identical to before this change. This is
intentionally NOT a reintroduction of Stage 1's recursive Kaggle-dataset-mount
scan (still out of scope here, per ``services.artifact_service``'s own
module docstring) -- just an unblocking of the two hardcoded absolute paths
that would otherwise make local (non-Kaggle) deployment impossible even when
every other artifact-discovery input (``configs.schema.DeploymentConfig.artifact_roots``)
is already fully configurable.

Source: sprint05-deployment.ipynb, Stage 1, lines ~39-104.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

SEED: int = 42
KAGGLE_INPUT_ROOT: Path = Path(os.environ.get("VISIONSERVE_INPUT_ROOT", "/kaggle/input"))
OUTPUT_ROOT: Path = Path(os.environ.get("VISIONSERVE_OUTPUT_ROOT", "/kaggle/working/sprint05_deployment"))
STAGE_DIR_NAME: str = "stage01_environment"
MAX_SCAN_DEPTH: int = 6

# Fingerprint files used to IDENTIFY which mounted Kaggle dataset satisfies
# each required artifact source. Purely content-based -- no dataset slug or
# username is ever hardcoded, per the original robust-discovery requirement.
FINGERPRINTS: Dict[str, List[str]] = {
    "sprint03": [
        "disease_registry.json",
        "dataset_statistics.json",
        "train_manifest.csv",
    ],
    "sprint04_training": [
        "best_model.pt",
        "training_summary.json",
        "checkpoint_summary.json",
    ],
    "sprint04_evaluation": [
        "evaluation_summary.json",
        "optimal_thresholds.json",
        "deployment_recommendations.json",
    ],
    "nih_chest_xray": [
        "Data_Entry_2017.csv",
        "BBox_List_2017.csv",
    ],
}

# Files whose absence is FATAL (stage fails loudly, fail-fast).
CRITICAL_FILES: Dict[str, List[str]] = {
    "sprint03": ["disease_registry.json"],
    "sprint04_training": ["best_model.pt", "training_summary.json"],
    "sprint04_evaluation": ["evaluation_summary.json"],
    "nih_chest_xray": ["Data_Entry_2017.csv"],
}

# Files that are recorded but only WARN if missing (nice-to-have, informational).
OPTIONAL_FILES: Dict[str, List[str]] = {
    "sprint03": [
        "class_distribution.json",
        "class_weights.json",
        "dataset_statistics.json",
        "dataset_summary.json",
        "train_manifest.csv",
        "val_manifest.csv",
        "test_manifest.csv",
    ],
    "sprint04_training": [
        "checkpoint_summary.json",
        "optimizer_configuration.json",
        "scheduler_configuration.json",
        "training_report.json",
        "training_history.json",
    ],
    "sprint04_evaluation": [
        "evaluation_report.json",
        "deployment_readiness.json",
        "publication_readiness.json",
        "optimal_thresholds.json",
        "calibration_summary.json",
    ],
    "nih_chest_xray": ["BBox_List_2017.csv"],
}
