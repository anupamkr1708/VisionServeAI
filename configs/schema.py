"""
Typed configuration schema for VisionServeAI.

Migrated verbatim from Sprint 05 Stage 1 ("Deployment Environment & Artifact
Discovery") of the archived notebook. No fields, types, or defaults were
changed -- the only modification is that ``ExportConfig.export_dir`` now
derives from ``configs.defaults.OUTPUT_ROOT`` via import instead of
referencing a notebook-local global, per the "replace notebook globals with
proper imports" migration rule. Behaviourally identical to the original.

Source: sprint05-deployment.ipynb, Stage 1, lines ~169-224.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from configs.defaults import OUTPUT_ROOT, SEED


@dataclass
class LoggingConfig:
    """Logging behaviour for a single run."""

    log_dir: str
    log_level: str = "INFO"
    log_format: str = "%(asctime)s | %(levelname)-8s | %(message)s"
    log_to_file: bool = True
    log_to_console: bool = True


@dataclass
class BenchmarkConfig:
    """Latency/throughput benchmarking parameters.

    ``enabled`` was reserved (False) at the point this class was authored in
    the notebook, ahead of the benchmarking stage being implemented -- kept
    as-is; benchmarking itself lands in the deployment/benchmarking.py phase.
    """

    enabled: bool = False
    warmup_iterations: int = 10
    benchmark_iterations: int = 100
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16])


@dataclass
class ExportConfig:
    """TorchScript / ONNX export parameters.

    ``onnx_enabled`` / ``torchscript_enabled`` were reserved (False) at the
    point this class was authored in the notebook, ahead of the export stage
    being implemented -- kept as-is; export itself lands in the
    deployment/export/ phase.
    """

    onnx_enabled: bool = False
    onnx_opset: int = 17
    torchscript_enabled: bool = False
    export_dir: str = str(OUTPUT_ROOT / "exports")


@dataclass
class APIConfig:
    """FastAPI serving parameters.

    ``reserved=True`` in the original notebook signalled that serving was
    not yet activated at Stage 1 -- kept as-is; the backend/ package is a
    later migration phase.
    """

    host: str = "0.0.0.0"
    port: int = 8000
    reserved: bool = True


@dataclass
class RuntimeConfig:
    """Reproducibility / execution parameters."""

    seed: int = SEED
    deterministic: bool = True
    num_workers: int = 2
    pin_memory: bool = True
    use_amp: bool = False  # reserved -- no inference happens at Stage 1


@dataclass
class DeploymentConfig:
    """Top-level deployment configuration, composing the configs above."""

    device: str
    dtype: str
    artifact_roots: Dict[str, Optional[str]]
    output_root: str
    logging: LoggingConfig
    benchmark: BenchmarkConfig
    export: ExportConfig
    api: APIConfig
    runtime: RuntimeConfig
    tensorrt_enabled_future: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
