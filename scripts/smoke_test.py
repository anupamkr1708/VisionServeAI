"""
Production smoke test for VisionServeAI.

Exercises the *entire* serving pipeline strictly through the service layer,
in the required dependency order::

    ArtifactService -> ModelService -> RuntimeService -> PredictionService
        -> ExplainabilityService -> HealthService -> ServiceRegistry

Never instantiates ``inference/`` modules directly except where a service
already does so internally (e.g. ``ServiceRegistry`` building an
``ExplainabilityEngine`` -- this script only ever calls
``registry.explainability.generate_*``, never
``inference.explainability.engine.ExplainabilityEngine`` itself).

This is verification infrastructure, not a notebook migration -- nothing
here reimplements or changes any production module. It is purely additive.

Two ways to run it
-------------------
1. **Synthetic self-test** (no real artifacts needed) -- proves the wiring
   itself is correct, on a tiny randomly-initialized model built by
   ``scripts._synthetic_fixtures``::

       python -m scripts.smoke_test --synthetic

2. **Real artifacts** -- the actual verification this script exists for::

       python -m scripts.smoke_test --artifact-root /path/to/artifacts \\
           --export-dir /path/to/exports

   ``--artifact-root`` is resolved via ``scripts.resolve_artifact_roots``
   (see that script for what it does and its own ``AMBIGUOUS``-match
   caveats). TorchScript/ONNX runtime checks are automatically skipped
   (not failed) if no ``model.ts`` / ``model.onnx`` is found under
   ``--export-dir`` -- Stage 3 (export) is intentionally not migrated into
   this repository yet (see ``inference/preprocessing.py``'s own
   docstring), so real deployments without an export step won't have them.

Exit code is 0 if every check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from inference.postprocessing import top_k_predictions
from inference.utils.environment import get_environment_info
from services.runtime_service import RuntimeService
from services.service_registry import ServiceRegistry

BANNER_WIDTH = 56


class Checklist:
    """Tracks ✓/✗ lines for the final PASS/FAIL summary. Verification
    bookkeeping only -- no production behavior."""

    def __init__(self) -> None:
        self.lines: List[str] = []
        self.failed = False

    def ok(self, label: str) -> None:
        self.lines.append(f"✓ {label}")
        print(f"✓ {label}")

    def fail(self, label: str, detail: str = "") -> None:
        self.failed = True
        suffix = f" -- {detail}" if detail else ""
        self.lines.append(f"✗ {label}{suffix}")
        print(f"✗ {label}{suffix}", file=sys.stderr)

    def skip(self, label: str, reason: str) -> None:
        self.lines.append(f"– {label} (skipped: {reason})")
        print(f"– {label} (skipped: {reason})")


def _banner(title: str) -> None:
    print("=" * BANNER_WIDTH)
    print()
    print(title)
    print()
    print("=" * BANNER_WIDTH)


def _step(logger: logging.Logger, checklist: Checklist, label: str, fn, *args, **kwargs):
    """Run ``fn``, record a ✓/✗ line, and return ``fn``'s result (or
    ``None`` on failure) -- avoids repeating the same try/except/log
    boilerplate for every one of the ~20 checks below."""
    try:
        result = fn(*args, **kwargs)
        checklist.ok(label)
        return result
    except Exception as exc:  # noqa: BLE001 -- isolate failure, keep testing the rest
        logger.error("SMOKE_TEST step failed: %s: %s", label, exc, exc_info=True)
        checklist.fail(label, str(exc))
        return None


# ----------------------------------------------------------------------
# Individual phase implementations (each raises on failure; _step() wraps)
# ----------------------------------------------------------------------


def check_environment() -> Dict[str, Any]:
    info = get_environment_info()
    required_keys = {"python_version", "torch_version", "torchvision_version", "device"}
    missing = required_keys - set(info)
    if missing:
        raise RuntimeError(f"environment report missing keys: {missing}")
    return info


def check_artifacts(registry: ServiceRegistry) -> Dict[str, Any]:
    assert registry.artifact is not None
    status = registry.artifact.artifact_status()
    if not status["healthy"]:
        raise RuntimeError(f"serving-critical artifacts missing: {status['serving_critical_missing']}")
    return status


def check_model(registry: ServiceRegistry) -> Dict[str, Any]:
    assert registry.model is not None
    mr = registry.model.model_registry
    if not mr.backbone or mr.num_classes <= 0:
        raise RuntimeError("model registry has no backbone / non-positive num_classes")
    if mr.total_parameters <= 0:
        raise RuntimeError("model registry reports zero parameters")
    if len(mr.checkpoint_sha256) != 64:
        raise RuntimeError(f"checkpoint fingerprint is not a sha256 hex digest: {mr.checkpoint_sha256!r}")
    return mr.to_dict()


def check_runtime(runtime_service: RuntimeService, input_shape) -> Dict[str, Any]:
    """load() + warmup() + validate() already ran inside
    ``initialize_runtime`` (called by the caller before this function).
    This additionally exercises ``metadata()`` and ``predict()`` directly
    on the runtime object ``RuntimeService`` owns -- not a bypass of the
    service layer, since ``PredictionService`` itself calls exactly these
    same two methods on whatever ``RuntimeService.get_runtime()`` returns.
    """
    runtime = runtime_service.get_runtime()
    if not runtime.is_loaded:
        raise RuntimeError("runtime.is_loaded is False after initialize_runtime()")
    if not runtime.is_validated:
        raise RuntimeError("runtime.is_validated is False after initialize_runtime()")

    meta = runtime.metadata()
    if not meta.runtime_version:
        raise RuntimeError("runtime metadata has no runtime_version")

    dummy = torch.randn(*input_shape, dtype=torch.float32)
    output = runtime.predict(dummy)
    if output.shape[0] != input_shape[0]:
        raise RuntimeError(f"runtime.predict() batch dimension mismatch: input={input_shape} output={tuple(output.shape)}")
    return {"metadata": meta.to_dict(), "output_shape": tuple(output.shape)}


def check_single_prediction(registry: ServiceRegistry, image_path: str):
    """Single-image predict + top-k + thresholding + determinism. Returns
    the successful :class:`~inference.postprocessing.PredictionResult` so
    :func:`check_batch_prediction` doesn't need to re-derive class names."""
    assert registry.prediction is not None
    assert registry.model is not None

    single = registry.prediction.predict_image(image_path)
    if not single.success:
        raise RuntimeError(f"single-image prediction failed: {single.error}")
    if single.error is not None:
        raise RuntimeError(f"successful PredictionResult unexpectedly carries an error: {single.error}")
    class_names = registry.model.class_names()
    if set(single.probabilities.keys()) != set(class_names):
        raise RuntimeError("PredictionResult.probabilities keys don't match the resolved class ordering")
    if set(single.thresholds_used.keys()) != set(class_names):
        raise RuntimeError("PredictionResult.thresholds_used keys don't match the resolved class ordering")

    # -- thresholding correctness: predicted_diseases must be exactly the
    #    classes whose probability exceeds their own threshold, no more,
    #    no less --
    expected_positive = {
        name for name in class_names
        if single.probabilities[name] >= single.thresholds_used[name]
    }
    if set(single.predicted_diseases) != expected_positive:
        raise RuntimeError(
            f"thresholding mismatch: predicted={sorted(single.predicted_diseases)} "
            f"expected={sorted(expected_positive)}"
        )

    # -- top-k --
    top5 = top_k_predictions(single, k=min(5, len(class_names)))
    if len(top5) != min(5, len(class_names)):
        raise RuntimeError(f"top_k_predictions returned {len(top5)} items, expected {min(5, len(class_names))}")
    sorted_desc = all(top5[i][1] >= top5[i + 1][1] for i in range(len(top5) - 1))
    if not sorted_desc:
        raise RuntimeError(f"top_k_predictions is not sorted descending by confidence: {top5}")

    # -- determinism: identical input -> identical output, run twice --
    repeat = registry.prediction.predict_image(image_path)
    if repeat.probabilities != single.probabilities:
        raise RuntimeError(
            "prediction is not deterministic across repeated calls on the same image "
            f"(first={single.probabilities} second={repeat.probabilities})"
        )

    return {"single": single.to_dict(), "top_k": top5, "deterministic": True}


def check_batch_prediction(registry: ServiceRegistry, image_paths: List[str]) -> Dict[str, Any]:
    assert registry.prediction is not None
    batch_paths = (image_paths * 2)[:max(2, len(image_paths))]
    batch_results = registry.prediction.predict_batch(batch_paths)
    if len(batch_results) != len(batch_paths):
        raise RuntimeError(f"predict_batch returned {len(batch_results)} results for {len(batch_paths)} inputs")
    if not all(r.success for r in batch_results):
        failed = [r.error for r in batch_results if not r.success]
        raise RuntimeError(f"one or more batch predictions failed: {failed}")
    return {"batch_count": len(batch_results)}


def check_explainability(registry: ServiceRegistry, image_path: str) -> Dict[str, Any]:
    assert registry.explainability is not None
    from PIL import Image

    if not registry.explainability.is_available():
        raise RuntimeError(
            "ExplainabilityService has no engine injected -- ServiceRegistry should auto-build one "
            "unless enable_explainability=False was passed."
        )

    image = Image.open(image_path).convert("RGB")
    method_calls = {
        "gradcam": registry.explainability.generate_gradcam,
        "gradcam_plus": registry.explainability.generate_gradcam_plus,
        "scorecam": registry.explainability.generate_scorecam,
        "eigencam": registry.explainability.generate_eigencam,
        "guided_backprop": registry.explainability.generate_guided_backprop,
        "integrated_gradients": registry.explainability.generate_integrated_gradients,
        "occlusion": registry.explainability.generate_occlusion,
    }

    results: Dict[str, Any] = {}
    failures: List[str] = []
    for method_name, fn in method_calls.items():
        result = fn(image, sample_id=f"smoke_{method_name}")
        results[method_name] = result.to_dict()
        if not result.success:
            failures.append(f"{method_name}: {result.error}")
            continue
        for path_attr in ("heatmap_path", "overlay_path"):
            path_value = getattr(result, path_attr)
            if not path_value or not Path(path_value).exists():
                failures.append(f"{method_name}: {path_attr} missing or not written to disk: {path_value}")

    if failures:
        raise RuntimeError("explainability failures: " + "; ".join(failures))

    return results


def check_health(registry: ServiceRegistry) -> Dict[str, Any]:
    assert registry.health is not None
    snapshot = registry.health.health()
    if snapshot["status"] not in ("healthy", "degraded"):
        raise RuntimeError(f"unexpected health status: {snapshot['status']!r} (full snapshot: {snapshot})")
    if snapshot["status"] != "healthy":
        raise RuntimeError(f"health status is '{snapshot['status']}', expected 'healthy': {snapshot}")
    return snapshot


def check_service_registry(registry: ServiceRegistry) -> Dict[str, bool]:
    if not registry.is_initialized:
        raise RuntimeError("ServiceRegistry.is_initialized is False after initialize()")
    services = {
        "artifact": registry.artifact, "model": registry.model, "runtime": registry.runtime,
        "prediction": registry.prediction, "explainability": registry.explainability, "health": registry.health,
    }
    missing = [name for name, svc in services.items() if svc is None]
    if missing:
        raise RuntimeError(f"ServiceRegistry did not construct: {missing}")
    return {name: True for name in services}


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------


def run_smoke_test(
    artifact_roots: Dict[str, Optional[str]],
    export_dir: Path,
    image_paths: List[str],
    device: Optional[torch.device],
    runtime_types: List[str],
    logger: logging.Logger,
    explainability_output_dir: Optional[Path] = None,
    seed: int = 42,
) -> bool:
    """Returns ``True`` if every check passed. Prints the checklist as it
    goes and the final PASS/FAIL summary at the end."""
    checklist = Checklist()
    _banner("VisionServe AI\n\nProduction Smoke Test")

    env_info = _step(logger, checklist, "Environment verified", check_environment)
    if env_info is not None:
        logger.info(
            "SMOKE_TEST environment: python=%s torch=%s torchvision=%s device=%s",
            env_info["python_version"], env_info["torch_version"],
            env_info["torchvision_version"], env_info["device"],
        )

    # ArtifactService -> ModelService -> RuntimeService(pytorch) ->
    # PredictionService/HealthService -> ExplainabilityService, all via one
    # ServiceRegistry.initialize() call, in the required dependency order.
    registry = ServiceRegistry(
        artifact_roots=artifact_roots,
        export_dir=export_dir,
        device=device,
        runtime_type="pytorch",
        explainability_output_dir=explainability_output_dir,
        seed=seed,
    )
    init_ok = _step(logger, checklist, "Service registry initialized (Artifact->Model->Runtime[pytorch])",
                     registry.initialize)
    if init_ok is None:
        _banner("Smoke Test FAILED (could not initialize ServiceRegistry -- see errors above)")
        return False

    _step(logger, checklist, "Artifacts discovered", check_artifacts, registry)
    _step(logger, checklist, "Model reconstructed", check_model, registry)

    cfg = registry.model.preprocessing_config
    input_shape = (1, cfg.channels, cfg.resize_height, cfg.resize_width)

    _step(logger, checklist, "Runtime loaded/validated/warmed up (pytorch)", check_runtime,
          registry.runtime, input_shape)

    # -- Additional runtime backends, each via its own RuntimeService
    #    (never a raw BaseRuntime constructed outside a service) --
    for extra_type in [t for t in runtime_types if t != "pytorch"]:
        path_getter = {"torchscript": registry.artifact.torchscript_path, "onnx": registry.artifact.onnx_path}[extra_type]
        runtime_path = path_getter()
        if runtime_path is None:
            checklist.skip(
                f"Runtime loaded/validated/warmed up ({extra_type})",
                f"no exported artifact found under {export_dir} (Stage 3/export not migrated -- see module docstrings)",
            )
            continue
        extra_service = RuntimeService(
            model_service=registry.model, logger=logger, runtime_type=extra_type, runtime_path=runtime_path,
        )
        extra_ok = _step(logger, checklist, f"Runtime initialized ({extra_type})",
                          extra_service.initialize_runtime, warmup=True, validate=True)
        if extra_ok is not None:
            _step(logger, checklist, f"Runtime loaded/validated/warmed up ({extra_type})", check_runtime,
                  extra_service, input_shape)

    _step(logger, checklist, "Prediction passed (single/top-k/thresholding/determinism)",
          check_single_prediction, registry, image_paths[0])
    _step(logger, checklist, "Batch prediction passed", check_batch_prediction, registry, image_paths)
    _step(logger, checklist, "Explainability passed (7 methods)", check_explainability, registry, image_paths[0])
    _step(logger, checklist, "Health service passed", check_health, registry)
    _step(logger, checklist, "Service registry passed", check_service_registry, registry)

    registry.shutdown()

    if checklist.failed:
        _banner("Smoke Test FAILED")
    else:
        _banner("Smoke Test PASSED")
    return not checklist.failed


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true",
                       help="Use a self-contained synthetic fixture (no real artifacts needed).")
    mode.add_argument("--artifact-root", type=str,
                       help="Local artifact directory tree, resolved via scripts.resolve_artifact_roots.")
    mode.add_argument("--artifact-roots-json", type=str,
                       help="Path to a JSON file containing an already-resolved artifact_roots mapping.")
    parser.add_argument("--export-dir", type=str, default=None,
                         help="Directory containing model.ts/model.onnx (ignored in --synthetic mode, which "
                              "exports its own into a temp directory).")
    parser.add_argument("--runtimes", type=str, default="pytorch,torchscript,onnx",
                         help="Comma-separated runtime types to test (default: all three).")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"],
                         help="Force a device; default auto-detects CUDA.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    logger = logging.getLogger("visionserve.smoke_test")

    device = torch.device(args.device) if args.device else None
    runtime_types = [t.strip() for t in args.runtimes.split(",") if t.strip()]

    with tempfile.TemporaryDirectory(prefix="visionserve_smoke_") as tmp_dir:
        tmp_path = Path(tmp_dir)

        if args.synthetic:
            from scripts._synthetic_fixtures import (
                build_sample_image, build_synthetic_artifact_tree, export_onnx, export_torchscript,
            )
            fixture = build_synthetic_artifact_tree(tmp_path / "artifacts", seed=args.seed)
            artifact_roots = fixture["artifact_roots"]
            export_dir = tmp_path / "export"
            export_dir.mkdir(parents=True, exist_ok=True)

            # Build the same model this fixture's checkpoint contains, so
            # TorchScript/ONNX can be exercised too, without needing
            # inference/model_loader touched a second time here.
            import json as _json
            from inference.model_loader import reconstruct_model
            training_summary = _json.loads(fixture["training_summary_path"].read_text())
            disease_registry = _json.loads(fixture["disease_registry_path"].read_text())
            model, _mr = reconstruct_model(
                training_summary=training_summary, disease_registry=disease_registry,
                checkpoint_path=fixture["checkpoint_path"], device=torch.device("cpu"), logger=logger,
            )
            input_shape = (1, 3, 224, 224)
            if "torchscript" in runtime_types:
                export_torchscript(model, export_dir / "model.ts", input_shape)
            if "onnx" in runtime_types:
                export_onnx(model, export_dir / "model.onnx", input_shape)

            image_paths = [str(build_sample_image(tmp_path / "sample_1.png", seed=1)),
                            str(build_sample_image(tmp_path / "sample_2.png", seed=2))]
            explainability_output_dir = tmp_path / "explainability"

        elif args.artifact_root:
            from scripts.resolve_artifact_roots import resolve_artifact_roots
            artifact_roots = resolve_artifact_roots(args.artifact_root, logger=logger)
            export_dir = Path(args.export_dir) if args.export_dir else Path("deployment") / "export"
            image_paths_arg = tmp_path / "probe.png"
            from scripts._synthetic_fixtures import build_sample_image
            image_paths = [str(build_sample_image(image_paths_arg))]
            explainability_output_dir = None

        else:
            with open(args.artifact_roots_json) as f:
                artifact_roots = json.load(f)
            export_dir = Path(args.export_dir) if args.export_dir else Path("deployment") / "export"
            from scripts._synthetic_fixtures import build_sample_image
            image_paths = [str(build_sample_image(tmp_path / "probe.png"))]
            explainability_output_dir = None

        passed = run_smoke_test(
            artifact_roots=artifact_roots,
            export_dir=export_dir,
            image_paths=image_paths,
            device=device,
            runtime_types=runtime_types,
            logger=logger,
            explainability_output_dir=explainability_output_dir,
            seed=args.seed,
        )

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
