# Migration notes: sprint05-deployment.ipynb -> VisionServeAI

This maps each completed phase back to its notebook source, and records
every place the migration is *not* a literal copy-paste -- i.e. every
documented, behavior-preserving architectural adaptation. Line numbers refer
to the archived `sprint05-deployment.ipynb`. Every module listed below
carries the same information in its own docstring; this file exists so the
mapping can be read in one place instead of across a dozen files.

## Stage -> module mapping

| Notebook stage | Repository location | Status |
|---|---|---|
| Stage 1 -- Environment & Artifact Discovery | `configs/schema.py`, `configs/defaults.py`, `inference/utils/*` (incl. `environment.py`, added this cleanup pass) | Complete |
| Stage 2 -- Model Registry & Reconstruction | `inference/model_loader.py`, `inference/model_registry.py` | Complete |
| Stage 3 -- Export (TorchScript/ONNX) | `deployment/export/` | **Not migrated** -- out of scope; production inference never re-derives export artifacts |
| Stage 4 -- Production Inference Runtime | `inference/preprocessing.py`, `inference/postprocessing.py`, `inference/thresholding.py`, `inference/engine.py` | Complete |
| (new, not a stage) -- Runtime Abstraction | `inference/runtimes/` (`base.py`, `pytorch_runtime.py`, `torchscript_runtime.py`, `onnx_runtime.py`, `runtime_factory.py`, `runtime_registry.py`) | Complete -- see "Runtime Abstraction Layer" below |
| (new, not a stage) -- Application Service Layer | `services/` (`artifact_service.py`, `model_service.py`, `runtime_service.py`, `prediction_service.py`, `explainability_service.py`, `health_service.py`, `service_registry.py`) | Complete |
| Stage 5 -- Deployment Validation / Benchmarking | -- | **Not migrated** -- out of scope (evaluation/benchmarking generation excluded from production inference code) |
| Stage 6 -- Explainability | `inference/explainability/` (`base.py`, `hooks.py`, `gradcam.py`, `gradcam_plus.py`, `scorecam.py`, `eigencam.py`, `guided_backprop.py`, `integrated_gradients.py`, `occlusion.py`, `engine.py`) | Complete |
| Stage 7 -- Deployment Packaging | `deployment/` | **Not migrated** -- out of scope |
| Stage 8 -- Release / Documentation | `deployment/`, `docs/` | **Not migrated** -- out of scope |

## Documented architectural adaptations (behavior-preserving, not bugs)

These are deliberate, disclosed differences from a literal 1:1 port. Each is
called out in the relevant module's own docstring; listed together here for
convenience.

1. **Runtime Abstraction Layer is new, not "preserved."** The notebook never
   had `BaseRuntime` / `PyTorchRuntime` / `TorchScriptRuntime` /
   `ONNXRuntime` / `RuntimeFactory` / `RuntimeRegistry` -- Stage 4's
   `InferenceEngine` always called `self.model(batch_tensor)` directly on a
   plain reconstructed `nn.Module`, and Stage 3's TorchScript/ONNX export
   was purely for artifact generation and numerical cross-validation, not
   for serving. This layer unifies those two notebook code paths (Stage 2's
   PyTorch model + Stage 3's export/validation logic) behind one interface
   so `PredictionService` can serve any of the three formats uniformly.
   `PyTorchRuntime.predict()` reproduces Stage 4's exact
   `with torch.no_grad(): logits = self.model(batch_tensor)` when
   `use_amp=False` (the default), so numerical behavior for the PyTorch path
   -- the only one Stage 4 itself ever exercised at serving time -- is
   unchanged.
2. **`resolve_preprocessing_config` / `resolve_input_signature` take raw
   values, not registry objects.** The notebook version pulled
   `input_shape` out of Stage 3's `export_registry.model_signature` and
   `training_metadata` out of Stage 2's `metadata_registry`. Neither
   registry exists in this repository (Stage 2's `MetadataRegistry` and all
   of Stage 3 are out of scope), so `services/model_service.py` ports
   Stage 3's `resolve_input_signature` (the `(3, 224, 224)`
   torchvision-default fallback, verbatim) and passes its result plus the
   already-loaded `training_summary` dict directly into
   `inference/preprocessing.py`'s `resolve_preprocessing_config`. The
   resolution *algorithm* (key search order, ImageNet fallback, logging) is
   unchanged.
3. **`InferenceEngine` (Stage 4) and `PredictionService` (new) are both
   kept, on purpose.** `InferenceEngine` is the direct, single-runtime,
   always-PyTorch path matching Stage 4 exactly -- useful standalone (e.g.
   scripts, tests) without needing a `ServiceRegistry`. `PredictionService`
   is the production path used by `ServiceRegistry`: it goes through the
   Runtime Abstraction Layer so it can serve PyTorch/TorchScript/ONNX
   uniformly, batches multiple input types (path/bytes/`PIL.Image`), and
   integrates with `HealthService`/`ExplainabilityService`. See task 7 of
   the audit report -- this is a documented design decision, not
   redundancy needing a merge.
4. **`RuntimeConfig.use_amp` is now consumed.** It existed as a schema
   field from Stage 1 but was never read by any notebook stage.
   `PyTorchRuntime` now honors it, defaulting to `False` (byte-identical to
   Stage 4's behavior); it only activates mixed precision if a caller
   explicitly opts in on a CUDA device.

## Gaps closed in this cleanup pass (see the full audit report for evidence)

- `pyproject.toml` was missing `services*` from `[tool.setuptools.packages.find]`
  -- `services/` was never actually part of the installed package (only
  importable by accident when the working directory happened to be the repo
  root). Fixed.
- `inference/__init__.py` only re-exported the Model Loader phase's symbols;
  `inference.engine.InferenceEngine` (and the rest of the Inference Pipeline
  phase) was unreachable from `inference` itself. Fixed.
- Stage 1's `set_seed()` and `get_environment_info()` were never migrated
  anywhere. Added as `inference/utils/environment.py`; `set_seed()` is now
  called once at `ServiceRegistry.initialize()` (matching Stage 1's own
  call site), and `get_environment_info()` is surfaced via
  `HealthService.environment_info()`.
- `KAGGLE_INPUT_ROOT` / `OUTPUT_ROOT` were hardcoded Kaggle paths with no
  override, which blocked local deployment outright. Both now honor
  `VISIONSERVE_INPUT_ROOT` / `VISIONSERVE_OUTPUT_ROOT` env vars, defaulting
  to the original Kaggle paths when unset (no behavior change unless a
  caller opts in).
- No local (non-Kaggle) equivalent of Stage 1's artifact-discovery existed;
  `ArtifactService` expects an already-resolved, flat `artifact_roots`
  mapping and does a non-recursive lookup by design. Added
  `scripts/resolve_artifact_roots.py`, which generalizes Stage 1's own
  fingerprint-matching discovery algorithm (ported, not reimplemented) to
  any local directory tree.
- Two unused imports in `inference/explainability/base.py`
  (`dataclasses.field`, `typing.Callable`). Removed.

See the full audit report delivered alongside this repository for complete
evidence, verification steps, and the remaining (non-blocking) technical
debt list.
