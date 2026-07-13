# VisionServeAI

Production deployment and serving layer for the chest X-ray multi-label
classifier. This repository is **deployment-only** — it consumes frozen
artifacts (`best_model.pt`, `model.ts`, `model.onnx`, thresholds, disease
registry, evaluation/deployment reports) produced by the archived Sprint
01–04 research pipeline. It contains no training, dataset-engineering, or
evaluation code, and does not retrain or re-evaluate the model.

Migrated module-by-module from the Sprint 05 deployment notebook, preserving
all engineering validations (logging, timing, SHA-256 verification, artifact
validation, deployment validation, readiness scoring, runtime validation,
benchmarking, robustness validation, packaging, release generation,
documentation generation) with no behavioural changes — only the notebook's
execution-order dependencies have been removed.

## Status

**Inference core complete; FastAPI backend not yet started.** The
following phases are complete, validated, and frozen (do not rewrite):

| Phase | Contents |
|---|---|
| Foundation | `configs/` (typed schema + defaults), `inference/utils/` (logging, hashing, IO, timers, resource monitoring, environment/seed reporting) |
| Model Loader | `WrappedClassifier`, `reconstruct_model()`, `ModelRegistry`, backbone resolution, checkpoint loading, fingerprinting, validation |
| Inference Pipeline | preprocessing, thresholding, postprocessing, `InferenceEngine`, `PredictionResult`, threshold registry |
| Runtime Abstraction Layer | `BaseRuntime`, `PyTorchRuntime`, `TorchScriptRuntime`, `ONNXRuntime`, `RuntimeFactory`, `RuntimeRegistry` |
| Application Service Layer | `ArtifactService`, `ModelService`, `RuntimeService`, `PredictionService`, `ExplainabilityService`, `HealthService`, `ServiceRegistry` |
| Explainability | GradCAM, GradCAM++, ScoreCAM, EigenCAM, Guided Backprop, Integrated Gradients, Occlusion, layer discovery, hooks, visualization, `ExplainabilityEngine` |

Not yet implemented (do not add in this phase): `backend/` (FastAPI app,
routers, OpenAPI/Swagger, auth, middleware), `deployment/` (export,
packaging, release engineering, documentation generation), `docker/`,
`frontend/`, `tests/` (scaffolding only, no test bodies yet), CI/CD.

For resolving `artifact_roots` against a local (non-Kaggle) artifact
directory tree, see `scripts/resolve_artifact_roots.py`.

## Structure

```
backend/            FastAPI app, routers, request/response schemas. Not yet implemented.
inference/           Model loading, runtimes (PyTorch/TorchScript/ONNX),
                     preprocessing, postprocessing, thresholding, the
                     InferenceEngine, and explainability. No FastAPI code.
  utils/             Canonical shared utilities (logging, JSON I/O, hashing,
                     timers, resource monitoring, environment/seed
                     reporting) -- single source of truth, consolidated
                     from duplicated per-stage copies in the original
                     notebook.
  runtimes/          BaseRuntime + PyTorch/TorchScript/ONNX backends,
                     RuntimeFactory, RuntimeRegistry.
  explainability/     GradCAM/GradCAM++/ScoreCAM/EigenCAM/Guided
                     Backprop/Integrated Gradients/Occlusion + ExplainabilityEngine.
services/            Application service layer used by (future) backend/:
                     ArtifactService, ModelService, RuntimeService,
                     PredictionService, ExplainabilityService,
                     HealthService, ServiceRegistry.
deployment/          Artifact registry, export, packaging, release engineering.
                     Not yet implemented (schema-only configs exist in configs/schema.py).
configs/             Typed config dataclasses (schema.py) + default values
                     (defaults.py), replacing hardcoded notebook globals.
artifacts/           No code -- exported models, thresholds, deployment
                     metadata, release packages land here at runtime.
docs/                Deployment guide, model card, API reference, release notes.
tests/               unit/, integration/, api/ -- scaffolding only, no test bodies yet.
scripts/             CLI entrypoints, incl. resolve_artifact_roots.py (resolves a
                     local artifact_roots mapping for ServiceRegistry -- see Status above).
docker/              Dockerfile, docker-compose.yml. Not yet implemented.
frontend/            Placeholder only.
```

See `docs/MIGRATION_NOTES.md` for the stage-by-stage mapping from the
Sprint 05 notebook to this repo, including every documented architectural
adaptation and every place a duplicated utility was consolidated.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest tests/unit -v
```

### Local (non-Kaggle) deployment

`configs.defaults.KAGGLE_INPUT_ROOT` / `OUTPUT_ROOT` default to Kaggle paths
(unchanged from the original notebook), but can be overridden without any
code change:

```bash
export VISIONSERVE_INPUT_ROOT=/path/to/local/artifacts
export VISIONSERVE_OUTPUT_ROOT=/path/to/local/output
```

To resolve `ServiceRegistry`'s `artifact_roots` mapping from a local
artifact directory tree (see `scripts/resolve_artifact_roots.py`):

```bash
python -m scripts.resolve_artifact_roots /path/to/artifacts
```
