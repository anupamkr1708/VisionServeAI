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

**Foundation phase (current).** Repository skeleton, typed config schema,
and canonical de-duplicated utilities only. No inference, FastAPI, deployment
pipeline, or export logic yet — those land in subsequent phases.

## Structure

```
backend/            FastAPI app, routers, request/response schemas. No model logic.
inference/           Model loading, runtimes (TorchScript/ONNX), preprocessing,
                     postprocessing, thresholding, explainability. No FastAPI code.
  utils/             Canonical shared utilities (logging, JSON I/O, hashing,
                     timers, resource monitoring) -- single source of truth,
                     consolidated from 8 duplicated per-stage copies in the
                     original notebook.
deployment/          Artifact registry, export, packaging, release engineering.
                     No training/evaluation code.
configs/             Typed config dataclasses (schema.py) + default values
                     (defaults.py), replacing hardcoded notebook globals.
artifacts/           No code -- exported models, thresholds, deployment
                     metadata, release packages land here at runtime.
docs/                Deployment guide, model card, API reference, release notes.
tests/               unit/, integration/, api/
scripts/             CLI entrypoints (build_release.py, export_model.py, ...)
docker/              Dockerfile, docker-compose.yml
frontend/            Placeholder only.
```

See `docs/MIGRATION_NOTES.md` (added once the first content-bearing phase
lands) for the stage-by-stage mapping from the Sprint 05 notebook to this
repo, including every place a duplicated utility was consolidated and exactly
what varied between copies.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest tests/unit -v
```
