# VisionServeAI

**A production-engineered serving layer for a 14-pathology chest X-ray classifier — modular FastAPI microservice, 3 pluggable inference runtimes, a 7-method explainability suite, and a validation pipeline honest enough to block its own model from shipping.**

[![Python](https://img.shields.io/badge/Python-3.12-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C)]()
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.17%2B-black)]()
[![Docker](https://img.shields.io/badge/Docker-multi--stage-2496ED)]()
[![Tests](https://img.shields.io/badge/tests-177%20passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## Quick Demo

Every endpoint returns one consistent JSON envelope. No frontend is bundled — this is an API-first serving layer, so the fastest way to see it work is `curl`:

```bash
# 1. Predict on a single chest X-ray
curl -X POST http://localhost:8000/predict \
  -F "file=@chest_xray.png" -F "identifier=patient_001"

# 2. Get a Grad-CAM explanation for a specific class
curl -X POST http://localhost:8000/explain/gradcam \
  -F "file=@chest_xray.png" -F "target_class=Effusion"

# 3. Check whether the model/runtime is actually ready to serve
curl http://localhost:8000/health/ready
```

```json
{
  "success": true,
  "message": "Prediction completed successfully.",
  "data": {
    "identifier": "patient_001",
    "predictions": [
      {"class_name": "Infiltration", "probability": 0.71, "predicted": true, "rank": 1},
      {"class_name": "Effusion", "probability": 0.44, "predicted": false, "rank": 2}
    ],
    "processing_time_ms": 187.3
  },
  "timestamp": "2026-07-22T09:12:04+00:00",
  "request_id": "3e5f0a5e-2f7e-4a41-9c8e-6b9a2b6f2d31"
}
```

Interactive Swagger docs are auto-generated at `/docs` once the service is running.

---

## What This Is

VisionServeAI is the **production deployment layer** for a multi-label chest X-ray disease classifier trained on NIH ChestX-ray14. The full model lifecycle — dataset exploration, dataset engineering, transfer-learning training, and a 7-stage evaluation/interpretability audit — was built and run as a sequence of Kaggle notebooks (GPU compute, large public dataset, no local infra needed). This repository is the result of taking the **final deployment notebook (Sprint 05)** and re-engineering it, module by module, into a tested, containerized Python service — the same transition every ML system has to make to go from "notebook that works" to "service that can be operated."

```
Kaggle: EDA → Dataset Engineering → Training → Evaluation/Interpretability → Deployment Notebook
                                                                                     │
                                                                     migrated module-by-module
                                                                                     ▼
                                                        VisionServeAI (this repository)
                                                        FastAPI + typed configs + service layer +
                                                        runtime abstraction + explainability engine +
                                                        Docker/Nginx + 177-test suite
```

Only the deployment notebook was ported to this codebase. Training, dataset engineering, and evaluation remain notebook-native and archived on Kaggle — this repo consumes their **frozen output artifacts** (checkpoint, thresholds, class registry, evaluation/deployment reports) and never retrains or re-evaluates the model itself.

---

## ML Lifecycle → Repository Mapping

| Stage | Notebook | What it produced | Lives here as |
|---|---|---|---|
| 1. Dataset exploration | `sprint-02-dataset-exploration.ipynb` | Class distribution, co-occurrence, image/label EDA on NIH ChestX-ray14 | Reference notebook only (Kaggle) |
| 2. Dataset engineering | `sprint03-v2-final-frozen.ipynb` | Disease registry, patient-aware train/val/test manifests, frozen splits | Reference notebook only (Kaggle) |
| 3. Model training | `sprint04-model-engineering.ipynb` | DenseNet-121 + `Linear(14)` multi-label classifier, `best_model.pt` checkpoint | Reference notebook only (Kaggle) |
| 4. Evaluation & interpretability | `sprint04-evaluation-ipynb.ipynb` | 8-stage audit: inference, metrics, calibration, thresholds, 7-method explainability, engineering checks, deployment scorecard | Reference notebook only — **see [Evaluation Findings](#evaluation-findings-honest-numbers) below** |
| 5. **Deployment** | `sprint05-deployment.ipynb` | Artifact loading, runtime construction, prediction/explainability serving, validation | **Migrated into this repository** (`backend/`, `services/`, `inference/`, `configs/`) |

`docs/MIGRATION_NOTES.md` maps every stage of the deployment notebook to its module here and documents each intentional architectural change (e.g. the new Runtime Abstraction Layer, which the notebook never had). Export (ONNX/TorchScript generation), benchmarking, and release packaging remain out of scope for this repository by design — it serves already-exported artifacts, it does not produce them.

---

## Architecture

```
                         ┌───────────────────────────┐
   Client ── HTTP ──────►│  Nginx (reverse proxy)    │  20MB upload cap, request-id passthrough
                         └─────────────┬─────────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │  FastAPI (backend/app.py)  │
                         │  RequestID → Logging →     │
                         │  TrustedHost → CORS → GZip │
                         └─────────────┬─────────────┘
                                       ▼
        ┌───────────────┬─────────────┼─────────────┬───────────────┐
        ▼               ▼             ▼             ▼               ▼
    /health          /predict      /explain      /model,/runtime  /info,/ping
    Health-        Prediction    Explainability  /classes,       /docs-info
    Service        Service       Service         /version,
                                                  /artifacts
        └───────────────┴─────────────┴─────────────┴───────────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │     ServiceRegistry        │  single init at startup,
                         │  (Artifact/Model/Runtime)  │  fail-fast on bad artifacts
                         └─────────────┬─────────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │   Runtime Abstraction      │  RuntimeFactory + RuntimeRegistry
                         │  PyTorch │ TorchScript │   │  — one interface, three backends
                         │        ONNX Runtime        │
                         └─────────────┬─────────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │  Frozen artifacts (mount    │  local bind mount, or
                         │  or HuggingFace Hub)        │  auto-downloaded via HF_REPO_ID
                         └───────────────────────────┘
```

**Design principles carried through the codebase:**
- **Runtime-agnostic serving** — `PredictionService` and `ExplainabilityService` talk to a `BaseRuntime` interface, not to PyTorch directly. Swapping `pytorch → torchscript → onnx` is a config change (`VISIONSERVE_RUNTIME_TYPE`), not a code change.
- **Fail loudly at startup, not silently at request time** — `ServiceRegistry.initialize()` validates the checkpoint, class registry, and runtime before the app accepts traffic; `/health/ready` reflects that state directly.
- **Cloud-native artifact provisioning** — artifacts can be bind-mounted locally *or* pulled from a HuggingFace Hub repo (`HF_REPO_ID`) at container start, resolved through the exact same fingerprint-matching logic either way.
- **One response envelope, everywhere** — every endpoint returns `{success, message, data, metadata, timestamp, request_id}`, so clients parse one shape regardless of route.

---

## Key Features

- **Multi-label chest X-ray classification** across 14 NIH ChestX-ray14 pathologies (DenseNet-121 backbone + `Linear(14)` head, 288×288 input).
- **Three pluggable inference runtimes** — PyTorch, TorchScript, and ONNX Runtime — behind a single `RuntimeFactory`/`RuntimeRegistry`, with a CPU-only Docker build by default.
- **Seven explainability methods** exposed as dedicated endpoints: Grad-CAM, Grad-CAM++, Score-CAM, Eigen-CAM, Guided Backprop, Integrated Gradients, and Occlusion sensitivity — all reused from the same audit pipeline that produced the evaluation report below.
- **Batch prediction** with preserved input ordering (`/predict/batch`), bounded by a configurable max batch size.
- **Typed, immutable configuration** (`configs/schema.py` + `backend/settings.py`) — every `VISIONSERVE_*` / `HF_*` environment variable resolves once, at process start, into a frozen dataclass. No hardcoded paths or magic constants.
- **Structured JSON logging with request correlation** — a `RequestIDMiddleware` assigns/propagates an `X-Request-ID` through every log line and response header for end-to-end traceability.
- **Multi-stage Docker build** — `builder → test → runtime`, where the `test` stage runs the full pytest suite as a build-time CI gate, and the shipped `runtime` image excludes tests, notebooks, and dev tooling entirely.
- **Nginx reverse proxy** in front of the API with upload-size limits, GradCAM-aware proxy timeouts, and `/docs`/`/redoc` blocked from public exposure by default.
- **177 automated tests** across unit, integration, and API layers (`pytest tests/`).

---

## Evaluation Findings (Honest Numbers)

The Sprint 04 evaluation notebook ran a full 8-stage audit of the trained checkpoint on a **25,596-image held-out test split** — inference, metrics, calibration, per-class threshold optimization, 7-method explainability, and an automated engineering/deployment scorecard. The results are reported here as-is, because a serving layer is only as trustworthy as the numbers it's built on top of:

| Metric | Value |
|---|---|
| Test samples | 25,596 images / 14 classes |
| Macro AUROC | **0.464** (micro: 0.584, weighted: 0.482) |
| Macro F1 | 0.048 |
| Macro ECE (calibration error) | 0.282 |
| Explainability coverage | 119 samples × 7 methods, **0 method failures** |
| Engineering checks | **123 / 127 passed** (96.9%), 1 fail, 3 warnings |
| Deployment scorecard | **50 / 100 → `DO_NOT_DEPLOY`** (14 / 14 classes flagged) |

**What actually failed, and why that's the point:** the audit's one hard check failure was a checkpoint parameter-count mismatch (7,051,975 loaded vs. 6,968,206 expected) — flagged, investigated, and traced to dropout layers contributing zero parameters, not a real architecture mismatch. Independently, the model's macro AUROC of 0.464 sits at/below random-guessing (0.5) for most pathologies, and calibration is poor (worst-case ECE of 0.96 on Edema) — a direct consequence of a short 10-epoch training run on a single Kaggle GPU session, not an infrastructure defect.

**Why this ships in the README rather than being hidden:** the value being demonstrated here is that the *pipeline* did its job. A 127-check engineering validation suite, a per-class calibration/threshold report, and an automated scorecard correctly computed `DO_NOT_DEPLOY` and blocked promotion — instead of a model silently shipping on the strength of a green build. That's the difference between a notebook that "ran successfully" and a system that knows when *not* to trust its own output. The recommended path forward (longer training, class-balanced sampling, per-class threshold calibration) is tracked in [Roadmap](#roadmap).

> **Not for clinical or diagnostic use.** This is a research/portfolio system trained on a single public dataset with a single held-out split; see `docs/` for full dataset and calibration limitations.

---

## API Reference

All routes are unauthenticated in this deployment phase (see `backend/settings.py` for CORS/host restriction knobs). Full schemas are in `/docs` (Swagger) and `/redoc`.

| Method | Route | Description |
|---|---|---|
| `POST` | `/predict` | Run inference on a single uploaded image |
| `POST` | `/predict/batch` | Run inference on multiple images, order preserved |
| `POST` | `/explain/gradcam` | Grad-CAM heatmap for a target class |
| `POST` | `/explain/gradcam_plus` | Grad-CAM++ heatmap |
| `POST` | `/explain/scorecam` | Score-CAM heatmap |
| `POST` | `/explain/eigencam` | Eigen-CAM heatmap |
| `POST` | `/explain/guided_backprop` | Guided Backpropagation saliency map |
| `POST` | `/explain/integrated_gradients` | Integrated Gradients attribution |
| `POST` | `/explain/occlusion` | Occlusion-sensitivity map |
| `GET` | `/model` | Reconstructed model architecture + checkpoint facts |
| `GET` | `/runtime` | Active runtime type, load state, execution providers |
| `GET` | `/classes` | The 14-class disease label registry |
| `GET` | `/version` | API/model version metadata |
| `GET` | `/artifacts` | Resolved artifact roots + validation status |
| `GET` | `/health` | Aggregate health: model, runtime, GPU, memory, artifacts |
| `GET` | `/health/live` | Liveness probe (always 200 once serving) |
| `GET` | `/health/ready` | Readiness probe (503 until model/runtime finish loading) |
| `GET` | `/info` | Process/environment diagnostics |
| `GET` | `/ping` | Trivial liveness ping for load balancers |
| `GET` | `/docs-info` | Where interactive docs are served from |

---

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI 0.110+, Uvicorn (ASGI), Pydantic v2 |
| Model / inference | PyTorch 2.2+ (CPU wheel), TorchVision, ONNX 1.16+, ONNX Runtime 1.17+ |
| Explainability | Custom Grad-CAM family, Guided Backprop, Integrated Gradients, Occlusion (`inference/explainability/`) |
| Artifact provisioning | HuggingFace Hub (`huggingface_hub`) with local bind-mount fallback |
| Imaging | Pillow, NumPy |
| Serving / infra | Docker (3-stage build), Nginx reverse proxy, Docker Compose |
| Testing | pytest, pytest-cov, httpx (async test client) |
| Tooling | ruff (lint), mypy (strict typing) |

---

## Repository Structure

```
backend/                FastAPI app: routers, schemas, middleware, settings, lifecycle, exceptions
  routers/               health, prediction, explainability, metadata, system
  schemas/               Pydantic request/response models (one envelope for every endpoint)
  dependencies/          DI wiring into the service layer
  artifacts_provider.py  HuggingFace Hub artifact download/verification

services/                Application service layer consumed by backend/
  service_registry.py     Single startup entry point — wires every service together
  artifact_service.py | model_service.py | runtime_service.py
  prediction_service.py | explainability_service.py | health_service.py

inference/                Model loading, runtimes, pre/postprocessing, explainability — no FastAPI code
  model_loader.py | model_registry.py           Checkpoint reconstruction + fingerprinting
  preprocessing.py | postprocessing.py | thresholding.py | engine.py
  runtimes/                BaseRuntime, PyTorchRuntime, TorchScriptRuntime, ONNXRuntime, RuntimeFactory
  explainability/          GradCAM, GradCAM++, ScoreCAM, EigenCAM, Guided Backprop, Integrated Gradients, Occlusion
  utils/                   Logging, hashing, timers, resource monitoring, environment/seed reporting

configs/                  Typed config schema (schema.py) + defaults (defaults.py)
deployment/               Artifact registry surface; export/packaging intentionally out of scope here
scripts/                  CLI entrypoints (resolve_artifact_roots, engineering_validation, smoke_test, ...)
tests/                    unit/, integration/, api/ — 177 tests
notebooks/                Reference Kaggle notebooks (sprint 02–05) — not executed by this service
docs/                     Migration notes, master architecture reference, sprint engineering reports
docker/nginx/             Reverse proxy config
Dockerfile, docker-compose.yml
```

---

## Getting Started

### Local (Python)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Point at a local artifact directory (checkpoint, thresholds, class registry, ...)
export VISIONSERVE_ARTIFACT_ROOT=/path/to/artifacts

uvicorn backend.app:app --reload
```

### Docker Compose (API + Nginx)

```bash
cp .env.example .env   # set VISIONSERVE_ARTIFACT_ROOT or HF_REPO_ID
docker compose up --build
curl http://localhost/health/ready
```

The `docker-compose.yml` `test` profile runs the full suite as a standalone service:

```bash
docker compose --profile test run --rm test
```

### Pulling artifacts from HuggingFace Hub instead of a local mount

```bash
export HF_REPO_ID=your-username/visionserveai-artifacts
export HF_TOKEN=hf_xxx   # only if the repo is private
```

`Settings.resolve_artifact_roots()` downloads and fingerprint-verifies the snapshot before anything else starts — set unset, this path is a complete no-op and behavior falls back to `VISIONSERVE_ARTIFACT_ROOT`.

---

## Configuration Reference

All variables have safe defaults and are read once at startup (`backend/settings.py`).

| Variable | Default | Purpose |
|---|---|---|
| `VISIONSERVE_ARTIFACT_ROOT` | *(unset)* | Local directory containing checkpoint/thresholds/registry |
| `HF_REPO_ID` / `HF_REVISION` / `HF_TOKEN` | *(unset)* | Pull artifacts from HuggingFace Hub instead of a local mount |
| `VISIONSERVE_RUNTIME_TYPE` | `pytorch` | `pytorch` \| `torchscript` \| `onnx` |
| `VISIONSERVE_WARMUP_ON_STARTUP` | `true` | Run warmup inference before accepting traffic |
| `VISIONSERVE_VALIDATE_RUNTIME_ON_STARTUP` | `true` | Numerically validate the runtime at boot |
| `VISIONSERVE_ENABLE_EXPLAINABILITY` | `true` | Toggle the `/explain/*` routes |
| `VISIONSERVE_FAIL_FAST_ON_STARTUP` | `true` | Crash on bad artifacts instead of degraded-serving |
| `VISIONSERVE_MAX_UPLOAD_SIZE_BYTES` | `15728640` (15MB) | Per-request upload cap |
| `VISIONSERVE_MAX_BATCH_SIZE` | `16` | Cap on `/predict/batch` |
| `VISIONSERVE_CORS_ORIGINS` / `VISIONSERVE_TRUSTED_HOSTS` | `*` | HTTP-layer policy |
| `VISIONSERVE_LOG_DIR` | `logs/` | Structured log output directory |

Full list in `.env.example` and `backend/settings.py`.

---

## Testing

```bash
pytest tests/unit -v          # runtime/service/model-loader logic in isolation
pytest tests/integration -v   # ServiceRegistry → PredictionService pipeline, wired together
pytest tests/api -v           # FastAPI TestClient against every router
pytest tests/ --cov           # full suite with coverage
```

177 tests across 22 files cover the model loader, all three runtimes, preprocessing/postprocessing/thresholding, every service, and every router — including failure paths (bad uploads, oversized batches, not-ready dependencies).

---

## Roadmap

- **Model quality** — longer/multi-session training, class-balanced sampling or focal loss for low-support classes (Hernia: 86 positives, Fibrosis: 435), and temperature scaling to fix calibration (Edema ECE 0.96 today).
- **Threshold-aware serving** — switch `/predict` from the fixed 0.5 cutoff to the per-class optimal thresholds already computed by the evaluation notebook.
- **Migrate remaining notebook stages**: ONNX/TorchScript export (currently notebook-only), benchmarking, and release packaging into this repository's `deployment/` module.
- **External-cohort validation** beyond the single NIH ChestX-ray14 split currently evaluated.

---

## License

MIT — see [LICENSE](LICENSE).
