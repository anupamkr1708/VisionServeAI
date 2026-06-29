# MASTER ARCHITECTURE DOCUMENT
## MedVision AI Platform — Production Medical Computer Vision System

**Version:** 1.0.0  
**Status:** Architecture Specification  
**Classification:** Internal Engineering Blueprint  
**Audience:** AI Engineers, MLOps Engineers, Backend Engineers, DevOps Engineers  
**Last Updated:** 2025

---

## TABLE OF CONTENTS

1. [Project Vision](#1-project-vision)
2. [Motivation](#2-motivation)
3. [Industry Problem](#3-industry-problem)
4. [Why Medical AI](#4-why-medical-ai)
5. [System Goals](#5-system-goals)
6. [Functional Requirements](#6-functional-requirements)
7. [Non-Functional Requirements](#7-non-functional-requirements)
8. [Success Metrics](#8-success-metrics)
9. [Technology Selection Matrix](#9-technology-selection-matrix)
10. [Alternative Technology Comparison](#10-alternative-technology-comparison)
11. [Dataset Analysis](#11-dataset-analysis)
12. [Folder Structure](#12-folder-structure)
13. [Repository Organization](#13-repository-organization)
14. [Module Responsibilities](#14-module-responsibilities)
15. [Microservice Architecture](#15-microservice-architecture)
16. [Backend Architecture](#16-backend-architecture)
17. [Frontend Architecture](#17-frontend-architecture)
18. [Training Architecture](#18-training-architecture)
19. [Inference Architecture](#19-inference-architecture)
20. [Serving Architecture](#20-serving-architecture)
21. [Deployment Architecture](#21-deployment-architecture)
22. [Docker Architecture](#22-docker-architecture)
23. [Networking](#23-networking)
24. [Storage Design](#24-storage-design)
25. [Database Schema](#25-database-schema)
26. [API Design](#26-api-design)
27. [Authentication (Future)](#27-authentication-future)
28. [Experiment Tracking Design](#28-experiment-tracking-design)
29. [Model Registry Design](#29-model-registry-design)
30. [Checkpoint Strategy](#30-checkpoint-strategy)
31. [Versioning Strategy](#31-versioning-strategy)
32. [Explainability Pipeline](#32-explainability-pipeline)
33. [Benchmark Pipeline](#33-benchmark-pipeline)
34. [Monitoring Architecture](#34-monitoring-architecture)
35. [Logging Architecture](#35-logging-architecture)
36. [Configuration Management](#36-configuration-management)
37. [Environment Variables](#37-environment-variables)
38. [Security Considerations](#38-security-considerations)
39. [Error Handling](#39-error-handling)
40. [Testing Strategy](#40-testing-strategy)
41. [CI/CD Roadmap](#41-cicd-roadmap)
42. [Documentation Strategy](#42-documentation-strategy)
43. [Coding Standards](#43-coding-standards)
44. [Dependency Management](#44-dependency-management)
45. [Scalability Strategy](#45-scalability-strategy)
46. [Cloud Migration Plan](#46-cloud-migration-plan)
47. [Future Research Hooks](#47-future-research-hooks)
48. [Future Vision-Language Expansion](#48-future-vision-language-expansion)
49. [Risk Analysis](#49-risk-analysis)
50. [Development Roadmap](#50-development-roadmap)

---

## 1. PROJECT VISION

### 1.1 One-Line Vision

> **MedVision AI** is a modular, production-grade AI platform designed for medical image analysis — starting with chest X-ray multi-label pathology classification and built to evolve into a full multi-modal clinical AI system.

### 1.2 Platform Vision

MedVision AI is not an experiment, a notebook, or a research demo. It is a **production AI platform** architected from first principles — designed in the same way that AI engineering teams at Google Health, Microsoft Azure AI, NVIDIA Clara, and Aidoc build their clinical AI systems.

The platform demonstrates command over the full AI lifecycle:

```
Dataset → EDA → Training → Evaluation → Experiment Tracking
→ Model Registry → Export → Optimization → Serving
→ Backend → Frontend → Docker → CI/CD → Cloud → Monitoring
```

Every design decision is made with **scalability**, **replaceability**, and **production realism** in mind.

### 1.3 Platform Pillars

| Pillar | Description |
|--------|-------------|
| **Modularity** | Every component can be swapped without touching others |
| **Separation of Concerns** | Training, serving, and inference are independent systems |
| **Production Realism** | No shortcuts that would not survive a production code review |
| **Extensibility** | New modalities, new tasks, new models slot in without rearchitecting |
| **Observability** | Everything is logged, traced, and monitored from day one |
| **Reproducibility** | All experiments, configs, and artifacts are version-controlled |

### 1.4 Platform Evolution

```
Version 1.0 ─── Chest X-ray Multi-label Classification
Version 1.1 ─── CT Scan Support
Version 1.2 ─── MRI + Retinal Imaging
Version 2.0 ─── Segmentation + Detection Heads
Version 2.1 ─── Histopathology + Satellite Vision
Version 3.0 ─── Vision-Language Models + Report Generation
Version 4.0 ─── Federated Learning + Continual Learning
```

### 1.5 Artifact Lifecycle

````
Training Run
     ↓
PyTorch Checkpoint (.pt)  ←── MLflow logs artifact
     ↓
ONNX Export + Validation  ←── diff < 1e-5 vs PyTorch
     ↓
MLflow Model Registry     ←── Stage: None → Staging
     ↓
Hugging Face Hub Upload   ←── versioned ONNX + metadata
     ↓
Triton / ONNX Runtime     ←── active serving
     ↓
Prometheus + Grafana      ←── latency, error rate, drift
````
 
### 1.6 Experiment Lifecycle
 
````
Idea
  ↓ create configs/experiments/{name}.yaml
Config
  ↓ python training/train.py +experiment={name}
Training
  ↓ MLflow logs params, metrics/epoch, artifacts
Metrics
  ↓ review MLflow UI: AUC curves, per-class breakdown
Comparison
  ↓ select best run by mean_auc on validation set
Promotion Check
  ↓ run mlops/registry/promote.py (see §29.2 for criteria)
Deployment
  ↓ upload ONNX → HF Hub → update serving → verify /health
````
````

---

## 2. MOTIVATION

### 2.1 Engineering Motivation

The AI engineering field is bifurcating into two distinct categories of practitioners:

1. **Research Engineers** — focused on novel algorithms, papers, and benchmarks
2. **Production AI Engineers** — focused on deploying, scaling, optimizing, and maintaining AI systems

The industry has a severe shortage of engineers who can bridge both. The engineers most valued by top AI companies are those who can take a model from a research notebook and ship it as a production system with proper serving infrastructure, monitoring, observability, and MLOps pipelines.

This platform was built to develop and demonstrate **all the skills expected of a Mid-to-Senior Production AI Engineer**.

### 2.2 Skill Gap Analysis

| Skill Area | Prior Experience | Platform Adds |
|------------|-----------------|---------------|
| LLM Applications | ✅ Strong | — |
| RAG / Graph RAG | ✅ Strong | — |
| FastAPI / Backend | ✅ Strong | Extended |
| Docker | ✅ Moderate | Extended |
| Computer Vision | ❌ Missing | ✅ Core focus |
| Transfer Learning | ❌ Missing | ✅ Core focus |
| ONNX / TensorRT | ❌ Missing | ✅ Core focus |
| Triton Inference Server | ❌ Missing | ✅ Core focus |
| MLflow / Experiment Tracking | ❌ Missing | ✅ Core focus |
| Model Registry | ❌ Missing | ✅ Core focus |
| Prometheus / Grafana | ❌ Missing | ✅ Core focus |
| CI/CD for ML | ❌ Missing | ✅ Core focus |
| Model Explainability | ❌ Missing | ✅ Core focus |
| Cloud AI Deployment | ❌ Missing | ✅ Core focus |

### 2.3 Portfolio Motivation

A GitHub repository structured like this platform signals to hiring teams and technical reviewers:

- System design literacy
- Production engineering mindset
- MLOps capability
- Computer vision expertise
- Deployment and infrastructure understanding

This is dramatically different from standard portfolio projects that consist only of training notebooks.

---

## 3. INDUSTRY PROBLEM

### 3.1 The Radiology Shortage Crisis

Medical imaging is one of the highest-volume, highest-stakes tasks in clinical medicine. Globally:

- There are approximately **300,000 radiologists** worldwide
- Radiology demand is growing at **~6% annually**
- Many low-income countries have fewer than 1 radiologist per million people
- In high-income countries, radiologist burnout and shortage are documented crises
- Radiologists read an average of **100–200 studies per day**, leading to fatigue errors

### 3.2 The Error Rate Problem

Studies indicate that:

- Radiologist miss rates for chest X-ray findings range from **4% to 30%** depending on the condition
- Lung nodule miss rates as high as 26% have been documented
- Pneumothorax is missed in an estimated 20% of cases on initial reads
- Fatigue significantly increases error rates during long shifts

### 3.3 AI as an Augmentation Layer

The clinical value of medical AI is not **replacing** radiologists but **augmenting** them:

- AI provides a second opinion
- AI flags high-priority cases for immediate review
- AI assists in understaffed environments
- AI enables preliminary screening in resource-limited settings
- AI tracks subtle longitudinal changes over time

### 3.4 Current Market Landscape

| Company | Focus | Status |
|---------|-------|--------|
| Aidoc | Radiology AI triage | FDA cleared, deployed in 1000+ hospitals |
| Viz.ai | Stroke AI triage | FDA cleared |
| Google Health | Pathology, Retinal | Research + partnerships |
| Microsoft Azure AI Health | Multi-modal clinical AI | Platform play |
| NVIDIA Clara | Medical imaging AI platform | Developer platform |
| Paige.ai | Pathology AI | FDA cleared |
| Zebra Medical | Multi-organ radiology AI | Acquired by Nanox |

The market is maturing. Production AI engineers are in high demand.

---

## 4. WHY MEDICAL AI

### 4.1 Technical Richness

Medical AI is one of the most technically challenging domains in computer vision:

| Challenge | Technical Complexity |
|-----------|---------------------|
| Multi-label classification | Multiple co-occurring pathologies per image |
| Class imbalance | Some diseases appear in <1% of images |
| Fine-grained features | Subtle pixel-level differences matter clinically |
| Domain shift | Different scanners produce different image characteristics |
| High-stakes predictions | Errors have clinical consequences |
| Explainability required | Clinicians must understand model reasoning |
| Regulatory constraints | FDA, CE marking, ISO 13485 compliance |

### 4.2 Dataset Availability

Unlike many specialized domains, high-quality, large-scale public medical imaging datasets exist:

- NIH ChestX-ray14 (112,000+ images)
- CheXpert (Stanford — 224,000+ images)
- MIMIC-CXR (377,000+ images)
- PadChest (160,000+ images)
- RSNA Pneumonia Detection dataset
- VinBigData Chest X-ray dataset

This means real, production-scale training is possible without proprietary data.

### 4.3 Domain Transfer

Skills developed on medical CV directly transfer to:

- Industrial defect detection (same fine-grained classification problems)
- Satellite imagery (same multi-label, multi-scale problems)
- Autonomous inspection systems
- Quality control in manufacturing

Medical AI is the hardest version of these problems.

### 4.4 Career Value

Medical AI and healthcare AI are among the fastest-growing segments in applied AI engineering. The combination of CV + MLOps + Medical domain knowledge is exceptionally rare and highly compensated.

---

## 5. SYSTEM GOALS

### 5.1 Primary Goals (Version 1.0)

| Goal | Description |
|------|-------------|
| **G1** | Implement production-grade multi-label chest X-ray classification |
| **G2** | Demonstrate complete transfer learning lifecycle from pretrained backbone to fine-tuned model |
| **G3** | Export model to ONNX and optimize for inference |
| **G4** | Serve model via NVIDIA Triton Inference Server |
| **G5** | Build FastAPI gateway with proper request routing and validation |
| **G6** | Build React + TypeScript frontend for clinical interface |
| **G7** | Implement Grad-CAM and SHAP explainability pipeline |
| **G8** | Implement Prometheus + Grafana monitoring |
| **G9** | Containerize every component with Docker |
| **G10** | Track all experiments with MLflow |
| **G11** | Maintain model registry with versioning |
| **G12** | Implement structured benchmarking pipeline |
| **G13** | Set up GitHub Actions CI/CD pipeline |

### 5.2 Architectural Goals

| Goal | Description |
|------|-------------|
| **A1** | Training and inference are completely decoupled systems |
| **A2** | Model can be swapped without modifying backend or frontend |
| **A3** | Platform supports future modalities without re-architecting |
| **A4** | All configuration is externalized via files and environment variables |
| **A5** | No hardcoded paths, credentials, or magic constants |
| **A6** | Every service is independently deployable |

### 5.3 Non-Goals (Version 1.0)

- Novel research contributions
- New loss functions or architectures
- FDA regulatory submission
- HIPAA-compliant production deployment
- Real patient data handling
- Multi-site federated learning
- Uncertainty quantification (beyond threshold confidence)

---

## 6. FUNCTIONAL REQUIREMENTS

### 6.1 Data Pipeline Requirements

| ID | Requirement |
|----|-------------|
| FR-D1 | System shall ingest and validate the NIH ChestX-ray14 dataset |
| FR-D2 | System shall perform and document EDA including class distribution, image statistics, and label correlations |
| FR-D3 | System shall support patient-wise train/val/test splits to prevent data leakage |
| FR-D4 | System shall apply configurable augmentation pipelines via Albumentations |
| FR-D5 | System shall support class-weighted sampling to handle label imbalance |
| FR-D6 | System shall produce reproducible dataset splits via fixed random seeds |
| FR-D7 | Dataset loading shall be modular — new datasets shall require only a new DataModule class |

### 6.2 Training Requirements

| ID | Requirement |
|----|-------------|
| FR-T1 | System shall support configurable backbone selection via timm model registry |
| FR-T2 | System shall support mixed-precision training (FP16) |
| FR-T3 | System shall log all hyperparameters, metrics, and artifacts to MLflow |
| FR-T4 | System shall support early stopping, LR scheduling, and gradient clipping |
| FR-T5 | System shall save checkpoints at configurable intervals |
| FR-T6 | System shall support resuming training from checkpoint |
| FR-T7 | System shall evaluate per-class AUC-ROC, mean AUC, F1, precision, recall |
| FR-T8 | Training shall run on Kaggle/Colab GPU without platform-specific code coupling |

### 6.3 Model Export Requirements

| ID | Requirement |
|----|-------------|
| FR-E1 | System shall export PyTorch models to ONNX format |
| FR-E2 | System shall validate ONNX export correctness against PyTorch output |
| FR-E3 | System shall produce TensorRT-optimized engines from ONNX models (GPU environments) |
| FR-E4 | System shall version and store exported models with associated metadata |
| FR-E5 | System shall document model input/output specifications |

### 6.4 Serving Requirements

| ID | Requirement |
|----|-------------|
| FR-S1 | System shall serve inference requests via NVIDIA Triton Inference Server |
| FR-S2 | Triton shall support dynamic batching |
| FR-S3 | Backend shall never load model weights directly |
| FR-S4 | Inference shall be accessible via Triton HTTP and gRPC endpoints |
| FR-S5 | System shall support ONNX Runtime as CPU fallback when GPU is unavailable |
| FR-S6 | Preprocessing pipeline shall be defined and version-controlled |

### 6.5 Backend Requirements

| ID | Requirement |
|----|-------------|
| FR-B1 | FastAPI gateway shall handle image upload, validation, and routing |
| FR-B2 | API shall return prediction probabilities and binary predictions per class |
| FR-B3 | API shall return Grad-CAM heatmap overlays |
| FR-B4 | API shall return structured JSON responses with prediction metadata |
| FR-B5 | API shall validate image format, size, and content-type |
| FR-B6 | Results shall be persisted to PostgreSQL |
| FR-B7 | API shall expose health check and readiness endpoints |
| FR-B8 | API shall expose Prometheus-compatible metrics endpoint |
| FR-B9 | Redis shall be used for caching repeated predictions on identical images |

### 6.6 Frontend Requirements

| ID | Requirement |
|----|-------------|
| FR-F1 | React + TypeScript frontend shall provide image upload interface |
| FR-F2 | Frontend shall display prediction results per disease class |
| FR-F3 | Frontend shall display Grad-CAM heatmap overlaid on original image |
| FR-F4 | Frontend shall display confidence scores with visual indicators |
| FR-F5 | Frontend shall be deployable to Vercel |
| FR-F6 | Frontend shall handle loading states and error states gracefully |
| FR-F7 | Frontend shall display model metadata including version and confidence thresholds |

### 6.7 Monitoring Requirements

| ID | Requirement |
|----|-------------|
| FR-M1 | Prometheus shall collect inference latency metrics |
| FR-M2 | Prometheus shall collect throughput and request rate metrics |
| FR-M3 | Grafana shall provide operational dashboards |
| FR-M4 | System shall alert on error rate spikes |
| FR-M5 | System shall log prediction distribution for data drift detection |

### 6.8 MLOps Requirements

| ID | Requirement |
|----|-------------|
| FR-O1 | MLflow shall track all training runs with full parameter and metric logging |
| FR-O2 | MLflow Model Registry shall manage model lifecycle stages |
| FR-O3 | All model artifacts shall be stored on Hugging Face Hub |
| FR-O4 | GitHub Actions shall run linting, testing, and build validation on every PR |
| FR-O5 | Configuration shall be managed via YAML files, not hardcoded values |

---

## 7. NON-FUNCTIONAL REQUIREMENTS

### 7.1 Performance

| ID | Requirement |
|----|-------------|
| NFR-P1 | Single image inference latency shall be < 500ms at p99 (ONNX Runtime CPU) |
| NFR-P2 | Single image inference latency shall be < 100ms at p99 (TensorRT GPU) |
| NFR-P3 | API gateway overhead shall be < 50ms excluding model inference |
| NFR-P4 | System shall support at least 10 concurrent requests without degradation |
| NFR-P5 | Grad-CAM generation shall add < 200ms overhead |

### 7.2 Availability

| ID | Requirement |
|----|-------------|
| NFR-A1 | API gateway shall expose health and readiness endpoints |
| NFR-A2 | Service shall restart automatically on failure in Docker Compose |
| NFR-A3 | System shall degrade gracefully when Triton is unavailable (fallback to ONNX Runtime) |

### 7.3 Scalability

| ID | Requirement |
|----|-------------|
| NFR-S1 | Architecture shall support horizontal scaling of the inference service |
| NFR-S2 | Architecture shall support adding new models without gateway changes |
| NFR-S3 | Database schema shall support multi-model, multi-task prediction storage |

### 7.4 Maintainability

| ID | Requirement |
|----|-------------|
| NFR-M1 | Every module shall have a single, clearly defined responsibility |
| NFR-M2 | Configuration shall be managed centrally, not scattered across code |
| NFR-M3 | All public interfaces shall be documented |
| NFR-M4 | Test coverage shall exceed 80% on business logic |

### 7.5 Security

| ID | Requirement |
|----|-------------|
| NFR-SEC1 | No credentials shall be committed to the repository |
| NFR-SEC2 | All secrets shall be managed via environment variables |
| NFR-SEC3 | Uploaded images shall be validated for file type and content |
| NFR-SEC4 | API rate limiting shall be implemented |

### 7.6 Reproducibility

| ID | Requirement |
|----|-------------|
| NFR-R1 | All experiments shall be reproducible via fixed seeds |
| NFR-R2 | All configurations shall be logged alongside experiment runs |
| NFR-R3 | Dataset splits shall be version-controlled or deterministically reproducible |

### 7.7 Portability

| ID | Requirement |
|----|-------------|
| NFR-PORT1 | All services shall run via Docker Compose on any machine |
| NFR-PORT2 | Training scripts shall run on Kaggle, Colab, and local CPU |
| NFR-PORT3 | No Windows-specific paths or OS-specific dependencies |

---

## 8. SUCCESS METRICS

### 8.1 Model Performance Metrics

| Metric | Target (Version 1.0) | Measurement Method |
|--------|---------------------|-------------------|
| Mean AUC-ROC (14 classes) | ≥ 0.80 | Test set evaluation |
| Per-class AUC-ROC | ≥ 0.75 for all classes | Per-class evaluation |
| Cardiomegaly AUC | ≥ 0.88 | NIH paper baseline reference |
| Pneumonia AUC | ≥ 0.77 | NIH paper baseline reference |
| Atelectasis AUC | ≥ 0.79 | NIH paper baseline reference |
| F1-Score (macro) | ≥ 0.40 | Multi-label F1 |

### 8.2 System Performance Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| API p50 latency | < 200ms | Prometheus histogram |
| API p99 latency | < 500ms | Prometheus histogram |
| Triton GPU p99 latency | < 100ms | Triton metrics |
| Throughput | ≥ 10 req/sec | Load test with locust |
| System uptime | ≥ 99% | Docker healthcheck |

### 8.3 Engineering Quality Metrics

| Metric | Target |
|--------|--------|
| Test coverage (business logic) | ≥ 80% |
| CI pipeline pass rate | 100% on main branch |
| Docker image build time | < 5 minutes |
| Code linting pass rate | 100% |
| Documentation coverage | All public APIs documented |

### 8.4 MLOps Metrics

| Metric | Target |
|--------|--------|
| Experiment reproducibility | 100% (same config = same results) |
| Model artifact traceability | Every deployed model traceable to training run |
| Config coverage | Zero hardcoded hyperparameters |

---

## 9. TECHNOLOGY SELECTION MATRIX

### 9.1 Core Technology Decisions

| Category | Selected Technology | Primary Reason |
|----------|--------------------|--------------:|
| **Deep Learning Framework** | PyTorch | Industry standard for research and production, dynamic graphs, excellent ecosystem |
| **Model Library** | timm (PyTorch Image Models) | 700+ pretrained models, standardized API, state-of-the-art backbones |
| **Augmentation** | Albumentations | 3-5x faster than torchvision transforms, medical imaging support, ONNX-compatible |
| **Image Processing** | OpenCV | Industry standard, hardware-accelerated, DICOM future support |
| **Model Export** | ONNX | Universal format, framework-agnostic, Triton-native |
| **CPU Inference** | ONNX Runtime | Fastest CPU inference, cross-platform, quantization support |
| **GPU Optimization** | TensorRT | Maximum NVIDIA GPU inference performance, FP16/INT8 support |
| **Model Serving** | NVIDIA Triton Inference Server | Production model serving, multi-model, dynamic batching, metrics |
| **API Gateway** | FastAPI | High performance, native async, automatic OpenAPI docs, Pydantic validation |
| **Frontend Framework** | React + TypeScript | Industry standard, strong typing, component-based |
| **Database** | PostgreSQL | ACID compliance, JSONB for flexible prediction storage |
| **Cache** | Redis | Sub-millisecond cache, image hash-based prediction caching |
| **Experiment Tracking** | MLflow | Open source, self-hosted, framework-agnostic |
| **Model Storage** | Hugging Face Hub | Free, versioned, widely used in AI community |
| **Containerization** | Docker + Docker Compose | Universal, reproducible, industry standard |
| **Monitoring** | Prometheus + Grafana | Industry standard, pull-based metrics, excellent dashboards |
| **Logging** | Python structlog | Structured JSON logs, context injection |
| **CI/CD** | GitHub Actions | Native GitHub integration, free for public repos |
| **Frontend Deploy** | Vercel | Zero-config deployment, free tier |
| **Configuration** | Hydra + YAML | Hierarchical config, override support, MLflow integration |

### 9.2 Data Science Stack

| Library | Version Target | Purpose |
|---------|----------------|---------|
| torch | ≥ 2.2.0 | Core framework |
| torchvision | ≥ 0.17.0 | Vision utilities |
| timm | ≥ 0.9.0 | Pretrained models |
| albumentations | ≥ 1.3.0 | Augmentation |
| opencv-python | ≥ 4.9.0 | Image processing |
| numpy | ≥ 1.26.0 | Numerical computing |
| pandas | ≥ 2.1.0 | Data manipulation |
| scikit-learn | ≥ 1.4.0 | Metrics, splits |
| matplotlib | ≥ 3.8.0 | Visualization |
| seaborn | ≥ 0.13.0 | Statistical plots |
| grad-cam | ≥ 1.5.0 | Explainability |
| shap | ≥ 0.44.0 | SHAP explanations |
| onnx | ≥ 1.16.0 | Model export |
| onnxruntime | ≥ 1.17.0 | CPU inference |
| mlflow | ≥ 2.11.0 | Experiment tracking |

### 9.3 Backend Stack

| Library | Version Target | Purpose |
|---------|----------------|---------|
| fastapi | ≥ 0.110.0 | API gateway |
| uvicorn | ≥ 0.29.0 | ASGI server |
| pydantic | ≥ 2.6.0 | Validation |
| sqlalchemy | ≥ 2.0.0 | ORM |
| alembic | ≥ 1.13.0 | DB migrations |
| asyncpg | ≥ 0.29.0 | Async PostgreSQL |
| redis | ≥ 5.0.0 | Redis client |
| tritonclient | ≥ 2.44.0 | Triton HTTP/gRPC |
| prometheus-client | ≥ 0.20.0 | Metrics |
| structlog | ≥ 24.1.0 | Structured logging |
| pillow | ≥ 10.2.0 | Image handling |
| httpx | ≥ 0.27.0 | Async HTTP client |
| python-multipart | ≥ 0.0.9 | File upload |

---

## 10. ALTERNATIVE TECHNOLOGY COMPARISON

### 10.1 Deep Learning Framework

| Framework | Pros | Cons | Decision |
|-----------|------|------|----------|
| **PyTorch** | Dynamic graphs, research-dominant, vast ecosystem, excellent debugging | Deployment requires more work than TF Serving | ✅ **Selected** |
| TensorFlow | TF Serving mature, TFX pipeline, Keras simplicity | Declining research mindshare, static graph complexity | ❌ |
| JAX | Fastest research iteration, functional paradigm | Immature production ecosystem, steep learning curve | ❌ |
| MXNet | Apache ecosystem | Declining community support | ❌ |

### 10.2 Model Serving

| Technology | Pros | Cons | Decision |
|-----------|------|------|----------|
| **Triton Inference Server** | Multi-framework, dynamic batching, GPU/CPU, ensemble, gRPC+HTTP, Prometheus native | Requires ONNX/TRT model format, complex config | ✅ **Selected** |
| TorchServe | Native PyTorch, simple deployment | Less production-mature, fewer features | ❌ Secondary |
| BentoML | Developer-friendly, flexible | Less performant at scale | ❌ |
| TF Serving | Mature, fast | TF-only | ❌ |
| Ray Serve | Flexible, Python-native | Heavy dependency | ❌ |
| Seldon Core | Kubernetes-native | Overkill for v1 | ❌ |

### 10.3 API Gateway

| Technology | Pros | Cons | Decision |
|-----------|------|------|----------|
| **FastAPI** | Async, auto OpenAPI, Pydantic v2, fast | Newer ecosystem than Flask | ✅ **Selected** |
| Flask | Mature, simple | Sync by default, no auto docs | ❌ |
| Django REST | Full-featured | Heavyweight | ❌ |
| gRPC directly | Maximum performance | Not browser-friendly | ❌ |

### 10.4 Experiment Tracking

| Technology | Pros | Cons | Decision |
|-----------|------|------|----------|
| **MLflow** | Open source, self-hosted, multi-framework, model registry | UI less polished than W&B | ✅ **Selected** |
| Weights & Biases | Best UI/UX, team collaboration | Paid for teams, requires account | ❌ Alternative |
| Neptune.ai | Good UI | Paid | ❌ |
| Comet ML | Good features | Paid for teams | ❌ |
| TensorBoard | Simple | Limited beyond TF/PT | ❌ |

### 10.5 Database

| Technology | Pros | Cons | Decision |
|-----------|------|------|----------|
| **PostgreSQL** | ACID, JSONB, mature, rich feature set | More complex than SQLite | ✅ **Selected** |
| Supabase | PostgreSQL + auth + API layer | Dependency on external service | ❌ Alternative for managed |
| MongoDB | Flexible schema | No ACID transactions | ❌ |
| SQLite | Zero setup | Not production-grade | ❌ Dev only |

### 10.6 Augmentation

| Technology | Pros | Cons | Decision |
|-----------|------|------|----------|
| **Albumentations** | Fastest, richest medical transforms, ONNX-compatible serialization | Slightly different API style | ✅ **Selected** |
| torchvision transforms | Native PyTorch | Slower, fewer transforms | ❌ Secondary |
| imgaug | Rich but slower | Less maintained | ❌ |
| Kornia | GPU augmentations | Extra complexity | ❌ Future |

### 10.7 Monitoring

| Technology | Pros | Cons | Decision |
|-----------|------|------|----------|
| **Prometheus + Grafana** | Industry standard, pull-based, free, rich dashboards | Setup complexity | ✅ **Selected** |
| Datadog | Excellent UX | Expensive | ❌ |
| New Relic | Full-stack | Paid | ❌ |
| CloudWatch | AWS-native | Vendor lock-in | ❌ |

---

## 11. DATASET ANALYSIS

### 11.1 NIH ChestX-ray14 Overview

| Property | Value |
|----------|-------|
| Total images | 112,120 |
| Unique patients | 30,805 |
| Image format | PNG |
| Resolution | 1024 × 1024 pixels |
| Color | Grayscale |
| Label type | Multi-label |
| Number of disease classes | 14 |
| Normal images | ~53,000 (~47%) |
| Bounding box annotations | 984 images |
| Source | NIH Clinical Center |
| License | Open (CC0-like for research) |

### 11.2 Disease Class Distribution

```
Atelectasis        ████████████████░░░░  11,559 (~10.3%)
Cardiomegaly       ████░░░░░░░░░░░░░░░░   2,776 (~2.5%)
Effusion           ██████████████░░░░░░  13,317 (~11.9%)
Infiltration       ██████████████████░░  19,894 (~17.7%)
Mass               █████░░░░░░░░░░░░░░░   5,782 (~5.2%)
Nodule             ████░░░░░░░░░░░░░░░░   6,331 (~5.6%)
Pneumonia          ██░░░░░░░░░░░░░░░░░░   1,431 (~1.3%)
Pneumothorax       ████░░░░░░░░░░░░░░░░   5,302 (~4.7%)
Consolidation      ████░░░░░░░░░░░░░░░░   4,667 (~4.2%)
Edema              ████░░░░░░░░░░░░░░░░   2,303 (~2.1%)
Emphysema          ████░░░░░░░░░░░░░░░░   2,516 (~2.2%)
Fibrosis           ██░░░░░░░░░░░░░░░░░░   1,686 (~1.5%)
Pleural Thickening ████░░░░░░░░░░░░░░░░   3,385 (~3.0%)
Hernia             █░░░░░░░░░░░░░░░░░░░     227 (~0.2%)
No Finding         ████████████████████  60,361 (~53.8%)
```

*Note: Images can have multiple labels simultaneously (multi-label)*

### 11.3 Class Imbalance Implications

The dataset exhibits **severe class imbalance** with implications for architecture decisions:

| Imbalance Strategy | Application |
|-------------------|-------------|
| Class-weighted BCE loss | Weight rare classes higher in loss |
| Weighted random sampling | Oversample underrepresented classes |
| Per-class threshold tuning | Different decision thresholds per class |
| Macro AUC-ROC evaluation | Fair evaluation across class sizes |
| Stratified splits | Maintain distribution across splits |

### 11.4 Data Leakage Prevention

A critical issue in the original NIH paper is **patient-level data leakage** — the same patient appearing in both train and test sets. This platform enforces **patient-wise splitting**:

```
Unique Patients: 30,805
  Train split: ~21,000 patients (~70%)
  Val split:   ~4,500 patients (~15%)
  Test split:  ~5,200 patients (~15%)
```

All images from a given patient are assigned to exactly one split.

### 11.5 Preprocessing Specification

| Step | Specification | Rationale |
|------|--------------|-----------|
| Resize | 224×224 or 320×320 | Model input requirement |
| Convert to RGB | 3-channel copy of grayscale | Pretrained model compatibility |
| Normalize | ImageNet mean/std | Transfer learning alignment |
| CLAHE enhancement | Optional | Chest X-ray contrast enhancement |
| Histogram equalization | Optional | Alternative contrast normalization |

### 11.6 Augmentation Policy

| Augmentation | Probability | Parameters | Purpose |
|--------------|-------------|------------|---------|
| HorizontalFlip | 0.5 | — | Symmetry |
| RandomBrightnessContrast | 0.3 | b=0.2, c=0.2 | Exposure variation |
| ShiftScaleRotate | 0.3 | shift=0.05, scale=0.05, rotate=10° | Positioning variation |
| GaussianNoise | 0.2 | var_limit=(10,50) | Noise robustness |
| GaussianBlur | 0.1 | blur_limit=3 | Scanner variation |
| GridDistortion | 0.1 | — | Geometric variation |
| ElasticTransform | 0.1 | alpha=1, sigma=50 | Tissue deformation |
| CLAHE | 0.2 | clip_limit=2.0 | Contrast normalization |

*Augmentations applied only during training. Validation/test use resize + normalize only.*

### 11.7 EDA Deliverables

The EDA notebook/script shall produce and log:

1. Class distribution bar chart
2. Co-occurrence heatmap (which diseases appear together)
3. Per-class image samples (3-5 examples each)
4. Image dimension statistics
5. Pixel intensity distribution
6. Patient age/gender distribution vs. disease
7. Bounding box annotation visualization
8. Sample augmented images

### 11.8 Future Dataset Integration

The data loading system is designed to support additional datasets via a common interface:

```
DataModule (abstract base)
  ├── NIHChestXrayDataModule (v1.0)
  ├── CheXpertDataModule (future)
  ├── MIMICCXRDataModule (future)
  ├── CTScanDataModule (future)
  └── RetinalDataModule (future)
```

---

## 12. FOLDER STRUCTURE

```
medvision-ai/
│
├── README.md                          # Project overview and quickstart
├── MASTER_ARCHITECTURE.md             # This document
├── CONTRIBUTING.md                    # Contribution guidelines
├── LICENSE                            # License
├── .gitignore                         # Git ignore rules
├── .env.example                       # Environment variable template
├── docker-compose.yml                 # Full stack orchestration
├── docker-compose.dev.yml             # Development overrides
├── docker-compose.gpu.yml             # GPU-enabled overrides
├── Makefile                           # Developer convenience commands
│
├── docs/                              # Documentation
│   ├── architecture/                  # Architecture diagrams and specs
│   │   ├── system-overview.md
│   │   ├── api-specification.md
│   │   └── database-schema.md
│   ├── deployment/                    # Deployment guides
│   │   ├── local-setup.md
│   │   ├── kaggle-training.md
│   │   └── cloud-deployment.md
│   ├── research/                      # Research notes and findings
│   └── api/                           # Auto-generated API docs
│
├── scripts/                           # Utility and automation scripts
│   ├── setup/
│   │   ├── install_dependencies.sh
│   │   ├── download_dataset.sh
│   │   └── setup_mlflow.sh
│   ├── export/
│   │   ├── export_to_onnx.py
│   │   ├── validate_onnx.py
│   │   └── export_to_tensorrt.py
│   ├── benchmark/
│   │   ├── benchmark_pytorch.py
│   │   ├── benchmark_onnx.py
│   │   └── benchmark_triton.py
│   └── deploy/
│       ├── push_to_hf.py
│       └── update_triton_config.py
│
├── configs/                           # All configuration files
│   ├── base/
│   │   ├── model.yaml                 # Model architecture config
│   │   ├── training.yaml              # Training hyperparameters
│   │   ├── data.yaml                  # Dataset config
│   │   ├── augmentation.yaml          # Augmentation config
│   │   └── inference.yaml             # Inference config
│   ├── experiments/                   # Named experiment configs
│   │   ├── densenet121_baseline.yaml
│   │   ├── efficientnetv2_baseline.yaml
│   │   └── convnext_small.yaml
│   ├── serving/                       # Triton server config
│   │   └── triton/
│   │       ├── models/
│   │       │   └── chestxray_classifier/
│   │       │       ├── 1/
│   │       │       │   └── model.onnx
│   │       │       └── config.pbtxt
│   │       └── ensemble/
│   └── monitoring/                    # Prometheus + Grafana config
│       ├── prometheus.yml
│       └── grafana/
│           ├── dashboards/
│           └── datasources/
│
├── data/                              # Data directory (gitignored)
│   ├── raw/                           # Original downloaded data
│   │   └── NIH_ChestXray14/
│   │       ├── images/
│   │       └── labels/
│   ├── processed/                     # Processed/cached data
│   ├── splits/                        # Train/val/test split files
│   └── external/                      # External reference data
│
├── training/                          # All training code
│   ├── __init__.py
│   ├── datasets/                      # Dataset and DataModule classes
│   │   ├── __init__.py
│   │   └── base.py                    # Abstract DataModule (used by domain plugins)
│   ├── models/                        # Model definitions
│   │   ├── __init__.py
│   │   ├── base.py                    # Abstract model interface
│   │   ├── classifier.py              # Multi-label classifier wrapper
│   │   └── backbones.py               # Backbone factory (timm)
│   ├── losses/                        # Loss functions
│   │   ├── __init__.py
│   │   ├── bce_weighted.py            # Weighted BCE for imbalance
│   │   └── focal_loss.py              # Focal loss
│   ├── metrics/                       # Metric computation
│   │   ├── __init__.py
│   │   └── multilabel_metrics.py      # AUC, F1, etc.
│   ├── trainer/                       # Training loop
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Main trainer class
│   │   └── callbacks.py               # Training callbacks
│   ├── utils/                         # Training utilities
│   │   ├── __init__.py
│   │   ├── seed.py
│   │   ├── checkpoint.py
│   │   └── mixup.py
│   └── notebooks/                     # EDA and experiment notebooks
│       ├── 01_eda.ipynb
│       ├── 02_baseline_experiment.ipynb
│       └── 03_results_analysis.ipynb
│
├── inference/                         # Inference and optimization
│   ├── __init__.py
│   ├── preprocessing/                 # Preprocessing pipeline
│   │   ├── __init__.py
│   │   └── image_preprocessor.py
│   ├── postprocessing/                # Output postprocessing
│   │   ├── __init__.py
│   │   └── prediction_formatter.py
│   ├── backends/                      # Inference backends
│   │   ├── __init__.py
│   │   ├── base.py                    # Abstract inference backend
│   │   ├── onnx_backend.py            # ONNX Runtime backend
│   │   ├── triton_backend.py          # Triton client backend
│   │   └── pytorch_backend.py        # PyTorch direct (dev only)
│   └── explainability/               # Explainability pipeline
│       ├── __init__.py
│       ├── gradcam.py                 # Grad-CAM implementation
│       └── shap_explainer.py          # SHAP explainer
│
├── serving/                           # Model serving configuration
│   ├── triton/                        # Triton server
│   │   ├── Dockerfile
│   │   └── model_repository/          # Triton model repository
│   │       └── chestxray_v1/
│   │           ├── config.pbtxt
│   │           └── 1/
│   └── onnx_runtime/                  # ONNX Runtime server
│       └── Dockerfile
│
├── backend/                           # FastAPI Gateway
│   ├── __init__.py
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                        # FastAPI application entry
│   ├── api/                           # Route handlers
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── predict.py             # Prediction endpoints
│   │   │   ├── health.py              # Health/readiness endpoints
│   │   │   └── models.py              # Model info endpoints
│   │   └── dependencies.py            # FastAPI dependencies
│   ├── core/                          # Core business logic
│   │   ├── __init__.py
│   │   ├── config.py                  # App settings
│   │   ├── inference_client.py        # Triton/ONNX client
│   │   └── explainability_service.py  # Grad-CAM service
│   ├── db/                            # Database layer
│   │   ├── __init__.py
│   │   ├── session.py                 # DB session management
│   │   ├── models.py                  # SQLAlchemy models
│   │   └── crud.py                    # CRUD operations
│   ├── cache/                         # Redis caching
│   │   ├── __init__.py
│   │   └── cache_manager.py
│   ├── schemas/                       # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── request.py                 # Request models
│   │   └── response.py                # Response models
│   ├── middleware/                    # FastAPI middleware
│   │   ├── __init__.py
│   │   ├── logging_middleware.py
│   │   └── metrics_middleware.py
│   └── monitoring/                   # Metrics
│       ├── __init__.py
│       └── prometheus_metrics.py
│
├── frontend/                          # React + TypeScript frontend
│   ├── Dockerfile
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── public/
│   │   └── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── components/               # React components
│       │   ├── upload/
│       │   │   ├── ImageUploader.tsx
│       │   │   └── DragDropZone.tsx
│       │   ├── results/
│       │   │   ├── PredictionResults.tsx
│       │   │   ├── DiseaseCard.tsx
│       │   │   └── HeatmapViewer.tsx
│       │   ├── layout/
│       │   │   ├── Header.tsx
│       │   │   └── Footer.tsx
│       │   └── common/
│       │       ├── LoadingSpinner.tsx
│       │       └── ErrorBoundary.tsx
│       ├── hooks/                    # Custom React hooks
│       │   ├── usePrediction.ts
│       │   └── useImageUpload.ts
│       ├── services/                 # API service layer
│       │   └── api.ts
│       ├── types/                    # TypeScript types
│       │   └── prediction.ts
│       └── styles/                   # CSS/Tailwind
│           └── globals.css
│
├── mlflow/                            # MLflow configuration
│   ├── Dockerfile
│   └── mlflow_server.sh
│
├── monitoring/                        # Monitoring stack
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── grafana/
│       ├── Dockerfile
│       ├── provisioning/
│       │   ├── dashboards/
│       │   └── datasources/
│       └── dashboards/
│           └── medvision_dashboard.json
│
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── conftest.py                    # Pytest fixtures
│   ├── unit/
│   │   ├── test_preprocessing.py
│   │   ├── test_postprocessing.py
│   │   ├── test_metrics.py
│   │   └── test_schemas.py
│   ├── integration/
│   │   ├── test_api_endpoints.py
│   │   ├── test_database.py
│   │   └── test_inference_pipeline.py
│   └── performance/
│       └── test_latency.py
│
├── mlops/                             # MLOps utilities (canonical Python logic)
│   ├── __init__.py
│   ├── export/                        # Export pipeline — canonical location
│   │   ├── onnx_exporter.py           # PyTorch → ONNX
│   │   ├── validate_export.py         # Diff check + MLflow artifact
│   │   ├── quantize.py                # ONNX INT8 dynamic quantization
│   │   └── tensorrt_optimizer.py      # TRT engine (GPU environments)
│   ├── artifacts/                     # Artifact distribution
│   │   └── hf_uploader.py             # Upload ONNX + metadata to HF Hub
│   ├── registry/                      # Model registry helpers
│   │   └── model_registry.py
│   └── drift/                         # Data drift detection
│       └── drift_detector.py
│
└── .github/                           # GitHub Actions
    └── workflows/
        ├── ci.yml                     # Continuous integration
        ├── lint.yml                   # Linting workflow
        ├── test.yml                   # Test workflow
        └── docker-build.yml           # Docker build validation
```

---

## 13. REPOSITORY ORGANIZATION

### 13.1 Top-Level Separation of Concerns

The repository is divided into clearly separated domains, each owned by a different concern:

| Directory | Owner Domain | Deployed As |
|-----------|-------------|-------------|
| `platform/` | ML Architecture | Shared Python package |
| `domains/` | Data Science / ML | Imported by training + inference |
| `training/` | Data Science / ML | Kaggle/Colab notebooks + scripts |
| `inference/` | ML Engineering | Shared library used by serving |
| `serving/` | MLOps / Infra | Docker containers |
| `backend/` | Backend Engineering | Docker container |
| `frontend/` | Frontend Engineering | Vercel / Docker container |
| `mlflow/` | MLOps | Docker container |
| `monitoring/` | DevOps | Docker containers |
| `tests/` | QA / All | CI/CD |
| `mlops/` | MLOps | Scripts and automation |
| `configs/` | All | Version-controlled config |
| `docs/` | All | Documentation site |

### 13.2 Cross-Cutting Modules

Some modules are shared across services. To avoid duplication while maintaining service independence:

```
inference/preprocessing/   →   Used by backend + serving
inference/postprocessing/  →   Used by backend
inference/backends/        →   Used by backend (abstracts Triton vs ONNX)
```

These are packaged as a local Python package (`pip install -e ./inference`) and imported by the backend.

### 13.3 Environment Segregation

| Environment | Config Source | DB | Cache | Serving |
|------------|--------------|-----|-------|---------|
| Local Dev | `.env.dev` | SQLite or PG container | Redis container | ONNX Runtime |
| CI/CD | `.env.ci` | Test DB | Mock Redis | Mock |
| Staging | `.env.staging` | Supabase | Redis | ONNX / Triton CPU |
| Production | `.env.prod` (secrets) | PostgreSQL | Redis cluster | Triton GPU |

---

## 14. MODULE RESPONSIBILITIES

### 14.1 Training Module

```
training/
```

**Responsibility:** Everything required to go from raw dataset to trained model checkpoint.

| Sub-Module | Responsibility |
|-----------|---------------|
| `datasets/` | Load, split, transform, and serve data via DataLoaders |
| `models/` | Define model architecture, backbone factory, forward pass |
| `losses/` | Loss function implementations (weighted BCE, focal) |
| `metrics/` | Metric computation and aggregation (AUC, F1, precision, recall) |
| `trainer/` | Training loop, validation loop, epoch management, callbacks |
| `utils/` | Seeding, checkpoint save/load, logging helpers |

**Does NOT:** export models, serve predictions, handle API requests.

### 14.2 Inference Module

```
inference/
```

**Responsibility:** The bridge between trained model artifacts and live prediction serving.

| Sub-Module | Responsibility |
|-----------|---------------|
| `preprocessing/` | Reproducible image preprocessing matching training pipeline |
| `postprocessing/` | Convert raw model outputs to structured predictions |
| `backends/` | Abstraction over Triton / ONNX Runtime / PyTorch |
| `explainability/` | Grad-CAM and SHAP computation |

**Does NOT:** train models, handle HTTP requests, interact with database.

### 14.3 Serving Module

```
serving/
```

**Responsibility:** Package and configure model for runtime serving.

| Sub-Module | Responsibility |
|-----------|---------------|
| `triton/` | Triton model repository structure, config.pbtxt, Dockerfile |
| `onnx_runtime/` | ONNX Runtime CPU serving Dockerfile |

**Does NOT:** contain business logic, interact with database, handle authentication.

### 14.4 Backend Module

```
backend/
```

**Responsibility:** FastAPI application serving as the system's external interface.

| Sub-Module | Responsibility |
|-----------|---------------|
| `api/` | Route handlers, request routing, response formatting |
| `core/` | Configuration, inference client, explainability service |
| `db/` | SQLAlchemy models, migrations, CRUD operations |
| `cache/` | Redis cache management |
| `schemas/` | Pydantic request/response models |
| `middleware/` | Logging, metrics, CORS |
| `monitoring/` | Prometheus metrics definitions |

**Does NOT:** load model weights, execute training, manage Triton configuration.

### 14.5 Frontend Module

```
frontend/
```

**Responsibility:** React + TypeScript clinical interface.

| Sub-Module | Responsibility |
|-----------|---------------|
| `components/` | UI components (upload, results, heatmap, layout) |
| `hooks/` | Custom React hooks for prediction and upload state |
| `services/` | API client layer |
| `types/` | TypeScript type definitions |

**Does NOT:** execute inference, access database, manage model artifacts.

### 14.6 MLOps Module

```
mlops/
```

**Responsibility:** Tools for model lifecycle management.

| Sub-Module | Responsibility |
|-----------|---------------|
| `registry/` | MLflow model registry interaction helpers |
| `export/` | ONNX and TensorRT export orchestration |
| `drift/` | Input data drift detection utilities |

---

````
### 14.7 Platform Module
 
````
platform/
````
 
**Responsibility:** Domain-agnostic abstract contracts that every domain plugin must satisfy.
 
| File | Responsibility |
|------|---------------|
| `domain_plugin.py` | `DomainPlugin` ABC + `DomainConfig` dataclass |
 
**Does NOT:** contain any domain-specific logic, labels, or data loading.
 
```python
# platform/domain_plugin.py
 
from abc import ABC, abstractmethod
from dataclasses import dataclass
 
@dataclass
class DomainConfig:
    domain_name: str           # e.g. "chest_xray"
    num_classes: int           # e.g. 14
    class_labels: list[str]   # e.g. ["Atelectasis", ...]
    task_type: str             # "multilabel" | "multiclass" | "binary"
    input_size: tuple[int, int]  # e.g. (224, 224)
 
class DomainPlugin(ABC):
    """Every domain must implement this interface.
    The platform core (backend, serving, monitoring) depends only on this
    contract — never on a specific domain implementation.
    """
 
    @property
    @abstractmethod
    def config(self) -> DomainConfig: ...
 
    @abstractmethod
    def get_datamodule(self): ...
 
    @abstractmethod
    def get_thresholds(self) -> dict[str, float]: ...
 
    @abstractmethod
    def format_predictions(self, logits: list[float]) -> list[dict]: ...
```
 
### 14.8 Domains Module
 
````
domains/
````
 
**Responsibility:** One subdirectory per domain. Each implements `DomainPlugin`.
 
| Domain | Status | Plugin Class |
|--------|--------|--------------|
| `chest_xray/` | v1.0 — active | `ChestXrayDomainPlugin` |
| `defect_detection/` | future | `DefectDetectionDomainPlugin` |
| `satellite/` | future | `SatelliteDomainPlugin` |
 
Adding a new domain requires **only** creating a new subdirectory that implements `DomainPlugin`. No changes to `platform/`, `backend/`, `inference/`, or `serving/` are required.
````
 
---

## 15. MICROSERVICE ARCHITECTURE

### 15.1 Architecture Overview

The system is composed of seven independently deployable services:

```mermaid
graph TB
    subgraph "User Layer"
        U[User / Clinician]
        B[Browser]
    end

    subgraph "Frontend Layer"
        FE[React Frontend\nVercel / Docker]
    end

    subgraph "API Layer"
        GW[FastAPI Gateway\nPort 8000]
    end

    subgraph "Inference Layer"
        TR[Triton Inference Server\nPort 8001 HTTP\nPort 8002 gRPC]
        OR[ONNX Runtime Fallback\nInternal]
    end

    subgraph "Data Layer"
        PG[(PostgreSQL\nPort 5432)]
        RD[(Redis Cache\nPort 6379)]
    end

    subgraph "MLOps Layer"
        ML[MLflow Tracking\nPort 5000]
        S3[Artifact Storage\nHugging Face Hub]
    end

    subgraph "Monitoring Layer"
        PR[Prometheus\nPort 9090]
        GR[Grafana\nPort 3001]
    end

    U --> B --> FE
    FE --> GW
    GW --> TR
    GW --> OR
    GW --> PG
    GW --> RD
    GW --> PR
    TR --> PR
    ML --> S3
    PR --> GR
```

### 15.2 Service Definitions

| Service | Image | Port(s) | Responsibility |
|---------|-------|---------|---------------|
| `frontend` | custom/node | 3000 | React UI |
| `gateway` | custom/python | 8000 | FastAPI API gateway |
| `triton` | nvcr.io/nvidia/tritonserver | 8001, 8002, 8003 | Model serving |
| `postgres` | postgres:15 | 5432 | Prediction storage |
| `redis` | redis:7 | 6379 | Prediction cache |
| `mlflow` | custom/mlflow | 5000 | Experiment tracking |
| `prometheus` | prom/prometheus | 9090 | Metrics collection |
| `grafana` | grafana/grafana | 3001 | Metrics dashboard |

### 15.3 Service Communication

```mermaid
sequenceDiagram
    participant C as Client (Browser)
    participant FE as Frontend
    participant GW as Gateway (FastAPI)
    participant RC as Redis Cache
    participant TR as Triton Server
    participant XA as Explainability Service
    participant PG as PostgreSQL

    C->>FE: Upload X-ray Image
    FE->>GW: POST /api/v1/predict (multipart)
    GW->>GW: Validate image (format, size)
    GW->>GW: Compute image hash
    GW->>RC: Check cache (image hash)
    
    alt Cache Hit
        RC-->>GW: Cached prediction
    else Cache Miss
        GW->>GW: Preprocess image
        GW->>TR: Inference request (HTTP/gRPC)
        TR-->>GW: Raw logits [14 floats]
        GW->>GW: Apply sigmoid + thresholds
        GW->>XA: Generate Grad-CAM
        XA-->>GW: Heatmap overlay (base64)
        GW->>RC: Store in cache (TTL: 1hr)
    end
    
    GW->>PG: Persist prediction record
    GW-->>FE: JSON response (predictions + heatmap)
    FE-->>C: Render results + heatmap
```

### 15.4 Request-Response Contract

**Request:** `POST /api/v1/predict`
- Content-Type: `multipart/form-data`
- Body: image file (JPEG/PNG, max 10MB)

**Response:** `application/json`
```json
{
  "prediction_id": "uuid",
  "model_version": "1.0.0",
  "processing_time_ms": 342,
  "predictions": [
    {
      "disease": "Atelectasis",
      "probability": 0.823,
      "binary_prediction": true,
      "threshold": 0.5,
      "confidence_level": "high"
    }
  ],
  "heatmap_base64": "data:image/png;base64,...",
  "metadata": {
    "image_dimensions": [1024, 1024],
    "preprocessing_version": "v1.0",
    "timestamp": "2025-01-01T12:00:00Z"
  }
}
```

### 15.5 Service Dependencies

```mermaid
graph LR
    FE --> GW
    GW --> TR
    GW --> PG
    GW --> RD
    TR -.->|Optional| OR[ONNX Fallback]
    ML --> FS[File Storage]
    PR --> GW
    PR --> TR
    GR --> PR
```

**Startup Order:**
1. PostgreSQL, Redis (data stores)
2. Triton (model server — loads model files)
3. Gateway (depends on Triton + PG + Redis)
4. MLflow (independent)
5. Prometheus (depends on Gateway + Triton)
6. Grafana (depends on Prometheus)
7. Frontend (depends on Gateway)

---

## 16. BACKEND ARCHITECTURE

### 16.1 FastAPI Application Structure

```mermaid
graph TD
    REQ[HTTP Request] --> MW[Middleware Stack]
    MW --> LOG[Logging Middleware]
    MW --> MET[Metrics Middleware]
    MW --> COR[CORS Middleware]
    MW --> LIM[Rate Limit Middleware]
    LOG --> RT[Router]
    MET --> RT
    COR --> RT
    LIM --> RT
    
    RT --> PR[/api/v1/predict]
    RT --> HL[/api/v1/health]
    RT --> MI[/api/v1/models]
    RT --> ME[/metrics]
    
    PR --> DEP[Dependencies Layer]
    DEP --> VC[Validation & Cache Check]
    VC --> IC[Inference Client]
    IC --> TR[Triton / ONNX]
    IC --> XS[Explainability Service]
    XS --> GC[Grad-CAM Engine]
    IC --> PF[Prediction Formatter]
    PF --> DB[Database CRUD]
    PF --> CA[Cache Write]
    PF --> RES[Response Builder]
    RES --> RESP[HTTP Response]
```

### 16.2 Application Layers

| Layer | Files | Responsibility |
|-------|-------|---------------|
| **Entry** | `main.py` | App factory, middleware registration, lifespan |
| **Router** | `api/v1/*.py` | HTTP route definitions, input parsing |
| **Schema** | `schemas/*.py` | Request/response Pydantic models |
| **Dependency** | `api/dependencies.py` | Shared DI (DB session, inference client) |
| **Service** | `core/*.py` | Business logic orchestration |
| **Persistence** | `db/crud.py` | Database read/write operations |
| **Cache** | `cache/cache_manager.py` | Redis operations |
| **Metrics** | `monitoring/*.py` | Prometheus counter/histogram definitions |

### 16.3 Inference Client Abstraction

The gateway communicates with the inference layer via an abstract `InferenceClient`:

```

InferenceBackend (abstract)
    ├── ONNXRuntimeBackend   ← Development default (CPU, zero extra containers)
    └── TritonHTTPBackend    ← Production / staging (GPU, dynamic batching)
```

The backend switches between implementations via configuration. This allows:
- Running locally with ONNX Runtime (no GPU required)
- Switching to Triton in staging/production
- Zero code changes in route handlers when switching backends

### 16.4 Caching Strategy

| Cache Key | Format | TTL | Invalidation |
|-----------|--------|-----|-------------|
| Prediction | `predict:{sha256_image_hash}` | 3600s | Manual or TTL |
| Model metadata | `model:info:{model_name}` | 300s | On deployment |

SHA-256 hash of the raw image bytes serves as the cache key, ensuring identical images get cached predictions without storing images in Redis.

### 16.5 Middleware Stack

```python
# Execution order (outermost first)
1. CORS Middleware
2. Rate Limiting Middleware  (slowapi)
3. Request ID Injection
4. Structured Logging Middleware
5. Prometheus Metrics Middleware
6. Exception Handler Middleware
```

### 16.6 Configuration Management in Backend

All backend configuration is managed via `core/config.py` using `pydantic-settings`:

```
Settings
  ├── app_name, version, debug
  ├── database_url
  ├── redis_url
  ├── triton_host, triton_http_port, triton_grpc_port
  ├── inference_backend ("triton" | "onnx" | "pytorch")
  ├── model_name, model_version
  ├── image_max_size_mb
  ├── cache_ttl_seconds
  └── prometheus_enabled
```

All values read from environment variables with sensible defaults.

---

## 17. FRONTEND ARCHITECTURE

### 17.1 Application Architecture

```mermaid
graph TD
    APP[App.tsx] --> HD[Header]
    APP --> MN[Main Content]
    APP --> FT[Footer]
    
    MN --> UP[ImageUploader]
    UP --> DD[DragDropZone]
    UP --> PV[ImagePreview]
    UP --> UH[useImageUpload hook]
    
    MN --> RS[ResultsPanel]
    RS --> PR[PredictionResults]
    PR --> DC[DiseaseCard × 14]
    RS --> HV[HeatmapViewer]
    RS --> MI[ModelInfo]
    
    UH --> AP[api.ts service]
    PR --> UP2[usePrediction hook]
    UP2 --> AP
    AP --> GW[Gateway API]
```

### 17.2 Component Responsibility

| Component | Responsibility |
|-----------|---------------|
| `App.tsx` | Global state, routing, layout |
| `ImageUploader` | File selection, drag-drop, preview, upload trigger |
| `DragDropZone` | File drop target UI |
| `PredictionResults` | Render all 14 disease predictions |
| `DiseaseCard` | Single disease probability display with visual indicator |
| `HeatmapViewer` | Overlay Grad-CAM heatmap on original image |
| `ModelInfo` | Show model version, preprocessing info |
| `LoadingSpinner` | Loading state display |
| `ErrorBoundary` | Catch and display rendering errors |

### 17.3 State Management

For Version 1.0, React's built-in state management is sufficient:

| State | Location | Type |
|-------|----------|------|
| Uploaded image file | `useImageUpload` | `File | null` |
| Image preview URL | `useImageUpload` | `string | null` |
| Upload loading state | `useImageUpload` | `boolean` |
| Prediction result | `usePrediction` | `PredictionResponse | null` |
| Prediction loading | `usePrediction` | `boolean` |
| Prediction error | `usePrediction` | `Error | null` |

If state complexity grows, Zustand is the preferred addition (lightweight, TypeScript-native).

### 17.4 API Service Layer

All API communication is centralized in `services/api.ts`:

```typescript
// All API operations abstracted here
class MedVisionAPIClient {
  predict(imageFile: File): Promise<PredictionResponse>
  getModelInfo(): Promise<ModelInfo>
  getHealth(): Promise<HealthStatus>
}
```

Axios or the Fetch API with typed wrappers is used. Error handling, retry logic, and base URL configuration are centralized here.

### 17.5 TypeScript Type Definitions

```typescript
// types/prediction.ts
interface DiseasePredict {
  disease: string;
  probability: number;
  binary_prediction: boolean;
  threshold: number;
  confidence_level: 'low' | 'medium' | 'high';
}

interface PredictionResponse {
  prediction_id: string;
  model_version: string;
  processing_time_ms: number;
  predictions: DiseasePredict[];
  heatmap_base64: string;
  metadata: PredictionMetadata;
}
```

### 17.6 Frontend Build and Deploy

| Environment | Toolchain | Deploy Target |
|------------|-----------|--------------|
| Development | Vite dev server | `localhost:3000` |
| Production (primary) | Vite build → Vercel | `vercel.app` |
| Production (Docker) | Vite build → nginx | Docker container |

Environment variables for frontend:
```
VITE_API_BASE_URL=https://api.medvision.ai
```

---

## 18. TRAINING ARCHITECTURE

### 18.1 Training Pipeline Overview

```mermaid
graph TD
    CFG[Hydra Config YAML] --> TP[Training Pipeline]
    
    TP --> DM[DataModule]
    DM --> DS[NIHChestXrayDataset]
    DS --> TF[Albumentations Transform]
    TF --> DL[DataLoader]
    
    TP --> MM[Model Factory]
    MM --> BB[timm Backbone]
    BB --> CH[Classification Head]
    CH --> MO[ChestXrayClassifier]
    
    TP --> LF[Loss Function\nWeighted BCE / Focal]
    TP --> OP[Optimizer\nAdamW / SGD]
    TP --> SC[LR Scheduler\nCosineAnnealing]
    
    TP --> TR[Trainer]
    TR --> EP[Epoch Loop]
    EP --> FW[Forward Pass]
    FW --> LS[Loss Compute]
    LS --> BW[Backward Pass]
    BW --> OPT[Optimizer Step]
    
    EP --> VL[Validation Loop]
    VL --> MT[Metric Compute\nAUC, F1]
    MT --> ML[MLflow Logger]
    ML --> CK[Checkpoint Save]
    
    CK --> BM[Best Model\nSelection]
    BM --> EX[Export to ONNX]
    EX --> HF[Push to HF Hub]
```

### 18.2 Model Architecture

**Base Architecture: Multi-label Classification Network**

```
Input: [B, 3, 224, 224]  (batch × channels × height × width)
  ↓
Backbone (timm pretrained)
  ├── DenseNet-121    ← Baseline (original NIH paper backbone)
  ├── EfficientNetV2-S ← Recommended v1.0
  ├── ConvNeXt-Small  ← Alternative
  └── ViT-Small/16    ← Future (ViT experiment)
  ↓
Global Average Pooling
  ↓
Dropout (p=0.3)
  ↓
Linear(backbone_features → 14)
  ↓
Output: [B, 14]  (raw logits)
  ↓
Sigmoid → Probabilities [0, 1]
  ↓
Threshold per class → Binary predictions
```

**Why not Softmax?**  
Multi-label classification allows multiple diseases simultaneously. Sigmoid treats each class independently, while Softmax would force mutual exclusivity.

### 18.3 Loss Function Selection

**Primary: Weighted Binary Cross-Entropy**

For each class `c` with weight `w_c`:
```
Loss = -Σ_c [w_c × (y_c × log(p_c) + (1-y_c) × log(1-p_c))]
```

Weights `w_c = total_samples / (num_classes × class_count_c)` — standard class frequency inverse weighting.

**Secondary: Focal Loss (experiment)**
```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```
Reduces loss contribution of easy negatives; better for extreme imbalance.

### 18.4 Training Configuration Schema

```yaml
# configs/experiments/efficientnetv2_baseline.yaml
model:
  backbone: "efficientnetv2_s"
  pretrained: true
  num_classes: 14
  dropout: 0.3
  freeze_backbone: false

training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 1e-4
  weight_decay: 1e-5
  optimizer: "adamw"
  scheduler: "cosine_annealing"
  scheduler_t_max: 50
  early_stopping_patience: 10
  mixed_precision: true
  gradient_clip: 1.0

data:
  dataset: "nih_chest_xray"
  data_root: "/data/raw/NIH_ChestXray14"
  image_size: 224
  num_workers: 4
  pin_memory: true
  use_weighted_sampling: true

augmentation:
  train:
    horizontal_flip_p: 0.5
    brightness_contrast_p: 0.3
    shift_scale_rotate_p: 0.3
    noise_p: 0.2
  val:
    enabled: false

loss:
  type: "weighted_bce"
  use_class_weights: true

experiment:
  name: "efficientnetv2_s_baseline"
  mlflow_tracking_uri: "http://localhost:5000"
  seed: 42
```

### 18.5 Transfer Learning Strategy

| Phase | Strategy | Duration |
|-------|----------|---------|
| Phase 1 | Freeze backbone, train head only | 5 epochs |
| Phase 2 | Unfreeze all, train end-to-end | Remaining epochs |

This strategy:
- Prevents destroying pretrained features in early training
- Allows head to reach reasonable predictions first
- Then fine-tunes backbone features for medical domain

### 18.6 Checkpoint Strategy

```
checkpoints/
└── {experiment_name}/
    ├── best_model.pt          ← Best val AUC checkpoint
    ├── last_model.pt          ← Last epoch checkpoint
    └── checkpoint_epoch_{N}.pt ← Periodic checkpoints (every 5 epochs)
```

Each checkpoint stores:
```python
{
    "epoch": int,
    "model_state_dict": dict,
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,
    "val_auc": float,
    "config": dict,
    "mlflow_run_id": str,
}
```

### 18.7 Metric Computation

| Metric | Computation | Aggregation |
|--------|-------------|-------------|
| Per-class AUC-ROC | `sklearn.metrics.roc_auc_score` | Per class |
| Mean AUC-ROC | Average of 14 per-class AUC scores | Macro mean |
| Per-class F1 | `sklearn.metrics.f1_score` | Per class |
| Macro F1 | Unweighted mean F1 | Macro |
| Per-class Precision | `precision_score` | Per class |
| Per-class Recall | `recall_score` | Per class |
| Loss | BCE / Focal | Mean over batch |

All metrics are logged to MLflow at every epoch.

### 18.8 Kaggle/Colab Training Setup

Training is designed to be environment-agnostic:

```
Training Environment Detection
  ├── Local CPU → batch_size=4, no mixed precision, small num_workers
  ├── Colab T4 → batch_size=32, mixed precision, num_workers=2
  └── Kaggle P100 → batch_size=64, mixed precision, num_workers=4
```

An `EnvironmentConfig` class detects available hardware and overrides batch/worker settings automatically.

---

## 19. INFERENCE ARCHITECTURE

### 19.1 Inference Pipeline

```mermaid
graph LR
    IMG[Raw Image\nJPEG/PNG] --> VAL[Format Validation]
    VAL --> PRE[Preprocessor]
    
    PRE --> RS[Resize 224×224]
    RS --> RGB[Grayscale → RGB]
    RGB --> NR[Normalize\nImageNet stats]
    NR --> TEN[→ NumPy Array\n1×3×224×224 float32]
    
    TEN --> BE[Inference Backend]
    BE --> TR[Triton HTTP Request]
    TR --> LG[Raw Logits\n1×14 float32]
    
    LG --> POST[Postprocessor]
    POST --> SG[Sigmoid → Probabilities]
    SG --> TH[Apply Thresholds]
    TH --> FMT[Format Predictions]
    FMT --> GC[Grad-CAM Engine]
    GC --> HM[Heatmap Generation]
    HM --> RESP[Structured Response]
```

### 19.2 Preprocessing Specification

The preprocessing pipeline must be **identical** during training and inference. Any discrepancy causes distribution shift and degrades performance.

**Training preprocessing (Albumentations):**
```
Resize(224, 224)
→ Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
→ ToTensorV2()
```

**Inference preprocessing (ONNX-compatible):**
```
PIL.Image.open()
→ Resize to (224, 224)
→ Convert to RGB
→ np.array / 255.0
→ Normalize with ImageNet stats
→ Transpose to CHW
→ Add batch dimension → shape (1, 3, 224, 224)
→ Cast to float32
```

Both pipelines produce identical numerical outputs.

### 19.3 Backend Abstraction

```mermaid
classDiagram
    class InferenceBackend {
        <<abstract>>
        +predict(image: ndarray) ndarray
        +health_check() bool
        +get_model_info() dict
    }
    
    class TritonHTTPBackend {
        -client: InferenceServerClient
        -model_name: str
        -model_version: str
        +predict(image: ndarray) ndarray
    }
    
    class TritonGRPCBackend {
        -channel: grpc.Channel
        +predict(image: ndarray) ndarray
    }
    
    class ONNXRuntimeBackend {
        -session: ort.InferenceSession
        +predict(image: ndarray) ndarray
    }
    
    InferenceBackend <|-- TritonHTTPBackend
    InferenceBackend <|-- TritonGRPCBackend
    InferenceBackend <|-- ONNXRuntimeBackend
```

The factory pattern selects the backend at startup based on configuration:
```python
INFERENCE_BACKEND = "triton"  # or "onnx" or "pytorch"
```

### 19.4 Postprocessing Pipeline

| Step | Input | Output | Details |
|------|-------|--------|---------|
| Sigmoid | Raw logits `[14]` | Probabilities `[14]` | `σ(x) = 1/(1+e^-x)` |
| Threshold apply | Probabilities | Binary predictions | Per-class thresholds (default 0.5) |
| Label mapping | Index → name | Named predictions | Maps index 0 → "Atelectasis", etc. |
| Confidence level | Probability | Enum | `high >0.8`, `medium 0.5-0.8`, `low <0.5` |
| Sort by probability | All predictions | Sorted list | Descending by probability |

### 19.5 Threshold Configuration

Default threshold of 0.5 is suboptimal for medical tasks. Per-class optimal thresholds are determined post-training using the Youden J-statistic on the validation set:

```
J = Sensitivity + Specificity - 1
Optimal threshold = argmax(J)
```

Thresholds are stored in:
```yaml
# configs/serving/thresholds.yaml
thresholds:
  Atelectasis: 0.43
  Cardiomegaly: 0.51
  Effusion: 0.38
  Infiltration: 0.46
  # ... etc
```

These are loaded at service startup and applied during postprocessing.

---

## 20. SERVING ARCHITECTURE

````
### 20.0 Serving Modes
 
The serving layer operates in two modes selected by `INFERENCE_BACKEND` in `.env`:
 
**Development Mode** (`INFERENCE_BACKEND=onnx`) — default
````
FastAPI Gateway
      ↓
ONNX Runtime (in-process)
      ↓
Predictions
````
- No additional containers required
- Runs on CPU-only, 8 GB RAM
- Full functionality: predictions, Grad-CAM, caching, persistence, monitoring
- Set by default in `docker-compose.yml`
 
**Production Mode** (`INFERENCE_BACKEND=triton`) — staging / cloud
````
FastAPI Gateway
      ↓
Triton Inference Server
      ↓
ONNX model  →  TensorRT engine (GPU)
````
- Requires GPU for TensorRT path
- Dynamic batching, multi-model serving
- Configured via `docker-compose.prod.yml`
 
The `InferenceBackend` abstraction (§16.3) ensures the gateway code is identical in both modes. Only the environment variable changes.
````
 
---

### 20.1 NVIDIA Triton Inference Server

Triton Inference Server is the core of the serving layer. It provides:

- Multi-model serving from a single server
- Dynamic batching (groups multiple requests into single batch)
- HTTP and gRPC endpoints
- ONNX, TensorRT, PyTorch, TensorFlow model support
- Built-in Prometheus metrics
- Model versioning (A/B testing capability)
- Ensemble pipelines (future)

### 20.2 Triton Model Repository Structure

```
model_repository/
└── chestxray_classifier/
    ├── config.pbtxt           ← Model configuration
    ├── 1/                     ← Version 1
    │   └── model.onnx         ← ONNX model file
    └── 2/                     ← Version 2 (future update)
        └── model.onnx
```

### 20.3 Triton config.pbtxt Specification

```protobuf
name: "chestxray_classifier"
backend: "onnxruntime"
max_batch_size: 32

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [3, 224, 224]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [14]
  }
]

dynamic_batching {
  preferred_batch_size: [4, 8, 16, 32]
  max_queue_delay_microseconds: 5000
}

instance_group [
  {
    count: 1
    kind: KIND_CPU      # → KIND_GPU for GPU deployment
  }
]

version_policy {
  latest { num_versions: 2 }
}
```

### 20.4 ONNX Runtime CPU Fallback

For CPU-only environments (local development, CI):

```
onnx_fallback/
└── Dockerfile
    ├── onnxruntime==1.17.0
    ├── numpy
    └── Simple HTTP server wrapping ORT session
```

This exposes the same HTTP interface as Triton (same request/response format) so the gateway can switch backends transparently.

### 20.5 Model Export Pipeline

All export logic lives in `mlops/export/`. `scripts/export/` contains only shell wrappers that call these Python modules. There is one canonical export flow:
 
```mermaid
graph LR
    PT[PyTorch .pt checkpoint] --> EX[mlops/export/onnx_exporter.py]
    EX --> ONNX[ONNX Model .onnx]
    ONNX --> VAL[mlops/export/validate_export.py\nOutput diff < 1e-5 on 100 samples]
    VAL -->|pass| OPT[onnx-simplifier]
    VAL -->|fail| ABORT[Abort — do not register]
    OPT --> QUANT[INT8 Dynamic Quantization\nmlops/export/quantize.py]
    OPT --> REG[mlops/artifacts/hf_uploader.py\nHugging Face Hub]
    QUANT --> REG
    REG --> TR[Triton model_repository/]
```
| File | Responsibility |
|------|---------------|
| `mlops/export/onnx_exporter.py` | Load checkpoint → export ONNX with opset 17 |
| `mlops/export/validate_export.py` | 100-sample diff check; log results to MLflow |
| `mlops/export/quantize.py` | ONNX INT8 dynamic quantization |
| `mlops/artifacts/hf_uploader.py` | Upload ONNX + metadata to HF Hub |
 
The `scripts/export/` shell scripts call these modules; they contain no export logic themselves.
````


### 20.6 Model Optimization Options

| Optimization | Tool | Benefit | Tradeoff |
|-------------|------|---------|---------|
| FP16 | TensorRT | 2× speedup | Minimal accuracy loss |
| INT8 | TensorRT + calibration | 4× speedup | Requires calibration dataset |
| ONNX Simplification | onnx-simplifier | Cleaner graph | Minor |
| Static shapes | ONNX export flag | Better optimization | Less flexibility |
| Dynamic batching | Triton config | Higher throughput | Latency may increase |

---

## 21. DEPLOYMENT ARCHITECTURE

### 21.1 Deployment Targets

| Stage | Environment | Components | Cost |
|-------|------------|-----------|------|
| **Local Dev** | Docker Compose (CPU) | All services, ONNX fallback | $0 |
| **Training** | Kaggle / Colab | Training scripts only | $0 |
| **Staging** | Single VM or free tier | Core services | $0-minimal |
| **Production v1** | Cloud VM (future) | All services with GPU | $$ |

### 21.2 Local Development Deployment

```
docker-compose up
```

Brings up 6 services on localhost (dev mode — ONNX backend, no Triton):
 
| Service | URL |
|---------|-----|
| Gateway API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Grafana | http://localhost:3001 |
| Prometheus | http://localhost:9090 |
| PostgreSQL | localhost:5432 (internal) |
 
`docker-compose.prod.yml` adds Triton and targets GPU environments.

### 21.3 Production Deployment (Future)

Target: AWS, GCP, or Azure

```
Cloud Architecture (Future)
├── Load Balancer (ALB/Nginx)
│   └── Frontend (CloudFront/CDN)
├── Application Layer
│   ├── FastAPI Gateway (ECS/Cloud Run)
│   └── Auto Scaling Group
├── Inference Layer
│   ├── Triton on GPU VM (g4dn.xlarge / A100)
│   └── Instance scaling
├── Data Layer
│   ├── RDS PostgreSQL
│   └── ElastiCache Redis
└── Storage
    ├── S3 (model artifacts)
    └── ECR (Docker images)
```

### 21.4 Vercel Frontend Deployment

```
GitHub Push → Vercel CI → Build (Vite) → Deploy to CDN
```

Environment variables in Vercel dashboard:
- `VITE_API_BASE_URL`: Backend gateway URL

---

## 22. DOCKER ARCHITECTURE

### 22.1 Docker Compose Overview

```yaml
# docker-compose.yml (dev — ONNX backend)
services:
  postgres:    # PostgreSQL database
  redis:       # Redis cache
  gateway:     # FastAPI backend (INFERENCE_BACKEND=onnx)
  mlflow:      # MLflow tracking server
  prometheus:  # Metrics collection
  grafana:     # Metrics visualization
 
# docker-compose.prod.yml (production — Triton backend)
services:
  triton:      # Triton Inference Server (GPU)
  # + all services above with INFERENCE_BACKEND=triton
```

### 22.2 Service-Level Docker Configurations

**Gateway Dockerfile:**
```
Base: python:3.11-slim
Install: requirements.txt
Copy: backend/ + inference/ source
Run: uvicorn main:app --host 0.0.0.0 --port 8000
Health: GET /api/v1/health
```

**Triton Dockerfile:**
```
Base: nvcr.io/nvidia/tritonserver:24.01-py3
Copy: model_repository/
Run: tritonserver --model-repository=/models
Health: GET /v2/health/ready
```

**Frontend Dockerfile:**
```
Stage 1 (build): node:20-alpine → npm install → npm run build
Stage 2 (serve): nginx:alpine → copy dist/ → expose 80
```

**MLflow Dockerfile:**
```
Base: python:3.11-slim
Install: mlflow[extras]
Run: mlflow server --backend-store-uri postgresql://...
```

### 22.3 Multi-Stage Builds

The frontend uses multi-stage builds to minimize image size:

```
Stage 1: builder (node:20)
  - Install dependencies
  - Build React application
  → Produces /app/dist/

Stage 2: production (nginx:alpine)
  - Copy /app/dist from builder
  - Copy nginx.conf
  - Expose port 80
  → Final image ~25MB vs ~500MB single stage
```

### 22.4 Docker Networks

```yaml
networks:
  medvision-internal:    # Backend communication (gateway ↔ triton, gateway ↔ postgres)
    driver: bridge
  medvision-monitoring:  # Monitoring (prometheus ↔ gateway, prometheus ↔ triton)
    driver: bridge
  medvision-public:      # External access (frontend ↔ gateway)
    driver: bridge
```

Services are only placed on networks they need, following the principle of least privilege.

### 22.5 Volume Management

```yaml
volumes:
  postgres_data:         # Persistent PostgreSQL data
  redis_data:            # Persistent Redis data (optional)
  mlflow_data:           # MLflow artifacts + database
  triton_models:         # Model repository (bind mount from ./serving/triton/model_repository)
  prometheus_data:       # Prometheus time-series data
  grafana_data:          # Grafana dashboard configs
```

### 22.6 Health Checks

Every service defines a Docker health check:

```yaml
gateway:
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 30s

triton:
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/v2/health/ready"]
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 60s  # Triton takes longer to start
```

---

## 23. NETWORKING

### 23.1 Port Allocation

| Service | Internal Port | External Port | Protocol |
|---------|--------------|--------------|---------|
| Frontend | 80 | 3000 | HTTP |
| Gateway | 8000 | 8000 | HTTP |
| Triton HTTP | 8000 | 8001 | HTTP |
| Triton gRPC | 8001 | 8002 | gRPC |
| Triton Metrics | 8002 | 8003 | HTTP |
| PostgreSQL | 5432 | 5432 | TCP |
| Redis | 6379 | 6379 | TCP |
| MLflow | 5000 | 5000 | HTTP |
| Prometheus | 9090 | 9090 | HTTP |
| Grafana | 3000 | 3001 | HTTP |

*Internal ports are what services listen on. External ports are what Docker binds on host.*

### 23.2 Service Discovery

In Docker Compose, services discover each other by service name (internal DNS):

```
gateway → http://triton:8000      (Triton HTTP)
gateway → postgres:5432           (PostgreSQL)
gateway → redis:6379              (Redis)
prometheus → http://gateway:8000/metrics
prometheus → http://triton:8003/metrics
```

### 23.3 CORS Configuration

The FastAPI gateway is configured with explicit CORS:

```python
origins = [
    "http://localhost:3000",      # Local dev frontend
    "https://*.vercel.app",       # Vercel deployments
    "https://medvision.ai",       # Production domain (future)
]
```

### 23.4 TLS/SSL (Future Production)

In production:
- Frontend served via CDN with HTTPS (Vercel handles this)
- API gateway behind Nginx reverse proxy with Let's Encrypt certificates
- Internal service-to-service communication can use HTTP (same VPC)
- Database connection uses SSL in production

---

## 24. STORAGE DESIGN

### 24.1 Storage Layers

| Layer | Technology | Content | Persistence |
|-------|-----------|---------|------------|
| Relational | PostgreSQL | Predictions, metadata | Permanent |
| Cache | Redis | Prediction cache | TTL-based |
| Model Artifacts | Hugging Face Hub | ONNX models, configs | Permanent |
| Experiment Artifacts | MLflow (local/S3) | Checkpoints, plots | Permanent |
| Application Logs | File + stdout | Structured logs | Rotated |
| Metrics | Prometheus | Time-series metrics | Configurable retention |

### 24.2 Model Artifact Lifecycle

```
Training
  ↓
PyTorch Checkpoint (.pt)
  ↓ MLflow logs
MLflow Experiment Store
  ↓ Export script
ONNX Model (.onnx)
  ↓ Validation
Validated ONNX
  ↓ Push script
Hugging Face Hub (versioned)
  ↓ Download
Triton Model Repository
  ↓ Serve
Live Predictions
```

### 24.3 Hugging Face Hub Organization

```
username/medvision-ai-models/
├── README.md                           # Model card
├── models/
│   ├── chestxray_v1.0.0/
│   │   ├── model.onnx                  # Exported ONNX model
│   │   ├── config.json                 # Model metadata
│   │   ├── thresholds.json             # Per-class thresholds
│   │   └── preprocessing.json         # Preprocessing specification
│   └── chestxray_v1.1.0/              # Next version
└── artifacts/
    ├── class_labels.json               # Class name mapping
    └── eval_metrics.json               # Evaluation metrics
```

### 24.4 Image Storage Policy

**Uploaded images are NOT stored permanently** in Version 1.0:
- Images are processed in memory
- SHA-256 hash is computed for cache lookup
- Images are not written to disk by the backend
- Only prediction results (not images) are persisted in PostgreSQL

This simplifies GDPR compliance and avoids storage costs.

---

## 25. DATABASE SCHEMA

### 25.1 Entity Relationship Diagram

```mermaid
erDiagram
    PREDICTIONS {
        uuid id PK
        timestamp created_at
        string image_hash
        string model_name
        string model_version
        jsonb raw_logits
        jsonb prediction_results
        jsonb metadata
        integer processing_time_ms
        string client_ip
        string session_id
    }

    DISEASE_PREDICTIONS {
        uuid id PK
        uuid prediction_id FK
        string disease_name
        float probability
        boolean binary_prediction
        float threshold_used
        string confidence_level
    }

    MODEL_VERSIONS {
        uuid id PK
        string model_name
        string version
        string onnx_artifact_path
        string mlflow_run_id
        float mean_auc
        jsonb per_class_metrics
        jsonb class_thresholds
        timestamp created_at
        boolean is_active
    }

    PREDICTION_FEEDBACK {
        uuid id PK
        uuid prediction_id FK
        string feedback_type
        jsonb corrected_labels
        string comment
        timestamp created_at
    }

    PREDICTIONS ||--o{ DISEASE_PREDICTIONS : contains
    PREDICTIONS }o--|| MODEL_VERSIONS : uses
    PREDICTIONS ||--o| PREDICTION_FEEDBACK : has
```

### 25.2 Table Specifications

**predictions table:**

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default gen_random_uuid() | Prediction identifier |
| created_at | TIMESTAMPTZ | NOT NULL, default now() | Creation timestamp |
| image_hash | VARCHAR(64) | NOT NULL, INDEX | SHA-256 of input image |
| model_name | VARCHAR(100) | NOT NULL | Model used for prediction |
| model_version | VARCHAR(20) | NOT NULL | Model version |
| raw_logits | JSONB | NOT NULL | Raw model output values |
| prediction_results | JSONB | NOT NULL | Full structured predictions |
| metadata | JSONB | NULLABLE | Image dimensions, preprocessing version |
| processing_time_ms | INTEGER | NOT NULL | Total processing time |
| client_ip | INET | NULLABLE | Client IP (for rate limiting analysis) |
| session_id | VARCHAR(100) | NULLABLE | Session identifier |

**model_versions table:**

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| model_name | VARCHAR(100) | e.g., "chestxray_classifier" |
| version | VARCHAR(20) | Semantic version e.g., "1.0.0" |
| onnx_artifact_path | TEXT | HF Hub path |
| mlflow_run_id | VARCHAR(32) | Linked MLflow run |
| mean_auc | FLOAT | Overall model performance |
| per_class_metrics | JSONB | Per-class AUC, F1, etc. |
| class_thresholds | JSONB | Per-class decision thresholds |
| created_at | TIMESTAMPTZ | Registration timestamp |
| is_active | BOOLEAN | Whether currently deployed |

### 25.3 Indexing Strategy

```sql
-- Primary query patterns
CREATE INDEX idx_predictions_image_hash ON predictions(image_hash);
CREATE INDEX idx_predictions_created_at ON predictions(created_at DESC);
CREATE INDEX idx_predictions_model_version ON predictions(model_name, model_version);
CREATE INDEX idx_disease_predictions_prediction_id ON disease_predictions(prediction_id);
CREATE INDEX idx_disease_predictions_disease_name ON disease_predictions(disease_name);
```

### 25.4 Migration Strategy

Alembic is used for database migrations:

```
alembic/
├── env.py
├── alembic.ini
└── versions/
    ├── 001_initial_schema.py
    ├── 002_add_feedback_table.py
    └── 003_add_model_versions.py
```

All migrations are version-controlled. Production deploys run migrations as a pre-startup step.

---

## 26. API DESIGN

### 26.1 API Versioning

All APIs are versioned via URL prefix: `/api/v1/`

This allows parallel versions for backward compatibility during upgrades.

### 26.2 Endpoint Catalog

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/api/v1/predict` | Submit image for prediction | None (v1) |
| `GET` | `/api/v1/predictions/{id}` | Retrieve stored prediction | None (v1) |
| `GET` | `/api/v1/predictions` | List recent predictions | None (v1) |
| `GET` | `/api/v1/models` | List available models | None |
| `GET` | `/api/v1/models/{name}` | Get model details | None |
| `GET` | `/api/v1/health` | Liveness check | None |
| `GET` | `/api/v1/ready` | Readiness check | None |
| `GET` | `/metrics` | Prometheus metrics | Internal |

### 26.3 Predict Endpoint Specification

**POST /api/v1/predict**

Request:
```
Content-Type: multipart/form-data

Fields:
  - image: File (required, JPEG/PNG, max 10MB)
  - model_name: string (optional, default: "chestxray_classifier")
  - model_version: string (optional, default: "latest")
  - include_heatmap: boolean (optional, default: true)
```

Response `200 OK`:
```json
{
  "status": "success",
  "prediction_id": "550e8400-e29b-41d4-a716-446655440000",
  "model_version": "1.0.0",
  "processing_time_ms": 342,
  "predictions": [
    {
      "disease": "Atelectasis",
      "probability": 0.823,
      "binary_prediction": true,
      "threshold": 0.43,
      "confidence_level": "high",
      "rank": 1
    },
    {
      "disease": "Effusion",
      "probability": 0.612,
      "binary_prediction": true,
      "threshold": 0.38,
      "confidence_level": "medium",
      "rank": 2
    }
  ],
  "positive_findings_count": 2,
  "heatmap_base64": "data:image/png;base64,iVBORw0KGgo...",
  "metadata": {
    "image_dimensions": [1024, 1024],
    "preprocessing": {
      "version": "v1.0",
      "resize": [224, 224],
      "normalization": "imagenet"
    },
    "inference_backend": "triton",
    "cache_hit": false,
    "timestamp": "2025-01-01T12:00:00.000Z"
  }
}
```

Response `422 Unprocessable Entity`:
```json
{
  "status": "error",
  "error_code": "INVALID_IMAGE",
  "message": "Image format not supported. Accepted: JPEG, PNG",
  "details": null
}
```

### 26.4 Health Endpoint Specification

**GET /api/v1/health**

```json
{
  "status": "healthy",
  "timestamp": "2025-01-01T12:00:00Z",
  "services": {
    "database": "healthy",
    "cache": "healthy",
    "inference_server": "healthy"
  }
}
```

**GET /api/v1/ready** (Kubernetes readiness probe pattern)

Returns `200` only when all dependent services are reachable. Returns `503` if any critical dependency is down.

### 26.5 Error Response Schema

All error responses follow a consistent structure:

```json
{
  "status": "error",
  "error_code": "ENUM_VALUE",
  "message": "Human-readable message",
  "details": null | { ... }
}
```

Error codes:
| Code | HTTP Status | Description |
|------|------------|-------------|
| `INVALID_IMAGE` | 422 | File is not a valid image |
| `IMAGE_TOO_LARGE` | 413 | File exceeds size limit |
| `UNSUPPORTED_FORMAT` | 415 | File format not supported |
| `INFERENCE_ERROR` | 500 | Model inference failed |
| `INFERENCE_TIMEOUT` | 504 | Inference server timeout |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Unexpected internal error |

### 26.6 OpenAPI Documentation

FastAPI auto-generates OpenAPI 3.0 documentation:
- Swagger UI: `GET /docs`
- ReDoc: `GET /redoc`
- OpenAPI JSON: `GET /openapi.json`

All endpoints, request/response schemas, and error responses are documented via Pydantic model docstrings and response_model declarations.

---

## 27. AUTHENTICATION (FUTURE)

### 27.1 Current State (v1.0)

Version 1.0 does not implement authentication. The API is publicly accessible for demonstration purposes.

**Security measures in v1.0 without auth:**
- Rate limiting (100 requests/minute per IP)
- File size validation
- Image content validation
- CORS restrictions

### 27.2 Future Authentication Architecture

For production clinical deployment, the following auth architecture is planned:

```
Auth Flow:
  User → Login → Auth Service → JWT Token
  User → API Request + JWT → Gateway → Token Validation → Service
```

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Auth Provider | Supabase Auth / Auth0 | Identity provider |
| Token Type | JWT (RS256) | Stateless auth |
| Token Validation | FastAPI dependency | Per-request validation |
| RBAC | Custom middleware | Role-based access |

**User Roles (future):**
- `public`: Rate-limited, no history
- `clinician`: Full access, prediction history
- `admin`: Model management, metrics access

### 27.3 API Key Authentication (Alternative)

For B2B API access:
```
X-API-Key: {api_key}
```

API keys stored in PostgreSQL with:
- Hashed key (never plaintext)
- Rate limit tier
- Organization association
- Expiry date

---

## 28. EXPERIMENT TRACKING DESIGN

### 28.1 MLflow Architecture

```mermaid
graph LR
    TR[Training Script] --> ML[MLflow Client]
    ML --> MLS[MLflow Tracking Server\nPort 5000]
    MLS --> PG[(PostgreSQL\nMLflow Backend)]
    MLS --> FS[Artifact Store\nLocal / S3]
    
    MLS --> UI[MLflow UI\nBrowser]
    MLS --> REG[Model Registry]
```

### 28.2 Experiment Hierarchy

```
MLflow Experiments
└── medvision-chestxray/
    ├── Run: densenet121_baseline_20250101
    │   ├── Parameters
    │   │   ├── backbone: densenet121
    │   │   ├── learning_rate: 0.0001
    │   │   ├── batch_size: 32
    │   │   └── ... all hyperparameters
    │   ├── Metrics (per epoch)
    │   │   ├── train_loss
    │   │   ├── val_loss
    │   │   ├── val_mean_auc
    │   │   ├── val_auc_atelectasis
    │   │   └── ... 14 per-class AUCs
    │   ├── Artifacts
    │   │   ├── best_model.pt
    │   │   ├── confusion_matrix.png
    │   │   ├── roc_curves.png
    │   │   └── config.yaml (copy of run config)
    │   └── Tags
    │       ├── environment: kaggle
    │       ├── dataset_version: v1.0
    │       └── phase: baseline
    └── Run: efficientnetv2_s_20250115
```

### 28.3 MLflow Logging Contract

The training system logs the following at every run:

**Parameters (logged once at start):**
- All config YAML keys (flattened with dot notation)
- Dataset version and split hash
- Environment info (GPU type, CUDA version, framework version)

**Metrics (logged per epoch):**
- `train_loss`, `val_loss`
- `val_mean_auc`
- `val_f1_macro`, `val_precision_macro`, `val_recall_macro`
- `val_auc_{disease_name}` × 14
- `learning_rate`
- `epoch_time_seconds`

**Artifacts (logged at end):**
- Best model checkpoint (.pt)
- Final ONNX export (.onnx)
- ROC curve plots (per class and mean)
- Confusion matrix
- Training curves (loss, AUC over epochs)
- Complete config file

### 28.4 MLflow Backend Configuration

MLflow can use different backends:

| Component | Dev | Production |
|-----------|-----|------------|
| Tracking URI | `http://localhost:5000` | `http://mlflow:5000` |
| Backend Store | SQLite / PostgreSQL | PostgreSQL |
| Artifact Store | Local filesystem | S3 / GCS |

---

## 29. MODEL REGISTRY DESIGN

### 29.1 MLflow Model Registry

MLflow provides a built-in model registry with lifecycle stages:

```
Model: chestxray_classifier
├── Version 1 (Stage: Archived)
│   ├── Registered from run: run_id_abc
│   ├── AUC: 0.81
│   └── Tags: backbone=densenet121
├── Version 2 (Stage: Staging)
│   ├── Registered from run: run_id_def
│   ├── AUC: 0.84
│   └── Tags: backbone=efficientnetv2_s
└── Version 3 (Stage: Production)
    ├── Registered from run: run_id_ghi
    ├── AUC: 0.86
    └── Tags: backbone=convnext_s
```

### 29.2 Model Lifecycle Stages

```
Training → None → Staging → Production → Archived
```

| Stage | Meaning | Who Sets |
|-------|---------|---------|
| `None` | Just registered | Automated after training completes |
| `Staging` | Passed promotion criteria (see below) | Automated via `mlops/registry/promote.py` |
| `Production` | Approved for live serving | Manual engineer sign-off |
| `Archived` | Superseded or degraded | Manual |

**Promotion Criteria — None → Staging**

All of the following must pass before a model is promoted to Staging:

| Criterion | Threshold | Checked By |
|-----------|-----------|-----------|
| Validation mean AUC | ≥ 0.80 | `validate_export.py` |
| ONNX export successful | No exceptions | `onnx_exporter.py` |
| ONNX/PyTorch output diff | max abs diff < 1e-5 on 100 samples | `validate_export.py` |
| Benchmark completed | p99 latency report generated | `mlops/benchmark/` |
| ONNX p99 latency (CPU) | < 500 ms | benchmark report |

**Promotion Criteria — Staging → Production**

| Criterion | Threshold | Checked By |
|-----------|-----------|-----------|
| All Staging criteria | Must still pass | — |
| No AUC regression vs current Production | mean AUC ≥ current Production − 0.01 | manual review |
| HF Hub artifact uploaded | model + metadata + thresholds present | `hf_uploader.py` |
| Manual engineer sign-off | PR approval or explicit CLI flag | engineer |

**Archival Criteria — Production → Archived**

A Production model is archived when a newer model is promoted to Production, or when mean AUC on the test set degrades more than 5% from its registered value.

### 29.3 Model Metadata Contract

Every registered model version must include:

```json
{
  "model_name": "chestxray_classifier",
  "version": "2.0",
  "mlflow_run_id": "abc123",
  "backbone": "efficientnetv2_s",
  "training_date": "2025-01-15",
  "dataset": "nih_chest_xray_v1.0",
  "dataset_split_hash": "sha256_of_split_file",
  "preprocessing_version": "v1.0",
  "input_shape": [1, 3, 224, 224],
  "output_shape": [1, 14],
  "class_labels": ["Atelectasis", "Cardiomegaly", ...],
  "class_thresholds": {"Atelectasis": 0.43, ...},
  "performance": {
    "mean_auc": 0.84,
    "per_class_auc": {...}
  },
  "onnx_path": "hf://username/medvision-models/chestxray_v2.0/model.onnx",
  "notes": "Improved baseline with EfficientNetV2-S backbone"
}
```

### 29.4 Model Promotion Pipeline

```mermaid
graph LR
    TR[Training Complete] --> LG[Log to MLflow]
    LG --> EX[Export ONNX]
    EX --> VP[Validate Performance\nAUC > 0.80?]
    VP -->|Pass| ST[Set Stage: Staging]
    VP -->|Fail| RJ[Reject Run]
    ST --> BT[Benchmark Test\nLatency < 500ms?]
    BT -->|Pass| UP[Upload to HF Hub]
    UP --> REG[Register in DB\nmodel_versions]
    REG --> DEP[Deploy to Triton]
    DEP --> PR[Set Stage: Production]
```

---

## 30. CHECKPOINT STRATEGY

### 30.1 Checkpoint Frequency

| Checkpoint Type | Frequency | Retention |
|----------------|-----------|-----------|
| Best validation AUC | Every improvement | Keep forever |
| Last epoch | Every epoch | Keep last 3 |
| Periodic | Every 5 epochs | Keep last 3 |
| Pre-unfreeze | Once (before Phase 2) | Keep forever |

### 30.2 Checkpoint File Naming

```
{experiment_name}/
├── best_model.pt                        # Best val AUC ever
├── last_model.pt                        # Last completed epoch
├── checkpoint_epoch_05.pt               # Periodic
├── checkpoint_epoch_10.pt
└── pre_unfreeze.pt                      # Before backbone unfreeze
```

### 30.3 Checkpoint Contents

```python
torch.save({
    # Model state
    "model_state_dict": model.state_dict(),
    
    # Training state
    "epoch": current_epoch,
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "scaler_state_dict": scaler.state_dict(),  # For AMP
    
    # Performance
    "val_mean_auc": best_auc,
    "per_class_auc": per_class_aucs,
    "val_loss": val_loss,
    
    # Reproducibility
    "config": OmegaConf.to_container(cfg),
    "mlflow_run_id": run.info.run_id,
    "random_seed": cfg.experiment.seed,
    
    # Provenance
    "training_date": datetime.utcnow().isoformat(),
    "dataset_split_hash": split_hash,
}, checkpoint_path)
```

### 30.4 Training Resume Logic

```python
if resume_checkpoint_path:
    checkpoint = torch.load(resume_checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_auc = checkpoint["val_mean_auc"]
    # MLflow: continue logging to same run
    active_run = mlflow.start_run(run_id=checkpoint["mlflow_run_id"])
```

---

## 31. VERSIONING STRATEGY

### 31.1 Version Taxonomy

The project uses separate versioning schemes for different artifacts:

| Artifact | Scheme | Example |
|----------|--------|---------|
| Platform version | Semantic (MAJOR.MINOR.PATCH) | `1.0.0` |
| Model version | Semantic (MAJOR.MINOR.PATCH) | `1.2.3` |
| API version | Integer prefix | `v1`, `v2` |
| Dataset version | Date-based | `nih_v1_2025-01` |
| Docker images | Git SHA + semantic | `gateway:1.0.0-abc1234` |
| Config schemas | Integer | `config_v2` |

### 31.2 Model Version Semantics

| Version Change | Meaning | Example |
|---------------|---------|---------|
| MAJOR | Architecture change (different backbone family) | `1.x.x → 2.0.0` |
| MINOR | Different backbone/hyperparameters, same task | `1.0.x → 1.1.0` |
| PATCH | Retraining with same config, bug fix | `1.0.0 → 1.0.1` |

### 31.3 API Versioning Policy

- New endpoints are added to current version
- Breaking changes require a new version prefix (`/api/v2/`)
- Old version is maintained for minimum 6 months
- Deprecation notices are added 30 days before removal

### 31.4 Git Branching Strategy

```
main                    # Production-ready code
├── develop             # Integration branch
├── feature/...         # Feature branches
├── experiment/...      # ML experiment branches
├── hotfix/...          # Emergency fixes
└── release/1.0.0       # Release preparation
```

All merges to `main` require:
- Passing CI pipeline
- Code review (even solo projects — self review via PR)
- Version tag

---

## 32. EXPLAINABILITY PIPELINE

### 32.1 Why Explainability

In clinical AI, a prediction without explanation is insufficient. Clinicians need to understand **what regions** of the image drove the model's prediction. This is:

1. **Clinically useful** — confirms model focus on correct anatomy
2. **Trust-building** — increases clinician adoption
3. **Debugging tool** — identifies spurious correlations
4. **Regulatory requirement** — FDA guidance on AI transparency

### 32.2 Grad-CAM Architecture

```mermaid
graph LR
    IMG[Input Image] --> MDL[Backbone + Head]
    MDL --> LOGIT[Output Logits]
    
    MDL --> LAYER[Target Layer\nLast Conv Block]
    LOGIT --> GRAD[Compute Gradients\n∂logit_c / ∂activation]
    LAYER --> FEAT[Feature Maps\nH×W×C]
    GRAD --> POOL[Global Average\nPool Gradients\nα_k^c]
    POOL --> WEIGHT[Weight Feature Maps]
    FEAT --> WEIGHT
    WEIGHT --> SUM[Weighted Sum\nΣ α_k^c A_k]
    SUM --> RELU[ReLU\n(positive influence only)]
    RELU --> UPSAMPLE[Upsample to\nOriginal Size]
    UPSAMPLE --> NORM[Normalize 0-1]
    NORM --> CMAP[Apply Colormap\nJET/turbo]
    CMAP --> OVERLAY[Overlay on\nOriginal Image]
    OVERLAY --> HM[Heatmap Output]
```

### 32.3 Grad-CAM Target Layer Selection

The target layer is the **last convolutional layer** before global average pooling. This layer preserves spatial information while having the highest semantic content.

| Backbone | Target Layer |
|----------|-------------|
| DenseNet-121 | `features.denseblock4` |
| EfficientNetV2-S | `blocks[-1]` |
| ConvNeXt-Small | `stages[-1]` |
| ResNet-50 | `layer4` |

Target layer is configurable in `configs/inference.yaml`.

### 32.4 Grad-CAM per Class

For multi-label prediction, a separate Grad-CAM is generated **per positive predicted class**:

```
For each disease with binary_prediction = True:
  → Generate Grad-CAM heatmap for that disease's logit
  → Return as separate overlay
```

The frontend displays the heatmap for the highest-probability positive prediction by default, with a selector for other positive classes.

### 32.5 SHAP Explainability (Secondary)

For deeper explanation:

```python
# Kernel SHAP for model-agnostic explanation
explainer = shap.KernelExplainer(
    model=onnx_predict_wrapper,
    background=background_dataset_samples
)
shap_values = explainer.shap_values(preprocessed_image)
```

SHAP is computationally expensive (~5-10 seconds per image). It is offered as an optional detailed explanation, not in the default inference path.

### 32.6 Explainability API Design

```
POST /api/v1/predict
  → Returns default Grad-CAM for top positive class

POST /api/v1/predict?explain=full
  → Returns Grad-CAM for all positive classes
  
POST /api/v1/explain/{prediction_id}
  → Generates SHAP explanation for stored prediction (async)
  
GET /api/v1/explain/{prediction_id}/status
  → Check SHAP generation status
```

---

## 33. BENCHMARK PIPELINE

### 33.1 Benchmark Dimensions

The benchmark pipeline evaluates three dimensions:

| Dimension | What It Measures | Tools |
|-----------|-----------------|-------|
| **Model Quality** | Prediction accuracy | AUC, F1, precision, recall |
| **Inference Latency** | Time per prediction | Python timeit, locust |
| **System Throughput** | Requests/second | locust, wrk |

### 33.2 Latency Benchmark Specification

```python
# Benchmark targets per backend

LATENCY_TARGETS = {
    "pytorch": {"p50": 800, "p99": 1200},       # ms, CPU
    "onnx_cpu": {"p50": 200, "p99": 500},        # ms, CPU
    "triton_cpu": {"p50": 250, "p99": 600},      # ms, CPU
    "triton_gpu": {"p50": 30, "p99": 100},       # ms, GPU
}
```

**Benchmark methodology:**
1. Warm-up: 10 inference calls (not measured)
2. Measurement: 1000 inference calls
3. Report: p50, p90, p95, p99, min, max, mean, std

### 33.3 Benchmark Output Format

```json
{
  "benchmark_id": "uuid",
  "timestamp": "2025-01-01T12:00:00Z",
  "backend": "triton_cpu",
  "model_version": "1.0.0",
  "batch_size": 1,
  "num_iterations": 1000,
  "latency_ms": {
    "min": 45.2,
    "max": 312.8,
    "mean": 187.4,
    "std": 28.3,
    "p50": 182.1,
    "p90": 231.5,
    "p95": 256.7,
    "p99": 298.4
  },
  "throughput_rps": 5.3,
  "hardware": {
    "cpu": "Intel Xeon",
    "memory_gb": 8,
    "gpu": null
  }
}
```

### 33.4 Backend Comparison Report

The benchmark script runs all three backends and produces a comparison:

```
Backend Benchmark Report — 2025-01-01
======================================
Metric          PyTorch     ONNX-CPU    Triton-CPU  Triton-GPU
────────────────────────────────────────────────────────────────
p50 latency     823ms       191ms       207ms       28ms
p99 latency     1143ms      487ms       534ms       89ms
Throughput      1.2 rps     5.2 rps     4.8 rps     35.6 rps
Memory (model)  120MB       45MB        45MB        45MB
────────────────────────────────────────────────────────────────
ONNX speedup vs PyTorch: 4.3×
Triton GPU speedup vs PyTorch: 29.4×
```

### 33.5 Model Quality Benchmark

```python
# Evaluated on held-out test set
benchmark_metrics = {
    "mean_auc": 0.843,
    "per_class_auc": {
        "Atelectasis": 0.812,
        "Cardiomegaly": 0.891,
        ...
    },
    "macro_f1": 0.423,
    "comparison_to_baseline": {
        "vs_densenet121_nih_paper": "+0.041 AUC"
    }
}
```

---

## 34. MONITORING ARCHITECTURE

### 34.1 Observability Stack

```mermaid
graph LR
    GW[FastAPI Gateway] --> PM[prometheus-client\n/metrics]
    TR[Triton Server] --> TM[Triton Metrics\n/metrics]
    PG[PostgreSQL] --> PE[postgres_exporter]
    RD[Redis] --> RE[redis_exporter]
    
    PM --> PR[Prometheus\nScrape every 15s]
    TM --> PR
    PE --> PR
    RE --> PR
    
    PR --> GR[Grafana\nDashboards]
    PR --> AL[Alert Manager\nFuture]
```

### 34.2 Metrics Catalog

**Gateway Metrics:**

| Metric Name | Type | Labels | Description |
|------------|------|--------|-------------|
| `medvision_requests_total` | Counter | method, endpoint, status | Total requests |
| `medvision_request_duration_seconds` | Histogram | endpoint | Request latency |
| `medvision_inference_duration_seconds` | Histogram | backend, model | Inference latency |
| `medvision_cache_hits_total` | Counter | hit/miss | Cache performance |
| `medvision_predictions_by_disease` | Counter | disease | Prediction distribution |
| `medvision_active_requests` | Gauge | — | Concurrent requests |
| `medvision_errors_total` | Counter | error_type | Error counts |

**Triton Metrics (native):**

| Metric Name | Description |
|------------|-------------|
| `nv_inference_request_success` | Successful inference count |
| `nv_inference_request_failure` | Failed inference count |
| `nv_inference_queue_duration_us` | Queue waiting time |
| `nv_inference_compute_infer_duration_us` | Compute time |
| `nv_gpu_memory_used_bytes` | GPU memory usage |

### 34.3 Grafana Dashboard Layout

**Dashboard: MedVision AI — Operational**

```
Row 1: System Health
  ├── [Gauge] API Status (healthy/degraded)
  ├── [Gauge] Inference Server Status
  └── [Gauge] Database Status

Row 2: Traffic
  ├── [Graph] Requests/second (5m rolling)
  ├── [Graph] Error rate (%)
  └── [SingleStat] Total predictions today

Row 3: Latency
  ├── [Graph] API latency p50/p95/p99 over time
  ├── [Graph] Inference latency p50/p99
  └── [Heatmap] Latency distribution

Row 4: Predictions
  ├── [Bar] Prediction count by disease (24h)
  ├── [Graph] Average probability per disease
  └── [Graph] Cache hit rate

Row 5: Infrastructure
  ├── [Graph] CPU usage by service
  ├── [Graph] Memory usage by service
  └── [Graph] GPU utilization (when GPU available)
```

### 34.4 Alert Rules (Future)

| Alert | Condition | Severity |
|-------|-----------|---------|
| High error rate | error_rate > 5% for 5min | Critical |
| High latency | p99 > 2000ms for 5min | Warning |
| Inference server down | health check fails for 1min | Critical |
| Low cache hit rate | cache_hit_rate < 20% | Info |
| Memory pressure | memory_usage > 80% | Warning |

---

## 35. LOGGING ARCHITECTURE

### 35.1 Logging Strategy

All logging uses **structured JSON logs** via `structlog`. This enables:
- Machine-parseable logs for log aggregation (ELK, Loki)
- Context injection (request_id, user_id, model_version)
- Consistent log format across all services

### 35.2 Log Levels

| Level | Usage |
|-------|-------|
| `DEBUG` | Development only, disabled in production |
| `INFO` | Normal operation (request received, prediction complete) |
| `WARNING` | Recoverable issues (cache miss, fallback used) |
| `ERROR` | Non-fatal errors (invalid input, inference timeout) |
| `CRITICAL` | Fatal errors (service crash, DB unreachable) |

### 35.3 Log Format

```json
{
  "timestamp": "2025-01-01T12:00:00.000Z",
  "level": "INFO",
  "service": "medvision-gateway",
  "event": "prediction_complete",
  "request_id": "req_abc123",
  "prediction_id": "pred_xyz789",
  "model_version": "1.0.0",
  "processing_time_ms": 342,
  "cache_hit": false,
  "positive_findings": ["Atelectasis", "Effusion"],
  "client_ip": "1.2.3.4"
}
```

### 35.4 Log Context Injection

Request ID is generated at gateway entry and propagated through all log messages within that request:

```python
# Middleware generates UUID per request
request_id = str(uuid.uuid4())

# structlog bound to request context
log = structlog.get_logger().bind(
    request_id=request_id,
    service="medvision-gateway"
)

# All downstream logs in this request include request_id automatically
log.info("prediction_started", model="chestxray_classifier")
log.info("inference_complete", latency_ms=234)
log.info("prediction_stored", prediction_id=pred_id)
```

### 35.5 Log Aggregation (Future)

For production multi-service deployment:

```
All Services → stdout/stderr → Docker logging driver
  ↓
Promtail (log shipper)
  ↓
Grafana Loki (log storage)
  ↓
Grafana (log visualization + search)
```

---

## 36. CONFIGURATION MANAGEMENT

### 36.1 Configuration Philosophy

Configuration follows the **12-Factor App** methodology:
- All config is external to code
- Config varies between environments (dev/staging/prod)
- Secrets are in environment variables, never in code
- Non-secret configuration is in version-controlled YAML files

### 36.2 Configuration Hierarchy

```
Priority (highest → lowest):
  1. Environment variables
  2. Docker Compose environment section
  3. .env file
  4. YAML config file
  5. Code defaults
```

### 36.3 Hydra Configuration (Training)

Training uses Hydra for hierarchical configuration with override support:

```python
# Run with defaults
python training/train.py

# Override specific values
python training/train.py training.batch_size=64 model.backbone=densenet121

# Use named experiment config
python training/train.py +experiment=efficientnetv2_baseline

# Multi-run sweep
python training/train.py --multirun model.backbone=densenet121,efficientnetv2_s training.learning_rate=0.0001,0.00001
```

### 36.4 Backend Configuration (pydantic-settings)

```python
class Settings(BaseSettings):
    # Application
    app_name: str = "MedVision AI Gateway"
    version: str = "1.0.0"
    debug: bool = False
    
    # Database
    database_url: PostgresDsn
    
    # Redis
    redis_url: RedisDsn = "redis://redis:6379/0"
    cache_ttl_seconds: int = 3600
    
    # Inference
    inference_backend: Literal["triton", "onnx", "pytorch"] = "triton"
    triton_host: str = "triton"
    triton_http_port: int = 8000
    triton_grpc_port: int = 8001
    model_name: str = "chestxray_classifier"
    model_version: str = "1"
    
    # Image validation
    max_image_size_mb: int = 10
    supported_formats: list[str] = ["JPEG", "PNG"]
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_period_seconds: int = 60
    
    class Config:
        env_file = ".env"
        case_sensitive = False
```

### 36.5 Config Schema Validation

All YAML configurations are validated against Pydantic schemas at startup. Invalid configuration fails fast with a clear error message rather than failing silently at runtime.

---

## 37. ENVIRONMENT VARIABLES

### 37.1 Complete Environment Variable Reference

**.env.example (committed to repo):**

```bash
# ─────────────────────────────────────────
# Application
# ─────────────────────────────────────────
APP_NAME=MedVision AI
APP_VERSION=1.0.0
DEBUG=false
LOG_LEVEL=INFO
ENVIRONMENT=development

# ─────────────────────────────────────────
# Database
# ─────────────────────────────────────────
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=medvision
POSTGRES_USER=medvision_user
POSTGRES_PASSWORD=changeme_in_production
DATABASE_URL=postgresql+asyncpg://medvision_user:changeme@postgres:5432/medvision

# ─────────────────────────────────────────
# Redis
# ─────────────────────────────────────────
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
REDIS_URL=redis://redis:6379/0
CACHE_TTL_SECONDS=3600

# ─────────────────────────────────────────
# Inference
# ─────────────────────────────────────────
INFERENCE_BACKEND=triton
TRITON_HOST=triton
TRITON_HTTP_PORT=8001
TRITON_GRPC_PORT=8002
MODEL_NAME=chestxray_classifier
MODEL_VERSION=1

# ─────────────────────────────────────────
# MLflow
# ─────────────────────────────────────────
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=medvision-chestxray

# ─────────────────────────────────────────
# Hugging Face
# ─────────────────────────────────────────
HUGGINGFACE_TOKEN=hf_your_token_here
HUGGINGFACE_REPO=username/medvision-ai-models

# ─────────────────────────────────────────
# Monitoring
# ─────────────────────────────────────────
PROMETHEUS_ENABLED=true

# ─────────────────────────────────────────
# Frontend (Vite)
# ─────────────────────────────────────────
VITE_API_BASE_URL=http://localhost:8000
```

### 37.2 Secret vs Non-Secret Classification

| Variable | Secret? | Storage |
|----------|---------|---------|
| `POSTGRES_PASSWORD` | ✅ YES | GitHub Secrets, Docker secret |
| `HUGGINGFACE_TOKEN` | ✅ YES | GitHub Secrets |
| `DATABASE_URL` | ✅ YES | Docker secret |
| `DEBUG` | ❌ NO | .env file |
| `TRITON_HOST` | ❌ NO | .env file |
| `MLFLOW_TRACKING_URI` | ❌ NO | .env file |

---

## 38. SECURITY CONSIDERATIONS

### 38.1 Input Validation

Every uploaded image goes through a multi-layer validation:

```
1. File size check (< 10MB)
2. Content-Type header check (image/jpeg or image/png)
3. Magic byte validation (not just trusting extension)
4. PIL.Image.open() validation (valid image can be decoded)
5. Dimension bounds check (minimum 32×32, maximum 8192×8192)
6. Sanitize filename (no path traversal)
```

### 38.2 Dependency Security

```
# Security tooling in CI
pip-audit     → Scan for known CVEs in Python dependencies
safety        → Alternative CVE scanner
bandit        → Static security analysis
npm audit     → Frontend dependency CVE scan
```

### 38.3 Container Security

```
- Run containers as non-root user
- Read-only filesystem where possible
- Drop all Linux capabilities except what's needed
- Use distroless or slim base images
- Scan Docker images with Trivy in CI
```

### 38.4 Network Security

```
- Services on internal Docker network not exposed externally
- Only Gateway and Frontend ports exposed to host
- Database not exposed to host in production
- Redis not exposed to host in production
```

### 38.5 No Patient Data

**Version 1.0 explicitly uses only public research datasets.** No patient PII is ever handled. HIPAA compliance is a future concern.

### 38.6 Rate Limiting

```python
# slowapi (FastAPI limiter)
@limiter.limit("100/minute")
async def predict(request: Request, ...):
    ...
```

Per-IP rate limiting prevents abuse while remaining open for demonstration.

---

## 39. ERROR HANDLING

### 39.1 Error Handling Philosophy

**Fail fast, fail loudly, never silently.**

- All exceptions are caught at the appropriate layer
- Errors are logged with full context (request_id, stack trace)
- Client receives a clean error response (no stack traces in production)
- Internal errors are tracked in metrics

### 39.2 Error Hierarchy

```
MedVisionError (base)
  ├── ValidationError
  │   ├── InvalidImageFormatError
  │   ├── ImageTooLargeError
  │   └── UnsupportedFormatError
  ├── InferenceError
  │   ├── InferenceTimeoutError
  │   ├── InferenceServerUnavailableError
  │   └── ModelNotFoundError
  ├── StorageError
  │   ├── DatabaseError
  │   └── CacheError
  └── ConfigurationError
```

### 39.3 Exception Handler Registration

```python
# All exceptions mapped to HTTP responses in main.py
app.add_exception_handler(InvalidImageFormatError, validation_error_handler)
app.add_exception_handler(InferenceTimeoutError, timeout_error_handler)
app.add_exception_handler(Exception, generic_error_handler)
```

### 39.4 Graceful Degradation

| Failure | Behavior |
|---------|---------|
| Triton unavailable | Fall back to ONNX Runtime backend |
| Redis unavailable | Skip cache, proceed with prediction |
| PostgreSQL unavailable | Return prediction without persisting |
| Grad-CAM fails | Return prediction without heatmap, log warning |

The system degrades gracefully — partial failures don't bring down the entire prediction pipeline.

### 39.5 Retry Strategy

```python
# Inference calls with retry
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(InferenceTimeoutError)
)
async def call_inference_server(payload):
    ...
```

---

## 40. TESTING STRATEGY

### 40.1 Testing Pyramid

```
                    ┌─────┐
                    │ E2E │         < 10 tests
                    └─────┘
                  ┌───────────┐
                  │Integration│     ~50 tests
                  └───────────┘
              ┌─────────────────┐
              │     Unit        │   ~200 tests
              └─────────────────┘
```

### 40.2 Unit Tests

**Scope:** Individual functions and classes in isolation.

| Module | Test Focus |
|--------|-----------|
| `preprocessing/` | Image resize, normalization, dtype correctness |
| `postprocessing/` | Sigmoid application, threshold logic, label mapping |
| `schemas/` | Pydantic validation, field constraints |
| `metrics/` | AUC computation on known inputs, F1 calculation |
| `cache/` | Cache key generation, hash consistency |
| `losses/` | Loss values on known inputs |

**Tools:** pytest, unittest.mock, pytest-asyncio

### 40.3 Integration Tests

**Scope:** Multiple components working together, with real infrastructure (test DB, mock Triton).

| Test | Scope |
|------|-------|
| `test_api_endpoints.py` | Full request → response cycle via TestClient |
| `test_database.py` | CRUD operations against test database |
| `test_cache.py` | Cache read/write with real Redis |
| `test_inference_pipeline.py` | Preprocess → ONNX inference → postprocess |

**Infrastructure:** Docker Compose test environment with PostgreSQL and Redis containers.

### 40.4 Performance Tests

**Scope:** Latency and throughput validation.

```python
# tests/performance/test_latency.py
def test_inference_latency_p99_under_500ms():
    """Ensure p99 latency stays within SLA."""
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        response = client.post("/api/v1/predict", files={"image": test_image})
        latencies.append((time.perf_counter() - start) * 1000)
    
    p99 = np.percentile(latencies, 99)
    assert p99 < 500, f"p99 latency {p99:.1f}ms exceeds 500ms SLA"
```

### 40.5 Model Quality Tests

```python
# tests/integration/test_model_quality.py
def test_mean_auc_above_baseline():
    """Regression test: model must not degrade below baseline AUC."""
    auc = evaluate_model_on_test_set(model_path=CURRENT_MODEL_PATH)
    assert auc["mean"] >= 0.80, f"Mean AUC {auc['mean']:.3f} below 0.80 baseline"
```

### 40.6 Test Configuration

```python
# tests/conftest.py
@pytest.fixture
def test_db():
    """Fresh database for each test."""
    engine = create_engine(TEST_DATABASE_URL)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)

@pytest.fixture
def test_client(test_db):
    """FastAPI test client with test dependencies."""
    app.dependency_overrides[get_db] = lambda: test_db_session
    return TestClient(app)

@pytest.fixture
def sample_chest_xray():
    """Synthetic chest X-ray image for testing."""
    img = Image.fromarray(np.random.randint(0, 255, (1024, 1024), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
```

### 40.7 CI Test Execution

```yaml
# .github/workflows/test.yml
- name: Run unit tests
  run: pytest tests/unit/ -v --cov=backend --cov-report=xml

- name: Run integration tests
  run: docker-compose -f docker-compose.test.yml up --abort-on-container-exit
  
- name: Upload coverage
  uses: codecov/codecov-action@v3
```

---

## 41. CI/CD ROADMAP

### 41.1 Pipeline Overview

```mermaid
graph LR
    PR[Pull Request] --> CI[CI Pipeline]
    CI --> LT[Lint]
    CI --> UT[Unit Tests]
    CI --> BD[Docker Build]
    LT --> PA[Pass/Fail]
    UT --> PA
    BD --> PA
    
    PA -->|Pass + Merge to main| CD[CD Pipeline]
    CD --> IT[Integration Tests]
    IT --> ST[Staging Deploy]
    ST --> MQ[Model Quality Check]
    MQ -->|Manual approval| PD[Production Deploy]
```

### 41.2 CI Workflow (Per PR)

**.github/workflows/ci.yml**

```yaml
name: CI Pipeline

on:
  pull_request:
    branches: [main, develop]
  push:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run ruff (Python linting)
        run: ruff check .
      - name: Run black (Python formatting)
        run: black --check .
      - name: Run mypy (type checking)
        run: mypy backend/ inference/ training/
      - name: Run eslint (TypeScript)
        run: cd frontend && npm run lint

  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - name: Run unit tests with coverage
        run: pytest tests/unit/ --cov --cov-fail-under=80

  docker-build:
    runs-on: ubuntu-latest
    steps:
      - name: Build all Docker images
        run: docker-compose build --no-cache
      
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - name: pip-audit
        run: pip-audit -r backend/requirements.txt
      - name: bandit
        run: bandit -r backend/ inference/ -ll
```

### 41.3 GitHub Actions Secrets Required

| Secret | Purpose |
|--------|---------|
| `DOCKERHUB_TOKEN` | Push Docker images |
| `HUGGINGFACE_TOKEN` | Push model artifacts |
| `POSTGRES_PASSWORD` | Integration test DB |

### 41.4 Future CD Pipeline

After staging deployment is established:

1. **Integration tests** run against staging environment
2. **Load test** validates latency SLAs
3. **Manual approval gate** before production
4. **Blue/green deployment** to production
5. **Smoke tests** after production deploy
6. **Automatic rollback** on smoke test failure

---

## 42. DOCUMENTATION STRATEGY

### 42.1 Documentation Layers

| Layer | Location | Audience | Format |
|-------|----------|---------|--------|
| Architecture | `MASTER_ARCHITECTURE.md` | Engineers | Markdown |
| API Reference | `GET /docs` (auto) | Developers | Swagger/OpenAPI |
| Setup Guide | `docs/deployment/local-setup.md` | New contributors | Markdown |
| Training Guide | `docs/deployment/kaggle-training.md` | ML engineers | Markdown |
| Code Comments | Source files | Maintainers | Docstrings |
| Experiment Notes | `training/notebooks/` | ML engineers | Jupyter |
| Change Log | `CHANGELOG.md` | All | Markdown |

### 42.2 Docstring Standard

All public functions and classes use Google-style docstrings:

```python
def predict(self, image: np.ndarray) -> PredictionResult:
    """Run inference on a preprocessed image.
    
    Args:
        image: Preprocessed image array of shape (1, 3, 224, 224) 
               with dtype float32 and ImageNet normalization applied.
    
    Returns:
        PredictionResult containing probabilities and binary predictions
        for all 14 disease classes.
    
    Raises:
        InferenceTimeoutError: If inference server does not respond within
            the configured timeout period.
        InferenceServerUnavailableError: If inference server is unreachable.
    
    Example:
        >>> preprocessor = ImagePreprocessor()
        >>> backend = ONNXRuntimeBackend(model_path="model.onnx")
        >>> image = preprocessor.preprocess(raw_image_bytes)
        >>> result = backend.predict(image)
        >>> print(result.probabilities)
    """
```

### 42.3 README Structure

The top-level README serves as the project's face:

```markdown
# MedVision AI — Production Medical Computer Vision Platform

[Badges: CI, Coverage, License, Docker, Model Version]

## What This Is
## Architecture Overview (Diagram)
## Quick Start (3 commands to run the system)
## Training on Kaggle/Colab
## API Documentation
## Technology Stack
## Project Structure
## Contributing
## License
```

---

## 43. CODING STANDARDS

### 43.1 Python Standards

| Standard | Tool | Configuration |
|----------|------|--------------|
| Formatting | black | line-length = 88 |
| Linting | ruff | E, W, F, I rules |
| Type checking | mypy | strict mode |
| Import sorting | ruff isort | standard |
| Security | bandit | level LOW+ |

### 43.2 TypeScript Standards

| Standard | Tool | Notes |
|----------|------|-------|
| Linting | eslint | react + typescript rules |
| Formatting | prettier | 2-space indent |
| Types | TypeScript strict | no implicit any |

### 43.3 Python Typing Requirements

All public functions must have complete type annotations:

```python
# Required
def preprocess(self, image_bytes: bytes) -> np.ndarray:

# NOT acceptable
def preprocess(self, image_bytes):
```

### 43.4 No Magic Numbers

All constants are named and defined in configuration or constants files:

```python
# WRONG
if image_size > 10485760:

# CORRECT
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
if image_size > MAX_IMAGE_SIZE_BYTES:
```

### 43.5 Commit Message Convention

[Conventional Commits](https://www.conventionalcommits.org/) standard:

```
feat(inference): add ONNX Runtime CPU fallback backend
fix(api): correct image hash collision in cache key generation
docs(architecture): update database schema section
refactor(training): extract metric computation to dedicated module
test(backend): add integration tests for predict endpoint
chore(deps): upgrade onnxruntime to 1.17.0
```

---

## 44. DEPENDENCY MANAGEMENT

### 44.1 Python Dependency Files

```
backend/requirements.txt          ← Production dependencies only
backend/requirements-dev.txt      ← + Testing, linting, type tools
training/requirements.txt         ← Training-specific (torch, timm, albumentations)
inference/requirements.txt        ← Inference-specific (onnxruntime, tritonclient)
```

### 44.2 Dependency Pinning Strategy

**Production:** Pinned to exact versions (`package==1.2.3`)  
**Development:** Pinned to compatible versions (`package>=1.2.0,<2.0.0`)

Rationale: Production deployments must be reproducible. Development allows minor updates for security patches.

### 44.3 Virtual Environment Strategy

```bash
# Python (uv - fast package manager)
uv venv .venv
uv pip install -r requirements.txt

# OR traditional
python -m venv .venv
pip install -r requirements.txt
```

`uv` is recommended for significantly faster dependency resolution and installation.

### 44.4 Frontend Dependencies

```json
// package.json key dependencies
{
  "react": "^18.2.0",
  "typescript": "^5.3.0",
  "vite": "^5.1.0",
  "tailwindcss": "^3.4.0",
  "axios": "^1.6.0",
  "@radix-ui/react-*": "latest"
}
```

### 44.5 Dependency Audit

```yaml
# .github/workflows/security.yml
- name: Audit Python dependencies
  run: pip-audit -r backend/requirements.txt --fail-on-vuln

- name: Audit npm dependencies
  run: cd frontend && npm audit --audit-level=high
```

---

## 45. SCALABILITY STRATEGY

### 45.1 Current Scalability (v1.0 — Docker Compose)

The v1.0 architecture on a single machine supports:

- ~10 concurrent users
- ~5 requests/second (ONNX CPU)
- ~35 requests/second (Triton GPU)

This is sufficient for a portfolio demonstration and small clinical pilots.

### 45.2 Horizontal Scaling Path

```
Current (Single node)
  ↓
Step 1: Scale inference horizontally
  Triton instances: 1 → N (behind load balancer)
  
Step 2: Scale gateway horizontally
  Gateway instances: 1 → N (behind load balancer)
  
Step 3: Scale database
  PostgreSQL → RDS with read replicas
  
Step 4: Kubernetes migration
  Docker Compose → Kubernetes/Helm
```

### 45.3 Stateless Design

Both the Gateway and Triton servers are stateless:
- No local session storage
- No local file dependencies during requests
- State only in PostgreSQL and Redis
- Any instance can serve any request

This is a prerequisite for horizontal scaling.

### 45.4 Database Scaling

| Scale Level | Strategy |
|------------|---------|
| Current | Single PostgreSQL container |
| Step 1 | Connection pooling via PgBouncer |
| Step 2 | Read replicas for analytics queries |
| Step 3 | Managed RDS on AWS |
| Step 4 | Partitioning by date for predictions table |

### 45.5 Cache Scaling

| Scale Level | Strategy |
|------------|---------|
| Current | Single Redis instance |
| Step 1 | Redis Sentinel (HA) |
| Step 2 | Redis Cluster (partitioned) |
| Step 3 | ElastiCache on AWS |

---

## 46. CLOUD MIGRATION PLAN

### 46.1 Migration Phases

**Phase 0 (Current): Local Docker Compose**
- Full stack on developer machine
- All services containerized
- Cost: $0

**Phase 1: Partial Cloud**
- Frontend → Vercel (free tier)
- Backend API → Render / Railway (free tier)
- Model serving → Still local or Kaggle inference API
- Database → Supabase (free tier PostgreSQL)
- Cost: $0

**Phase 2: Full Cloud (Small VM)**
- AWS EC2 t3.medium or GCP e2-medium
- Docker Compose on single cloud VM
- Cost: ~$30-50/month

**Phase 3: Production Cloud (GPU)**
- AWS EC2 g4dn.xlarge (NVIDIA T4 GPU)
- Triton with GPU-optimized TensorRT models
- RDS PostgreSQL, ElastiCache Redis
- Cost: ~$200-400/month

**Phase 4: Kubernetes**
- EKS / GKE cluster
- Auto-scaling inference pods
- Full MLOps pipeline on cloud
- Cost: ~$500-1000/month

### 46.2 Cloud Provider Comparison

| Provider | GPU Tier | Managed ML | Free Tier | Recommendation |
|----------|---------|-----------|-----------|---------------|
| AWS | g4dn.xlarge (T4) | SageMaker | ✅ (compute) | Best ecosystem |
| GCP | n1-standard + T4 | Vertex AI | ✅ ($300 credit) | Best ML tooling |
| Azure | NC6 (K80) | Azure ML | ✅ ($200 credit) | Good for MS stack |
| Lambda Labs | A100 / H100 | None | ❌ | Cheapest GPU |

### 46.3 Terraform Infrastructure (Future)

```
infra/
├── terraform/
│   ├── main.tf
│   ├── variables.tf
│   ├── modules/
│   │   ├── vpc/
│   │   ├── ecs/
│   │   ├── rds/
│   │   └── elasticache/
│   └── environments/
│       ├── staging.tfvars
│       └── production.tfvars
```

---

## 47. FUTURE RESEARCH HOOKS

### 47.1 Architecture Hooks for Future Research

The platform is architected to accommodate future research extensions without restructuring:

| Research Area | Hook Location | Integration Method |
|--------------|--------------|-------------------|
| Uncertainty Estimation | `inference/postprocessing/` | Add MC-Dropout wrapper around backend |
| Confidence Calibration | `mlops/calibration/` | Post-hoc temperature scaling |
| Focal Loss | `training/losses/` | New loss class implementing base |
| Mixup / CutMix | `training/datasets/` | Dataset transform wrapper |
| Semi-supervised Learning | `training/datasets/` | New DataModule subclass |
| Active Learning | `mlops/active_learning/` | Query strategy module |
| Ensemble | `inference/backends/` | EnsembleBackend combining multiple models |
| Segmentation Head | `training/models/` | New task head on shared backbone |
| Detection Head | `training/models/` | FCOS/DETR head attachment |
| Vision Transformer | `training/models/backbones.py` | New timm backbone string |

### 47.2 Research Extension Principles

When adding research features:

1. **Add, don't modify** — New components extend existing interfaces
2. **Config-driven** — New features activated via config flags
3. **Backward compatible** — Base pipeline still works identically
4. **Separately trackable** — MLflow tracks research experiments independently

---

## 48. FUTURE VISION-LANGUAGE EXPANSION

### 48.1 Vision-Language Architecture Hook

The serving layer is designed to accommodate multimodal models:

```
Current (v1.0):
  Image → Backbone → Classifier → Labels

Future (v3.0):
  Image + Text Prompt → VLM Encoder → Decoder → Medical Report
```

### 48.2 Report Generation Integration

Future architecture for radiology report generation:

```mermaid
graph LR
    IMG[Chest X-ray] --> VE[Vision Encoder\ne.g. CLIP ViT]
    TXT[Report Template\nor Prompt] --> TE[Text Encoder\nBERT/LLaMA]
    VE --> CR[Cross-Attention]
    TE --> CR
    CR --> DEC[Text Decoder\nGPT-style]
    DEC --> RPT[Generated\nRadiology Report]
```

### 48.3 Model Support for Future Modalities

| Modality | Dataset | Model Architecture |
|----------|---------|------------------|
| CT Scan | LUNA16 | 3D CNN (ResNet3D) |
| MRI | BraTS | UNet3D |
| Retinal | EyePACS | EfficientNet |
| Histopathology | TCGA | ViT + Patch Embedding |
| Satellite | SpaceNet | DeepLab / SAM |

Each new modality adds a `DataModule` subclass and potentially a new `ModelHead` — the serving infrastructure remains unchanged.

---

## 49. RISK ANALYSIS

### 49.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| ONNX export breaks on complex model | Medium | High | Validate ONNX output ≡ PyTorch output on 100 test inputs |
| Triton not available on CPU machine | High | Medium | ONNX Runtime fallback; dev mode uses ONNX |
| Class imbalance causes poor per-class AUC | High | High | Weighted loss + sampling + per-class threshold tuning |
| Data leakage in train/test split | Medium | Critical | Enforce patient-wise splitting via explicit split files |
| Model overfit to NIH-specific artifacts | Medium | High | Document limitation; future cross-dataset evaluation |
| Grad-CAM focus on spurious correlations | Medium | High | Visual inspection during development; future TCAV |
| Docker Compose fails on Windows | Low | High | Test on Linux VM; document Windows WSL2 requirement |

### 49.2 Project Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Free GPU quota exhaustion | High | Medium | Use both Kaggle + Colab; optimize batch size |
| HF Hub rate limits | Low | Low | Cache model locally; infrequent uploads |
| PostgreSQL migration conflicts | Low | Medium | Test migrations in isolation before applying |
| Triton version incompatibility with ONNX | Low | High | Pin Triton version; test ONNX opset compatibility |

### 49.3 Clinical Risks (Non-Technical)

| Risk | Mitigation |
|------|-----------|
| System used for actual clinical decisions | Clear disclaimer: "For Research and Education Only" |
| Biases in NIH dataset (gender, age, ethnicity) | Document known biases; not suitable for deployment without bias analysis |
| Missing rare conditions | Document that 14-class scope is explicit; no "unknown" class |

---

## 50. DEVELOPMENT ROADMAP

### 50.1 Phase 1: Foundation (Weeks 1–2)

**Goal:** Repository structure and infrastructure baseline.

| Task | Description |
|------|-------------|
| Repo setup | Initialize repository, folder structure, gitignore, README |
| Docker Compose | Base docker-compose.yml with all services defined |
| Database | PostgreSQL schema, Alembic migrations, CRUD layer |
| Config system | Hydra configs for training; pydantic-settings for backend |
| CI skeleton | GitHub Actions with lint + placeholder test |

**Deliverable:** `docker-compose up` brings up all services (without model).

### 50.2 Phase 2: Data & EDA (Week 3)

**Goal:** Complete understanding of the dataset.

| Task | Description |
|------|-------------|
| Dataset download | Script to download and verify NIH ChestX-ray14 |
| EDA notebook | Class distribution, image stats, co-occurrence, sample images |
| DataModule | `NIHChestXrayDataModule` with patient-wise splits |
| Augmentation | Albumentations pipeline with training and validation configs |
| DataLoader validation | Verify batches are correct shape, dtype, and label format |

**Deliverable:** EDA notebook with full analysis; DataLoader producing correct batches.

### 50.3 Phase 3: Training (Weeks 4–5)

**Goal:** Trained model with full experiment tracking.

| Task | Description |
|------|-------------|
| Model architecture | `ChestXrayClassifier` with timm backbone factory |
| Loss functions | Weighted BCE + Focal Loss implementations |
| Trainer | Training loop with mixed precision, callbacks |
| MLflow integration | Full parameter, metric, and artifact logging |
| Training on Kaggle | Run baseline experiment on Kaggle GPU |
| Evaluation | Per-class AUC, F1, confusion matrix, ROC curves |

**Deliverable:** Trained model achieving mean AUC ≥ 0.80. Full MLflow experiment logged.

### 50.4 Phase 4: Export & Optimization (Week 6)

**Goal:** Production-ready model artifacts.

| Task | Description |
|------|-------------|
| ONNX export | Export best checkpoint to ONNX, validate correctness |
| ONNX optimization | Apply onnx-simplifier |
| Benchmark (PyTorch vs ONNX) | Latency comparison on CPU |
| Threshold tuning | Per-class optimal thresholds on validation set |
| HF Hub upload | Push ONNX model + metadata to Hugging Face |

**Deliverable:** ONNX model on HF Hub. Benchmark report showing 4x+ speedup over PyTorch.

### 50.5 Phase 5: Serving (Week 7)

**Goal:** Model served via Triton Inference Server.

| Task | Description |
|------|-------------|
| Triton config | `config.pbtxt` with correct input/output specs |
| Triton Docker | Dockerfile and model repository setup |
| Inference backends | `TritonHTTPBackend` + `ONNXRuntimeBackend` |
| Preprocessing | `ImagePreprocessor` matching training pipeline |
| Postprocessing | `PredictionFormatter` with threshold application |
| Explainability | Grad-CAM implementation for ONNX model |

**Deliverable:** Triton serving predictions via HTTP. Grad-CAM heatmaps generated.

### 50.6 Phase 6: Backend (Week 8)

**Goal:** FastAPI gateway with full feature set.

| Task | Description |
|------|-------------|
| FastAPI app | App factory, middleware, router setup |
| Predict endpoint | Full predict pipeline: validate → cache → infer → explain → persist |
| Health/ready endpoints | Dependency health checks |
| Redis caching | Image hash cache with TTL |
| Database persistence | Store predictions in PostgreSQL |
| Prometheus metrics | All latency, cache, prediction metrics |
| Integration tests | Full API test coverage |

**Deliverable:** FastAPI serving predictions with caching, persistence, and metrics.

### 50.7 Phase 7: Frontend (Week 9)

**Goal:** Clinical-grade React interface.

| Task | Description |
|------|-------------|
| React scaffold | Vite + TypeScript + Tailwind setup |
| Image uploader | Drag-drop + file browser component |
| Results display | All 14 disease cards with probabilities |
| Heatmap viewer | Grad-CAM overlay component |
| Error handling | Loading states, error boundaries |
| API integration | `api.ts` service layer |
| Vercel deploy | Frontend live on Vercel |

**Deliverable:** Live frontend on Vercel connected to API.

### 50.8 Phase 8: Monitoring & Polish (Week 10)

**Goal:** Full observability and production polish.

| Task | Description |
|------|-------------|
| Grafana dashboards | Operational dashboard with all metrics |
| Alert rules | Error rate and latency alerts |
| Structured logging | structlog in all services |
| Documentation | Complete README, deployment guide, API docs |
| Benchmark report | Full backend comparison in docs |
| Security review | Dependency audit, bandit scan, input validation review |
| Final CI/CD | Complete CI pipeline with all checks |

**Deliverable:** Full production platform with monitoring. Repository ready for showcase.

---

### Summary Timeline

```
Week 1–2  ████░░░░░░  Foundation & Infrastructure
Week 3    ████░░░░░░  Dataset & EDA
Week 4–5  ████████░░  Training & MLflow
Week 6    ████░░░░░░  Export & Optimization
Week 7    ████░░░░░░  Serving (Triton)
Week 8    ████░░░░░░  Backend (FastAPI)
Week 9    ████░░░░░░  Frontend (React)
Week 10   ████░░░░░░  Monitoring & Polish
```

**Total: 10 weeks to a complete production AI platform.**

---

## APPENDIX A: Technology Quick Reference

| Tool | Version | Purpose | Docs |
|------|---------|---------|------|
| PyTorch | ≥2.2 | Deep learning | pytorch.org |
| timm | ≥0.9 | Pretrained models | timm.fast.ai |
| Albumentations | ≥1.3 | Augmentation | albumentations.ai |
| ONNX | ≥1.16 | Model format | onnx.ai |
| Triton | ≥24.01 | Model serving | docs.nvidia.com/deeplearning/triton |
| FastAPI | ≥0.110 | API gateway | fastapi.tiangolo.com |
| MLflow | ≥2.11 | Experiment tracking | mlflow.org |
| Prometheus | latest | Metrics | prometheus.io |
| Grafana | latest | Dashboards | grafana.com |

---

## APPENDIX B: NIH ChestX-ray14 Class Labels

| Index | Label | Prevalence |
|-------|-------|-----------|
| 0 | Atelectasis | 10.3% |
| 1 | Cardiomegaly | 2.5% |
| 2 | Effusion | 11.9% |
| 3 | Infiltration | 17.7% |
| 4 | Mass | 5.2% |
| 5 | Nodule | 5.6% |
| 6 | Pneumonia | 1.3% |
| 7 | Pneumothorax | 4.7% |
| 8 | Consolidation | 4.2% |
| 9 | Edema | 2.1% |
| 10 | Emphysema | 2.2% |
| 11 | Fibrosis | 1.5% |
| 12 | Pleural Thickening | 3.0% |
| 13 | Hernia | 0.2% |

---

## APPENDIX C: Glossary

| Term | Definition |
|------|-----------|
| AUC-ROC | Area Under the Receiver Operating Characteristic curve |
| BCE | Binary Cross-Entropy loss |
| DICOM | Digital Imaging and Communications in Medicine (medical image format) |
| Dynamic batching | Combining multiple inference requests into a single GPU batch |
| Grad-CAM | Gradient-weighted Class Activation Mapping |
| ONNX | Open Neural Network Exchange format |
| TensorRT | NVIDIA's inference optimization framework |
| Triton | NVIDIA Triton Inference Server |
| Transfer learning | Using features from a model trained on one task for another |
| Patient-wise split | Data split ensuring no patient appears in multiple sets |
| Multi-label | Single image can have multiple simultaneous class labels |
| p99 latency | 99th percentile latency (slowest 1% of requests) |
| MLflow | Open-source platform for ML lifecycle management |
| CLAHE | Contrast Limited Adaptive Histogram Equalization |

---

*End of MASTER_ARCHITECTURE.md*  
*Document version: 1.0.0*  
*This document should be treated as a living specification — update it as architectural decisions evolve.*
