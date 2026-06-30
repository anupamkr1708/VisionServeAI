# SPRINT 03 — PRODUCTION DATA PIPELINE SPECIFICATION
## VisionServeAI | NIH ChestXray14 Domain Plugin

**Status:** Specification for implementation — no code in this document
**Depends on:** `MASTER_ARCHITECTURE.md` (Phase 0, frozen), `PHASE_01_TRAINING_SPECIFICATION.md` (frozen), `SPRINT_02_DATASET_ENGINEERING_REPORT.md` (completed)
**Produces:** The blueprint for Sprint 03's implementation notebook/modules
**Out of scope:** Backbone architectures, training loop, optimizers, loss functions, MLflow, ONNX, TensorRT, Triton, FastAPI, Docker, cloud, deployment (later sprints/phases)

This specification continues exactly where Sprint 02 ended. Every numerical fact, schema detail, and risk cited below is carried forward from the Sprint 02 report — this document does not re-derive or re-question them. Its job is to specify, in implementation-ready detail, how the validated dataset becomes a verified PyTorch training batch.

---

## 1. Executive Summary

Sprint 02 answered the question "what is actually in this dataset." Sprint 03 answers the question "how does that dataset become something a model can consume, safely and reproducibly." This sprint converts the audited NIH ChestXray14 metadata and the 112,120 verified PNG files into a production-grade PyTorch input pipeline: cleaned metadata, a frozen disease registry, multi-hot encoded labels, a patient-disjoint train/validation/test split, a `Dataset` abstraction, a transform pipeline, and a `DataLoader` configuration — with verification gates at every stage.

Sprint 03 is the connective tissue between Sprint 02 and Sprint 04. Sprint 02 produced *facts about the data*; Sprint 04 will produce *a trained model*. Neither can be skipped to get to the other safely: Sprint 04's training loop is only as trustworthy as the batches Sprint 03 hands it, and Sprint 03's design decisions are only justified because of what Sprint 02 measured (the imbalance profile, the patient long tail, the nested-folder layout, the integrity audit result). No model code, loss function, or training loop is introduced here — this sprint ends at "a verified, inspectable training batch," not at a trained weight.

---

## 2. Sprint Goals

### Completed objectives (target state for this sprint)
- Canonical, validated metadata schema with documented cleaning rules.
- A frozen, deterministically-ordered 14-class disease registry.
- A multi-hot label encoder/decoder with defined edge-case behavior.
- A patient-wise train/validation/test split with an automated, non-bypassable leakage check.
- A PyTorch `Dataset` abstraction over the cleaned metadata, registry, and split.
- Separate, explicitly scoped transform pipelines for training, validation, and inference.
- A configured `DataLoader` for each split.
- A pipeline verification suite that checks schema, images, labels, batches, and patient leakage end-to-end.
- A visual inspection step confirming a real, transformed batch looks correct.

### Out-of-scope objectives
- Any backbone architecture (DenseNet121, ConvNeXt, ViT, or otherwise).
- Any training loop, optimizer, scheduler, or loss function.
- MLflow experiment tracking, ONNX export, TensorRT, Triton.
- FastAPI, Docker, or any cloud/deployment concern.
- Bounding-box data usage (still deferred, per Sprint 02 Section 17).

### Deliverables
See Section 17 for the complete artifact list.

### Acceptance criteria
See Section 18. Sprint 04 does not begin until every item there is satisfied.

---

## 3. Pipeline Architecture

The pipeline is a strict, one-directional sequence of stages. Each stage consumes only the previous stage's validated output — never reaches back to raw inputs — which is what makes the pipeline testable stage-by-stage rather than only end-to-end.

```
Raw Metadata CSV               Raw PNG Files (112,120, on disk, Sprint 02-verified)
(Data_Entry_2017.csv)                 │
        │                             │
        ▼                             │
[Stage 1] Metadata Validation         │
  schema check, row-count check       │
        │                             │
        ▼                             │
[Stage 2] Metadata Cleaning           │
  drop Unnamed: 11, rename            │
  fragmented columns                  │
        │                             │
        ▼                             │
   Canonical Metadata (v1)            │
        │                             │
        ▼                             │
[Stage 3] Disease Registry            │
  discover + freeze 14-class order    │
        │                             │
        ▼                             │
[Stage 4] Label Encoding              │
  Finding Labels → 14-dim multi-hot   │
        │                             │
        ▼                             │
[Stage 5] Patient-wise Split          │
  train_val_list.txt → train/val      │
  test_list.txt → test (untouched)    │
  leakage assertion (hard gate)       │
        │                             │
        ▼                             │
   Split Manifests (frozen, hashed)   │
        │                             │
        ├─────────────────────────────┤
        ▼                             ▼
[Stage 6] Dataset Construction  ◄──────┘
  (per split: metadata rows + registry
   + image-root path → indexable samples)
        │
        ▼
[Stage 7] Transform Pipeline
  train: mandatory + augmentation
  val/test: mandatory only
        │
        ▼
[Stage 8] DataLoader
  batching, shuffling, workers
        │
        ▼
[Stage 9] Pipeline Verification
  schema / image / label / batch /
  leakage / visual checks
        │
        ▼
   Verified Training Batch
```

### Stage responsibilities

| Stage | Responsibility | Section |
|---|---|---|
| 1–2 | Turn the raw, audited CSV into one canonical, versioned metadata artifact | 4 |
| 3 | Establish the one fixed mapping from disease name to label index, forever | 5 |
| 4 | Turn `Finding Labels` strings into model-consumable multi-hot vectors | 6 |
| 5 | Produce three patient-disjoint sample lists with a proof they don't overlap | 7 |
| 6 | Wrap (image path, label vector) pairs in the PyTorch `Dataset` contract | 8 |
| 7 | Apply the correct, split-appropriate image transform | 9–10 |
| 8 | Batch samples efficiently for the target hardware (Kaggle/Colab free-tier GPU) | 11 |
| 9 | Prove the previous eight stages did what they claim, before any training code touches the output | 12 |

---

## 4. Metadata Pipeline

### 4.1 Canonical schema
Sprint 02 audited the raw 12-column schema and identified exactly one column with zero information content (`Unnamed: 11`, 112,120/112,120 null) and two conceptually-single fields that were split into awkwardly-named column pairs by a CSV parsing artifact (`OriginalImage[Width` / `Height]`, `OriginalImagePixelSpacing[x` / `y]`). Sprint 03 resolves both, producing an 11-column canonical schema:

| Canonical column | Source column | Action |
|---|---|---|
| `image_index` | `Image Index` | kept, renamed to snake_case |
| `finding_labels` | `Finding Labels` | kept, renamed |
| `follow_up_number` | `Follow-up #` | kept, renamed |
| `patient_id` | `Patient ID` | kept, renamed |
| `patient_age` | `Patient Age` | kept, renamed |
| `patient_gender` | `Patient Gender` | kept, renamed |
| `view_position` | `View Position` | kept, renamed |
| `original_image_width` | `OriginalImage[Width` | kept, renamed (fragment merged conceptually) |
| `original_image_height` | `Height]` | kept, renamed |
| `original_pixel_spacing_x` | `OriginalImagePixelSpacing[x` | kept, renamed |
| `original_pixel_spacing_y` | `y]` | kept, renamed |
| — | `Unnamed: 11` | **dropped** — 100% null, zero information, confirmed in Sprint 02 |

Renaming to `snake_case` is a deliberate, low-risk cleanup: it removes the bracket-fragment naming defect without discarding any data, and gives every downstream module (encoder, dataset, future domain plugins) a stable, predictable column-naming convention to code against instead of the raw NIH header text.

### 4.2 Metadata validation rules (enforced at load time, before cleaning)
- Row count must equal **112,120** (the Sprint 02-verified figure). A different count means the dataset version on disk has drifted from what this pipeline was designed against — **hard failure**, not a warning.
- All 12 raw columns must be present with the expected raw names, prior to renaming.
- `Image Index` must be non-null and follow the expected filename pattern (`########_###.png`).
- `Finding Labels` must be non-null and non-empty for every row (consistent with Sprint 02's finding of zero missing labels).
- `Patient ID` must be a positive integer for every row.
- `Patient Age` is validated for **type** (integer) but not for **plausibility** — Sprint 02 already documented implausible age outliers, and age is a reserved-for-future column, not a model input (Phase 1, Problem Definition). Gating pipeline construction on a column the model never consumes would be over-engineering; the outlier is simply not corrected.

### 4.3 Dropped columns
Only `Unnamed: 11`. This is the single column Sprint 02 explicitly flagged as a parsing artifact with no information (Section 5.3 of the Sprint 02 report) — dropping it now, in a dedicated cleaning stage, is exactly the deferred action that sprint promised.

### 4.4 Future metadata support
The canonical schema is intentionally a superset of what the v1 baseline model consumes — `original_image_width/height` and `original_pixel_spacing_x/y` are retained (renamed, not dropped) because a future domain plugin or a future research extension (e.g., physical-size-aware preprocessing) may need them. This mirrors the platform's domain-agnostic design principle: the canonical-metadata contract should outlive any one model's input requirements.

### 4.5 Schema versioning
The canonical metadata is persisted as a versioned artifact: `canonical_metadata_v1.csv` (or `.parquet`), accompanied by `schema_manifest_v1.json` listing every canonical column name, dtype, and nullability constraint. Any future change to the schema (a new NIH metadata release, a different domain plugin's metadata format) produces `v2`, not an in-place mutation of `v1`. This is the same versioning discipline the training specification already requires for split files and model exports (Phase 1, Sections 13.1 and 14.1) — applied here one stage earlier, at the metadata layer, where it is cheapest to enforce.

---

## 5. Disease Registry

### 5.1 Canonical disease ordering
The 14 pathology classes, in the dataset's natural alphabetical order, are frozen as the canonical registry:

```
0  Atelectasis
1  Cardiomegaly
2  Consolidation
3  Edema
4  Effusion
5  Emphysema
6  Fibrosis
7  Hernia
8  Infiltration
9  Mass
10 Nodule
11 Pleural_Thickening
12 Pneumonia
13 Pneumothorax
```

`No Finding` is **not** assigned an index — consistent with the training specification's decision (Phase 1, Section 2) to treat it as the derived absence of all 14 positive findings, not a 15th trained class.

### 5.2 Disease discovery (verification, not re-derivation)
The registry is not "discovered" fresh in Sprint 03 — it is the same 14-name set Sprint 02 already enumerated via the `Counter` over `Finding Labels` tokens (Sprint 02 report, Section 8.3). Sprint 03's discovery step exists purely as a **regression check**: parse `finding_labels` again, collect the unique non-`"No Finding"` tokens, and assert the resulting set is exactly these 14 names, no more, no fewer. If a 15th name appeared, that would indicate a dataset version change since Sprint 02 — a hard failure, surfaced before any encoding happens.

### 5.3 Registry as a frozen artifact
The registry is persisted once as `disease_registry_v1.json`: an ordered list plus an index map, with a `schema_version` field. It is loaded by every downstream component (encoder, decoder, model export bundle) — never redefined inline in multiple places.

### 5.4 Encoding policy
Index assignment is **append-only and write-once**. Once `disease_registry_v1.json` exists, its existing index assignments are never reordered or reused, even if a future dataset version adds a 15th disease — that would become index 14 in a `v2` registry, with the first 14 indices left untouched, not a re-sort of all 15 names alphabetically.

### 5.5 Why deterministic ordering is mandatory
A model's output is a 14-dimensional vector with no inherent labels attached — index 7 means "Hernia" only because the registry says so. If the registry were rebuilt (e.g., re-running `sorted(set(...))` fresh) at a different point with even slightly different input data, an index could silently shift meaning between training and inference, or between two model versions, without any error being raised. This is precisely the risk the training specification's `labels.json` export artifact (Phase 1, Section 14.1) exists to prevent at export time — Sprint 03 is where that frozen ordering is actually established, one sprint earlier than where it's finally packaged for export.

### 5.6 Future extensibility
A future domain plugin (industrial inspection, retinal imaging, etc.) defines its own registry following the same contract (ordered list, frozen indices, versioned JSON artifact) — the encoder/decoder logic in Section 6 depends only on "a registry object," not on anything ChestXray14-specific, so it transfers unchanged.

---

## 6. Label Engineering

### 6.1 Multi-hot encoding
For a given row's `finding_labels` string:
1. If the string is exactly `"No Finding"`, the encoded vector is all zeros (length 14).
2. Otherwise, split on `|`, deduplicate the resulting tokens (6.4), look up each token's index in the frozen registry, and set that position to 1 in a 14-length zero vector.

This is a pure function of `(finding_labels, registry) → 14-dim vector` — it has no dependency on split assignment, image data, or anything else, which is what makes it independently unit-testable (Section 15).

### 6.2 Label decoder
The inverse operation takes a 14-dim vector (either ground-truth, 0/1, or model output passed through per-class thresholds — thresholds are a Phase-1/Sprint-04 concern, not produced here) and returns the list of disease names whose position is set. If every position is unset, the decoder returns the literal string `"No Finding"` — making round-trip encode→decode→encode idempotent and exactly reversible for every ground-truth label observed in Sprint 02.

### 6.3 Validation
The encoder/decoder pair is validated against the dataset itself, not just synthetic examples: for a sample of real `finding_labels` values (including the rarest combinations, e.g., 9-disease rows identified in Sprint 02 Section 9), `decode(encode(x)) == sorted(x.split("|"))` must hold exactly.

### 6.4 Edge cases
| Case | Behavior | Rationale |
|---|---|---|
| `No Finding` | All-zero vector | Consistent with Phase 1's decision not to train a 15th class |
| Unknown disease token (not in registry) | **Hard failure** (raised exception, pipeline halts) | An unrecognized token means the data violates the exact 14-class schema Sprint 02 verified — silently dropping it would hide a real upstream data problem, not handle a benign edge case |
| Duplicate token within one row (e.g., `"Mass|Mass"`) | Deduplicated before encoding (idempotent — multi-hot is set membership, not a count) | Not observed in Sprint 02's data, but the encoder is defensive against malformed input rather than assuming it can't occur |
| Empty / null `finding_labels` | **Hard failure** | Sprint 02 confirmed zero missing labels in the current dataset; a null value appearing would indicate the metadata-validation stage (Section 4.2) was bypassed |

### 6.5 Testing strategy
Unit tests cover: every individual disease name encodes to a vector with exactly one set bit at its registry index; `"No Finding"` encodes to all-zero; a known multi-disease combination from Sprint 02's sampled combinations (Section 8 of the Sprint 02 report) encodes and decodes correctly; an injected unknown token raises the expected exception; an injected duplicate token does not produce an out-of-range or incorrect vector.

---

## 7. Patient-wise Data Splitting

### 7.1 Algorithm
Directly implementing the procedure the training specification already justified (Phase 1, Section 3.3) and Sprint 02 already provided the statistical evidence for (median 1 image/patient, max 184, Sprint 02 Section 7):

1. Load `test_list.txt` — these images, and the patients they belong to, are the **held-out test set**, untouched by any further splitting logic.
2. Load `train_val_list.txt` — extract the unique `patient_id` values present.
3. Build a coarse per-patient label-prevalence signature (does this patient ever have each of the 14 diseases, across any of their images).
4. Split the `train_val` patient set 80/20 into train and validation, stratified by that signature.
5. Assign every image belonging to a train-split patient to train, and every image belonging to a val-split patient to validation.

### 7.2 Split ratios
- **Test:** fixed by NIH's official `test_list.txt` — approximate size ~25,596 images per the file's documented purpose (to be confirmed exactly when the file is loaded in this sprint; only its *existence* was verified in Sprint 02, not its row count).
- **Train / Validation:** 80% / 20% of the `train_val_list.txt` patient pool.

### 7.3 Random seed policy
A single fixed seed (consistent with the seed already used for reproducible sampling in Sprint 02's notebook, `random_state=42`) governs the stratified patient split. The seed is recorded in the split manifest (7.6) so the exact split can be regenerated byte-for-byte if ever needed — but regeneration is a recovery procedure, not a normal operation; the split is treated as a frozen artifact once produced (7.6).

### 7.4 Leakage prevention
**Hard, automated, non-bypassable check:** after assignment, the three resulting `patient_id` sets (train, val, test) must have **zero pairwise intersection**. This check runs every time the split is constructed or loaded — not once, manually, at creation time — so a future code change that accidentally reintroduces leakage is caught immediately rather than discovered after a suspiciously high validation score.

### 7.5 Verification rules / quality checks
- Total image count across the three splits must equal **112,120** (the Sprint 02-verified total) — a discrepancy means the split logic dropped or duplicated rows.
- Per-split disease prevalence is computed and compared against the Sprint 02 global prevalence table (Section 8.3 of that report) as a sanity check — large deviations (a class accidentally concentrated almost entirely in one split) are flagged for review, though some deviation is expected and tolerated given that splitting is patient-wise, not image-wise.
- Patient ID overlap check (7.4) must report zero before the split is considered usable by any downstream stage.

### 7.6 Future reproducibility considerations
The split is persisted exactly once as three static files (e.g., `train_patient_ids.json`, `val_patient_ids.json`, plus the test list derived directly from NIH's file) along with a `split_manifest_v1.json` recording the seed, the stratification method, the source file hashes, and the resulting per-split counts. Later sprints **load** this manifest; they do not regenerate the split by re-running the random stratification. This is the same "frozen artifact, not a re-derived one" discipline already applied to the disease registry (Section 5.3) and is what allows the training specification's MLflow logging contract (Phase 1, Section 13.1: "dataset split file hashes") to refer to something concrete and stable.

---

## 8. Dataset Class Design

(Specification only — no code.)

### 8.1 Responsibilities
A single `Dataset` abstraction, parameterized by which split it represents, is responsible for exactly one thing: given an index, return one fully-prepared sample. It does not know how to split patients, encode labels from scratch, or apply augmentation policy decisions — those are the responsibilities of Sections 5–7 and 9–10, injected into the dataset at construction time.

### 8.2 Constructor inputs
- The canonical metadata (Section 4), already cleaned.
- The frozen disease registry (Section 5).
- The patient ID list for the split this instance represents (Section 7).
- The image root path (the same canonical `PROJECT_ROOT` established during Sprint 02's filesystem discovery, respecting the nested `images_NNN/images/` layout).
- A transform callable (Section 10) — the dataset applies whatever transform it's given; it does not choose between train/val/inference transforms itself.

At construction time, the dataset filters the canonical metadata down to only the rows whose `patient_id` is in the supplied split list, and resolves each `image_index` to a concrete file path. Given Sprint 02's confirmed zero-missing/zero-orphan integrity result, this resolution is expected to succeed for every row — but the constructor still performs the resolution explicitly rather than assuming it, so that a future dataset-version drift would be caught here rather than at first access.

### 8.3 Returned sample format
A dictionary (not a bare tuple) per sample, containing at minimum:
- `image`: the transformed image tensor.
- `label`: the 14-dim multi-hot float tensor (Section 6).
- `image_id`: the original `image_index` string, retained for traceability and for the GradCAM-based failure analysis the training specification already requires (Phase 1, Section 11.6).
- `patient_id`: retained for any future patient-level aggregation or debugging, not consumed by the model itself.

A dictionary return type is chosen over a bare tuple because it is self-describing at every call site (`sample["label"]` vs. an easily-miscounted `sample[1]`) and because it can absorb additional fields later (e.g., per-class threshold-ready metadata in Sprint 04+) without breaking every existing call site that unpacks a fixed-length tuple.

### 8.4 Error handling
Per Section 8.2, file resolution happens at construction, not at `__getitem__` time. This means `__getitem__` can assume the index is always valid and the file always exists — it does not need defensive per-call error handling for the missing-file case. A defensive corrupted-image check (the file exists but fails to decode) is still performed at construction time by attempting to verify each image once (mirroring Sprint 02's integrity-audit pattern); any image failing this check is excluded from the split's usable index list and logged (Section 13/14), never silently replaced with a placeholder tensor.

### 8.5 Caching strategy
**No in-memory image caching.** At 112,120 images, even decoded-and-resized tensors would not comfortably fit in the 8GB local RAM ceiling stated in the project's hardware constraints, and on Kaggle/Colab the per-epoch decode cost is acceptable without it. What *is* cached, cheaply, is the lightweight per-sample lookup structure (file path, encoded label vector) — this is a few megabytes at most (comparable to the 10.3 MB metadata footprint Sprint 02 measured), not the image bytes themselves.

### 8.6 Future extensibility
The dataset class depends only on a generic contract — "a row with an image path and a label vector" — not on anything ChestXray14-specific. A future domain plugin supplies its own metadata-cleaning stage (Section 4) and its own disease/defect registry (Section 5) but can reuse this dataset class unmodified, consistent with the platform's domain-agnostic design goal stated in the master architecture.

---

## 9. Image Loading Strategy

| Aspect | Decision | Rationale |
|---|---|---|
| Loading library | `PIL.Image` | Already used and validated in Sprint 02's exploration; no need to introduce a second image library |
| Color conversion | Open and force-convert to single-channel grayscale (`"L"` mode) | Source files are grayscale; an explicit conversion guards against any file that happens to carry an unexpected mode, rather than trusting the source file's stored mode implicitly |
| Grayscale-to-3-channel policy | Channel replication happens in the **transform pipeline** (Section 10), not at load time | Keeps the loaded/cached representation as small as possible; replication is a cheap, deterministic tensor operation applied only once a sample is actually being transformed for model input |
| Resize policy | 224×224, bilinear interpolation | Matches the training specification (Phase 1, Section 5) exactly — this sprint does not re-decide preprocessing, it implements it |
| Normalization policy | ImageNet mean/std, applied after resize and channel replication | Same source as above |
| Error handling | Construction-time verification (Section 8.4); any image that fails to open or decode is excluded from the split's index and logged with a reason code | Never substitute a blank/zero tensor for a failed load — doing so would inject a fake, mislabeled-looking negative sample |
| Corrupted image policy | Identical to Sprint 02's and Phase 1's policy: skip and log, do not attempt repair | Consistency with already-established project policy; repairing a corrupted medical image file is out of scope and risky |
| Future optimization considerations | Pre-resizing the corpus to a smaller on-disk cache (e.g., a 224×224 pre-resized mirror) is a viable future optimization if per-epoch I/O becomes a bottleneck on free-tier compute | Explicitly **deferred** — Sprint 02's resolution audit already confirmed a consistent 1024×1024 source size on a 2,000-image sample, so there is no immediate evidence this is needed yet |

---

## 10. Transform Pipeline

### 10.1 Three distinct pipelines, not one configurable pipeline with flags
**Training**, **validation**, and **inference** each get their own explicit transform composition, rather than one shared pipeline toggled by a boolean. This is a deliberate redundancy: it makes it structurally impossible to accidentally leave an augmentation flag on for validation, because the validation pipeline simply never references the augmentation steps in the first place.

| Pipeline | Steps |
|---|---|
| **Training** | Resize (224×224, bilinear) → grayscale-to-3-channel replication → mandatory augmentations (±10° rotation, brightness/contrast jitter, per Phase 1 Section 6.2) → horizontal flip (p=0.5, flagged per Phase 1 Section 6.5) → ImageNet normalization |
| **Validation** | Resize (224×224, bilinear) → grayscale-to-3-channel replication → ImageNet normalization. **No randomness.** |
| **Inference** | Identical, step-for-step, to the validation pipeline | 

### 10.2 Mandatory preprocessing vs. augmentation
- **Mandatory** (applied in every pipeline, train/val/inference alike): resize, channel replication, normalization. These exist to put the image into the numeric form the model architecture requires — they are not a modeling choice, they are a contract.
- **Augmentation** (training only): rotation, brightness/contrast jitter, horizontal flip. These exist to improve generalization and are not part of the model's input contract — removing them changes training dynamics, not what a valid input tensor looks like.

### 10.3 Why augmentation belongs only to training
Validation and test metrics must be reproducible and comparable across runs and across model versions. If validation transforms were stochastic, two evaluations of the *same* checkpoint on the *same* validation set could report different AUROC values purely from transform randomness — making it impossible to tell whether a metric change between two experiments reflects a real model difference or just transform noise. Inference must additionally match validation byte-for-byte, because the per-class thresholds the training specification requires (Phase 1, Section 11.3) are optimized against validation-transform outputs specifically — applying a different transform at inference time would silently invalidate those thresholds.

### 10.4 Forbidden/deferred augmentations carried forward unchanged
Vertical flip, aggressive color/hue jitter, and elastic deformation remain forbidden; CLAHE, random crop/zoom, Gaussian noise, cutout, mixup/cutmix remain deferred with their original switch criteria (Phase 1, Sections 6.3–6.4). Sprint 03 does not revisit these decisions — it implements the mandatory/optional split exactly as already specified.

---

## 11. DataLoader Design

| Aspect | Decision | Rationale |
|---|---|---|
| Batch size | Default 32, exposed as a config parameter | Sized to fit comfortably on a free-tier T4/P100 at 224×224×3, consistent with the training specification's optimizer/mechanics section (Phase 1, Section 10) |
| Shuffle policy | `True` for training, `False` for validation and test | Training benefits from shuffled batch composition; validation/test must be order-stable for reproducible, comparable evaluation runs |
| Worker configuration | Small worker count (e.g., 2), exposed as a config parameter | Free-tier Kaggle/Colab CPU core counts are limited; over-provisioning workers competes with the main process for the same scarce cores rather than helping |
| Pin memory | `True` when a CUDA device is available, `False` otherwise | Standard practice that speeds up host-to-device transfer; conditional on hardware so the same config works on the CPU-only local Windows machine too |
| Persistent workers | `True` whenever `num_workers > 0` | Avoids respawning the worker pool every epoch, which is otherwise a fixed per-epoch cost paid for no benefit |
| Prefetching | Left at the library default, exposed as a config parameter rather than hand-tuned | Avoids premature optimization; revisit only if profiling on actual free-tier hardware shows it matters |
| `drop_last` | `True` for training, `False` for validation/test | A small, irregular final training batch can destabilize batch-statistics-sensitive layers; for evaluation every sample must be counted, so the final partial batch is kept |
| Determinism | Seed governs shuffling and any worker-level augmentation RNG via an explicit worker-init function | Exact bitwise reproducibility across multi-worker data loading is **not** guaranteed by PyTorch by default and is treated here as best-effort, not a hard requirement — the leakage and content correctness checks (Section 12) are the hard requirements; reproducible *randomness* is a secondary, best-effort goal |
| Future distributed training support | DataLoader construction is specified as a factory function parameterized by sampler, so a future multi-GPU `DistributedSampler` can be substituted without altering the `Dataset` class itself | Keeps today's single-GPU design from blocking a later scale-up, without building distributed-training support now (avoiding the project's stated anti-overengineering principle) |

---

## 12. Pipeline Verification

Every stage in Section 3's architecture has a corresponding, automated verification step. None of these are one-off manual notebook checks (as Sprint 02's were, appropriately, for an EDA sprint) — in Sprint 03 they are checks the pipeline itself runs every time it is constructed.

| Verification | What it checks | Failure behavior |
|---|---|---|
| **Schema validation** | Canonical metadata has the expected 11 columns, expected dtypes, expected row count (112,120) | Hard failure |
| **Image verification** | Every `image_index` referenced by a split resolves to a real, decodable file under the canonical image root | Excluded + logged if isolated; hard failure if the exclusion rate is anomalously high (suggesting a systemic problem, not isolated bad files) |
| **Label verification** | Every encoded label vector round-trips through the decoder (Section 6.3) to the original disease set; aggregate positive-bit count across an entire split matches the expected total from Sprint 02's verified disease-frequency table (Section 8.3 of that report) as a regression check | Hard failure |
| **Batch verification** | Batch image tensor shape is `[batch_size, 3, 224, 224]`, label tensor shape is `[batch_size, 14]`, dtypes are the expected float type, no `NaN`/`Inf` values present | Hard failure |
| **Tensor value-range verification** | Post-normalization pixel values fall within the expected band implied by ImageNet normalization statistics | Flagged for review (soft check — wide but not unbounded valid range) |
| **Patient leakage verification** | Zero `patient_id` intersection across train/val/test (Section 7.4), re-checked at pipeline construction time, not just once at split-creation time | Hard failure |
| **Visual inspection** | A real, fully-transformed training batch is denormalized and rendered as a sample grid, analogous to Sprint 02's 5-image spot-check (Section 10 of that report) but now *post-augmentation* — confirming rotation/jitter/flip produce anatomically plausible, non-corrupted images | Manual review; not an automated gate, but a required deliverable (Section 17) |

**Acceptance tie-in:** Section 18's acceptance criteria require every hard-failure-class check above to pass cleanly at least once, with the visual inspection artifact produced and reviewed, before Sprint 04 may begin.

---

## 13. Error Handling Strategy

| Scenario | Response |
|---|---|
| Missing image file (metadata references a filename absent from disk) | Excluded from the usable index at `Dataset` construction time, logged with the `image_index` and reason code. Given Sprint 02's confirmed zero-missing-file result, this is expected to be a no-op in practice — but the handling path must exist defensively, since dataset mirrors can drift between sprints. |
| Corrupted image (file exists, fails to decode) | Same as above — excluded and logged, never substituted with a placeholder tensor. |
| Unknown disease label (a token in `finding_labels` not present in the frozen registry) | **Hard failure**, pipeline construction halts. This is treated as a data-integrity violation, not a recoverable per-row issue — it indicates the dataset no longer matches the schema this entire pipeline was designed and verified against. |
| Missing metadata (an `image_index` exists on disk but has no corresponding row, or a required field is null) | Excluded from the usable set, logged. (Sprint 02's integrity audit found zero orphan files and zero missing required values, so this is also an expected no-op, handled defensively.) |
| Invalid patient (a `patient_id` appearing in metadata but absent from all three split manifests, e.g., due to a split-construction bug) | **Hard failure** at split-loading time — every patient in the canonical metadata must be accounted for in exactly one split; an unassigned patient is a split-construction defect, not a data quality issue to quietly work around. |
| Unexpected file format (a non-PNG file matched by a loosely-specified glob pattern) | Image discovery is restricted to the `*.png` extension exactly as established in Sprint 02 (`rglob("*.png")`); any other extension is never considered a candidate image and requires no special handling. |
| Future dataset expansion (a new image shard, a new metadata row format, additional diseases) | Handled at the schema-versioning layer (Section 4.5) and the registry's append-only policy (Section 5.4) — not by ad hoc exceptions scattered through the pipeline code. |

---

## 14. Logging Strategy

Logging exists to make failures actionable, not to produce a record of every operation. Per-image logging at this dataset's scale (112,120 images) would flood any log output and bury the signal that actually matters.

- **INFO level:** stage-level progress and summary counts — e.g., "canonical metadata loaded: 112,120 rows, 11 columns," "disease registry verified: 14 classes," "split constructed: train=N, val=N, test=N, patient overlap=0."
- **WARNING level:** individual exclusions (a specific missing or corrupted image, with its `image_index` and reason) — these are genuinely actionable per-item, and given Sprint 02's clean integrity result, the expected volume is zero or near-zero, so logging them individually is cheap and useful rather than noisy.
- **ERROR level:** any hard-failure-class event from Sections 12–13 (schema mismatch, unknown label, patient-leakage detection, unassigned patient) — these halt the pipeline and must be impossible to miss in any log review.
- **Explicitly not logged at INFO/WARNING volume:** per-sample `__getitem__` calls, per-batch `DataLoader` iteration events, or per-epoch-equivalent verification passes during normal operation — these belong in metrics/progress bars, not the log stream, once the pipeline is operating normally.

---

## 15. Testing Strategy

(Specification of test coverage — no implementation.)

| Component | Test type | Coverage |
|---|---|---|
| Metadata pipeline | Unit | Schema validation rejects a wrong row count / wrong column set; cleaning correctly drops `Unnamed: 11` and renames the fragmented columns; round-trips a known sample of rows unchanged in content |
| Disease registry | Unit | Discovered token set equals exactly the 14 known names; registry ordering is stable across repeated construction from the same input; loading a persisted `disease_registry_v1.json` reproduces the same mapping |
| Label encoder/decoder | Unit | Every individual disease encodes correctly; `No Finding` encodes to all-zero; known multi-disease rows from Sprint 02's sampled combinations round-trip correctly; unknown token raises; duplicate token does not corrupt the vector |
| Patient split | Unit + integration | Zero patient overlap across all three splits; total image count across splits equals 112,120; re-loading a persisted split manifest reproduces the identical patient assignment given the recorded seed |
| Dataset class | Unit | `__len__` matches the filtered row count for a given split; `__getitem__` returns the documented dictionary shape and tensor dtypes for a sample of indices, including boundary indices (first/last) |
| Transform pipelines | Unit | Validation/inference transforms are deterministic (identical output across repeated calls on the same input); training transform output shape matches validation transform output shape despite the added randomness |
| DataLoader | Integration | A full batch from each split has the expected tensor shapes and dtypes, contains no `NaN`/`Inf`, and (for training) reflects shuffling across two separate epoch iterations |
| End-to-end pipeline | Integration | Starting from the raw CSV and raw image root, producing one verified batch from each split, with all Section 12 verification checks passing |

---

## 16. Production Considerations

**Reproducibility.** Every stage that involves a choice with randomness (the patient split, training-time augmentation) is seeded, and the seed is recorded in a persisted manifest rather than left as an in-memory default. The disease registry and split assignments are frozen, versioned artifacts (Sections 5.3, 7.6), not values re-derived by re-running exploratory code.

**Maintainability.** All pipeline parameters (paths, image size, batch size, split ratio, seed, worker count) live in one configuration object rather than as scattered magic numbers/constants across notebook cells — consistent with the project's stated principle of moving from exploratory notebook code toward maintainable modules.

**Scalability.** The dataset/transform/dataloader design does not assume in-memory image caching (Section 8.5) and does not assume single-GPU-only operation (Section 11's factory-function note) — both decisions trade a small amount of present-day simplicity for avoiding a rewrite when the project scales past the current free-tier hardware constraints.

**Notebook-to-package migration.** Sprint 03's implementation is structured so each specification section maps to one future Python module: a metadata module (Section 4), a registry/label module (Sections 5–6), a splitting module (Section 7), a dataset module (Section 8), a transforms module (Sections 9–10), and a dataloader/verification module (Sections 11–12). Writing Sprint 03 as notebook cells that already respect this module boundary (rather than one long undifferentiated script) is what makes the eventual extraction into a real Python package mechanical rather than a redesign.

**Configuration management.** A single versioned config (e.g., `pipeline_config_v1.json/yaml`) captures every tunable parameter referenced throughout this document, so that re-running the pipeline against a different image size, batch size, or split ratio is a config change, not a code change.

**Future cloud compatibility.** Because Sprint 02 already established the practice of deriving every path from one canonical root variable (rather than hardcoding `/kaggle/input/...` throughout), swapping that root for a cloud object-storage-backed path later is a one-line change, not a refactor.

---

## 17. Deliverables

- `canonical_metadata_v1.csv` (or `.parquet`) — cleaned, schema-validated metadata.
- `schema_manifest_v1.json` — canonical column names, types, version.
- `disease_registry_v1.json` — frozen, ordered 14-class disease registry.
- Label encoder/decoder module (specification implemented as code in this sprint).
- `train_patient_ids.json`, `val_patient_ids.json` (derived patients), test patient set (derived from NIH's `test_list.txt`).
- `split_manifest_v1.json` — seed, stratification method, source file hashes, per-split counts, leakage-check result.
- `Dataset` class module, parameterized by split.
- Transform pipeline module (training / validation / inference compositions).
- `DataLoader` factory module/configuration.
- `pipeline_config_v1.json` (or `.yaml`) — all tunable pipeline parameters.
- Pipeline verification report (output of Section 12's checks, run end-to-end).
- Visual batch inspection artifact (rendered sample grid, post-transform, per split).
- Unit/integration test suite covering Section 15's specified coverage.

---

## 18. Acceptance Criteria

Sprint 04 does not begin until every item below is satisfied:

- [ ] Canonical metadata produced, schema-validated, with `Unnamed: 11` dropped and fragmented columns renamed; row count confirmed at 112,120.
- [ ] Disease registry frozen as a versioned artifact; the 14-name set verified to exactly match Sprint 02's findings, with no unexpected additions or omissions.
- [ ] Label encoder/decoder implemented and unit-tested, including the edge cases in Section 6.4.
- [ ] Patient-wise split constructed and persisted as a versioned manifest; automated leakage check passes with **zero** patient overlap across train/val/test; total image count across splits equals 112,120.
- [ ] `Dataset` class implemented for all three splits, returning the documented sample dictionary shape with correct dtypes.
- [ ] Training, validation, and inference transform pipelines implemented as specified, with validation and inference confirmed identical and deterministic.
- [ ] `DataLoader` constructed for all three splits with the specified configuration (shuffle, workers, pin memory, persistent workers, `drop_last` policy).
- [ ] All hard-failure-class verification checks in Section 12 pass cleanly at least once end-to-end.
- [ ] Visual inspection artifact produced and manually reviewed for anatomical plausibility post-augmentation.
- [ ] Unit and integration tests specified in Section 15 implemented and passing.
- [ ] No backbone, loss function, optimizer, training loop, or experiment-tracking code introduced — scope boundary respected throughout.

---

## 19. Lessons Expected

Sprint 03 is designed to teach, in order of emphasis:

**Data engineering discipline.** Schema validation, versioned artifacts, and append-only registries are general data-engineering practices, not PyTorch-specific ones — this sprint is where they're applied for the first time in this project, on a dataset already well understood from Sprint 02.

**PyTorch dataset/dataloader internals.** The `Dataset`/`DataLoader` contract (what belongs in `__getitem__` vs. construction time, why worker count and `persistent_workers` matter, why shuffle/`drop_last` differ by split) is learned by designing it deliberately here, not by copying a tutorial's defaults.

**Medical AI pipeline correctness concerns.** Patient-wise leakage prevention, the "never substitute a blank tensor for a failed load" rule, and the requirement that validation/inference transforms be byte-for-byte identical are all concerns that matter disproportionately in medical imaging compared to generic computer vision tutorials, and this sprint is where they're made concrete and enforced by code rather than left as a paragraph in a specification.

**Dataset abstraction design.** Separating "what a sample looks like" (Section 8) from "how a sample gets transformed" (Section 10) from "how samples get batched" (Section 11) is a transferable abstraction skill — the same separation is what lets a future domain plugin reuse this design with a different dataset underneath it.

**Production software engineering habits.** Config-driven parameters, versioned artifacts instead of regenerated-on-the-fly values, and a test suite specified before implementation are habits this sprint exercises directly, ahead of model training — exactly because retrofitting them after a model is already training is far more disruptive.

---

## 20. Sprint 04 Preview

Sprint 04 introduces the baseline model and the first training loop, consuming Sprint 03's `DataLoader` output directly and without modification. Per the training specification (Phase 1, Section 7), the baseline backbone is DenseNet121, brought in via the staged unfreezing schedule (Phase 1, Section 8); per Section 9, the loss is weighted `BCEWithLogitsLoss`, with per-class `pos_weight` values computed from the **training split's** label statistics — statistics that Sprint 03's verified label-encoding output (Section 6) makes a direct, already-validated computation rather than a fresh derivation.

Sprint 03's job in service of Sprint 04 is narrow but load-bearing: by the time Sprint 04 begins, "does the data going into the model make sense" must already be a fully answered, verified question — so that any problem encountered during the first training run can be attributed to model/training choices, not to an unverified data pipeline. Sprint 04 does not redesign anything in this document; it consumes it.