# Sprint 02 — Dataset Engineering & Exploratory Data Analysis Report

**Project:** VisionServeAI
**Sprint:** 02 — Dataset Exploration & Integrity Analysis
**Dataset:** NIH ChestXray14
**Sprint type:** Dataset engineering / EDA only — no model training performed
**Source artifact:** `sprint-02-dataset-exploration.ipynb`

---

## 1. Executive Summary

Sprint 02 establishes the empirical ground truth about the NIH ChestXray14 dataset before any training-pipeline code is written. The sprint's scope was deliberately narrow: discover the real on-disk dataset structure, load and audit the metadata schema, characterize patient and label distributions, and verify that the metadata and the filesystem agree with each other. No preprocessing, no label encoding, no `Dataset`/`DataLoader` construction, and no model code were introduced.

The dataset explored is the full NIH ChestXray14 release as mirrored on Kaggle: **112,120 images** across **30,805 unique patients**, with a 12-column metadata file (`Data_Entry_2017.csv`), a bounding-box annotation file (`BBox_List_2017.csv`), and official train/test split lists. The principal engineering outcomes of this sprint are:

- A confirmed, non-obvious on-disk layout (each `images_NNN` shard contains a nested `images/` subfolder, not flat PNGs), discovered through direct filesystem inspection and resolved with recursive discovery.
- A verified 1:1 correspondence between every metadata row and every file on disk — zero missing files, zero orphan files.
- A quantified, severe class-imbalance profile (from 53.84% "No Finding" down to 227 Hernia-positive images) that will directly shape the loss-function and evaluation design in the training phase.
- A confirmed long-tailed patient distribution (median 1 image/patient, maximum 184 images for a single patient) that confirms patient-wise splitting is not optional.
- An identified, but intentionally not-yet-cleaned, metadata schema defect (`Unnamed: 11`, entirely null).

This report documents what was measured, what was decided, what was deferred, and why — so Sprint 03 can begin from a fully understood dataset rather than re-deriving these facts from scratch.

---

## 2. Sprint Goals

### Completed objectives
- Discover and document the real directory structure of the dataset as mounted in the Kaggle environment.
- Load `Data_Entry_2017.csv` and audit its schema (column count, types, nulls).
- Quantify patient-level statistics (unique patients, images per patient, distribution shape).
- Quantify label-level statistics (unique label combinations, "No Finding" prevalence, per-disease frequency, label cardinality per image).
- Perform a full filesystem-vs-metadata integrity audit (missing/orphan file detection).
- Perform a qualitative visual inspection of a small image sample.
- Perform an image-resolution audit on a sample of images.

### Out-of-scope objectives (explicitly not attempted this sprint)
- Any metadata cleaning or column removal.
- Label encoding (multi-hot vector construction).
- Patient-wise train/validation/test split implementation.
- Any `torch.utils.data.Dataset` or `DataLoader` construction.
- Any image transform, augmentation, or preprocessing pipeline.
- Any model loading, training, or evaluation.
- Bounding-box data analysis (file existence was confirmed; contents were not loaded or analyzed).
- Loading or using `train_val_list.txt` / `test_list.txt` (existence was confirmed only).

### Future work
- All items above under "out-of-scope" are the working backlog for Sprint 03 (Section 17 details the deferral rationale for each).

---

## 3. Dataset Overview

| Attribute | Value |
|---|---|
| Dataset name | NIH ChestXray14 |
| Medical domain | Chest radiography (frontal-view X-ray), multi-label thoracic pathology classification |
| Hosting/mount path | `/kaggle/input/datasets/organizations/nih-chest-xrays/data` |
| Total images (metadata rows) | 112,120 |
| Image shard folders | 12 (`images_001` through `images_012`) |
| Metadata file | `Data_Entry_2017.csv` (confirmed present) |
| Bounding-box file | `BBox_List_2017.csv` (confirmed present, not yet loaded) |
| Official split files | `train_val_list.txt`, `test_list.txt` (both confirmed present, not yet loaded) |
| Supporting documentation | `README_CHESTXRAY.pdf`, `FAQ_CHESTXRAY.pdf`, `LOG_CHESTXRAY.pdf` |
| Research paper | `ARXIV_V5_CHESTXRAY.pdf` (present in the dataset root) |

**Why this dataset was selected.** Per the project's frozen architecture, VisionServeAI's v1 domain plugin targets medical chest radiography as the first proof of the platform's domain-agnostic design. NIH ChestXray14 is the largest publicly available multi-label chest X-ray dataset with an established literature baseline (CheXNet), official patient-disjoint train/test lists, and accompanying documentation/paper — making it well suited both for learning production CV/MLOps practices and for producing results that are externally sanity-checkable against published work.

---

## 4. Filesystem Discovery

### 4.1 Initial exploration
The first filesystem probe iterated only two levels deep from `/kaggle/input` (`ROOT.iterdir()` plus one nested `iterdir()` call). This returned only:

```
📂 datasets
   └── organizations
```

This was insufficient — the actual dataset content lives several directory levels deeper, and a fixed two-level loop could not see it.

### 4.2 Recursive structure discovery
A depth-bounded recursive `explore()` function (max depth 5) was written to walk the full tree. This revealed the true hierarchy:

```
input/datasets/organizations/nih-chest-xrays/data/
├── images_001 … images_012
├── ARXIV_V5_CHESTXRAY.pdf
├── BBox_List_2017.csv
├── Data_Entry_2017.csv
├── FAQ_CHESTXRAY.pdf
├── LOG_CHESTXRAY.pdf
├── README_CHESTXRAY.pdf
├── test_list.txt
└── train_val_list.txt
```

This is the point at which `PROJECT_ROOT` was canonically fixed to `.../nih-chest-xrays/data`, and all subsequent paths (`CSV_METADATA`, `CSV_BBOX`, `TRAIN_LIST`, `TEST_LIST`, `IMAGE_FOLDERS`) were derived from it rather than hardcoded independently — confirmed via existence checks that all four reference files (`True`, `True`, `True`, `True`) and all 12 image folders resolve correctly.

### 4.3 Nested image-directory discovery (the structural surprise)
Inspecting the first shard folder directly (`images_001.iterdir()`) did **not** return PNG files. It returned a single entry:

```
Contents:

images
```

In other words, the real layout is `images_001/images/*.png`, not `images_001/*.png`. Every one of the 12 shard folders nests its PNGs one additional level deeper than a naive reading of the folder name (`images_001`) would suggest.

### 4.4 Why recursive discovery via `pathlib.rglob()` became necessary
Given the nesting discovered in 4.3, any image-loading logic written against the assumption `images_NNN/*.png` would silently enumerate zero or near-zero images per shard — not raise an exception, just under-collect. This is a worse failure mode than a crash, because it can produce a pipeline that runs, trains on whatever the (wrong) glob did happen to match, and never signals that anything is wrong.

The fix applied was to discover PNGs with full recursion from the project root:

```python
png_files = list(PROJECT_ROOT.rglob("*.png"))
```

This returned **112,120 PNG files** — an exact match to the metadata row count — confirming the recursive approach correctly accounts for the extra nesting level regardless of which shard a file lives under.

### 4.5 Engineering lesson
Directory layout should never be assumed from a dataset's published documentation or from folder naming conventions alone. The shard folder name (`images_001`) implied a flat layout; the actual layout had one more level of nesting. The corrective practice adopted going forward: any new dataset (including future domain plugins, per the platform's domain-agnostic design goal) gets a direct, empirical filesystem probe before any loading code is written against assumed paths.

---

## 5. Metadata Analysis

### 5.1 Schema
`Data_Entry_2017.csv` loads as **112,120 rows × 12 columns**.

| # | Column | Dtype | Non-null count | Meaning |
|---|---|---|---|---|
| 0 | `Image Index` | object | 112,120 | Filename of the corresponding PNG; the primary identifier joining metadata to the filesystem |
| 1 | `Finding Labels` | object | 112,120 | Pipe-delimited (`\|`) string of one or more of the 14 pathology names, or the literal string `No Finding` |
| 2 | `Follow-up #` | int64 | 112,120 | Sequential visit/study index for a given patient |
| 3 | `Patient ID` | int64 | 112,120 | Patient identifier; **not** unique per row — the patient-level join key |
| 4 | `Patient Age` | int64 | 112,120 | Patient age at time of study |
| 5 | `Patient Gender` | object | 112,120 | `M` / `F` |
| 6 | `View Position` | object | 112,120 | `PA` / `AP` acquisition view |
| 7 | `OriginalImage[Width` | int64 | 112,120 | Original image width in pixels (see naming note below) |
| 8 | `Height]` | int64 | 112,120 | Original image height in pixels |
| 9 | `OriginalImagePixelSpacing[x` | float64 | 112,120 | Physical pixel spacing, x-axis |
| 10 | `y]` | float64 | 112,120 | Physical pixel spacing, y-axis |
| 11 | `Unnamed: 11` | float64 | **0** | Entirely null — see 5.3 |

**Column-naming note.** Columns 7/8 and 9/10 are the result of the source CSV's header using bracket notation across what was originally a single conceptual field (`OriginalImage[Width Height]`, `OriginalImagePixelSpacing[x y]`); pandas split each on the embedded space, producing the awkwardly-named but fully usable column pairs shown above. This is documented here rather than silently renamed, for the same reason given in 5.3.

### 5.2 Primary identifier and important columns
- **Primary identifier:** `Image Index` — the join key between metadata and the filesystem (used directly in Section 11's integrity audit).
- **Patient-level key:** `Patient ID` — not unique per row, and the key that all future leakage-prevention logic (Section 7) must split on.
- **Target source column:** `Finding Labels` — the single column from which all 14 binary pathology targets must be derived.

### 5.3 The `Unnamed: 11` column
`Unnamed: 11` is present in every row and **null in all 112,120 of them** (confirmed by both `.info()` and `.isnull().sum()`). It is a parsing artifact, almost certainly produced by a trailing delimiter in the source CSV's header row rather than a deliberate data field.

**Decision: do not drop it this sprint.** Removing it now would be a metadata-cleaning action, and Sprint 02 is explicitly EDA-only — the metadata CSV is being treated as a read-only source of truth during this sprint (Section 12) so that every later cleaning decision is made deliberately, in one place, in Sprint 03, rather than scattered across exploratory cells. Dropping a column "while just looking" is exactly the kind of small, unreviewed mutation that makes a dataset pipeline hard to audit later. The column is documented here precisely so it isn't silently rediscovered (or silently forgotten) when cleaning does happen.

### 5.4 Columns reserved for future work
`Follow-up #`, `Patient Age`, `Patient Gender`, `View Position` are not used by the v1 image-only classification baseline, but are retained as candidates for: stratifying the patient-wise split (Section 7) by demographic balance, secondary subgroup-performance analysis (Section 11's "Acceptance Criteria" concerns in the training spec), or future metadata-conditioned modeling.

### 5.5 Columns excluded from baseline
`OriginalImage[Width` / `Height]` / `OriginalImagePixelSpacing[x` / `y]` describe the *original*, pre-Kaggle-mirror image properties and physical pixel spacing. The training specification's preprocessing contract resizes every image to a fixed 224×224 input regardless of original size, so these four columns carry no information the baseline model consumes. `Unnamed: 11` is excluded outright (5.3).

---

## 6. Dataset Statistics

All figures below are taken directly from notebook output; none are estimated or extrapolated unless explicitly marked "derived."

| Statistic | Value |
|---|---|
| Total images (metadata rows) | 112,120 |
| Total metadata columns | 12 |
| Unique patients | 30,805 |
| Average images per patient (mean) | 3.64 |
| Median images per patient | 1.0 |
| Std. dev. of images per patient | 7.27 |
| 75th percentile, images per patient | 3.0 |
| Maximum images for a single patient | 184 |
| Patients with exactly 1 image | 17,503 |
| Metadata memory footprint (in-memory `DataFrame`) | 10.3+ MB |
| Missing values — all columns except `Unnamed: 11` | 0 |
| Missing values — `Unnamed: 11` | 112,120 (100%) |
| Image shard folders | 12 |
| PNG files discovered (recursive) | 112,120 |
| Unique label combinations (`Finding Labels`) | 836 |
| Images labeled `No Finding` | 60,361 (53.84%) |
| Average pathology labels per image | 0.72 |
| Maximum pathology labels on one image | 9 |
| Unique image resolutions (2,000-image sample) | 1 — all `(1024, 1024)` |

**Derived cross-check (not a direct notebook output):** summing the per-class disease counts in Section 8 (excluding `No Finding`) gives 81,176 total pathology-label instances. Independently, summing the label-cardinality distribution in Section 9 weighted by label count (`1×30,963 + 2×14,306 + … + 9×2`) also gives 81,176. These two independently-computed figures agreeing exactly is a useful internal consistency check on the reported counts — both derive from the same `Finding Labels` column but through different code paths (`Counter` over split tokens vs. token-count-per-row).

---

## 7. Patient Analysis

### 7.1 Why patient IDs matter
`Patient ID` is the only column in this dataset that captures the fact that many images are not independent samples — they are repeated studies of the same underlying patient. A model that sees one image of a given patient during training and a different image of the *same* patient during validation can appear to generalize well while actually partly memorizing patient-specific anatomy or imaging artifacts rather than learning the pathology itself. This is patient-wise leakage, and it is the single most common silent correctness bug in medical imaging pipelines built on this kind of dataset.

### 7.2 Long-tail patient distribution — what was actually measured
The patient-count distribution is sharply right-skewed:

- Median images per patient: **1.0**
- Mean images per patient: **3.64**
- 75th percentile: **3** images
- Maximum: **184** images, for a single patient
- **17,503 of 30,805 patients (≈56.8%, derived)** contribute exactly one image each

A mean nearly 3.6× the median, combined with a maximum of 184 against a median of 1, confirms a classic long tail: most patients contribute a single image, while a small number of patients (likely those with extensive follow-up imaging) contribute disproportionately many.

### 7.3 Importance for the train/validation/test split
Because a small number of patients carry a large number of images, naive image-level random splitting risks two compounding problems: (1) leakage, as described in 7.1, and (2) split imbalance — a heavily-imaged patient could end up effectively "dominating" one split's label statistics if not handled deliberately. The training specification's patient-wise split design (stratified by approximate label signature, then expanded to all of that patient's images) is the correct response to this measured distribution, not a generic best practice applied without justification.

### 7.4 Future pipeline implications
Any later subgroup analysis (e.g., by `Patient Age` or `Patient Gender`, Section 5.4) must also respect the patient-wise split boundary — these are reserved-for-future-work columns specifically because using them well requires the same leakage discipline already established here for the primary split.

---

## 8. Label Engineering

### 8.1 Multi-label structure
`Finding Labels` is a pipe-delimited string field, not a categorical label. A given image can carry zero pathology labels (`No Finding`), exactly one, or several simultaneously (up to 9, see Section 9). This is fundamentally a multi-label problem, not a multi-class one — confirmed empirically by the existence of **836 unique label combinations** found in the data, far more than the 14 individual disease names plus `No Finding` would imply if labels never co-occurred.

### 8.2 No Finding analysis
- Count: **60,361** images
- Percentage of dataset: **53.84%**

This single label/value covers more than half the dataset. It is the dominant outcome by a wide margin, which is the primary driver of the imbalance discussed in 8.4.

### 8.3 Disease frequency

| Disease | Count | % of all images |
|---|---|---|
| No Finding | 60,361 | 53.84% |
| Infiltration | 19,894 | 17.74% |
| Effusion | 13,317 | 11.88% |
| Atelectasis | 11,559 | 10.31% |
| Nodule | 6,331 | 5.65% |
| Mass | 5,782 | 5.16% |
| Pneumothorax | 5,302 | 4.73% |
| Consolidation | 4,667 | 4.16% |
| Pleural_Thickening | 3,385 | 3.02% |
| Cardiomegaly | 2,776 | 2.48% |
| Emphysema | 2,516 | 2.24% |
| Edema | 2,303 | 2.05% |
| Fibrosis | 1,686 | 1.50% |
| Pneumonia | 1,431 | 1.28% |
| **Hernia** | **227** | **0.20%** |

(Percentages computed from the printed counts; these are direct counts of pathology occurrence, not mutually exclusive — they do not sum to 100%.)

### 8.4 Class imbalance — clinical and engineering observations
The frequency table spans more than two orders of magnitude: Infiltration (19,894) is roughly **88× more common** than Hernia (227). This is not a dataset artifact to be fixed — it reflects genuine real-world prevalence differences in thoracic disease — but it is a hard constraint the training pipeline must be designed around rather than ignore.

### 8.5 Training implications
A model trained with an unweighted loss against this distribution will trivially minimize loss by predicting "absent" for Hernia and other rare classes almost regardless of input, since doing so is correct ~99.8% of the time by raw frequency. The training specification's response — per-class `pos_weight` inside `BCEWithLogitsLoss`, computed from the training split only — exists specifically to counteract the gradient signal this imbalance would otherwise produce, without altering what data the model actually sees.

### 8.6 Evaluation implications
Aggregate accuracy is close to meaningless on this label distribution — a model could exceed 99% raw accuracy on Hernia by never predicting it. This is why the training specification mandates per-class AUROC and per-class Average Precision (not just a single aggregate number) as first-class reporting requirements, with explicit extra scrutiny on the rarest classes (Hernia, Fibrosis, Pneumonia).

---

## 9. Label Cardinality Analysis

| Diseases per image | Image count |
|---|---|
| 0 (No Finding) | 60,361 |
| 1 | 30,963 |
| 2 | 14,306 |
| 3 | 4,856 |
| 4 | 1,247 |
| 5 | 301 |
| 6 | 67 |
| 7 | 16 |
| 8 | 1 |
| 9 | 2 |

- **Average diseases per image:** 0.72
- **Maximum diseases on a single image:** 9

### 9.1 Why cardinality matters
The distribution confirms that co-occurrence is common but high-order co-occurrence is rare: roughly 81% of images have 0 or 1 pathology label, while only 86 images (8 of "7", 1 of "8", 2 of "9", per the table) carry five or more simultaneous findings. This tail exists but is small enough that it should not be allowed to dominate any per-batch loss statistics disproportionately.

### 9.2 Loss function implications
This distribution supports the training specification's choice of independent per-class binary cross-entropy rather than a loss that explicitly models label *interactions* (e.g., a structured multi-label loss conditioning on co-occurrence). With the bulk of the mass at cardinality 0–2, modeling each class independently is a reasonable starting assumption; a higher-order interaction-aware loss would be solving a problem this data only thinly supports evidence for in v1.

### 9.3 Threshold-tuning implications
Because most images have low cardinality, per-class threshold optimization (as specified in the training spec, Section 11.3) can be performed largely independently per class without needing to jointly optimize across classes for typical cases — though the rare high-cardinality images (cardinality 6–9) are exactly the kind of case the failure-analysis step should sample deliberately, since they're underrepresented in any random validation sample.

### 9.4 Future research possibilities
The 836 unique label combinations (Section 8.1) are a resource for a future co-occurrence-aware analysis (e.g., a 14×14 co-occurrence matrix) that this sprint's notebook did not compute. That remains a candidate addition for a future EDA pass rather than something to claim as already measured.

---

## 10. Image Analysis

### 10.1 Visual inspection process
Five PNG files were sampled at random (`random.sample(png_files, 5)`) from the recursively-discovered file list and rendered in a single row using `matplotlib`/`PIL`, displayed with a grayscale colormap.

### 10.2 Observed characteristics
- All five sampled images rendered correctly as grayscale chest radiographs with no visible corruption, truncation, or decode errors.
- This was a small, random spot-check (5 of 112,120 images) intended to catch gross loading/rendering problems early — it is not a substitute for the systematic integrity audit in Section 11, and should not be read as a claim that all 112,120 images were visually reviewed.

### 10.3 Resolution findings
A resolution audit was run across the **first 2,000** discovered PNG files:

- **Unique resolutions found: 1**
- All 2,000 sampled images: **1024 × 1024**

### 10.4 Why standardized resolution simplifies preprocessing
A confirmed single resolution across the sample means the resize step in the training specification (Section 5: direct resize to 224×224, no letterboxing) is operating on a consistent starting aspect ratio and scale, rather than having to handle a mix of source resolutions/aspect ratios. This directly de-risks one of the more fragile parts of an image pipeline.

**Important caveat — sample size.** This audit checked 2,000 of 112,120 images (≈1.8% of the dataset). The result is a strong, consistent signal, but it is a sample, not an exhaustive check. A full-corpus resolution sweep (or at minimum a larger stratified sample spanning all 12 shard folders) is a reasonable low-cost addition for Sprint 03 before fully relying on "all images are 1024×1024" as a hard pipeline assumption.

### 10.5 Acquisition variability and medical artifacts
The notebook's visual inspection step did not include a systematic review of exposure variability, patient positioning, or imaging artifacts across a representative sample — only the five randomly-sampled images were viewed. No quantitative or qualitative claims about acquisition variability or artifact prevalence are made in this report beyond noting that this is a known general property of multi-site medical imaging datasets (per the dataset's own documentation, not measured here) and is the reason the training specification treats moderate brightness/contrast jitter as a mandatory augmentation rather than an optional one.

---

## 11. Dataset Integrity Audit

### 11.1 Filesystem verification
- 12 image shard folders confirmed present (`images_001`–`images_012`).
- Each shard folder confirmed to contain a nested `images/` subdirectory rather than flat PNGs (Section 4.3).
- Recursive discovery (`rglob("*.png")`) confirmed **112,120** PNG files across all shards.

### 11.2 Metadata verification
- `Data_Entry_2017.csv` confirmed to load cleanly as 112,120 rows × 12 columns.
- Zero missing values in every column except the known-artifact `Unnamed: 11` (Section 5.3).

### 11.3 Image count verification (metadata ↔ filesystem)

| Check | Result |
|---|---|
| Metadata image count | 112,120 |
| Filesystem image count | 112,120 |
| Missing files (in metadata, absent on disk) | **0** |
| Orphan files (on disk, absent from metadata) | **0** |
| Synchronization status | **Perfectly synchronized** |

### 11.4 Why integrity audits are mandatory before training
A missing file discovered mid-training (rather than before it) typically surfaces as a `DataLoader` crash hours into a Kaggle/Colab session, at exactly the point where free-tier compute time is most expensive to lose. An orphan file (present on disk, absent from metadata) is lower-risk but indicates a possible mismatch between the dataset version on disk and the metadata version being used, which is the kind of silent versioning problem that's far cheaper to catch with one `set`-difference check than to debug after a confusing training run. The zero/zero result here is a genuinely good outcome — it means the Kaggle mirror of this dataset is internally consistent — but it is a result that had to be checked, not assumed, precisely because of the directory-nesting surprise already encountered in Section 4.

---

## 12. Engineering Decisions

| Decision | Rationale |
|---|---|
| Use `pathlib.Path` exclusively for all filesystem operations | Cleaner cross-platform path composition than raw string concatenation; required for `.rglob()` |
| Fix one canonical `PROJECT_ROOT` and derive all other paths from it | Avoids hardcoding multiple independent absolute paths that could drift out of sync if the mount point changes |
| Use recursive discovery (`rglob`) rather than a fixed-depth glob pattern | Directly forced by the nested `images_NNN/images/*.png` layout discovered in Section 4.3; a shallow glob would have silently under-collected files |
| Treat `Data_Entry_2017.csv` as the source of truth, and verify the filesystem *against* the metadata (not the reverse) | The metadata is the labeled, authoritative record; the filesystem audit's job is to confirm the images claimed by the metadata actually exist, not to discover labels from the filesystem |
| Defer all metadata cleaning (including dropping `Unnamed: 11`) | Keeps this sprint's output a faithful, unmodified record of the raw dataset; cleaning decisions are deliberately bundled into a dedicated Sprint 03 step instead of being made ad hoc inside exploratory cells |
| Keep EDA strictly separate from preprocessing/training code | No transforms, `Dataset`, `DataLoader`, or model code were introduced this sprint — this sprint answers "what is in the data," not "how do we feed it to a model" |
| Avoid premature model training | Consistent with the project's stated principle (Phase 0/Phase 1 specs) of not optimizing for speed-to-model at the expense of understanding the data first |
| Document patient-wise split necessity without yet implementing it | The decision is justified by data already in hand (Section 7); the implementation is correctly sequenced into Sprint 03, after metadata cleaning, not before |

---

## 13. Bugs Encountered

### 13.1 Image discovery bug — nested shard directories

**Symptom.** Listing the contents of the first image shard folder (`images_001`) did not return PNG files; it returned a single subdirectory entry, `images`.

**Root cause.** The dataset's on-disk layout (at least as mirrored on Kaggle) nests each shard's images one directory level deeper than the shard folder name implies: the real path to any image is `images_NNN/images/<filename>.png`, not `images_NNN/<filename>.png`.

**Investigation.** The nesting was found through direct, manual inspection of one shard folder's contents — not inferred from documentation. This was only possible because the recursive structure-discovery step (Section 4.2) had already been written generically enough to support drilling into any single folder on demand.

**Resolution.** Image discovery was switched from any shard-relative flat-glob assumption to a single, fully recursive search from `PROJECT_ROOT`: `PROJECT_ROOT.rglob("*.png")`. This returned exactly 112,120 files, matching the metadata row count exactly, and was subsequently confirmed correct end-to-end by the zero-missing/zero-orphan result in Section 11.

**Lessons learned.** (1) Folder naming conventions are not reliable evidence of folder contents' depth. (2) A single successful existence check (e.g., confirming `images_001` exists) is not sufficient validation that a planned access pattern (`images_001/*.png`) will actually work — the *contents* of a sampled directory must be inspected directly. (3) Recursive discovery is a reasonable default for this class of problem precisely because it is robust to an extra nesting level that a fixed-depth pattern is not.

---

## 14. Risks Identified

| Risk | Evidence from this sprint | Severity |
|---|---|---|
| **Severe class imbalance** | Hernia: 227 images (0.20%) vs. No Finding: 60,361 (53.84%) — an ~88× gap between the most and least common pathology classes | High — directly shapes loss-function design (Section 9 training spec) |
| **Patient-level leakage if split naively** | Long-tailed patient distribution: median 1 image/patient, max 184; 17,503 patients with only 1 image | High — must be resolved before any train/val/test split is implemented |
| **Rare-disease statistical fragility** | Hernia (227), Pneumonia (1,431), Fibrosis (1,686) — all under 1,700 positive examples in a 112,120-image dataset | Medium-High — affects both achievable recall and the reliability of any rare-class metric computed on a small validation slice |
| **Multi-label complexity** | 836 unique label combinations; up to 9 simultaneous labels on a single image | Medium — rules out simple multi-class assumptions in any downstream code |
| **Metadata schema quality** | `Unnamed: 11` fully null across all 112,120 rows; fragmented bracket-style column headers (`OriginalImage[Width` / `Height]`) | Low-Medium — no missing label data, but the schema requires a deliberate cleaning pass before being used downstream without caveats |
| **Resolution-audit sample coverage** | Only 2,000 of 112,120 images (≈1.8%) were checked for resolution consistency | Low — result was clean, but the claim "all images are 1024×1024" is not yet exhaustively verified |
| **Unverified bounding-box and split-list contents** | `BBox_List_2017.csv`, `train_val_list.txt`, `test_list.txt` existence was confirmed but contents were not loaded or validated this sprint | Low — deferred work, not a defect, but worth tracking so it isn't forgotten before Sprint 03 relies on these files |

---

## 15. Sprint Deliverables

The following artifacts were produced and are considered complete:

- `sprint-02-dataset-exploration.ipynb` — the full exploration notebook (source of this report).
- Filesystem structure map of the dataset, including the discovered nested-image-folder layout.
- Canonical dataset path definitions (`PROJECT_ROOT` and all derived paths).
- Metadata schema audit (column count, dtypes, missing-value report).
- Patient distribution statistics and patient-count summary table.
- Label engineering audit (unique combinations, No Finding count/percentage, sample combinations).
- Disease frequency table and bar chart (14 pathology classes + No Finding).
- Label cardinality table and bar chart (0–9 diseases per image).
- Recursive PNG file discovery (112,120 files located).
- Metadata-vs-filesystem integrity audit (missing/orphan file detection).
- Visual sample inspection (5 randomly sampled images rendered).
- Image resolution audit (2,000-image sample).
- This report, `SPRINT_02_DATASET_ENGINEERING_REPORT.md`.

---

## 16. Acceptance Criteria

Sprint 02 is considered complete because all of the following hold:

- [x] The real, on-disk dataset structure is documented, including the non-obvious nested-folder layout — not assumed from documentation.
- [x] The metadata schema is fully audited: column count, types, and null counts are known for every column, including the identification of the `Unnamed: 11` artifact.
- [x] Patient-level statistics are measured directly (unique patient count, mean/median/max images per patient), establishing the evidence base for the patient-wise split requirement.
- [x] Label-level statistics are measured directly (unique combinations, per-disease frequency, No Finding prevalence, cardinality distribution), establishing the evidence base for the imbalance-aware loss design.
- [x] A full metadata-vs-filesystem integrity audit was run and returned a clean result (zero missing, zero orphan files).
- [x] A qualitative visual spot-check and a sampled resolution audit were performed, with sample-size caveats explicitly recorded rather than overstated.
- [x] No training, preprocessing, or model code was introduced — the sprint's scope boundary (per the sprint objective: dataset engineering and EDA only) was respected throughout.

No metric thresholds gate this sprint, because this sprint produces no model — completion is defined by coverage and correctness of the data-understanding work, all of which is satisfied above.

---

## 17. Decisions Deferred to Sprint 03

| Deferred item | Why it was not done in Sprint 02 |
|---|---|
| Metadata cleaning (drop `Unnamed: 11`, address fragmented `OriginalImage[Width` / `Height]` and `OriginalImagePixelSpacing[x` / `y]` column naming) | Sprint 02 treats the raw metadata as a read-only source of truth; cleaning is a deliberate, reviewable step, not an incidental one |
| Label encoding (multi-hot vector construction from `Finding Labels`) | Requires the cleaning step above to be settled first, and is properly a preprocessing concern, not an EDA concern |
| Patient-wise train/validation/test split implementation | This sprint established *why* it's required (Section 7) and the statistical evidence for it, but implementing the actual split (with the leakage assertion specified in the training spec) is correctly sequenced after metadata cleaning |
| `torch.utils.data.Dataset` construction | Out of scope per the sprint objective; depends on the split and encoded labels above |
| `DataLoader` construction | Same as above |
| Image transforms / augmentation pipeline | Defined at the specification level in Phase 1's training spec, but not implemented against real data until the `Dataset` exists |
| Model training of any kind | Explicitly out of scope for this sprint by project design |
| Bounding-box data loading and analysis (`BBox_List_2017.csv`) | File existence was confirmed only; analysis was not part of this sprint's stated deliverables |
| Loading and using `train_val_list.txt` / `test_list.txt` | Existence confirmed only; these become relevant once the actual split implementation begins |
| Full-corpus (or larger-sample) resolution verification | The 2,000-image sample gave a clean, consistent result; expanding coverage is a low-priority but worthwhile addition before fully relying on the 1024×1024 assumption |

---

## 18. Lessons Learned

**Medical datasets carry structural assumptions that must be verified, not inherited.** The nested `images/` subdirectory (Section 4, 13) is a small detail with an outsized consequence: any code written against the "obvious" flat layout would have failed silently rather than loudly. The general principle this sprint reinforces is that dataset packaging on a third-party mirror (Kaggle, in this case) cannot be assumed to match the layout implied by the original dataset's documentation, filenames, or folder naming.

**Filesystem assumptions should be tested against actual directory contents, not against existence checks alone.** Confirming a path *exists* (`images_001.exists() == True`) says nothing about what's *inside* it. This sprint's workflow — discover structure broadly, then drill into one concrete folder to inspect contents directly — is the pattern worth repeating for any future domain plugin's dataset.

**Dataset validation (the metadata-vs-filesystem integrity audit) is cheap insurance against expensive failures.** A single `set` difference between metadata image names and discovered filenames caught (or in this case, confirmed the absence of) exactly the class of problem — missing files, orphaned files, version mismatches — that otherwise surfaces as a runtime failure deep into a training run, on exactly the kind of time-limited free-tier GPU session this project depends on.

**EDA earns its place by directly informing concrete downstream decisions, not by being thorough for its own sake.** Every major statistic gathered this sprint maps to a specific decision already made or pending in the training specification: the patient distribution justifies patient-wise splitting; the disease frequency table justifies per-class loss weighting; the label cardinality distribution justifies the choice of independent per-class BCE over a joint multi-label loss.

**Patient-aware thinking has to start at the EDA stage, not at the splitting stage.** Measuring the patient distribution now — before any split code exists — means the split, when implemented, can be designed against real, known numbers (a median of 1 image/patient, a max of 184) rather than generic best-practice assumptions about what a "typical" medical imaging dataset looks like.

**Engineering-first development means being willing to defer obviously-needed work.** Several items in this report (cleaning `Unnamed: 11`, implementing the split) are clearly necessary and were clearly identified — and were still deliberately not done this sprint, because doing them now would have blurred the EDA sprint's scope and made this report's findings harder to trust as an unmodified baseline.

---

## 19. Sprint Summary

Sprint 02 produced a complete, verified picture of the NIH ChestXray14 dataset as it actually exists on disk and in its metadata — not as documentation or folder names suggest it exists. The dataset's true nested-folder layout was discovered and correctly handled via recursive discovery; the metadata schema was fully audited, including identification of a null-only artifact column; patient and label distributions were quantified with exact figures (112,120 images, 30,805 patients, 836 unique label combinations, a 53.84%/0.20% prevalence range from No Finding to Hernia); and a full integrity audit confirmed perfect synchronization between metadata and filesystem (zero missing, zero orphan files). No training-adjacent code was written, preserving this sprint as a clean, trustworthy baseline of dataset understanding for Sprint 03 to build on.

---

## 20. Sprint 03 Preview

Sprint 03 will pick up directly from this report's deferred-work list (Section 17): cleaning the metadata schema, encoding the 14-class multi-hot label vectors, implementing the patient-wise train/validation/test split with an automated leakage assertion, and beginning the `Dataset`/`DataLoader` construction needed to feed images into the training pipeline. No model training is expected to begin in Sprint 03 either — it remains a data-pipeline-construction milestone, building the concrete implementation of decisions this sprint only justified on paper.
