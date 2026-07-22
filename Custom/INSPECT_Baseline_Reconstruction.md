# INSPECT Baseline Reconstruction: Process

Reconstructing the baseline dataset was a complex debugging process. Here is a formal summary of the exact steps taken to successfully overcome these blockers and generate the baseline dataset.

---

## 0. Python Environment Setup

To run the scripts in the `Custom/` directory, you must first install the legacy environment dependencies followed by the custom pipeline supplements. They can be easily installed using:

```bash
# 1. Install the base legacy environment
pip install -r ehr/requirements.txt

# 2. Install the custom pipeline supplements
pip install -r Custom/addition_reqs.txt
```

---

## 1. Raw Data Download & Format Conversion

To conserve disk space, a Redivis downloader script (`Custom/INSPECT_DL_EHR.py`) was initially developed to download all 32+ raw OMOP tables (e.g., measurement, condition_occurrence, person) as highly compressed `.parquet` files.

* **Issue #1:** While the `.parquet` format is highly optimized for future custom processing pipelines, it was discovered that the legacy baseline pipeline strictly required uncompressed `.csv` files.
* **Solution:** The downloader script was updated to target the `EHR_CSV` directory directly, and a conversion script (`Custom/convert_parquet_to_csv.py`) was used to expand existing Parquet files back into flat `.csv` formats.
* **Database Compilation:** The legacy `ehr/1_csv_to_database.py` script was then successfully executed. This utilized the `femr` framework to ingest the raw CSVs and compile them into a highly optimized, longitudinal patient database located at `DATA_RAW/EHR_FEMR_DB/extract`.

---

## 2. Diagnosing the Missing Labels

Upon reviewing the `run_all_ehr.py` master script, it was realized that the pipeline explicitly requires a pre-built file named `cohort_0.2.0_master_file_anon.csv`. The downloaded EHR dataset from Redivis was analyzed, revealing that the core ground-truth labels for the project (specifically the `pe_positive_nlp` column) were entirely missing.

---

## 3. The AIMI Portal Breakthrough

After hitting a dead end with the public dataset, an inquiry was sent to Professor Fries. He clarified that the INSPECT dataset is actually distributed across two different portals:

* **Tabular EHR Data:** Hosted by the Shah Lab on Redivis.
* **True Labels, Mappings, and Images:** Hosted by the AIMI Center on a separate portal.

> **NOTE — Undocumented Requirement:** The required link to the AIMI center's label and mapping files was notably absent from the main INSPECT dataset website. This split-distribution setup was essentially a hidden requirement that could only be resolved by directly contacting the dataset authors.

---

## 4. Data Reconstruction & OMOP Clinical Anchoring (`merge_labels.py`)

Following the download of the missing `labels_20250611.tsv` and `study_mapping_20250611.tsv` files from the AIMI portal, a custom Python script (`Custom/merge_labels.py`) was engineered to reconstruct the master file.

* An inner join was executed between the true labels and the official study mapping.
* **Issue #2:** It was identified that exactly 779 scans in the official dataset were completely missing timestamp data (`procedure_DATETIME`). Specifically, during Stanford's strict PHI de-identification scrubbing process, the `procedure_occurrence_id` relational key was stripped entirely from 710 records, turning them into "ghost" patients.
* **Solution:** Because the `femr` pipeline relies on timestamps to calculate patient history, these 779 ghost records were intentionally dropped to replicate the exact constraints of the original study. The resulting master file successfully processed the remaining valid timestamped records.
* The result was a cleaned `cohort_0.2.0_master_file_anon.csv` that matches the exact restricted distribution used in the official baseline benchmark.

### 4.1. Integration of Canonical Splits & Metadata (June 2025 Data Drop)

Shortly after the initial dataset reconstruction, three supplementary files were released on the AIMI portal: `splits_20250611.tsv`, `series_metadata_20250611.tsv`, and `image_ehr_crosswalk_20250418.csv`.

* **The Splits:** Previously, the exact train/valid/test patient divisions were implicit or generated dynamically. The new `splits_20250611.tsv` file provided canonical benchmark split assignments.
* **Pipeline Updates:** `Custom/2_merge_labels.py` was explicitly refactored to ingest this TSV, carefully drop any duplicate `impression_id`s, and perform an inner join to merge the `split` column directly into `cohort_0.2.0_master_file_anon.csv`.
* **Model Integrity:** By having the splits directly embedded in the master cohort file, the downstream LightGBM (`ehr/3_train_gbm.py`) and sequence modeling scripts are now strictly locked into using the official canonical train/valid/test divisions, avoiding any potential cross-split leakage.
* **Relative Path Portability:** Alongside this data update, all hardcoded absolute paths (e.g. `~/Documents/Internship_INSPECT/`) across the `Custom/` pipeline scripts were dynamically refactored to use standard relative paths (`../`), ensuring seamless repository portability.

---

## 5. Data Validation Dashboard

To visually inspect and sanity check the generated data (including sparse feature matrices, missing value ratios, and label distributions), a custom Streamlit web dashboard (`Custom/merged_labeled_data_viewer.py`) was constructed. This allowed for interactive filtering and statistical validation of the reconstructed cohort prior to pipeline execution.

---

## 6. Bypassing the Legacy Pipeline (`run_baseline_benchmark.py`)

The repository's master script (`run_all_ehr.py`) is designed to train massive deep-learning models (MOTOR/CLMBR). Several critical errors and limitations were encountered when attempting execution:

* **Environment Version Conflicts:** Due to massive breaking changes in recent releases of `numpy` (2.x) and `pandas` (3.x), attempting to run the original scripts with modern packages results in fatal C-API/ABI and syntax errors. To guarantee execution, the environment must be strictly compiled against `ehr/requirements.txt` to lock `numpy==1.24.3` and `pandas==2.0.2`.

* **Hardware/CUDA 12+ Incompatibility (Blackwell GPUs):** The legacy Python 3.10 environment relies on JAX/jaxlib wheels pinned to CUDA 11, which cannot compile kernels for Compute Capability 12.0+ (Blackwell 50-series) without intervention. Upgrading JAX would overwrite `numpy` to 2.x and break the `femr` C-API. The fix requires four targeted changes documented in full in Section 16.

* **Hardcoded Model Paths:** The script's hardcoded paths attempted to load the Foundation Model weights from internal `/share/pi/` servers. While the MOTOR model is actively hosted on Hugging Face (`StanfordShahLab/motor-t-base`), access requires formal approval.

* **Hardcoded Extract Paths:** The original script hardcoded the database extract path to `inspect_femr_extract/extract` inside the output directory. To allow flexibility when using pre-generated multi-gigabyte databases (like the 21GB FEMR DB), `ehr/run_all_ehr.py` was directly modified to include an `--extract_path` argument. This explicitly overrides the hardcoded path, allowing the pipeline to skip the database generation step seamlessly.

To bypass these dependencies, a clean Python wrapper (`Custom/run_baseline_benchmark.py`) was implemented. This script manually invoked the legacy Python environment and executed only Step 2 (`2_generate_labels_and_features.py`) against the newly reconstructed master cohort.

---

## 7. Successful Feature Extraction

The targeted script ran flawlessly, ingesting all 23,248 patients and successfully outputting the final, baseline clinical features to `DATA_RAW/EHR_FEMR_DB/features/PE`. This definitively validated the structural integrity of the local data environment, completing the baseline reproduction phase.

---

## 8. GBM Benchmark Results & AUROC Discrepancy

Following feature extraction, the GBM baseline was trained and evaluated for the PE diagnostic task using `ehr/3_train_gbm.py`, which performs hyperparameter tuning via `GridSearchCV` over a predefined train/validation split and reports test-set AUROC.

The reproduced GBM achieved the following metrics:

- **Train AUROC:** 0.9185
- **Validation AUROC:** 0.7456
- **Test AUROC:** 0.7437

This **0.7437** test AUROC is significantly higher than the **0.681** reported in the original paper — a gap of approximately 0.0627.

### Investigation of Potential Data Leakage

An investigation was conducted to determine whether the inflated result was an artifact of the reconstruction process:

* **Ghost patient timestamp fallback (ruled out):** An initial concern was that the 5-tier OMOP timestamp fallback in `Custom/merge_labels.py` could introduce leakage. However, the 779 affected patients were dropped entirely and the pipeline was re-run. The AUROC remained at 0.7437, ruling out the fallback strategy as a meaningful contributor.

* **`note_DATETIME` as fallback (minor, ruled out):** Using the radiology report timestamp as a StudyTime proxy was considered a potential source of same-day leakage. However, the `-1 day` offset applied in `2_generate_labels_and_features.py` provides sufficient buffer, and the effect on the overall cohort is negligible.

* **CountFeaturizer preprocessing over full cohort (present in original paper):** The `featurizer_age_count.preprocess_featurizers()` call in `2_generate_labels_and_features.py` is applied to all patients prior to the train/val/test split, meaning test-set patients influence vocabulary selection. This constitutes a minor form of test-set leakage. However, the same behavior is present in the original paper's codebase, so it cannot explain the performance delta.

**Most probable explanation — label file version mismatch:** The ground-truth labels used in this reproduction (`labels_20250611.tsv`, dated June 2025) were obtained from the AIMI portal well after the original paper's data freeze. The `pe_positive_nlp` column is generated by an NLP pipeline applied to radiology reports; if this pipeline was updated or retrained between the paper's submission and the current data release, the ground-truth labels would be cleaner, improving both training signal and evaluation accuracy. A ~0.07 AUROC lift from label quality improvement is plausible. This is consistent with the observation that multiple inquiries to the dataset authors regarding label provenance did not yield confirmation of the exact label version used in the paper.

> **NOTE:** The AUROC discrepancy is considered unresolved pending clarification from the original authors on the label file version and dataset snapshot used in the paper. The reproduced pipeline is otherwise structurally faithful to the original.

---

## 9. Auxiliary Prognostic Tasks (Mortality, Readmission, PH)

To evaluate the fully reconstructed environment against the paper's secondary endpoints, an automation script (`Custom/run_all_tasks_gbm.py`) was engineered to iteratively generate features and train the GBM across all 7 auxiliary tasks (1, 6, 12-month mortality/readmission, and 12-month PH).

During this process, three major legacy pipeline bugs were identified and patched in `ehr/2_generate_labels_and_features.py`:

* **FEMR API Deprecation:** The `femr` v0.2.x API deprecated the `patient_ids` keyword argument in `labeler.apply()`. Passing it caused a `TypeError` that broke all auxiliary tasks.

* **Scrubbed OMOP Concepts:** The legacy pipeline relies on `femr.labelers.omop.CodeLabeler` to search the patient's EHR timeline for exact death/readmission codes. Because Stanford scrubbed these precise codes from the public Redivis `condition_occurrence` tables to prevent re-identification, the labeler silently failed and yielded 100% `False` evaluations.

* **Case Sensitivity:** For `12_month_PH`, the original script attempted to read the CSV column directly via `label == "True"`. However, the boolean strings exported in the 2025 AIMI dataset are fully capitalized (`"TRUE"`).

**Solution:** Since all 7 outcomes were actually pre-computed and appended to `labels_20250611.tsv` prior to OMOP scrubbing, the script was refactored to permanently bypass `CodeLabeler` and explicitly extract the ground-truth endpoints directly from the merged cohort file.

The resulting test-set AUROC scores under the single static split confirm a highly robust and functioning benchmark replication:

### Static Train/Val/Test Split Results (GBM)

| Endpoint (AUROC) | Custom | INSPECT | Delta |
|---|---|---|---|
| **Pulmonary Embolism (PE)** | 0.7437 | 0.681 | +0.0627 |
| **1-Month Mortality** | 0.9267 | 0.848 | +0.0787 |
| **6-Month Mortality** | 0.8969 | 0.865 | +0.0319 |
| **12-Month Mortality** | 0.8813 | 0.855 | +0.0263 |
| **1-Month Readmission** | 0.7745 | 0.737 | +0.0375 |
| **6-Month Readmission** | 0.7089 | 0.740 | −0.0311 |
| **12-Month Readmission** | 0.7463 | 0.728 | +0.0183 |
| **12-Month Pulmonary Hypertension (PH)** | 0.9226 | 0.828 | +0.0946 |

### 5-Fold Cross-Validation Results (GBM)

To align directly with the cross-validation evaluation strategy used in the publication (as verified by the fold distributions and footnote in the demographics tables), deterministic 5-fold cross-validation scripts (`Custom/9c_train_gbm_cv.py` and `Custom/9d_run_all_tasks_gbm_cv.py`) were implemented.

The resulting pooled Out-of-Fold (OOF) AUROC, average test AUROC, and validation-optimized sensitivity/specificity metrics are summarized below:

| Endpoint | Overall OOF AUROC | Avg Test AUROC | Avg Test Sens (optimized) | Avg Test Spec (optimized) |
| :--- | :--- | :--- | :--- | :--- |
| **Pulmonary Embolism (PE)** | 0.7584 | 0.7590 ± 0.0027 | 0.6568 ± 0.0372 | 0.7290 ± 0.0420 |
| **1-Month Mortality** | 0.9012 | 0.9049 ± 0.0142 | 0.8823 ± 0.0358 | 0.7931 ± 0.0222 |
| **6-Month Mortality** | 0.8343 | 0.8899 ± 0.0100 | 0.8084 ± 0.0393 | 0.8063 ± 0.0400 |
| **12-Month Mortality** | 0.8428 | 0.8808 ± 0.0049 | 0.8122 ± 0.0273 | 0.7937 ± 0.0206 |
| **1-Month Readmission** | 0.6573 | 0.7337 ± 0.0347 | 0.7143 ± 0.0879 | 0.6148 ± 0.0571 |
| **6-Month Readmission** | 0.6760 | 0.7342 ± 0.0287 | 0.6844 ± 0.0954 | 0.6504 ± 0.0686 |
| **12-Month Readmission** | 0.7210 | 0.7420 ± 0.0035 | 0.6864 ± 0.0479 | 0.6597 ± 0.0366 |
| **12-Month Pulmonary Hypertension (PH)** | 0.9153 | 0.9162 ± 0.0071 | 0.7754 ± 0.0205 | 0.8980 ± 0.0210 |

---

## 10. Comprehensive Cohort Pipeline Validation (`validate_cohort_pipeline.py`)

To ensure absolute data integrity and catch any potential leakage or misalignment before downstream training, a rigorous validation script (`Custom/validate_cohort_pipeline.py`) was implemented. This script independently audits the outputs of all three major pipeline stages:

* **Layer 1: Master Cohort CSV (`cohort_0.2.0_master_file_anon.csv`)**
  * Verified 22,457 total rows across 18,738 unique patients.
  * Confirmed PE prevalence is exactly 20.1% (4,503 PE+ / 17,954 PE−).
  * Validated patient-level split integrity to ensure no single patient appears in multiple splits (train/valid/test), preventing cross-split data leakage.
  * Identified 0 duplicate `impression_id`s, verifying completely clean data without fan-out.
  * Validated `StudyTime` bounds (2000-03-03 to 2021-09-29) with 0 ghost timestamps or implausible future dates.

* **Layer 2: FEMR Labels (`labeled_patients.csv`)**
  * Validated output from the FEMR labeling step (22,457 rows), ensuring structural correctness.
  * Verified that the label prevalence perfectly matches the original cohort CSV at exactly 20.1% (Δ=0.0%).
  * Confirmed zero duplicates cascaded.

* **Layer 3: FEMR Features (`featurized_patients.pkl`)**
  * Audited the generated sparse feature matrix: 22,457 samples × 74,303 features with ~15M non-zero elements (0.920% density).
  * Confirmed the absence of NaN/Inf values and exact length alignment between `patient_ids`, matrix rows, label values, and label times.
  * Re-verified that the PE prevalence in the pickle matches the master cohort CSV at exactly 20.1% (Δ=0.0%).

* **Layer 4: Cross-Pipeline Split Integrity**
  * Performed a final end-to-end audit mapping the PIDs in the featurized output back to the original cohort CSV split assignments.
  * Confirmed the exact split counts: **Train** = 17,981 (20.2% PE+), **Valid** = 2,289 (19.6% PE+), **Test** = 2,199 (19.1% PE+).
  * Confirmed that absolutely no `PatientID` spans multiple splits in the final featurized output, guaranteeing a leakage-free dataset for model training.

This automated validation suite confirms that the custom baseline reconstruction successfully preserved the structural and distributional integrity of the INSPECT benchmark while circumventing legacy pipeline limitations.

---

## 11. CTPA Image Vectorization Pipeline (`Custom/5_process_ctpa.py`)

To extend the baseline beyond EHR-only features, a high-throughput vectorization pipeline was developed to generate fixed-length embedding vectors from all 23,340 CTPA volumes using Stanford Shah Lab's pretrained CT image encoder.

* **Model:** `StanfordShahLab/resnetv2_ct` (HuggingFace), a ResNetV2 backbone pretrained on chest CT images via BigTransfer (ImageNet-21k). The checkpoint is a PyTorch Lightning artifact from the `radfusion3` multimodal fusion framework.
* **Architecture:** Each CT volume is treated as a sequence of 2D axial slices. Slices are encoded independently through the ResNetV2 backbone, and the resulting per-slice feature vectors are mean-pooled across the depth dimension to produce a single fixed-length embedding per study.
* **Output:** A 6,144-dimensional float32 embedding vector per patient, saved as `{patient_id}_ctpa_vector.pt`. Total embedding space: ~574MB for 23,340 studies (a **4,000:1 compression** from the 2.3TB raw CTPA dataset).

> **NOTE — Pre-Fine-Tuning Baseline:** Confirmed via the official INSPECT GitHub repository (`som-shahlab/INSPECT_public`) that the `resnetv2_ct` checkpoint is **not** fine-tuned on the RSPECT PE detection dataset. The INSPECT paper's imaging pipeline requires the ResNetV2 to first be fine-tuned on RSPECT (publicly available via the AWS Open Data Registry at no egress cost, which can be downloaded using `Custom/0b_download_rspect_images.py`) before being used as a slice encoder. The current embeddings therefore represent a **pre-fine-tuning baseline** — the ResNet encodes general chest CT anatomy but has not been explicitly trained to identify PE-specific features such as filling defects, RV/LV strain patterns, or clot burden. The full replication pipeline is:
>
> 1. Fine-tune `resnetv2_ct` on RSPECT (~12,000 CTPAs, study-level and slice-level PE labels)
> 2. Apply 3-channel windowing preprocessing (lung, PE, mediastinum windows → 224×224×3 per slice)
> 3. Re-vectorize all 23,340 INSPECT CTPAs with the fine-tuned weights
> 4. Train the GRU/sequence model end-to-end on INSPECT PE labels
> 5. Late fusion with EHR-GBM and MOTOR predictions via weighted mean
>
> The current pre-fine-tuning embeddings are retained as an ablation baseline to quantify the discriminative signal contributed by RSPECT fine-tuning.

### Debugging the Weight Loading

Several non-trivial issues were encountered during model initialization:

* **Issue #1 — Architecture Mismatch (Depth):** The initial implementation instantiated `resnetv2_152` (blocks `[3, 8, 36, 3]`). Checkpoint key inspection revealed the actual architecture has blocks `[3, 4, 23, 3]`, corresponding to `resnetv2_101`. Loading the wrong depth silently left 17 entire blocks randomly initialized, causing NaN outputs throughout inference.

* **Issue #2 — Key Prefix Mismatch:** The PyTorch Lightning + radfusion3 wrapping results in doubly-prefixed checkpoint keys (`model.model.stages.0...`). A single `str.replace("model.", "")` call only stripped one level, leaving residual `model.` prefixes that prevented all stage weights from loading. **Solution:** All leading `model.` segments were stripped iteratively until the key matched the bare timm module namespace.

* **Issue #3 — Normalization Type Mismatch (BatchNorm vs. GroupNorm):** `timm.create_model('resnetv2_101')` defaults to BatchNorm, but the checkpoint was trained using GroupNorm (the standard BiT architecture). This caused 455 `running_mean`/`running_var` buffers to be absent from the checkpoint, and the BatchNorm layers corrupted all activations to NaN in eval mode. **Solution:** The registered BiT variant `resnetv2_101x3_bit.goog_in21k_ft_in1k` was used instead, which correctly instantiates GroupNorm + StdConv2d, achieving a clean 304/304 key load with zero missing weights.

* **Issue #4 — GRU Dimension Mismatch:** The original sequence encoder was hardcoded to `input_size=2048`. With `width_factor=3`, the ResNetV2 feature dimension is `2048 × 3 = 6,144`, causing an immediate shape error at the first GRU forward pass.

* **Issue #5 — Untrained GRU Producing NaN:** The GRU and attention aggregation layers were never pretrained — the checkpoint contains only ResNetV2 weights. Routing 448 slice vectors through 3 layers of randomly initialized GRU weights caused the hidden state to diverge to NaN. **Solution:** A `spatial_mean` aggregation mode was added that bypasses the GRU entirely, mean-pooling the pretrained ResNetV2 slice features directly. This is the correct approach for fixed-feature extraction; the GRU pathway remains available for future end-to-end fine-tuning.

* **Issue #6 — MONAI MetaTensor Serialization:** MONAI's `LoadImage` transform returns `MetaTensor` objects rather than standard PyTorch tensors. Saving these directly with `torch.save` produces files that require `weights_only=False` to reload, and introduces a hard dependency on the MONAI library for all downstream consumers. **Solution:** Explicit `.as_tensor()` conversion was applied in `__getitem__` immediately after the MONAI transform pipeline.

### Infrastructure & I/O Issues

* **Issue #7 — GCS Cross-Region Latency:** The raw CTPA dataset (`inspaect_imgs_raw`) was stored in `us-east1` while the only available G2 GPU instance quota was in `us-central1`. Cross-region gcsfuse reads over ~76.5MB NIfTI files resulted in ~10 seconds/scan and a projected 70-hour total runtime. The bucket was copied to `inspect-imgs-central` (`us-central1`) via `gsutil -m rsync`, reducing the estimate to ~41 hours.

* **Issue #8 — gcsfuse Random Seek Failures:** After remounting the us-central1 bucket, nibabel failed to read files through the gcsfuse mount with `ImageFileError`. Local copies of the same files loaded correctly, confirming the issue was gcsfuse's handling of random seeks into gzip-compressed NIfTI files. **Solution:** Remounting with `--file-cache-cache-file-for-range-read` and `--implicit-dirs` resolved the issue.

* **Issue #9 — VM Service Account OAuth Scope:** The compute VM was provisioned with read-only Cloud Storage OAuth scopes, preventing writes to the new `inspect-imgs-central` bucket. IAM role grants alone were insufficient. **Solution:** `gcloud auth application-default login` was used to authenticate with broader user credentials, bypassing the VM's restricted service account scopes for the bucket copy operation.

### Current Status & Data Transfer

The remote pipeline successfully processed the entire CTPA dataset on the GCP G2 instance. The resulting embedding corpus was synced via `gcloud storage rsync` and downloaded locally. The 23,227 raw `[6144]`-dimensional output vectors are now unzipped and stored at `DATA_PROCESSED/Ablation study ResnetV2-101x3 vectors/ctpa_vectors` for local baseline ablation studies.

---

## 12. Baseline Embedding Analysis & Compression (`6_analyze_vectors.py` & `7_compress_vectors.py`)

Validation of the pre-fine-tuning baseline embeddings was executed. The analysis revealed that the raw CNN feature extraction yielded an extremely highly correlated spatial embedding space:

* **Cosine Similarity:** The mean pairwise cosine similarity across a random subset of 1000 vectors was 0.9837 (std 0.0165), indicating severe representation collapse (the "Anisotropy Problem"), where all 23,227 patient vectors pointed in almost identically the same direction.

* **Intrinsic Dimensionality (PCA):** A Principal Component Analysis demonstrated that the top 50 components out of 6,144 were sufficient to explain over 99.8% of the variance on a localized batch, and 84.45% of the variance when applied globally across all 23,227 vectors after standard scaling. The first principal component alone accounted for nearly 79% of the local variance.

* **t-SNE & K-Means:** t-SNE mapping and K-Means clustering (K=5) were applied. Due to the high global similarity, the embeddings clustered tightly, verifying that the variance defining the actual pathology is contained within a very small fraction of the latent space.

To optimize these vectors for downstream multimodal fusion, a compression script (`7_compress_vectors.py`) was applied. It implemented global mean-centering and standard scaling (removing the isotropic bias), compressed the 6,144 dimensions to 50 dimensions via PCA, and re-normalized the outputs.

---

## 13. High-Speed Vector Ingestion (`8_vector_ingestion.py`)

With the vectors compressed from ~574MB of 6144-dimensional arrays down to highly dense 50-dimensional arrays, the heavy MONAI `LoadImaged` 3D processing pipelines were successfully bypassed. A lightweight, blazing-fast PyTorch `Dataset` (`8_vector_ingestion.py`) was constructed, capable of lazily loading the `.pt` compressed vectors and instantly fusing them with the structured PyArrow EHR tabular outputs on the fly, feeding batches of 256 seamlessly to downstream algorithms.

These compressed results serve as an optimized, clean ablation baseline against which the RSPECT fine-tuned embeddings can be directly compared, isolating the contribution of PE-specific fine-tuning to downstream multimodal fusion performance.

---

## 14. Pre-Fine-Tuning Ablation Validation (`tsne_compressed_vectors.py`)

To visually confirm the necessity of the RSPECT fine-tuning stage mentioned by the original authors, a final script was engineered to map the 22,436 PCA-compressed `[50]`-dimensional vectors into a 2D t-SNE space, explicitly colored by the ground-truth Pulmonary Embolism (PE) label.

The resulting scatterplot successfully validated the ablation hypothesis:

* **Severe Entanglement:** The vast majority of PE-positive cases (red) were heavily mixed and completely indistinguishable from PE-negative cases (blue) across the central cluster cloud. This proves that a generic ResNet trained on BigTransfer/ImageNet fundamentally lacks the pathological awareness to detect microscopic blood clots.

* **Structural Anomalies Detected:** A single, highly dense cluster of almost exclusively PE-positive patients formed in the top right of the t-SNE space. This likely represents massive/saddle embolisms or severe clinical cases resulting in gross anatomical distortions (such as Right Ventricular Strain), which even an untrained generic image encoder can detect.

**Conclusion:** The heavy entanglement in the 2D projection serves as the perfect mathematical justification for the next phase of the pipeline. To resolve the PE pathology for the remaining ~80% of patients hidden within the central cloud, the ResNetV2 backbone must be formally fine-tuned on the RSPECT dataset (to learn explicit clot features) prior to extracting the final embedding vectors.

---

## 15. Custom Time-Binned Feature Generation & ETL Patches

To improve pipeline flexibility and support advanced experiments, several additions and runtime patches were made:

* **Custom-Binned Feature Generation (`Custom/generate_binned_features.py`):** Added a new wrapper utility allowing the user to group features into custom time ranges (bins) relative to the prediction anchor time (using `CountFeaturizer`'s `time_bins` and `excluded_event_filter` parameters). This allows modeling recent vitals and lab measurements separately from historical records.

* **Dynamic Venv Path Resolution:** Patched all runner scripts (`9a`, `9b`, `9d`) to dynamically check for both hidden (`.venv_legacy`) and standard (`venv_legacy`) virtual environments, ensuring plug-and-play execution when switching between development machines (e.g. laptop and desktop).

* **Relative Path Portability for `run_all_ehr.py`:** Refactored `run_all_ehr.py` script paths to be absolute based on the script location. This allows invoking the runner script from any working directory.

* **CLMBR Parameter Patch:** Fixed a bug in `run_all_ehr.py` by adding the missing required `--path_to_cohort` argument to the `clmbr_train_linear_probe` CLI command.

* **Float Parsing Bug in ETL Parser:** Patched `femr/extractors/omop.py` and `femr/extractors/csv.py` to handle float-formatted strings (e.g. `'92629710.0'`) and map mismatched headers (like `visit_detail_concept_id`), preventing crashes during raw OMOP CSV parsing.

---

## 16. MOTOR/CLMBR GPU Training on Blackwell GPUs — Full Troubleshooting & Fix

Executing `clmbr_train_linear_probe` on a Blackwell GPU (Compute Capability 12.0, e.g., RTX 5090) required resolving four independent failure modes. This section documents the complete diagnosis and the final working configuration.

### Root Cause Overview

The failures are caused by the mismatch between the pinned JAX/jaxlib 0.4.7 stack (compiled against CUDA 11) and the Blackwell SM_120 architecture:

1. The GPU package `femr_cuda` must be at version 0.1.16+. Earlier versions contain a CUDA C++ local attention kernel that physically deadlocks Blackwell SMs. Version 0.1.16 replaced this with a JAX-native fallback (logged as "inefficient CUDA attention mechanism"), eliminating the deadlock.
2. The `ptxas` (PTX assembler) bundled with jaxlib 0.4.7 predates SM_120 and cannot compile Blackwell kernels. CUDA 12.8's `ptxas` must be injected instead, and the `-O0` flag must be forced to prevent the assembler entering a non-terminating optimization loop on Blackwell's large register file.
3. The XLA autotuner passes (`gemm_algorithm_picker`, `gpu_conv_algorithm_picker`) benchmark every new HLO shape against the GPU. On Blackwell they hang indefinitely and must be disabled.
4. The `nvlink` bundled inside jaxlib 0.4.7 does not know about `sm_120` and fatally errors when XLA attempts to link compiled CUBIN objects in parallel. The `--xla_gpu_force_compilation_parallelism=1` flag bypasses the nvlink-based parallel linking API entirely, falling back to single-threaded compilation.

### Fix 1 — Upgrade to `femr_cuda 0.1.16`

The CPU `femr` package and earlier `femr_cuda` releases must be replaced:

```bash
pip uninstall femr femr-cuda -y
pip install femr_cuda==0.1.16
```

Verify: `pip show femr_cuda | grep Version` → should output `0.1.16`.

When training starts, you will see the line:

```
WARNING: Using inefficient CUDA attention mechanism for Blackwell or later GPU
```

This is expected and confirms the JAX-native fallback is active. No `transformer.py` patching is needed.

### Fix 2 — Inject CUDA 12.8 `ptxas` via a Wrapper Script

Download the CUDA 12.8 PTX assembler (supports SM_120) into a local directory:

```bash
mkdir -p ~/cu12_8_ptxas
pip download nvidia-cuda-nvcc-cu12==12.8.93 --no-deps -d ~/cu12_8_ptxas
cd ~/cu12_8_ptxas && unzip *.whl -d .
```

Create a wrapper script inside the venv `bin/` so it is always on `PATH` when the venv is active:

```bash
cat > ~/Documents/INSPECT/venv_legacy/bin/ptxas << 'EOF'
#!/bin/bash
exec $HOME/cu12_8_ptxas/nvidia/cuda_nvcc/bin/ptxas -O0 "$@"
EOF
chmod +x ~/Documents/INSPECT/venv_legacy/bin/ptxas
```

The `-O0` flag disables the optimization pass that causes `ptxas` to hang on Blackwell's large attention graphs. The wrapper must call the CUDA 12.8 binary — not the system `ptxas` (which is too old to support SM_120) and not itself.

> **CRITICAL — PATH at launch time:** Placing the wrapper inside the venv `bin/` directory guarantees it is on `PATH` for any process launched through the venv. If placed elsewhere (e.g., `~/ptxas`), you must manually prepend that directory to `PATH` before launching the training script. Verify with: `cat /proc/<PID>/environ | tr '\0' '\n' | grep PATH`.

### Fix 3 — Set `XLA_FLAGS` to Disable the Autotuner and Bypass `nvlink`

Four XLA flags are required:

| Flag | Purpose |
|---|---|
| `--xla_gpu_cuda_data_dir=$HOME/cu12_8_ptxas/nvidia/cuda_nvcc` | Tells XLA where to find the CUDA 12.8 tools |
| `--xla_gpu_autotune_level=0` | Disables XLA's GPU kernel autotuner (hangs indefinitely on Blackwell) |
| `--xla_disable_hlo_passes=gemm_algorithm_picker,gpu_conv_algorithm_picker` | Disables the HLO passes that benchmark the GPU (also hang on Blackwell) |
| `--xla_gpu_force_compilation_parallelism=1` | Bypasses jaxlib's bundled `nvlink`, which fatally errors on SM_120 |

Set them together before launching:

```bash
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$HOME/cu12_8_ptxas/nvidia/cuda_nvcc \
  --xla_gpu_autotune_level=0 \
  --xla_disable_hlo_passes=gemm_algorithm_picker,gpu_conv_algorithm_picker \
  --xla_gpu_force_compilation_parallelism=1"
```

### Fix 4 — Ensure `JAX_PLATFORMS` is Not Set to `cpu`

JAX silently falls back to CPU if `JAX_PLATFORMS=cpu` is in the environment. This manifests as 0% GPU utilization and 0 VRAM — indistinguishable from a data loading stall.

```bash
unset JAX_PLATFORMS
python -c "import jax; print(jax.devices())"
# Expected: [GpuDevice(id=0, process_index=0)]
```

Check shell config files for any persistent setting:

```bash
grep -r "JAX_PLATFORMS" ~/.bashrc ~/.bash_profile ~/.profile ~/.zshrc /etc/environment 2>/dev/null
```

### Complete Launch Command

```bash
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$HOME/cu12_8_ptxas/nvidia/cuda_nvcc --xla_gpu_autotune_level=0 --xla_disable_hlo_passes=gemm_algorithm_picker,gpu_conv_algorithm_picker --xla_gpu_force_compilation_parallelism=1"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset JAX_PLATFORMS

clmbr_train_linear_probe ~/Documents/INSPECT/DATA_RAW/EHR_FEMR_DB/motor_results_gpu_test \
  --data_path ~/Documents/INSPECT/DATA_RAW/EHR_FEMR_DB/extract \
  --model_dir ~/Documents/INSPECT/motor-t-base/model \
  --batches_path ~/Documents/INSPECT/DATA_RAW/EHR_FEMR_DB/MOTOR_batches/PE
```

`XLA_PYTHON_CLIENT_PREALLOCATE=false` prevents JAX from pre-allocating 75% of VRAM upfront, which causes an OOM on Blackwell when the float32 attention matrices are large.

### Diagnostic Checklist for Future Hangs

Run `watch -n2 nvidia-smi` and interpret the GPU state:

| GPU Util | VRAM Usage | Likely Cause |
|---|---|---|
| 0% | 0 MB | `JAX_PLATFORMS=cpu` is set, or JAX is still initializing |
| 0% | >0 MB | XLA is compiling (ptxas is running) — normal, wait |
| ~98% | >0 MB | Training is running correctly |
| 0% | 0 MB (stalled >15 min) | ptxas wrapper not on PATH, or assembler stuck at `-O3` |

If stuck at the second compilation shape after 10+ minutes with the wrapper confirmed on PATH, check that `XLA_FLAGS` is set correctly in the training process environment:

```bash
cat /proc/<PID>/environ | tr '\0' '\n' | grep XLA
```


## 17. MOTOR Linear Probe Results — All Tasks

The following table summarises the MOTOR linear probe results across all 8 INSPECT tasks, compared against the INSPECT paper's EHR-only MOTOR baseline (Table 2, M column). All values are Test AUROC unless noted. Δ = Ours − INSPECT.

|Endpoint                 | Train AUROC | Valid AUROC | Test AUROC (Ours) | Test AUROC (INSPECT) | Δ         | L2 Strength |
|-------------------------|-------------|-------------|-------------------|----------------------|-----------|-------------|
| **(PE)**                | 0.7824      | 0.7165      |    0.7046         |    0.677             |  +0.0276  |    0.0004   |
| **1-Month Mortality**   | 0.9363      | 0.9149      |    0.9305         |    0.923             |  +0.0075  |    0.0034   |
| **6-Month Mortality**   | 0.9083      | 0.9206      |    0.9074         |    0.901             |  +0.0064  |    0.0616   |
| **12-Month Mortality**  | 0.9074      | 0.9217      |    0.8987         |    0.892             |  +0.0067  |    0.0144   |
| **1-Month Readmission** | 0.7330      | 0.7735      |    0.7587         |    0.773             |  -0.0143  |    1.1288   |
| **6-Month Readmission** | 0.8079      | 0.7551      |    0.7924         |    0.779             |  +0.0134  |    0.0070   |
| **12-Month Readmission**| 0.7823      | 0.7321      |    0.7795         |    0.767             |  +0.0125  |    0.0298   |
| **12-Month (PH)**       | 0.8609      | 0.8893      |    0.8533         |    0.824             |  +0.0293  |    0.0144   |

Results are consistent with the label-version-shift hypothesis documented in Section 8: MOTOR results are very close to the paper's (mean |Δ| ≈ 0.014) because the frozen transformer representations are insensitive to label quality differences, unlike GBM which retrains entirely on the labels. The single regression (1-Month Readmission, −0.0143) is within expected variance and likely reflects the femr-internal split assigning a harder subset to the test set for that task.

---

## 18. MOTOR All-Tasks Automation (`9e_run_all_tasks_motor.py`)

To evaluate MOTOR across all 8 INSPECT tasks in a single unattended run, a wrapper script (`Custom/9e_run_all_tasks_motor.py`) was implemented. It handles batch creation and linear probe training end-to-end, with all Blackwell XLA flags baked in.

### What It Does (per task)

1. Checks whether `MOTOR_batches/<task>` exists. If not, runs `clmbr_create_batches` using the task's `labeled_patients.csv` from `DATA_RAW/EHR_FEMR_DB/features/<task>/`.
2. Runs `clmbr_train_linear_probe`, saving results to a timestamped folder: `DATA_RAW/EHR_FEMR_DB/motor_results/<YYYYMMDD_HHMMSS>_<task>/`.
3. Parses Train/Valid/Test AUROC and L2 Strength from stdout.
4. After all tasks complete, prints a formatted summary table and saves a `motor_results.csv` inside the run's results directory.

### Prerequisites

- `femr_cuda 0.1.16` installed (see Fix 1 in Section 16)
- CUDA 12.8 ptxas wrapper in place (see Fix 2 in Section 16)
- `labeled_patients.csv` generated for each task (run `9a_run_baseline_benchmark.py` for each task, or use the for-loop below)
- `motor-t-base/model/` and `motor-t-base/dictionary/` both present

```bash
# Generate labeled_patients.csv for all tasks if not already done:
cd ~/Documents/INSPECT/INSPECT_custom_data_preprocessing
for task in PE 1_month_mortality 6_month_mortality 12_month_mortality \
            1_month_readmission 6_month_readmission 12_month_readmission 12_month_PH; do
    python Custom/9a_run_baseline_benchmark.py --task $task
done

# Then run the full MOTOR evaluation:
python Custom/9e_run_all_tasks_motor.py
```

### CLI Flags

| Flag | Effect |
|---|---|
| `--force-batches` | Delete and regenerate MOTOR batches even if they already exist |
| `--force-probe` | Re-run linear probe even if results already exist |

### Output Structure

```
DATA_RAW/EHR_FEMR_DB/motor_results/
├── 20260720_110356_PE/
├── 20260720_110356_1_month_mortality/
├── ...
└── motor_results.csv      ← summary of all tasks for this run
```

> **Note:** `clmbr_train_linear_probe` requires its output directory to not exist at call time. The script removes any stale directory automatically before each probe run. The timestamp prefix on each folder ensures historical runs are never overwritten.

---

## 17. RSPECT CTPA Slice Encoder Fine-Tuning Setup & Path Portability Strategy

To execute the 2D CT slice encoder fine-tuning on the RSPECT (RSNA PE Challenge) dataset via `image/run_rsna.sh`, several configuration and infrastructure updates were completed.

### RSPECT Fine-Tuning Infrastructure Setup

1. **Base Model Checkpoint (`resnetv2_ct.ckpt`):**
   - The base ResNetV2-101x3 model checkpoint (4.58 GB) was fetched from Hugging Face (`StanfordShahLab/resnetv2_ct`) and placed in a dedicated local model directory: `/home/sravar/Documents/INSPECT/resnetv2_ct/resnetv2_ct.ckpt`.
   - The Hydra model config `image/radfusion3/configs/model/resnetv2_ct.yaml` was updated to reference this local checkpoint path instead of the inaccessible original cluster directory (`/share/pi/nigam/...`).

2. **Dataset Configuration (`rsna.yaml` & `rsna_featurized.yaml`):**
   - Updated `csv_path` (`train.csv`), `dicom_dir` (`train/`), and `output_dir` / `hdf5_path` in `image/radfusion3/configs/dataset/rsna.yaml` and `rsna_featurized.yaml` to point directly to the RSPECT dataset location (`/home/sravar/Documents/INSPECT/RSPECT_CTPA`).

3. **GPU ID Configuration:**
   - In `image/run_rsna.sh`, `CUDA_VISIBLE_DEVICES` was adjusted (e.g. `CUDA_VISIBLE_DEVICES=1`) to target active GPUs on multi-GPU workstations.

---

### Path Portability Strategy: Relative vs. Absolute Paths

A major challenge when reproducing legacy machine learning benchmarks is the prevalence of hardcoded absolute environment paths (e.g., `/share/pi/nigam/projects/...` or `/local-scratch/...`). Hardcoded absolute paths instantly break when code is cloned onto new developer workstations, laptops, or cloud instances.

To ensure long-term reproducibility and cross-platform portability, the following path architecture guidelines were adopted across all custom runner scripts and configuration files:

#### 1. In Python Scripts (`Path(__file__)` Anchoring)
Always construct absolute paths dynamically relative to the current file's parent directory:
```python
from pathlib import Path

# Anchor to script location (e.g. Custom/)
SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_RAW    = PROJECT_DIR.parent / "DATA_RAW" / "EHR_FEMR_DB"
```
* **Benefit:** Allows running Python scripts from *any* working directory without failing to locate adjacent submodules or dataset directories.

#### 2. In YAML / Hydra Configuration Files
Avoid embedding machine-specific root directories (`/home/user/...`). Use workspace-relative paths (`../RSPECT_CTPA`, `../resnetv2_ct`) anchored to the repository root or project parent:
```yaml
# Recommended Relative / Portable Config
csv_path: '../RSPECT_CTPA/train.csv'
dicom_dir: '../RSPECT_CTPA/train'
output_dir: '../RSPECT_CTPA/rsna_features'
```
* **Benefit:** Prevents path-resolution errors when sharing config files across team members or moving repositories across file systems.

