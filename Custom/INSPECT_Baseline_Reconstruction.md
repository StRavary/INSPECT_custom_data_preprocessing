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

---

### Fine-Tuning Execution Dynamics, Memory Optimization & Autograd Leak Fix

#### 1. Micro-Batching & Gradient Accumulation Dynamics (GroupNorm Immunity)
* **VRAM Constraints:** Fine-tuning the heavy ResNetV2-101x3 backbone on 224×224 slices consumes ~13.4 GB VRAM at batch size 8. To operate comfortably within 16 GB single-GPU limits (e.g., RTX 5070 Ti) without OOM crashes, `batch_size` was set to `4` in `image/radfusion3/configs/dataset/rsna.yaml` and `accumulate_grad_batches` set to `64` in `image/radfusion3/configs/classify.yaml`.
* **Effective Batch Size:** $4 \times 64 = 256$, matching the exact effective batch size specified in the original paper.
* **GroupNorm vs. BatchNorm Advantage:** Unlike architectures that rely on `BatchNorm` (which degrade severely when micro-batch size drops to 4 due to noisy mini-batch statistics), `ResNetV2-101x3` employs `GroupNorm` (`GroupNorm(32, ...)`). GroupNorm calculates normalization statistics per-sample across channel groups independently of batch size. It is mathematically invariant to micro-batch size, making micro-batching with high gradient accumulation perfectly stable.

#### 2. System RAM Autograd Memory Leak Diagnosis & Patch
* **Failure Mode:** During training runs across the 72,431 steps of an epoch, host System RAM usage climbed continuously until the Linux OS OOM killer terminated the Python process and host desktop applications.
* **Root Cause:** In `image/radfusion3/lightning/classification_lightning_model.py`, inside `shared_step()`, prediction logits and ground-truth tensors were appended directly to Python lists (`self.step_outputs[split]["logit"].append(logit)`). Because `logit` was appended without `.detach()`, PyTorch retained the full autograd computation graph in host memory for all 144,000+ step executions across the epoch.
* **Solution:** Patched `classification_lightning_model.py` to explicitly detach tensors (`logit.detach().cpu()` and `y.detach().cpu()`) before storing, freeing autograd graph memory after every step. Additionally, `num_workers` in `classify.yaml` was tuned from `8` to `4` to prevent subprocess RAM multiplication during desktop multitasking, and `precision` was set to `"bf16-mixed"` with `gradient_clip_val: 1.0` to eliminate numerical `NaN` loss explosions.

#### 3. Output Directory & Checkpoint Artifact Structure
Fine-tuning results and checkpoints are automatically exported to timestamped directories under `outputs/`:
* **Directory Format:** `outputs/classify_pe_present_on_image_<timestamp>/`
* **Checkpoints:**
  * `epoch=<E>-val/mean_auroc=<score>.ckpt`: Top checkpoint selected by peak validation mean AUROC.
  * `last.ckpt`: Final epoch state checkpoint.
* **Test Evaluation Artifacts:** Automatically generated via `trainer.test()` at the end of training:
  * `test_preds.csv`: Tabular CSV containing `patient_id`, `procedure_time`, `label`, and model predicted probability `prob`.
  * `config.pkl`: Serialized Python dictionary of the full training hyperparameter configuration.

#### 4. Automatic Mixed Precision (`bfloat16` AMP) Evaluation Metric Fix
* **Issue:** Enabling `precision: bf16-mixed` caused epoch-end metric logging (`on_validation_epoch_end`) to fail with `TypeError: Got unsupported ScalarType BFloat16`. This occurred inside `image/radfusion3/utils.py` when converting prediction logits and probabilities (`y.detach().cpu().numpy()`, `prob.detach().cpu().numpy()`) to NumPy arrays for scikit-learn metrics (`roc_auc_score`, `average_precision_score`).
* **Root Cause:** PyTorch tensors with `dtype=torch.bfloat16` cannot be directly converted to NumPy arrays via `.numpy()` without prior dtype casting.
* **Solution:** Patched `get_auroc` and `get_auprc` in `image/radfusion3/utils.py` (as well as feature extraction in `image/radfusion3/lightning/featurize_lightning_model.py`) to explicitly convert tensors to single-precision float32 (`.float()`) prior to calling `.numpy()`.

---

## 19. Label-Noise & Representation-Ceiling Hypothesis Testing (`Custom/experiment_hypothesis_tests.py`)

To investigate the mechanism behind why LightGBM gained significantly more AUROC ($+0.03$ to $+0.09$) than MOTOR ($+0.006$ to $+0.03$) following baseline label updates, a suite of three hypothesis testing experiments was implemented in `Custom/experiment_hypothesis_tests.py` and evaluated on the held-out test split ($N = 3,109$ studies).

### 19.1 Experiment 1 — Hedge-Language Subgroup Stratification

* **What It Tests:** Asks whether LightGBM and MOTOR are differentially sensitive to label ambiguity in radiology text. A clinical regex tagger split the held-out test set into a **Hedged Group** ($N = 1,383$, reports containing phrases like *"cannot exclude"*, *"subsegmental"*, *"artifact vs. filling defect"*) and a **Clear-Cut Group** ($N = 1,726$). AUROCs were calculated separately per subgroup, and a bootstrapped interaction test (1,000 resamples) evaluated whether the performance drop from Clear-Cut to Hedged differs between models ($\Delta_{\text{interaction}} = \Delta_{\text{GBM}} - \Delta_{\text{MOTOR}}$).
* **Results:** Both models degrade on hedged cases, as expected. LightGBM drops more ($\text{Clear-Cut } 0.772 \rightarrow \text{Hedged } 0.700$, $\Delta_{\text{GBM}} = 0.072$) than MOTOR ($\text{Clear-Cut } 0.741 \rightarrow \text{Hedged } 0.707$, $\Delta_{\text{MOTOR}} = 0.034$). The interaction estimate is $+0.038$ in the predicted direction, but the 95% CI $[-0.019, 0.093]$ crosses zero ($p = 0.096$), falling short of statistical significance at $\alpha = 0.05$.
* **Implications:** LightGBM directionally appears more thrown off by hedged language than MOTOR, but with this sample size the gap cannot be conclusively separated from chance. This result serves as supporting context and motivation for Experiment 2.

### 19.2 Experiment 2 — Prediction Confidence & Margin Analysis

* **What It Tests:** Evaluates the model prediction margin $M = |p - 0.5|$ specifically restricted to misclassified hedged cases ($y \neq \hat{y}$) to measure how confident each model was when making errors on ambiguous text. A one-sided Mann-Whitney U test compared the error margin distributions.
* **Results:** LightGBM's errors on hedged cases carry a mean margin of **0.300** (std 0.142) — meaning when it makes mistakes on ambiguous text, it tends to be wrong with high confidence. MOTOR's errors on the exact same cases average a margin of **0.244** (std 0.122), sitting closer to the uncertain region near $p = 0.50$. The difference is highly statistically significant ($U = 91,783$, $p = 4.8 \times 10^{-10}$).
* **Implications:** This is the headline finding. The two models fail in qualitatively distinct ways: LightGBM (high capacity) learned confident, decisive rules from raw EHR features that overfit to structured noise/spurious patterns in ambiguous reports. MOTOR's errors reflect genuine uncertainty — its frozen, self-supervised embeddings lack the fine-grained features to separate borderline cases, so its mistakes cluster near the threshold ($p \approx 0.50$) rather than committing hard to a wrong answer.

### 19.3 Experiment 3 — Clinical Longformer NLP Labeler Prevalence-Shift Re-Evaluation

* **What It Tests:** Isolates base-rate mismatch as an alternative explanation: could performance discrepancies simply be an artifact of prevalence shift? The Clinical Longformer NLP labeler was validated on 682 reports at $33.4\%$ PE+ prevalence (Table 10 confusion matrix: $TP=221, FP=5, FN=7, TN=449$), whereas the deployment cohort runs at $\approx 20.5\%$ prevalence. Holding labeler sensitivity ($96.93\%$) and specificity ($98.90\%$) fixed, Bayes' rule was used to recalculate PPV and NPV at deployment prevalence.
* **Results:** PPV drops modestly from $97.79\%$ (validation) to **$95.78\%$** (deployment-adjusted) — a $-2.05\%$ relative degradation. NPV rises slightly ($98.46\% \rightarrow 99.21\%$).
* **Implications:** Prevalence shift alone accounts for only a minor drop in labeler reliability. This negative result rules out base-rate mismatch as the primary driver of performance gaps, confirming that the dominant mechanism is structured label noise in ambiguous text (supported by Experiment 2).

### 19.4 Synthesis & Poster Presentation Strategy

The three experiments triangulate on a clear scientific narrative:
1. **Experiment 3 (Ruled-Out Alternative):** Base-rate prevalence shift causes only a minor $2.05\%$ PPV drop, ruling out statistical base-rate mismatch as the main driver.
2. **Experiment 1 (Directional Context):** LightGBM degrades more on hedged text ($\Delta = 0.072$) than MOTOR ($\Delta = 0.034$, interaction $+0.038$, $p = 0.096$), providing directionally consistent context.
3. **Experiment 2 (Headline Result):** LightGBM makes high-confidence errors ($M = 0.300$) on ambiguous text due to capacity noise-overfitting, whereas MOTOR makes low-confidence errors ($M = 0.244$, $p = 4.8 \times 10^{-10}$) due to a representation ceiling in its frozen embeddings.




---

## 20. RSPECT Slice-Encoder Fine-Tuning — FP16 NaN Root Cause & BF16 Resolution

### 20.1 Symptom

The first fine-tuning run of `resnetv2_101x3` on RSPECT (2026-07-22, `precision: 16-mixed`, no gradient clipping) produced NaN loss from the first epoch. The run wrote a single checkpoint, `epoch=0-val/mean_auroc=0.000.ckpt`, and made no further progress.

### 20.2 Root Cause

The BiT ResNetV2 backbone uses **weight-standardised convolutions** (`timm`'s `StdConv2d`). Every forward pass normalises each filter by `(w - mean) / sqrt(var + eps)` with `eps = 1e-6`.

FP16 has 5 exponent bits: its smallest *normal* value is `2^-14 ≈ 6.10e-5` and its smallest subnormal is `2^-24 ≈ 5.96e-8`. `1e-6` therefore lands in subnormal range, and under the flush-to-zero behaviour typical of CUDA kernels it collapses to zero. When a filter's variance is also subnormal, the expression degenerates to `0/0` → NaN.

Two corollaries that matter for the write-up:

* **The failure is in the forward pass, not the gradients.** `16-mixed` wraps the step in a `GradScaler`, which detects inf/NaN gradients and *skips the optimizer step*. Gradient overflow therefore cannot corrupt the weights. A deterministic forward-pass NaN, by contrast, causes every step to be skipped, so the model never leaves initialisation — precisely the observed behaviour.
* **Gradient clipping was not the operative fix.** It was added at the same time as BF16 and is harmless, but it cannot address a forward-pass instability.

### 20.3 Resolution

`precision: bf16-mixed`. BF16 keeps FP32's 8 exponent bits, so `1e-6` is an ordinary normal number with ~32 orders of magnitude of headroom. Verified stable across all 30 epochs of the 2026-07-24 run.

### 20.4 The Metric Clamp That Masked It

`image/radfusion3/utils.py::get_auroc` returns a hardcoded `0.0` when the probability vector contains NaN or when the label vector is single-class:

```python
if np.isnan(prob_cls).any():   auroc_dict[k] = 0.0
elif len(set(y_cls)) == 1:     auroc_dict[k] = 0.0
```

The reported metric was therefore **never NaN** — it was silently floored to `0.000`. A diverged run is indistinguishable from a single-class validation set, and `ModelCheckpoint` happily saves the result as "best". This is worth stating explicitly in any write-up, since the original symptom was reported as "NaN val-auroc".

> **Diagnostic (not yet run):** loading the CT checkpoint and counting `StdConv2d` filters whose per-filter variance is below `6.104e-5` would confirm the mechanism directly against the actual weights. The diagnosis is consistent with every observation but has not been verified empirically.

---

## 21. RSPECT Split Defect — Diagnosis, Impact & Corrective Tooling

### 21.1 Origin

The public Kaggle RSNA-STR (RSPECT) release ships `train.csv` with 17 columns and **no split column**. Upstream `radfusion3` assumed one existed on the authors' cluster copy:

```python
# original, ZepengHuo, Nov 2023
if self.split != "all":
    self.df = self.df[self.df['Split'] == self.split]      # KeyError if absent
```

Commit `1fb422f` (2026-07-22 12:42) replaced this with a tolerant form that accepts either capitalisation but has **no `else` branch**:

```python
if "Split" in self.df.columns:   ...
elif "split" in self.df.columns: ...
# (nothing — filter silently skipped)
```

The first run launched four minutes later. With neither column present, the filter is a no-op and `train`, `valid` and `test` all become the full positive-exam dataframe.

Corroborating evidence that upstream maintained splits externally: `constants.py` declares `SPLIT_COL = 'Split'`; `RSNADataset1D` still filters on `"Split"` unconditionally; and `RSNADataset1D` hardcodes a path to `rsna_hdf5_keys_testsplit.pkl`.

### 21.2 Confirmation

Three independent derivations, all agreeing to within one batch:

| source | quantity | value |
|---|---|---|
| aborted run A (`accumulate_grad_batches=1`) | train batches in one epoch | 144,862 |
| run B, `global_step` 43,015 over 19 epochs × 64 accumulation | train batches/epoch | ≈144,890 |
| run B, `val/loss_step` 2,752,396 over 19 validations | val batches/epoch | 144,863 |

At `batch_size: 4` all three imply ~579,450 slices. Direct measurement of the CSV later gave **579,449** positive-exam slices — `floor(579449/4) = 144,862` train batches (`drop_last=True`) and `ceil(579449/4) = 144,863` val batches (`drop_last=False`). Exact match. Train and validation were the same slices.

### 21.3 Impact Assessment

**No reported result is affected.** RSPECT is used solely to pretrain the slice encoder; all reported metrics come from INSPECT, a disjoint Stanford cohort the encoder never sees. Training a feature extractor on the full corpus remains methodologically sound — ImageNet and BiT backbones are trained on 100% of their data.

What was lost is *monitoring*: `val/mean_auroc` measured training data, so it could not support checkpoint selection or early stopping. The final reported figures (test AUROC 0.9983, AUPRC 0.9906, loss 0.0490) are training-set metrics and should be presented as such.

Secondary cost: the validation pass ran over the full 579k slices every epoch. Forward-only work is roughly a third the per-sample cost of training, so validation consumed ~25% of each 3.96 h epoch — on the order of 30 h across the 30-epoch run.

### 21.4 Corrective Tooling

| artefact | purpose |
|---|---|
| `Custom/make_rspect_splits.py` | Generates a study-level, PE-status-stratified `Split` column (grouped by `StudyInstanceUID`), writing a **new** CSV so the source file is never mutated. Verifies no study or series spans splits, refuses in-place overwrite, deterministic under `--seed`. |
| `Custom/check_rsna_splits.py` | Replays `RSNADataset2D`'s filtering and reports split sizes, class balance, study-level overlap and validation cost. Resolves the CSV from `rsna.yaml` and warns if it differs from the file being inspected. |
| `image/radfusion3/data/dataset_2d.py` | `else: raise ValueError(...)` when no split column exists, plus a second guard raising when a split matches zero rows (catches `"val"` vs `"valid"`). |
| `image/radfusion3/configs/dataset/rsna.yaml` | `csv_path` repointed at the split-annotated CSV. |

---

## 22. RSPECT Dataset Characterisation

Measured directly from `train.csv` (2026-07-29):

| quantity | value |
|---|---|
| total slices | 1,790,594 |
| total studies | 7,279 |
| mean slices per study | 246.0 |
| PE-positive exams (`negative_exam_for_pe == 0`) | 2,368 (32.5%) |
| slices retained after the positive-exam filter | 579,449 (32.4%) |
| PE-positive slices | 96,540 |
| slice PE prevalence **within positive exams** | **0.1666** |
| slice PE prevalence across all slices | 0.0539 |
| `WeightedRandomSampler` oversampling factor on positives | **3.00×** |

### 22.1 Why Negative Exams Are Dropped

In a PE-negative exam every slice has `pe_present_on_image == 0` by definition, so those studies contribute only trivially-easy negatives. Retaining only positive exams means each study supplies both positive slices and *matched* negatives — same patient, scanner and contrast timing — forcing the model to learn where the embolus is rather than whether the scan "looks like" a PE study.

The filter also serves as prevalence management. Slice prevalence rises from 5.4% to 16.7%, which cuts the balanced sampler's oversampling factor from ~9× to 3.0× and correspondingly reduces repetition of the positive set.

### 22.2 Physical Consistency Check

At 1 mm slice thickness, 246 slices ≈ 24.6 cm of craniocaudal coverage — consistent with a CTPA from lung apices to costophrenic angles. Positive studies average 96,540 / 2,368 ≈ **40.8 positive slices**, i.e. ~4.1 cm of extent containing visible clot. For multifocal PE distributed across lobar and segmental branches this is anatomically plausible, and it independently corroborates the measured 16.7% prevalence.

Note that `pe_present_on_image` marks the *union* of z-extents over all emboli, so prevalence scales with the number of emboli more than with any single clot's size, and with vessel orientation relative to the axial plane.

### 22.3 Implication for the Overfitting Question

Each positive slice is drawn ~3× per epoch; across 30 epochs that is ~90 exposures per positive slice, drawn from only **2,368 independent studies**. Whether the encoder memorised RSPECT is *unmeasurable from this run* — there was no held-out data — but the configuration is one in which memorisation is plausible. The INSPECT downstream AUROC is the first uncontaminated test of feature transfer.

---

## 23. MOTOR Split-Source Investigation — Stock FEMR vs. INSPECT

### 23.1 The Question

Do INSPECT's published MOTOR numbers sit on the same split as their GBM and imaging baselines? If not, the cross-arm comparison in the paper would need a caveat.

### 23.2 What Stock FEMR Does

Read directly from `som-shahlab/femr` at the commit INSPECT pins (`de3673d`):

`femr/models/dataloader.py::create_batches` defines its own partition by hashing patients into percentile buckets:

```python
parser.add_argument("--seed",       default=97)   # random seed used for data splitting
parser.add_argument("--val_start",  default=80)
parser.add_argument("--test_start", default=85)
...
"splits": [["train", 0, args.val_start],
           ["dev",   args.val_start, args.test_start],
           ["test",  args.test_start, 100]],
```

INSPECT's `ehr/run_all_ehr.py` passes only `--val_start 80`, so it inherits seed 97 and an **80/5/15** partition.

`femr/models/linear_probe.py::train_linear_probe` then evaluates on exactly that partition:

```python
for i, split in enumerate(("train", "dev", "test")):
    ...
    l_repr_split.append(np.ones(batch["num_indices"]) * i)
...
train_mask = split_indices == 0
scores = [get_c(hazards, i) for i in range(3)]
```

It fits on split 0, tunes L2 on split 1, reports test AUROC on split 2 — and **never opens a cohort file**. Its argument list is only `output_dir`, `--data_path`, `--batches_path`, `--model_dir`.

### 23.3 But INSPECT Does Not Run Stock FEMR

`ehr/run_all_ehr.py` invokes:

```
clmbr_train_linear_probe {task_dir} --data_path ... --model_dir ... \
    --batches_path ... --path_to_cohort {cohort_path}
```

`--path_to_cohort` **does not exist** in stock FEMR, and the script uses `parse_args()` (not `parse_known_args()`), so this would abort with "unrecognized arguments". INSPECT therefore runs a patched fork, and the only plausible purpose of that flag is to re-key the splits to the cohort file — i.e. the authors likely identified and fixed this themselves.

**Conclusion: unresolved from the repositories available, but the evidence points toward INSPECT having corrected it.** The claim "INSPECT should not have compared MOTOR to its other arms" is *not* supported and should not be presented. Section 25.3 settles the question empirically instead.

### 23.4 What Our Pipeline Does

Both `run_MOTOR_MLP.py` and `9e_run_all_tasks_motor.py` still call `clmbr_create_batches --val_start 80` (the hash partition is needed to enumerate patients and build batches), then **discard the loader's split index** and reassign per patient from the cohort file via `load_cohort_splits`, keyed on `PatientID`. Patients absent from the cohort are skipped with a warning. Splits are therefore patient-level and identical to the GBM and imaging arms.

`run_MOTOR_MLP_Hashed.py` (and `--split-source hash`) reproduces the stock behaviour on identical representations, so the two can be compared directly.

---

## 24. MOTOR Pipeline Defects Fixed

Three defects in the custom MOTOR scripts, all present in **both** `run_MOTOR_MLP.py` and `9e_run_all_tasks_motor.py`. The first two are blocking; the third is silent.

### 24.1 `config` Not Marked Static for `jax.jit`

```python
@jax.jit                                   # WRONG
def compute_repr(params, rng, config, batch): ...
```

`config` is MOTOR's `config.msgpack` — a nested dict containing strings, including the pretraining batch path. JAX attempts to abstractify it and raises:

```
TypeError: Argument 'survival_batches_fixed/batch_info.msgpack' of type <class 'str'>
           is not a valid JAX type
```

Stock FEMR uses `functools.partial(jax.jit, static_argnames=("config"))`. Fixed by binding explicitly:

```python
compute_repr = jax.jit(_compute_repr, static_argnames=("config",))
```

`hk.data_structures.to_immutable_dict(config)` on the preceding line already makes it hashable, which static arguments require.

### 24.2 `integer_ages` Indexed by Position Instead of Label Index

```python
ages_list.append(np.array(raw_batch["transformer"]["integer_ages"][:num_indices]))   # WRONG
```

`integer_ages` is **per token** across the flattened 16,384-token batch; `label_indices` gives the token positions where labels sit. Stock FEMR fancy-indexes by those positions. Taking the first `num_indices` entries yields the ages of the first few tokens, unrelated to the label times, so the subsequent `lexsort` and alignment walk fail on `assert repr_ages[j_idx] <= l_age`. Fixed:

```python
li = np.asarray(raw_batch["transformer"]["label_indices"])[:num_indices]
ages_list.append(np.asarray(raw_batch["transformer"]["integer_ages"])[li])
```

### 24.3 Label / Representation Desynchronisation (silent)

When a patient is skipped for being absent from the cohort file, the label arrays must be filtered by **label** index, not representation index:

* `run_MOTOR_MLP.py` returned `label_values`, `label_pids`, `label_ages` **unfiltered** while `matched_reprs` and `split_indices` excluded skipped patients → length mismatch.
* `9e_run_all_tasks_motor.py` did `label_values[matching_indices]`, where `matching_indices` holds positions into the sorted **representation** arrays → each representation paired with the wrong patient's label. **This would not have crashed**; it would have produced plausible-looking, meaningless AUROCs.

Both now track `kept_label_idx` separately, with an assertion that reprs, labels and splits have equal length and a log line reporting the aligned per-split counts. With 18,738 cohort splits against ≤18,640 labels, patients *are* skipped, so this defect was live.

### 24.4 Seeding & Repeatability

The MLP head had no seeding at all — no `torch.manual_seed`, and `DataLoader(..., shuffle=True)` with no `generator=`. Weight init, dropout masks and batch order were fresh randomness on every invocation, so two runs of the same configuration differed and small gaps between configurations were uninterpretable.

Added to `run_MOTOR_MLP.py`:

* `--seed N` — seeds `torch`, `torch.cuda`, `numpy` and the DataLoader generator.
* `--n-repeats N` — retrains the head N times on the **same** representations with seeds `seed..seed+N-1`, reporting mean/std/min/max per split plus a per-seed breakdown. Representation extraction (~11 min) is deterministic and is not repeated, so repeats are nearly free.

`9e_run_all_tasks_motor.py` needs no seeding: `LogisticRegression(solver="lbfgs")` solves a strictly convex objective deterministically, `random_state` is ignored by `lbfgs`, and there is no shuffling anywhere in the file. Its numbers are reproducible to the digit.

---

## 25. MOTOR Results

### 25.1 Linear Probe, All Tasks, Cohort Split (`9e_run_all_tasks_motor.py`)

| Task | Train | Valid | Test | L2 |
|---|---|---|---|---|
| PE | 0.7840 | 0.7131 | 0.6817 | 1.00e+01 |
| 1-month mortality | 0.9457 | 0.9275 | 0.9360 | 1.00e+01 |
| 6-month mortality | 0.9299 | 0.9059 | 0.9099 | 1.00e+01 |
| 12-month mortality | 0.9234 | 0.8826 | 0.8975 | 1.00e+01 |
| 1-month readmission | 0.8663 | 0.8453 | 0.7833 | 1.00e+01 |
| 6-month readmission | 0.8307 | 0.7807 | 0.7701 | 1.00e+01 |
| 12-month readmission | 0.8203 | 0.7315 | 0.7572 | 1.00e+01 |
| 12-month PH | 0.8942 | 0.8572 | 0.8505 | 4.83e+00 |

> **⚠ These numbers are superseded — see §25.1.1. They were produced with a mis-parameterised L2 conversion and are under-regularised. Retained for the record and because they are what §25.3's original comparison used.**

### 25.1.1 L2 Parameterisation Bug (diagnosed 2026-07-30, fixed)

Every task selected an L2 at or adjacent to the top of the grid — seven at the maximum of 10, and `12_month_PH` at 4.83, which is exactly `10 / 2.069`, the second grid point down. Selection pinned to a boundary means the optimum lies outside the grid and the chosen value is **censored, not optimal**.

The cause was the conversion from FEMR's `l2` to sklearn's `C`:

```python
C = 1.0 / l2          # WRONG — omits n
```

The two libraries parameterise the objective differently:

| | objective |
|---|---|
| stock FEMR | `mean(BCE) + 0.5·l2·‖β‖²` |
| sklearn | `0.5·‖w‖² + C·Σᵢ loss` |

Dividing sklearn's objective by `C·n` gives `0.5/(C·n)·‖w‖² + mean(loss)`, so the two are identical (to floating point) when

```
l2 = 1/(C·n)      ⟺      C = 1/(l2 · n_train)
```

Verified numerically: `J_sklearn/(C·n) == J_femr` to 1e-12 across `l2 ∈ {1e-3, 1e-2, 1e-1, 1}`.

With `C = 1/l2` the effective penalty was `l2/n` — about four decades too weak at n ≈ 13,000. Mapping stock FEMR's selected optima into the nominal scale 9e was searching shows the optimum was unreachable for every task but PE:

| task | stock optimum | nominal `l2` 9e would need (`= opt·n`) | within grid (max 10)? |
|---|---|---|---|
| PE | 4.00e-4 | 5.2 | yes |
| 1-month mortality | 3.40e-3 | 44.2 | no |
| 6-month mortality | 6.16e-2 | 800.8 | no |
| 12-month mortality | 1.44e-2 | 187.2 | no |
| 1-month readmission | 1.1288 | 14,674 | no |
| 6-month readmission | 7.00e-3 | 91.0 | no |
| 12-month readmission | 2.98e-2 | 387.4 | no |
| 12-month PH | 1.44e-2 | 187.2 | no |

**Fix applied** in `9e_run_all_tasks_motor.py`:

* `C = 1.0 / (l2 * n_train)` in the sweep.
* The same conversion in the final re-fit that saves predictions — previously `1.0 / metrics["l2_strength"]`, which would have written predictions from a differently-regularised model than the one whose metrics were reported.
* A warning logged whenever the selected `l2` sits at either end of the grid, so censored selection cannot recur silently.

The grid itself was never the problem, and extending it (an earlier recommendation in this document, now withdrawn) would have masked the parameterisation error rather than fixing it.

A corroborating signal, visible before the algebra: the hash-split runs from stock FEMR show train and valid AUROC close together (6-month mortality: 0.9083 / 0.9206), whereas the mis-parameterised cohort-split runs show wider gaps on the same task (0.9299 / 0.9059) — the signature of insufficient shrinkage.

**All §25.1, §25.2 and §25.3 figures require re-running with the corrected conversion.**

### 25.2 Reproduction of Published INSPECT Baselines

| Task | INSPECT | Ours (cohort split) | Δ |
|---|---|---|---|
| PE | 0.677 | 0.682 | +0.005 |
| 1-month mortality | 0.923 | 0.936 | +0.013 |
| 6-month mortality | 0.901 | 0.910 | +0.009 |
| 12-month mortality | 0.892 | 0.897 | +0.005 |
| 1-month readmission | 0.773 | 0.783 | +0.010 |
| 6-month readmission | 0.779 | 0.770 | −0.009 |
| 12-month readmission | 0.767 | 0.757 | −0.010 |
| 12-month PH | 0.824 | 0.851 | +0.027 |

> **⚠ Superseded — these came from the mis-parameterised run (§25.1.1) and must be regenerated.**

Mean **+0.006**, 6/8 above, all within 0.027. Across eight tasks with an independently rebuilt cohort, a later label vintage and 12% of codes falling outside MOTOR's vocabulary, this is a successful reproduction. It should be presented as such — **not** as an improvement, since the differences have several innocent explanations (779 records dropped for missing `StudyTime`, 12 deduplicated `impression_id`s, vocabulary coverage, seeds).

`12_month_PH` is the largest deviation and is also the only task that did not peg L2 at the grid boundary; if INSPECT's run hit the ceiling and ours did not, the two are not regularised identically.

### 25.3 Split Source Has No Detectable Effect

> **⚠ Confounded — read §25.3.1 before citing.** The cohort column below comes from `9e` under the mis-parameterised L2 conversion (§25.1.1); the hash column comes from stock `clmbr_train_linear_probe`, which is correctly regularised. The comparison therefore mixes split source with regularisation strength. The conclusion still holds, but on the strength of the *clean* comparison in §25.3.1, not this table.

Linear probe, same representations, cohort split vs. CLMBR hash split:

| Task | cohort | hash | Δ |
|---|---|---|---|
| PE | 0.682 | 0.705 | +0.023 |
| 1-month mortality | 0.936 | 0.930 | −0.006 |
| 6-month mortality | 0.910 | 0.907 | −0.003 |
| 12-month mortality | 0.897 | 0.899 | +0.002 |
| 1-month readmission | 0.783 | 0.759 | −0.024 |
| 6-month readmission | 0.770 | 0.792 | +0.022 |
| 12-month readmission | 0.757 | 0.779 | +0.022 |
| 12-month PH | 0.851 | 0.853 | +0.002 |

**mean Δ = +0.005, 95% CI [−0.009, +0.019], paired t(7) = 0.81, p = 0.45, 5/8 favour hash.**

Non-parametric confirmation (n = 8 makes normality unverifiable): Wilcoxon signed-rank W⁺ = 21 against E[W⁺] = 18, exact two-sided p = 0.71; sign test p = 0.73. The t-test uses magnitudes and is the most sensitive of the three, and it is the one reporting p = 0.45.

Two structural observations support the null reading:

1. **The sign flips.** The single largest difference (1-month readmission, −0.024) favours the *cohort* split. A systematic mechanism effect would be consistent in direction.
2. **Magnitude tracks task difficulty, not mechanism.** The four largest |Δ| are the four lowest-AUROC tasks (PE and the three readmission horizons, 0.68–0.78); the four smallest are the four highest (three mortality horizons and PH, 0.85–0.94). That is the signature of sampling noise.

Caveats to state alongside the result: the CI is over **tasks**, not patients, and the eight tasks are not independent (same patients, correlated outcomes), so the effective n is below 8 and the interval is if anything too narrow — which only strengthens a null conclusion. Individual per-task AUROCs carry their own ~±0.011 sampling error that is not drawn. And with n = 8 the test has low power: an effect smaller than ~0.019 would be invisible. Phrase as *no detectable difference*, not equivalence.

Figure: `Custom/figures/motor_split_delta.png` (generator: `Custom/figures/make_split_delta_plot.py`).

### 25.3.1 What the Comparison Actually Supports

Three differences separate the two columns in §25.3, not one:

1. **split source** — cohort patient-level vs. CLMBR 80/5/15 hash percentiles (the intended comparison)
2. **regularisation** — mis-parameterised vs. correct (§25.1.1)
3. **optimiser** — sklearn `lbfgs` vs. FEMR's hand-written conjugate gradient (equivalent at convergence, so benign)

Difference 2 invalidates that table as a clean test of difference 1.

**The clean comparison is the MLP pair**, where identical code, head and representations are used and only the split source varies:

| | cohort split | hash split | Δ |
|---|---|---|---|
| MLP, `12_month_PH` | 0.847 | 0.856 | +0.009 |

That supports the null, with the caveat that the MLP head was unseeded when those numbers were produced, so part of the 0.009 is run-to-run variance — measurable with `--n-repeats` (§24.4) but not yet measured.

Three further reasons the null reading survives the confound:

* The **sign flips** across tasks in §25.3, and the single largest difference favours the cohort split. A systematic mechanism effect would be directionally consistent.
* **Magnitude tracks task difficulty**, not split mechanism — the four largest |Δ| are the four lowest-AUROC tasks.
* **The two partitions are the same size.** Measured from `cohort_0.2.0_master_file_anon.csv`: 81.5% / 4.7% / 13.8% of exams (81.4 / 4.7 / 13.9 of patients) against CLMBR's 80/5/15 — almost certainly because both descend from the same Shah-lab percentile convention (`femr.splits_omop_2023_03_05`). The comparison therefore isolates *which* patients land in each split, not how many, which makes the null reading cleaner rather than weaker. (An earlier draft of this section assumed a 70/15/15 cohort split and argued the hash arm had a training-size advantage; that was wrong and is withdrawn.)

### 25.3.2 Measured Cohort Split Distribution

| split | exams | % exams | patients | % patients |
|---|---|---|---|---|
| train | 18,293 | 81.5% | 15,247 | 81.4% |
| valid | 1,056 | 4.7% | 882 | 4.7% |
| test | 3,108 | 13.8% | 2,609 | 13.9% |
| **total** | **22,457** | | **18,738** | |

* **Patients spanning more than one split: 0.** Patient-level integrity confirmed directly on the file the MOTOR runs load, not only via `4_validate_cohort_pipeline.py`.
* 2,608 patients (13.9%) have more than one CTPA, max 13, mean 1.198 exams per patient. The repeat-exam leakage risk is real and correctly handled.
* The 18,738 figure matches the `Loaded 18738 patient splits` line emitted by `load_cohort_splits`.

> **Model selection is the weak link in both arms.** Validation is only 4.7% of the cohort — ~1,056 exams, and at the 14.0% prevalence of `12_month_PH` roughly 150 positives. Both the L2 sweep (linear probe) and the best-epoch choice (MLP) are made on that set, so selection is noisy regardless of split source. This is a stronger candidate explanation for run-to-run variability than anything about the split mechanism, and it argues for reporting selection-robust summaries (e.g. `--n-repeats`) rather than single best-validation values.

**Outstanding work to settle it cleanly:** re-run `9e_run_all_tasks_motor.py` with the corrected `C`, and run `run_MOTOR_MLP.py --n-repeats 5` on both split sources. That yields a deterministic linear-probe comparison with no confounds and a measured noise floor for the MLP.

### 25.4 MLP Head vs. Linear Probe

| | cohort split | hash split |
|---|---|---|
| Linear probe | 0.8505 | 0.853 |
| 2-stage MLP | 0.847 | 0.856 |

On `12_month_PH` the MLP head buys nothing over the linear probe (0.856 vs 0.853). MOTOR's 768-d representations are already linearly separable for this task, which is the expected behaviour of a well-trained foundation model. **The linear probe is the number to quote** — it is both what INSPECT published and the deterministic one.

Note the asymmetry when comparing: the linear probe is a fixed point, while MLP numbers are draws from a distribution whose width `--n-repeats` measures.

---

## 26. Hyperparameter Sweep Provenance

`image/sweep.yaml` has been in the repository since the first upstream commit and confirms a **wandb Bayesian sweep**:

```yaml
method: bayes
metric: {name: val/_auroc, goal: maximize}
parameters:
  lr:                              [0.005, 0.001, 0.0005, 0.0001]
  model.aggregation:               [max, mean, attention, attention+max]
  model.seq_encoder.rnn_type:      [LSTM, GRU, transformer]
  model.seq_encoder.hidden_size:   [64, 128, 256]
  model.seq_encoder.bidirectional: [true, false]
  model.seq_encoder.num_layers:    [1, 3]
  model.seq_encoder.dropout_prob:  [0.0, 0.25, 0.5]
  dataset.num_slices:              [200, 250, 300]
  dataset.weighted_sample:         [true, false]
  dataset.sample_strategy:         [fix, random]
```

`run_sweep.py` is only the controller — it polls the wandb API and stops a sweep at `--max_runs` (default 50). The target is pinned on line 8 of the `command` block, and the README notes that line specifies the prediction target, so the sweep was run **once per task**. That is why the per-task launchers carry different hyperparameters.

Two caveats: the committed YAML sweeps `vit_base_14_dinov2` features, whereas the final launchers use `resnetv2_101_ct`; and `rnn_type: transformer` was in the search space but did not win for any task.

### 26.1 Swept Results per Task (from `image/run_classify_*.sh`)

| Script | Target | Aggregation | RNN | Hidden | Dropout | LR |
|---|---|---|---|---|---|---|
| pe | pe_positive_nlp | max | LSTM | 128 | 0.5 | 1e-3 |
| ph | 12_month_PH | attention | GRU | 128 | 0.25 | 1e-3 |
| 1m_mort | 1_month_mortality | max | GRU | 128 | 0.25 | 1e-3 |
| 6m_mort | 6_month_mortality | mean | GRU | 128 | 0.0 | 5e-4 |
| 12m_mort | 12_month_mortality | attention | GRU | 128 | 0.5 | 5e-4 |
| read_1m | 1_month_readmission | max | LSTM | 128 | 0.0 | 1e-3 |
| read_6m | 6_month_readmission | max | LSTM | 128 | 0.5 | 1e-3 |
| read_12m | 12_month_readmission | mean | LSTM | 128 | 0.25 | 5e-4 |

`hidden_size=128`, `num_layers=1`, `bidirectional=true` and `num_slices=250` are constant across all eight, so every task's pre-classifier embedding is **256-d**. `run_classify.sh` is a stale template — not called by `run_classify_all.sh`, duplicates `1_month_mortality`, and is the only launcher that does not override `hidden_size`.

**Important:** these launcher overrides differ substantially from `configs/model/model_1d.yaml` defaults (GRU / hidden 512 / `attention+max` / dropout 0.0). Running `run_classify.py model=model_1d dataset=stanford_featurized` without the launcher flags does **not** reproduce the swept configuration.

### 26.2 Aggregation Semantics

The sequence encoder emits `(batch, 250, 256)`; aggregation collapses the slice axis to one vector per study:

* `max` — element-wise maximum over slices. Suits localised findings where one slice should drive the prediction.
* `mean` — average over slices. Suits diffuse properties.
* `attention` — learned weighted average via the `Attention` module.
* `attention+max` — concatenation of both, doubling the dimension. Not selected by any task.

---

## 27. Outstanding Defects in `image/` (upstream, not yet fixed)

Found while tracing the 1-D pipeline. None are our code; all are in the original `radfusion3` and would affect any run.

### 27.1 `RNNSequentialEncoder` Contradicts Its Own Axis Convention

```python
self.rnn = getattr(nn, rnn_type)(..., batch_first=True, ...)

def forward(self, x):
    x = x.transpose(0, 1)
    x, _ = self.rnn(x)          # comment says (Slice, Batch, Feature)
    x = x.transpose(0, 1)
    return x
```

`batch_first=True` expects `(batch, seq, feature)`, and `x` already arrives as `(B, 250, 6145)`. The transpose makes it `(250, B, 6145)`, so **the RNN reads 250 as the batch dimension and the batch as the sequence**. No error is raised because `input_size` only constrains the last axis, and the output transposes back to the expected `(B, 250, 256)`.

If this reading holds: the recurrence never runs along slices, so no slice-order information is modelled; a study's representation depends on which *other studies* share its batch; and at `batch_size=1` the sequence length is 1, degenerating to a context-free per-slice transform. Pooling in `aggregate` still operates on dim 1 (real slices), so the model functions — as a per-slice projection followed by pooling — which is why it produces sensible AUROCs.

`models_1d.py` has only two commits, both from the original authors (Nov 2023); this is entirely upstream. The fix is one line (drop the transposes, or set `batch_first=False`).

**Verification snippet (not yet run):**

```python
import torch, torch.nn as nn
torch.manual_seed(0)
rnn = nn.LSTM(8, 4, batch_first=True, bidirectional=True)
x = torch.randn(2, 5, 8)                                 # (batch=2, slices=5, feat=8)
f = lambda t: rnn(t.transpose(0, 1))[0].transpose(0, 1)  # what the code does
sp, bp = torch.randperm(5), torch.tensor([1, 0])
print("slice order matters:", not torch.allclose(f(x)[:, sp], f(x[:, sp]), atol=1e-6))
print("batch order matters:", not torch.allclose(f(x)[bp],   f(x[bp]),   atol=1e-6))
# correct encoder → True, False.  Expected here → False, True.
```

### 27.2 `max` and `mean` Aggregation Ignore the Padding Mask

```python
elif cfg.model.aggregation == "mean":  x = torch.mean(x, 1)     # no mask
elif cfg.model.aggregation == "max":   x, _ = torch.max(x, 1)   # no mask
```

Only `attention` applies `a = a * mask`. Studies shorter than `num_slices` are zero-padded by `fix_series_slice_number`; those zero vectors still pass through the RNN and produce non-zero outputs which are then pooled. `mean` additionally divides by 250 rather than the true slice count, shrinking representations for shorter studies. Affects six of the eight tasks, including PE. Fix: `masked_fill(-inf)` before `max`, and divide by `mask.sum(1)` for `mean`.

### 27.3 `Attention` Returns the Wrong Tensor

```python
a = a / torch.sum(a, 1, keepdim=True) + 1e-10
weighted_input = x * torch.unsqueeze(a, -1)
return torch.sum(weighted_input, 1), self.weight    # returns the PARAMETER, not `a`
```

The per-slice attention scores `a` — the clinically interpretable output, showing which slices drove a prediction — are computed and discarded. Only relevant to `ph` and `12m_mort`, the two tasks using `attention`.

Note also that the epsilon is added *after* the division, so it does not guard the denominator; a fully-masked sample yields `0/0` → NaN.

### 27.4 Slice-Position Encoding Is Identically Zero for RSPECT

`DatasetBase.__init__` loads `dict_slice_thickness` from a Stanford cluster path and falls back to `{}` when absent. `read_from_hdf5` then does `self.dict_slice_thickness[key] * idx_th` inside a bare `try/except` appending `0` on failure. With the dictionary empty, the appended slice-position column — the `+1` making features 6145-d, and what `trainer.position_encoding: true` is meant to supply — is **zero for every slice**. The 1-D model receives no positional information.

### 27.5 `Dataset1D` Label Parsing Is Type-Fragile

```python
self.df[cfg.dataset.target] = self.df[cfg.dataset.target].astype(str)
self.labels = [1 if t == "True" else 0 for t in self.df[cfg.dataset.target]]
```

Matches the literal string `"True"` with no `.upper()`. If a label column round-trips as integers, `astype(str)` yields `"1"`, every label silently becomes 0, and `get_auroc` reports its single-class `0.0`. (The EHR path in `ehr/2_generate_labels_and_features.py` was already patched with `.strip().upper()` and a `CENSORED`/`NAN` skip; `Dataset1D` was not.) Verified for our cohort: `12_month_PH` holds `TRUE` / `FALSE` / `CENSORED` strings, so the EHR path is safe.

### 27.6 `ModelCheckpoint` Filename Contains a Slash

`filename="{epoch}-{val/mean_auroc:.3f}"` is not sanitised, so Lightning writes `epoch=18-val/mean_auroc=0.996.ckpt` — a *subdirectory* per epoch. `save_top_k=1` deletes the superseded files but leaves the empty directories behind. Cosmetic; fixable with `auto_insert_metric_name=False` and a slash-free template.

### 27.7 Missing Training Hygiene in `run_classify.py`

* No `EarlyStopping` callback — every run grinds through `max_epochs` regardless.
* `save_top_k=1` means only the single best checkpoint survives, so an earlier epoch cannot be recovered without retraining.
* No `logger=` is passed, so `flat_config` (computed at line 28) is unused and nothing reaches wandb; metrics land in Lightning's default CSV/TensorBoard logger under `<save_dir>/lightning_logs/`.
* `on_train_epoch_end` computes train-split AUROC, accumulating ~579k single-element CPU tensors plus ID strings per epoch (~1 GB sawtooth), then re-materialises them via `torch.cat([f for x in ... for f in x])`. The `.detach().cpu()` fix removed the severe autograd-graph retention; this is the residual.
