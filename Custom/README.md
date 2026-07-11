# Custom INSPECT Baseline Reproduction Pipeline

This directory contains the custom scripts engineered to successfully reproduce the INSPECT baseline benchmark cohort, bypassing undocumented missing data errors, un-anonymization artifacts, and hardcoded infrastructure dependencies.

## Quickstart & Prerequisites

To completely rebuild the baseline dataset from scratch using these custom scripts, you must ensure the following four prerequisites are met. Failure to do so will result in missing data or authentication errors.

### 1. Redivis Authentication
Script `1_INSPECT_DL_EHR.py` connects to the Redivis API to download the raw OMOP tables. You must authenticate your session before running it.
* Generate a Redivis API token from your account settings.
* Export it to your terminal environment:
  ```bash
  export REDIVIS_API_TOKEN="your_token_here" #Safeguard API token via .env importing and .env in .gitignore
  ```

### 2. Manual AIMI Portal Downloads
The INSPECT dataset is split across two portals. The ground-truth labels and imaging crosswalks are NOT hosted on Redivis and cannot be downloaded programmatically due to strict Data Use Agreements (DUA).
* Log in to the [Stanford AIMI Portal](https://stanford.redivis.com/datasets/2n96-d71hggrbf).
* Manually download `labels_20250611.tsv`, `study_mapping_20250611.tsv`, `splits_20250611.tsv`, `series_metadata_20250611.tsv`, and `image_ehr_crosswalk_20250418.csv` (or use `download_aimi_labels.py` -  manal download recommended).
* Place labels, mapping, splits, and metadata files in `DATA_RAW/LABELS/`. The crosswalk file should be placed in `DATA_PROCESSED/`.

### 3. The "Hidden" FEMR Compilation Step
After downloading the raw CSVs (Step 1) and before merging the labels (Step 2), you **must** compile the longitudinal patient database using the legacy `femr` framework script provided in the original repository.
* **Execution Order:**
  1. Run `Custom/1_INSPECT_DL_EHR.py`
  2. **Run `python ehr/1_csv_to_database.py` (from the original repository)**
  3. Run `Custom/2_merge_labels.py`

### 4. Python Environment Dependencies
To execute the legacy portions of the pipeline (like `femr` extraction and the baseline GBM models), you **must** strictly use the exact environment specifications provided by the original authors to avoid severe C-API crashes (e.g., Numpy 2.x incompatibilities) and Pandas syntax deprecations.

1. **First, install the exact legacy base environment:**
   ```bash
   pip install -r ehr/requirements.txt
   ```
2. **Next, install the custom pipeline supplements:**
   This custom pipeline relies on modern utility libraries for the new modalities (e.g., PyArrow for massive matrix processing, Streamlit for data validation, PyTorch/MONAI for image ingestion). These are defined in [addition_reqs.txt](addition_reqs.txt) and should be installed *after* the base requirements:
   ```bash
   pip install -r Custom/addition_reqs.txt
   ```

## 📂 Execution Order

Once the prerequisites are satisfied, execute the scripts in the following numbered order. The pipeline is separated into four logical phases:

### Phase 1: Downloading Raw Data
0. `0a_download_aimi_labels.py`, `0b_download_rspect.py`, `0c_download_ctpa_images.py`, & `0d_download_rspect_images.py`: Fetch standard clinical tags, RSPECT image data, raw CTPA scans, and the full RSPECT dataset from the AWS Open Data Registry respectively.

### Phase 2: EHR Tabular Processing & Sanity Checks
1. `1_INSPECT_DL_EHR.py`: Downloads raw OMOP tables from Redivis.
2. *(Run legacy `ehr/1_csv_to_database.py`)*
3. `2_merge_labels.py`: Reconstructs the master cohort using OMOP clinical event anchoring to recover "ghost" patients.
4. `3_custom_sanity_checks.py`: Validates the integrity of the generated sparse feature matrices.
5. `4_validate_cohort_pipeline.py`: Runs comprehensive checks on the dataset split sizes and potential target leakages.

### Phase 3: 3D Image Ingestion & Vector Processing
6. `5_process_ctpa.py`: Extracts 6144-dim pre-trained vectors from the raw CTPA 3D volumes.
7. `6_analyze_vectors.py`: Performs analytical calculations on the vectors, including PCA variance explanation, cosine similarity clustering, and t-SNE mapping.
8. `7_compress_vectors.py`: Drops the isotropic dimensionality by standard-scaling and performing PCA to retain 50 components (holding ~84.5% variance globally) to optimize PyArrow/MONAI I/O performance.

### Phase 4: Datasets, ML Training & Benchmarks
9. `8_vector_ingestion.py`: High-speed multimodal PyTorch dataset to seamlessly fuse EHR tabular PyArrow frames with the compressed 50-dim image vectors.
10. `9a_run_baseline_benchmark.py`: Wrapper to execute legacy feature extraction while bypassing hardcoded cluster weights. That is done because the scripts were developed on a system with only CPU available so we are unable to run MOTOR for baseline.

> **Note:** For a highly detailed breakdown of the exact engineering steps and debugging taken to reconstruct the baseline (including the 5-tier OMOP fallback logic), see `INSPECT_Baseline_Reconstruction.md`.

## EHR Baseline Evaluation & Auxiliary Tasks

To evaluate the extracted EHR features against the pulmonary embolism (PE) endpoint and all 7 auxiliary prognostic endpoints (1, 6, 12-month mortality/readmission, and 12-month PH), an automated evaluation script was introduced.

### Execution Step
11. `9b_run_all_tasks_gbm.py`: Iteratively trains and evaluates the GBM baseline across all tasks, extracting and saving test-set AUROC scores.

### Modifications to Original `ehr/` Scripts
Executing the auxiliary tasks and the master pipeline successfully required patching legacy bugs in the original `ehr/` files:

#### `ehr/2_generate_labels_and_features.py`
1. **Bypassed `CodeLabeler`:** The original script searched the OMOP `condition_occurrence` tables for precise death/readmission codes. Because Stanford scrubbed these exact codes from the public dataset to preserve patient privacy, the labeler silently failed and yielded 100% `False` labels. The script was refactored to extract the true, pre-computed outcomes directly from our merged master cohort CSV.
2. **Fixed FEMR API Deprecation:** The `femr` v0.2.x API deprecated the `patient_ids` keyword argument in `labeler.apply()`. Passing it caused a `TypeError` that completely broke the pipeline. This argument was removed.
3. **Corrected Case Sensitivity:** For the `12_month_PH` endpoint, the original script strictly checked `label == "True"`. Because the 2025 AIMI dataset exports this column as fully capitalized (`"TRUE"`), a simple string normalization was implemented to prevent false negatives.

#### `ehr/run_all_ehr.py`
1. **Added `--extract_path` Argument:** The original master script hardcoded the FEMR database location to `inspect_femr_extract/extract` within the output directory. A custom `--extract_path` argument was added. This allows the pipeline to point directly to pre-generated multi-gigabyte database extracts (like the 21GB `event_metadata` extract) located anywhere on disk, seamlessly bypassing the expensive database creation step.

> **Note on New Data Drops (June 2025):** Although new `splits_20250611.tsv`, `series_metadata_20250611.tsv`, and crosswalk files were added to the pipeline to finalize the cohort, the underlying Redivis clinical data is still heavily scrubbed. Therefore, the custom bypasses implemented in the `/ehr` scripts (skipping ghost patients missing from Redivis and avoiding the OMOP `CodeLabeler`) **must remain completely intact** and should not be reverted.
