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
* Log in to the [Stanford AIMI Portal](https://aimi.stanford.edu).
* Manually download `labels_20250611.tsv`, `study_mapping_20250611.tsv`, `splits_20250611.tsv`, and `series_metadata_20250611.tsv` (or use `download_aimi_labels.py`).
* Place all files exactly in `DATA_RAW/LABELS/`.

### 3. The "Hidden" FEMR Compilation Step
After downloading the raw CSVs (Step 1) and before merging the labels (Step 2), you **must** compile the longitudinal patient database using the legacy `femr` framework script provided in the original repository.
* **Execution Order:**
  1. Run `Custom/1_INSPECT_DL_EHR.py`
  2. **Run `python ehr/1_csv_to_database.py` (from the original repository)**
  3. Run `Custom/2_merge_labels.py`

### 4. Python Environment Dependencies
This custom pipeline requires specific libraries that may not be present in the base legacy environment (e.g., PyArrow for massive matrix processing, Streamlit for data validation).
Ensure your active virtual environment has the following installed:
```bash
pip install pandas numpy pyarrow streamlit python-dotenv redivis torch monai
```
*(Note: `femr` must also be installed in your active virtual environment to execute the legacy portions of the pipeline).*

## 📂 Execution Order

Once the prerequisites are satisfied, execute the scripts in the following numbered order:

1. `1_INSPECT_DL_EHR.py`: Downloads raw OMOP tables.
2. *(Run legacy `ehr/1_csv_to_database.py`)*
3. `2_merge_labels.py`: Reconstructs the master cohort using OMOP clinical event anchoring to recover "ghost" patients.
4. `3_run_baseline_benchmark.py`: Wrapper to execute legacy feature extraction while bypassing hardcoded cluster weights. That is done because the scripts were developed on a system with only CPU available so we are unable to run MOTOR for baseline.
5. `4_custom_sanity_checks.py`: Validates the integrity of the generated sparse feature matrices.

> **Note:** For a highly detailed breakdown of the exact engineering steps and debugging taken to reconstruct the baseline (including the 5-tier OMOP fallback logic), see `INSPECT_Baseline_Reconstruction.md`.

## EHR Baseline Evaluation & Auxiliary Tasks

To evaluate the extracted EHR features against the pulmonary embolism (PE) endpoint and all 7 auxiliary prognostic endpoints (1, 6, 12-month mortality/readmission, and 12-month PH), an automated evaluation script was introduced.

### Execution Step
6. `run_all_tasks_gbm.py`: Iteratively trains and evaluates the GBM baseline across all tasks, extracting and saving test-set AUROC scores.

### Modifications to `ehr/2_generate_labels_and_features.py`
Executing the auxiliary tasks successfully required patching three major legacy bugs in the original `ehr/2_generate_labels_and_features.py` file:
1. **Bypassed `CodeLabeler`:** The original script searched the OMOP `condition_occurrence` tables for precise death/readmission codes. Because Stanford scrubbed these exact codes from the public dataset to preserve patient privacy, the labeler silently failed and yielded 100% `False` labels. The script was refactored to extract the true, pre-computed outcomes directly from our merged master cohort CSV.
2. **Fixed FEMR API Deprecation:** The `femr` v0.2.x API deprecated the `patient_ids` keyword argument in `labeler.apply()`. Passing it caused a `TypeError` that completely broke the pipeline. This argument was removed.
3. **Corrected Case Sensitivity:** For the `12_month_PH` endpoint, the original script strictly checked `label == "True"`. Because the 2025 AIMI dataset exports this column as fully capitalized (`"TRUE"`), a simple string normalization was implemented to prevent false negatives.

> **Note on New Data Drops (June 2025):** Although new `splits_20250611.tsv`, `series_metadata_20250611.tsv`, and crosswalk files were added to the pipeline to finalize the cohort, the underlying Redivis clinical data is still heavily scrubbed. Therefore, the custom bypasses implemented in the `/ehr` scripts (skipping ghost patients missing from Redivis and avoiding the OMOP `CodeLabeler`) **must remain completely intact** and should not be reverted.
