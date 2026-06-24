# Custom INSPECT Baseline Reproduction Pipeline

This directory contains the custom scripts engineered to successfully reproduce the INSPECT baseline benchmark cohort, bypassing undocumented missing data errors, un-anonymization artifacts, and hardcoded infrastructure dependencies.

## 🚀 Quickstart & Prerequisites

To completely rebuild the baseline dataset from scratch using these custom scripts, you must ensure the following four prerequisites are met. Failure to do so will result in missing data or authentication errors.

### 1. Redivis Authentication
Script `1_INSPECT_DL_EHR.py` connects to the Redivis API to download the raw OMOP tables. You must authenticate your session before running it.
* Generate a Redivis API token from your account settings.
* Export it to your terminal environment:
  ```bash
  export REDIVIS_API_TOKEN="your_token_here"
  ```

### 2. Manual AIMI Portal Downloads
The INSPECT dataset is split across two portals. The ground-truth labels and imaging crosswalks are NOT hosted on Redivis and cannot be downloaded programmatically due to strict Data Use Agreements (DUA).
* Log in to the [Stanford AIMI Portal](https://aimi.stanford.edu).
* Manually download `labels_20250611.tsv` and `study_mapping_20250611.tsv`.
* Place both files exactly in `DATA_RAW/LABELS/`.

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
6. `5_ehr_ingestion.py`: Extracts dense feature matrices via PyArrow for custom MONAI pipelines.
7. `6_image_ingestion.py`: Lazy-loads 3D CTPA volumes via MONAI.

> **Note:** For a highly detailed breakdown of the exact engineering steps and debugging taken to reconstruct the baseline (including the 5-tier OMOP fallback logic), see `INSPECT_Baseline_Reconstruction.md`.
