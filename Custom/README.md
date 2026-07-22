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
3. **GPU Support for Blackwell 50-series GPUs (CUDA 12+, Compute Capability 12.0):**
   The legacy environment uses JAX/jaxlib pinned to CUDA 11. Running MOTOR/CLMBR training on Blackwell GPUs requires four fixes. Do **not** upgrade JAX or jaxlib — doing so overwrites `numpy` to 2.x and breaks the `femr` C-API.

   **Step A — Install `femr_cuda 0.1.16` (GPU package):**
   Earlier `femr_cuda` versions contain a CUDA C++ attention kernel that deadlocks Blackwell SMs. Version 0.1.16 replaces it with a JAX-native fallback that works on Blackwell. No `transformer.py` patching is needed.
   ```bash
   pip uninstall femr femr-cuda -y
   pip install femr_cuda==0.1.16
   ```

   **Step B — Inject CUDA 12.8 `ptxas` via a wrapper script:**
   The `ptxas` bundled with jaxlib 0.4.7 predates SM_120 and cannot compile Blackwell kernels. Download CUDA 12.8's assembler and create a wrapper that forces `-O0` (prevents an infinite optimization loop on Blackwell's large register file):
   ```bash
   mkdir -p ~/cu12_8_ptxas
   pip download nvidia-cuda-nvcc-cu12==12.8.93 --no-deps -d ~/cu12_8_ptxas
   cd ~/cu12_8_ptxas && unzip *.whl -d .

   cat > venv_legacy/bin/ptxas << 'EOF'
   #!/bin/bash
   exec $HOME/cu12_8_ptxas/nvidia/cuda_nvcc/bin/ptxas -O0 "$@"
   EOF
   chmod +x venv_legacy/bin/ptxas
   ```
   > Placing the wrapper inside the venv `bin/` ensures it is always on `PATH` when the venv is active.

   **Step C — Set `XLA_FLAGS` and launch:**
   Three XLA flags are required to disable the autotuner (hangs on Blackwell), direct XLA to the CUDA 12.8 tools, and bypass jaxlib's bundled `nvlink` (which fatally errors on SM_120):
   ```bash
   export XLA_FLAGS="--xla_gpu_cuda_data_dir=$HOME/cu12_8_ptxas/nvidia/cuda_nvcc --xla_gpu_autotune_level=0 --xla_disable_hlo_passes=gemm_algorithm_picker,gpu_conv_algorithm_picker --xla_gpu_force_compilation_parallelism=1"
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   unset JAX_PLATFORMS
   # Verify GPU is detected:
   python -c "import jax; print(jax.devices())"
   # Expected: [GpuDevice(id=0, process_index=0)]
   ```
   Check whether `JAX_PLATFORMS` is persisted in a shell config: `grep -r "JAX_PLATFORMS" ~/.bashrc ~/.bash_profile ~/.profile 2>/dev/null`

   > **For the full diagnostic table and troubleshooting decision tree**, see section 16 of `INSPECT_Baseline_Reconstruction.md`.

## 📂 Execution Order

Once the prerequisites are satisfied, execute the scripts in the following numbered order. The pipeline is separated into four logical phases:

### Phase 1: Downloading Raw Data
0. `0a_download_aimi_labels.py` & `0b_download_rspect_images.py`: Fetch standard clinical tags (via Stanford AIMI Portal) and the full RSPECT dataset (from the AWS Open Data Registry) respectively.

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
10. `9a_run_baseline_benchmark.py`: Wrapper to execute legacy feature extraction (labels + FEMR features) for a single task. Must be run for each task before MOTOR batch creation.
11. `9c_train_gbm_cv.py`: Trains and evaluates the LightGBM baseline on a task using a deterministic 5-fold cross-validation scheme. It outputs fold-specific models, fold-level scores, and pooled out-of-fold (OOF) predictions.
12. `9e_run_all_tasks_motor.py`: Runs the full MOTOR/CLMBR evaluation pipeline across all 8 tasks in a single unattended run. For each task it generates MOTOR batches (if absent) and trains a linear probe, saving per-task results to timestamped folders and a summary CSV. All Blackwell GPU XLA flags are baked in — no manual `export` required.

> **Note:** For a highly detailed breakdown of the exact engineering steps and debugging taken to reconstruct the baseline (including the 5-tier OMOP fallback logic), see `INSPECT_Baseline_Reconstruction.md`.

## EHR Baseline Evaluation & Auxiliary Tasks

To evaluate the extracted EHR features against the pulmonary embolism (PE) endpoint and all 7 auxiliary prognostic endpoints (1, 6, 12-month mortality/readmission, and 12-month PH), automated evaluation wrappers were introduced.

### Execution Steps
12. `9b_run_all_tasks_gbm.py`: Iteratively trains and evaluates the GBM baseline across all tasks on the static train/val/test split, extracting and saving test-set AUROC scores.
13. `9d_run_all_tasks_gbm_cv.py`: Iteratively trains and evaluates the GBM baseline using **5-Fold Cross-Validation** across all tasks, extracting and tabulating pooled OOF AUROCs and average test metrics (AUROC, Sensitivity, and Specificity with Youden's J threshold optimization).
14. `9e_run_all_tasks_motor.py`: Runs the MOTOR foundation model linear probe across all 8 tasks. Handles `clmbr_create_batches` and `clmbr_train_linear_probe` automatically. Requires a Blackwell-compatible environment (see Step 4 above) and `labeled_patients.csv` for each task. Results are saved to timestamped per-task folders under `DATA_RAW/EHR_FEMR_DB/motor_results/` with a consolidated `motor_results.csv`.

## Custom Time-Binned Feature Generation
To support modeling time-binned historical features, we introduced:
* `generate_binned_features.py`: An alternative script to generate features where event counts are grouped into custom time windows (bins) relative to the prediction anchor time. Categories like vitals/labs (measurements) and diagnoses/procedures (conditions/procedures/devices) can have independent time ranges configured via CLI flags (`--vitals_labs_bins`, `--diag_proc_bins`).

### Modifications to Original `ehr/` Scripts
Executing the auxiliary tasks and the master pipeline successfully required patching legacy bugs in the original `ehr/` files:

#### `ehr/2_generate_labels_and_features.py`
1. **Bypassed `CodeLabeler`:** The original script searched the OMOP `condition_occurrence` tables for precise death/readmission codes. Because Stanford scrubbed these exact codes from the public dataset to preserve patient privacy, the labeler silently failed and yielded 100% `False` labels. The script was refactored to extract the true, pre-computed outcomes directly from our merged master cohort CSV.
2. **Fixed FEMR API Deprecation:** The `femr` v0.2.x API deprecated the `patient_ids` keyword argument in `labeler.apply()`. Passing it caused a `TypeError` that completely broke the pipeline. This argument was removed.
3. **Corrected Case Sensitivity:** For the `12_month_PH` endpoint, the original script strictly checked `label == "True"`. Because the 2025 AIMI dataset exports this column as fully capitalized (`"TRUE"`), a simple string normalization was implemented to prevent false negatives.

#### `ehr/run_all_ehr.py`
1. **Added `--extract_path` Argument:** The original master script hardcoded the FEMR database location to `inspect_femr_extract/extract` within the output directory. A custom `--extract_path` argument was added. This allows the pipeline to point directly to pre-generated multi-gigabyte database extracts (like the 21GB `event_metadata` extract) located anywhere on disk, seamlessly bypassing the expensive database creation step.
2. **Dynamic Script Path Resolution:** Made script references resolve absolute to the parent runner script directory so it can be called from any workspace folder.
3. **Fixed Missing CLMBR Parameter**: Patched `clmbr_train_linear_probe` system execution call by adding the missing required `--path_to_cohort` flag.

#### `image/` Configuration & RSPECT Fine-Tuning Setup
1. **RSPECT Dataset & Model Checkpoint Paths:** The original configs contained hardcoded Stanford cluster paths (`/share/pi/nigam/...` and `/local-scratch/nigam/...`). These have been updated to point to local/relative workspace paths:
   - **Base Model Checkpoint:** Downloaded `resnetv2_ct.ckpt` (4.58 GB) from `StanfordShahLab/resnetv2_ct` on Hugging Face into `../resnetv2_ct/resnetv2_ct.ckpt`.
   - **`image/radfusion3/configs/model/resnetv2_ct.yaml`:** Updated `checkpoint_path` to `../resnetv2_ct/resnetv2_ct.ckpt`.
   - **`image/radfusion3/configs/dataset/rsna.yaml` & `rsna_featurized.yaml`:** Updated `csv_path` (`train.csv`), `dicom_dir` (`train/`), `output_dir`, and `hdf5_path` to point to the local RSPECT dataset (`../RSPECT_CTPA/`).
2. **Path Portability Guidelines (Relative vs. Absolute Paths):**
   - **Avoid Hardcoded Absolute Paths:** Hardcoding absolute server paths (e.g. `/share/pi/...` or `/home/username/...`) causes instant pipeline failures when transferring code across environments, laptops, or cloud VMs.
   - **Use Script-Anchored Relative Paths:** In Python scripts, resolve paths dynamically relative to the script file location using `Path(__file__).resolve().parent`.
   - **Use Workspace-Relative Config Paths:** In YAML/Hydra configuration files, use paths relative to the project root (`../RSPECT_CTPA`, `../resnetv2_ct`) so the pipeline executes seamlessly across any developer setup without manual edit steps.
3. **Batch Size & Gradient Accumulation Dynamics:** Configured `batch_size: 4` in `rsna.yaml` and `accumulate_grad_batches: 64` in `classify.yaml` to achieve an effective batch size of 256 while operating cleanly within 16 GB single-GPU VRAM limits (e.g. RTX 5070 Ti). ResNetV2-101x3 utilizes GroupNorm (`GroupNorm(32, ...)`), which evaluates statistics per-sample independently of batch size, guaranteeing mathematical invariance and stability regardless of micro-batch size.
4. **PyTorch Autograd System RAM Leak Fix:** Fixed a host RAM leak in `classification_lightning_model.py` by detaching output tensors (`logit.detach().cpu()`, `y.detach().cpu()`) in `shared_step()`. This prevents PyTorch from holding 144,000+ autograd computation graphs in host RAM over the course of an epoch. Tuned `num_workers: 4` in `classify.yaml`.
5. **Output Checkpoints & Evaluation Predictions:** Fine-tuning checkpoints (`.ckpt`), serialized configurations (`config.pkl`), and test evaluation predictions (`test_preds.csv`) are automatically exported to timestamped run directories under `outputs/classify_pe_present_on_image_<timestamp>/`.

> **Note on New Data Drops (June 2025):** Although new `splits_20250611.tsv`, `series_metadata_20250611.tsv`, and crosswalk files were added to the pipeline to finalize the cohort, the underlying Redivis clinical data is still heavily scrubbed. Therefore, the custom bypasses implemented in the `/ehr` scripts (skipping ghost patients missing from Redivis and avoiding the OMOP `CodeLabeler`) **must remain completely intact** and should not be reverted.

