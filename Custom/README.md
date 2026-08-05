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

### Phase 3: Datasets, ML Training & Benchmarks
6. `9a_run_baseline_benchmark.py`: Wrapper to execute legacy feature extraction (labels + FEMR features) for a single task. Must be run for each task before MOTOR batch creation.
7. `9c_train_gbm_cv.py`: Trains and evaluates the LightGBM baseline on a task using a deterministic 5-fold cross-validation scheme. It outputs fold-specific models, fold-level scores, and pooled out-of-fold (OOF) predictions.
8. `9e_run_all_tasks_motor.py`: Runs the full MOTOR/CLMBR evaluation pipeline across all 8 tasks in a single unattended run. For each task it generates MOTOR batches (if absent) and trains a linear probe, saving per-task results to timestamped folders and a summary CSV. All Blackwell GPU XLA flags are baked in — no manual `export` required.

> **Note:** For a highly detailed breakdown of the exact engineering steps and debugging taken to reconstruct the baseline (including the 5-tier OMOP fallback logic), see `INSPECT_Baseline_Reconstruction.md`.

## EHR Baseline Evaluation & Auxiliary Tasks

To evaluate the extracted EHR features against the pulmonary embolism (PE) endpoint and all 7 auxiliary prognostic endpoints (1, 6, 12-month mortality/readmission, and 12-month PH), automated evaluation wrappers were introduced.

### Execution Steps
9. `9b_run_all_tasks_gbm.py`: Iteratively trains and evaluates the GBM baseline across all tasks on the static train/val/test split, extracting and saving test-set AUROC scores.
10. `9d_run_all_tasks_gbm_cv.py`: Iteratively trains and evaluates the GBM baseline using **5-Fold Cross-Validation** across all tasks, extracting and tabulating pooled OOF AUROCs and average test metrics (AUROC, Sensitivity, and Specificity with Youden's J threshold optimization).
11. `9e_run_all_tasks_motor.py`: Runs the MOTOR foundation model linear probe across all 8 tasks. Handles `clmbr_create_batches` and `clmbr_train_linear_probe` automatically. Requires a Blackwell-compatible environment (see Step 4 above) and `labeled_patients.csv` for each task. Results are saved to timestamped per-task folders under `DATA_RAW/EHR_FEMR_DB/motor_results/` with a consolidated `motor_results.csv`.

## MOTOR Foundation-Model Evaluation

Three entry points share one implementation. `run_MOTOR_MLP.py` holds all the logic; the other two are a thin wrapper and an all-tasks driver, so fixes cannot drift between them.

| Script | Head | Split source | Tasks |
|---|---|---|---|
| `9e_run_all_tasks_motor.py` | sklearn `LogisticRegression` (L2 sweep) | cohort file | all 8 |
| `run_MOTOR_MLP.py` | PyTorch 2-stage MLP | cohort file (default) | `--task <name>` or `all` |
| `run_MOTOR_MLP_Hashed.py` | PyTorch 2-stage MLP | CLMBR hash partition | `--task <name>` or `all` |

**Split handling.** All three call `clmbr_create_batches --val_start 80` to build batches, then discard CLMBR's own 80/5/15 hash partition and reassign train/valid/test **per patient** from the cohort file's `split` column — matching the GBM and imaging arms. `run_MOTOR_MLP_Hashed.py` (equivalently `--split-source hash`) keeps the hash partition instead, so the two can be compared on identical representations. Measured difference across 8 tasks: mean +0.005, p = 0.45 — see §25.3 of `INSPECT_Baseline_Reconstruction.md`.

**Prerequisites.** Each task needs `labeled_patients.csv` under `DATA_RAW/EHR_FEMR_DB/features/<task>/`, produced by `9a_run_baseline_benchmark.py`. The cohort file supplies the *split*; the labels reaching MOTOR come from that FEMR artefact, which additionally carries prediction times and drops patients absent from the FEMR extract. Generate any missing ones first:

```bash
cd <repo root>          # NOT Custom/ — 9a builds paths with bare ../
for t in PE 1_month_mortality 6_month_mortality 12_month_mortality \
         1_month_readmission 6_month_readmission 12_month_readmission 12_month_PH; do
  [ -f ../DATA_RAW/EHR_FEMR_DB/features/$t/labeled_patients.csv ] \
    || python Custom/9a_run_baseline_benchmark.py --task $t
done
```

**Running.** These are multi-hour GPU jobs; use `tmux` and tee the output. Do **not** run two in parallel — JAX preallocates most of VRAM and the second process will fail to allocate.

```bash
tmux new -s motor
python Custom/run_MOTOR_MLP.py --task 12_month_PH --n-repeats 5 \
  2>&1 | tee ~/logs/motor_ph_$(date +%Y%m%d_%H%M%S).log
# detach: Ctrl-b then d      reattach: tmux attach -t motor
```

**`run_MOTOR_MLP.py` flags:** `--task` (required), `--seed`, `--n-repeats`, `--split-source {cohort,hash}`, `--epochs`, `--batch-size`, `--lr`, `--dropout`, `--hidden-dim`, `--force-batches`, `--motor-dir`, `--cohort-file`.

**L2 conversion (fixed 2026-07-30).** Stock FEMR minimises `mean(BCE) + 0.5·l2·‖β‖²` while sklearn minimises `0.5·‖w‖² + C·Σ loss`, so the equivalence is `C = 1/(l2·n_train)`, not `C = 1/l2`. The earlier conversion omitted `n`, making the effective penalty ~4 decades too weak and pushing L2 selection to the top of the grid on all 8 tasks. Corrected in both the sweep and the final re-fit, with a warning logged if selection lands at either grid boundary. **Results produced before this fix are under-regularised and need regenerating** — see §25.1.1 of `INSPECT_Baseline_Reconstruction.md`.

**Reproducibility.** `9e` is deterministic — `lbfgs` on a strictly convex objective, no shuffling, `random_state` ignored — so it needs no seed. The MLP head *is* stochastic; `--seed` fixes it and `--n-repeats N` retrains N times on the same (deterministic, ~11 min) representations to report mean ± std. Use repeats before claiming any small AUROC difference is real.

## Split Diagnostics

| Script | Purpose |
|---|---|
| `check_rsna_splits.py` | Replays `RSNADataset2D`'s filtering on the RSPECT CSV. Reports whether a split column exists, whether values match the literal `train`/`valid`/`test` the DataModule requests, per-split row counts and class balance, study-level overlap, and validation cost. Resolves the CSV from `rsna.yaml` and warns if it differs from the file being inspected. |
| `make_rspect_splits.py` | Adds a study-level, PE-status-stratified `Split` column to the RSPECT CSV, grouped by `StudyInstanceUID` so no study or series spans splits. Writes a **new** file (`train_with_splits.csv`); refuses to overwrite the input. Deterministic under `--seed`; defaults 70/15/15. |
| `check_inspect_splits.py` | Replays `Dataset1D`'s preprocessing for the INSPECT/Stanford path — the `image_id` → `patient_datetime` derivation, dedup, and both merges — then checks merge key coverage, whether the lowercase `split` column exists so the filter actually fires, and **patient-level** (not just exam-level) integrity, since some INSPECT patients have more than one CTPA. |

The RSPECT diagnostics exist because the public Kaggle release ships no split column and a local refactor removed the guard that would have failed loudly. See §21 of `INSPECT_Baseline_Reconstruction.md`.

## Figures

`figures/make_split_delta_plot.py` generates `motor_split_delta.png` / `.svg` — the paired per-task delta of MOTOR test AUROC between the two split sources, with the 95% CI of the mean. Regenerate after updating the numbers at the top of the script.

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
6. **BFloat16 AMP Evaluation Metric Fix (`TypeError: Got unsupported ScalarType BFloat16`):**
   - **Issue:** When training with Automatic Mixed Precision (`precision: bf16-mixed`), output prediction tensors are saved as `torch.bfloat16`. Calling `.numpy()` directly on these tensors inside evaluation metrics (`utils.get_auroc` and `utils.get_auprc`) raised `TypeError: Got unsupported ScalarType BFloat16` at epoch end because NumPy cannot directly wrap PyTorch `bfloat16` tensors.
   - **Fix:** Patched `image/radfusion3/utils.py` (`get_auroc`, `get_auprc`) and `image/radfusion3/lightning/featurize_lightning_model.py` to explicitly cast PyTorch tensors to single-precision float32 (`.float()`) prior to calling `.numpy()`.

#### `image/radfusion3/data/dataset_2d.py` — restored loud failure on missing splits
The original `RSNADataset2D` filtered unconditionally (`self.df[self.df['Split'] == self.split]`), which raises `KeyError` on a CSV without that column. A later refactor made it tolerant of either capitalisation but omitted the `else`, so a CSV with **neither** column silently skipped the filter and made train, valid and test the full dataframe. Restored:
1. **`else: raise ValueError`** naming the CSV and pointing at `make_rspect_splits.py`.
2. **Zero-row guard** raising when a split matches nothing, listing the values actually present — this catches `"val"` vs the `"valid"` the DataModule requests.

#### `image/radfusion3/configs/dataset/rsna.yaml`
`csv_path` repointed at the split-annotated CSV produced by `make_rspect_splits.py`.

#### `Custom/run_MOTOR_MLP.py` & `Custom/9e_run_all_tasks_motor.py` — MOTOR extraction fixes
Three defects, both files (details in §24 of `INSPECT_Baseline_Reconstruction.md`):
1. **`config` not static for `jax.jit`** — MOTOR's `config.msgpack` contains strings, so JAX refused to trace it (`TypeError: ... is not a valid JAX type`). Now bound as `jax.jit(fn, static_argnames=("config",))`, matching stock FEMR.
2. **`integer_ages` indexed by position** rather than by `label_indices` — `integer_ages` is per *token* across the flattened batch, so the ages were unrelated to the label times and the alignment walk failed its assertion.
3. **Label/representation desynchronisation** when a patient is skipped for being absent from the cohort. `run_MOTOR_MLP.py` returned labels unfiltered; `9e` indexed labels by *representation* positions, which would not have crashed but would have paired each representation with the wrong patient's label. Both now track `kept_label_idx` and assert equal lengths.

> **Note on New Data Drops (June 2025):** Although new `splits_20250611.tsv`, `series_metadata_20250611.tsv`, and crosswalk files were added to the pipeline to finalize the cohort, the underlying Redivis clinical data is still heavily scrubbed. Therefore, the custom bypasses implemented in the `/ehr` scripts (skipping ghost patients missing from Redivis and avoiding the OMOP `CodeLabeler`) **must remain completely intact** and should not be reverted.

