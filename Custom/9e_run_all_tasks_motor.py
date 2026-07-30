"""
9e_run_all_tasks_motor.py

Runs clmbr_create_batches + a Python-based L2-regularised logistic-regression
linear probe (MOTOR foundation model) across all 8 INSPECT tasks.

Splits are assigned from the authoritative cohort file ('split' column:
train / valid / test), matching every other INSPECT baseline (GBM, etc.).

Train/Valid/Test AUROC and best L2 strength are written to a results CSV and
printed in a summary table.

Prerequisites
-------------
- femr_cuda 0.1.16 installed in the venv
- CUDA 12.8 ptxas downloaded to ~/cu12_8_ptxas/  (see README.md Step B)
- Labeled patients CSVs generated for each task under:
    DATA_RAW/EHR_FEMR_DB/features/<task>/labeled_patients.csv
  (produced by 9a_run_baseline_benchmark.py)
- motor-t-base model + dictionary at ~/Documents/INSPECT/motor-t-base/
- cohort_0.2.0_master_file_anon.csv in DATA_PROCESSED/ (standard location)

Usage
-----
    python Custom/9e_run_all_tasks_motor.py

    # Force-recreate batches even if they already exist:
    python Custom/9e_run_all_tasks_motor.py --force-batches

    # Override cohort file location:
    python Custom/9e_run_all_tasks_motor.py --cohort-file /path/to/cohort.csv
"""

import argparse
import csv
import logging
import os
import pickle
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import sklearn.metrics
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TASKS = [
    "PE",
    "1_month_mortality",
    "6_month_mortality",
    "12_month_mortality",
    "1_month_readmission",
    "6_month_readmission",
    "12_month_readmission",
    "12_month_PH",
]

SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_RAW    = PROJECT_DIR.parent / "DATA_RAW" / "EHR_FEMR_DB"

RUN_TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")

EXTRACT_PATH   = DATA_RAW / "extract"
FEATURES_ROOT  = DATA_RAW / "features"
BATCHES_ROOT   = DATA_RAW / "MOTOR_batches"
RESULTS_ROOT   = DATA_RAW / "motor_results"

_motor_candidates = [
    Path.home() / "Documents" / "INSPECT" / "motor-t-base",
    Path.home() / "Documents" / "Internship_INSPECT" / "motor-t-base",
    PROJECT_DIR.parent / "motor-t-base",
    PROJECT_DIR / "motor-t-base",
]
MOTOR_ROOT     = next((p for p in _motor_candidates if p.is_dir()), _motor_candidates[0])
MODEL_DIR      = MOTOR_ROOT / "model"
DICTIONARY_DIR = MOTOR_ROOT / "dictionary"

_venv_candidates = [
    PROJECT_DIR.parent / ".venv_legacy",
    PROJECT_DIR / ".venv_legacy",
    PROJECT_DIR / "venv_legacy",
    Path.home() / "Documents" / "INSPECT" / "venv_legacy",
]
VENV_DIR = next((p for p in _venv_candidates if p.is_dir()), None)

# Cohort file — authoritative train/valid/test split column
_cohort_candidates = [
    PROJECT_DIR.parent / "DATA_PROCESSED" / "cohort_0.2.0_master_file_anon.csv",
    Path.home() / "Documents" / "Internship_INSPECT" / "DATA_PROCESSED" / "cohort_0.2.0_master_file_anon.csv",
    Path.home() / "Documents" / "INSPECT" / "DATA_PROCESSED" / "cohort_0.2.0_master_file_anon.csv",
    Path.home() / "sravar" / "Documents" / "INSPECT" / "DATA_PROCESSED" / "cohort_0.2.0_master_file_anon.csv",
]
COHORT_FILE = next((p for p in _cohort_candidates if p.is_file()), None)

# ---------------------------------------------------------------------------
# XLA / JAX environment for Blackwell GPUs (CC 12.0 / SM_120)
# ---------------------------------------------------------------------------

CU128_PTXAS = Path.home() / "cu12_8_ptxas" / "nvidia" / "cuda_nvcc"

XLA_FLAGS = " ".join([
    f"--xla_gpu_cuda_data_dir={CU128_PTXAS}",
    "--xla_gpu_autotune_level=0",
    "--xla_disable_hlo_passes=gemm_algorithm_picker,gpu_conv_algorithm_picker",
    "--xla_gpu_force_compilation_parallelism=1",
])

os.environ["XLA_FLAGS"] = XLA_FLAGS
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_NUMPY_RANK_PROMOTION"] = "raise"
os.environ.pop("JAX_PLATFORMS", None)


def build_env() -> dict:
    """Return a copy of os.environ with XLA/JAX flags applied, for subprocess calls."""
    env = os.environ.copy()
    env["XLA_FLAGS"] = XLA_FLAGS
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["JAX_NUMPY_RANK_PROMOTION"] = "raise"
    env.pop("JAX_PLATFORMS", None)
    return env


# ---------------------------------------------------------------------------
# Cohort split loading
# ---------------------------------------------------------------------------

def load_cohort_splits(cohort_file: Path) -> dict:
    """
    Read the 'split' column from the cohort CSV/TSV.
    Returns {patient_id (int) -> split_idx (0=train, 1=valid, 2=test)}.
    """
    split_map = {"train": 0, "valid": 1, "test": 2}
    pid_to_split = {}
    with open(cohort_file) as f:
        reader = csv.DictReader(f, delimiter="\t" if cohort_file.suffix == ".tsv" else ",")
        for row in reader:
            pid_col = next((c for c in ("PatientID", "patient_id") if c in row), None)
            if pid_col is None or "split" not in row:
                raise ValueError(
                    f"Cohort file missing 'PatientID'/'patient_id' or 'split' column: {cohort_file}"
                )
            pid = int(row[pid_col])
            s   = row["split"].strip().lower()
            if s in split_map:
                pid_to_split[pid] = split_map[s]
    logging.info(f"Loaded {len(pid_to_split)} patient splits from {cohort_file}")
    return pid_to_split


# ---------------------------------------------------------------------------
# Batch creation
# ---------------------------------------------------------------------------

def create_batches(task: str, clmbr_create_bin: Path, force: bool) -> bool:
    """Run clmbr_create_batches for a single task. Returns True on success."""
    batches_dir      = BATCHES_ROOT / task
    labeled_patients = FEATURES_ROOT / task / "labeled_patients.csv"

    if not labeled_patients.exists():
        logging.error(f"labeled_patients.csv not found for {task}: {labeled_patients}")
        logging.error(f"Run 9a_run_baseline_benchmark.py --task {task} first.")
        return False

    if batches_dir.exists():
        if force:
            logging.info(f"Removing existing batches for {task} (--force-batches).")
            shutil.rmtree(batches_dir)
        else:
            logging.info(f"Batches already exist for {task}, skipping creation.")
            return True

    cmd = [
        str(clmbr_create_bin),
        str(batches_dir),
        "--data_path",             str(EXTRACT_PATH),
        "--task",                  "labeled_patients",
        "--labeled_patients_path", str(labeled_patients),
        "--val_start",             "80",
        "--dictionary_path",       str(DICTIONARY_DIR),
        "--is_hierarchical",
    ]

    logging.info(f"Creating batches: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=build_env())
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()

    if proc.returncode != 0:
        logging.error(f"clmbr_create_batches failed for {task} (exit {proc.returncode})")
        return False

    logging.info(f"Batches created at {batches_dir}")
    return True


# ---------------------------------------------------------------------------
# Representation extraction
# ---------------------------------------------------------------------------

def extract_representations(task: str, pid_to_split: dict):
    """
    Load MOTOR-t-base, extract per-patient representations from all batches,
    and assign train/valid/test splits from the cohort file mapping.

    Returns (reprs, labels, pids, ages, split_indices).
    """
    import haiku as hk
    import jax
    import jax.numpy as jnp
    import msgpack
    import femr.datasets
    import femr.extension.dataloader
    import femr.models.transformer

    batches_path    = BATCHES_ROOT / task
    batch_info_path = batches_path / "batch_info.msgpack"

    logging.info(f"Loading database extract from {EXTRACT_PATH}")
    database = femr.datasets.PatientDatabase(str(EXTRACT_PATH))

    logging.info(f"Loading MOTOR model weights from {MODEL_DIR}")
    with open(MODEL_DIR / "best", "rb") as f:
        params = pickle.load(f)
    params = femr.models.transformer.convert_params(params, dtype=jnp.float16)

    with open(batch_info_path, "rb") as f:
        batch_info = msgpack.load(f, use_list=False)

    with open(MODEL_DIR / "config.msgpack", "rb") as f:
        config = msgpack.load(f, use_list=False)
    config = hk.data_structures.to_immutable_dict(config)

    rng = jax.random.PRNGKey(config.get("seed", 42))

    loader = femr.extension.dataloader.BatchLoader(str(EXTRACT_PATH), str(batch_info_path))
    logging.info(
        f"Loaded batches: Train={loader.get_number_of_batches('train')}, "
        f"Dev={loader.get_number_of_batches('dev')}, "
        f"Test={loader.get_number_of_batches('test')}"
    )

    def model_fn(config, batch):
        model = femr.models.transformer.EHRTransformer(config)(batch, no_task=True)
        return model

    model = hk.transform(model_fn)

    # `config` must be a STATIC argument. It is MOTOR's config.msgpack -- a nested
    # dict containing strings (e.g. the pretraining batch path), which JAX cannot
    # abstractify. Matches stock femr/models/linear_probe.py.
    def _compute_repr(params, rng, config, batch):
        repr, _ = model.apply(params, rng, config, batch)
        return repr

    compute_repr = jax.jit(_compute_repr, static_argnames=("config",))

    reprs_list   = []
    ages_list    = []
    pids_list    = []
    offsets_list = []

    for split in ("train", "dev", "test"):
        num_batches = loader.get_number_of_batches(split)
        logging.info(f"Processing {split} split ({num_batches} batches)...")
        for j in range(num_batches):
            raw_batch = loader.get_batch(split, j)
            batch = jax.tree_map(
                lambda a: jax.device_put(a, device=jax.devices("gpu")[0]), raw_batch
            )
            repr = compute_repr(params, rng, config, batch)

            num_indices = batch["num_indices"]
            p_index = (
                batch["transformer"]["label_indices"] // batch["transformer"]["length"]
            )[:num_indices]

            # integer_ages is per TOKEN across the flattened batch; label_indices
            # gives the token positions where labels sit. Must fancy-index by those
            # positions, not take the first num_indices entries.
            li = np.asarray(raw_batch["transformer"]["label_indices"])[:num_indices]

            reprs_list.append(np.array(repr[:num_indices]))
            ages_list.append(np.asarray(raw_batch["transformer"]["integer_ages"])[li])
            pids_list.append(np.array(raw_batch["patient_ids"][p_index]))
            offsets_list.append(np.array(raw_batch["offsets"][p_index]))

    reprs       = np.concatenate(reprs_list,   axis=0)
    repr_ages   = np.concatenate(ages_list,    axis=0)
    repr_pids   = np.concatenate(pids_list,    axis=0)
    repr_offsets = np.concatenate(offsets_list, axis=0).astype(np.int32)

    # Label alignment (same logic as clmbr_train_linear_probe)
    task_labels  = batch_info["config"]["task"]["labels"]
    label_pids   = np.array([v[0] for v in task_labels], dtype=np.uint64)
    label_ages   = np.array([v[1] for v in task_labels], dtype=np.uint32)
    label_values = np.array([v[2] for v in task_labels], dtype=np.float32)

    sort_idx_label = np.lexsort((label_ages, label_pids))
    label_pids   = label_pids[sort_idx_label]
    label_ages   = label_ages[sort_idx_label]
    label_values = label_values[sort_idx_label]

    sort_idx_repr = np.lexsort((-repr_offsets, repr_ages, repr_pids))
    repr_offsets  = repr_offsets[sort_idx_repr]
    repr_ages     = repr_ages[sort_idx_repr]
    repr_pids     = repr_pids[sort_idx_repr]
    reprs_sorted  = reprs[sort_idx_repr]

    matching_indices = []
    split_indices    = []
    kept_label_idx   = []   # which LABELS survived; keeps labels aligned with reprs

    j = 0
    for i_lab, (l_pid, l_age) in enumerate(zip(label_pids, label_ages)):
        while True:
            if j + 1 == len(repr_pids):
                break
            elif repr_pids[j] < l_pid:
                pass
            else:
                if repr_pids[j + 1] != l_pid or repr_ages[j + 1] > l_age:
                    break
            j += 1

        assert repr_pids[j] == l_pid
        assert repr_ages[j] <= l_age

        # Assign split from the authoritative cohort file
        s = pid_to_split.get(int(l_pid))
        if s is None:
            logging.warning(f"Patient {l_pid} not in cohort file — skipping.")
            continue
        split_indices.append(s)
        matching_indices.append(j)
        kept_label_idx.append(i_lab)

    kept_label_idx = np.array(kept_label_idx, dtype=np.int64)
    matched_reprs = reprs_sorted[matching_indices, :]

    # Labels must be filtered by LABEL index, not representation index. Indexing
    # the label arrays with matching_indices (which holds positions into the
    # sorted REPRESENTATION arrays) silently pairs each repr with the wrong label.
    label_values  = label_values[kept_label_idx]
    label_pids    = label_pids[kept_label_idx]
    label_ages    = label_ages[kept_label_idx]
    split_indices = np.array(split_indices)

    assert len(matched_reprs) == len(label_values) == len(split_indices), (
        f"alignment mismatch: reprs={len(matched_reprs)} "
        f"labels={len(label_values)} splits={len(split_indices)}"
    )

    logging.info(
        f"Split counts → Train: {(split_indices==0).sum()} | "
        f"Valid: {(split_indices==1).sum()} | Test: {(split_indices==2).sum()}"
    )

    return matched_reprs, label_values, label_pids, label_ages, split_indices, database


# ---------------------------------------------------------------------------
# Linear probe (L2-regularised logistic regression, sklearn)
# ---------------------------------------------------------------------------

def run_linear_probe(reprs: np.ndarray, labels: np.ndarray, split_indices: np.ndarray) -> dict:
    """
    Train an L2-regularised logistic regression probe on MOTOR representations,
    sweeping 20 regularisation strengths (matching clmbr_train_linear_probe range)
    and selecting the best model by validation AUROC.

    Returns a metrics dict with train/valid/test AUROC and best L2 strength.
    """
    train_mask = split_indices == 0
    val_mask   = split_indices == 1
    test_mask  = split_indices == 2

    X = reprs.astype(np.float32)
    X_train, y_train = X[train_mask], labels[train_mask]
    X_val,   y_val   = X[val_mask],   labels[val_mask]
    X_test,  y_test  = X[test_mask],  labels[test_mask]

    logging.info(
        f"Linear probe split counts — Train: {len(X_train)} | "
        f"Valid: {len(X_val)} | Test: {len(X_test)}"
    )
    logging.info(
        f"Prevalence — Train: {y_train.mean():.4f} | "
        f"Valid: {y_val.mean():.4f} | Test: {y_test.mean():.4f}"
    )

    # Sweep L2 strengths from 10^1 down to 10^-5, matching clmbr_train_linear_probe.
    l2_values = list(10 ** e for e in np.linspace(1, -5, 20)) + [0.0]

    # --- C conversion -------------------------------------------------------
    # Stock FEMR minimises      mean(BCE) + 0.5 * l2 * ||beta||^2
    # sklearn minimises         0.5 * ||w||^2 + C * SUM(loss)
    #                         = n*C * [ mean(loss) + 0.5/(C*n) * ||w||^2 ]
    # Matching the penalty coefficients gives  l2 = 1/(C*n)  ->  C = 1/(l2*n).
    #
    # The earlier `C = 1/l2` omitted n, making the effective penalty l2/n --
    # about four decades too weak at n~13k. Every task then selected the top of
    # the grid because the optimum lay above the reachable ceiling. See
    # INSPECT_Baseline_Reconstruction.md section 25.1.
    n_train = len(X_train)

    best_val_auroc   = -1.0
    best_train_auroc = None
    best_test_auroc  = None
    best_l2          = None

    for l2 in l2_values:
        C = 1.0 / (l2 * n_train) if l2 > 0 else 1e12
        clf = LogisticRegression(
            C=C, solver="lbfgs", max_iter=1000,
            fit_intercept=True, tol=1e-4
        )
        clf.fit(X_train, y_train)

        train_auroc = sklearn.metrics.roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
        val_auroc   = sklearn.metrics.roc_auc_score(y_val,   clf.predict_proba(X_val)[:, 1])
        test_auroc  = sklearn.metrics.roc_auc_score(y_test,  clf.predict_proba(X_test)[:, 1])

        logging.info(f"  L2={l2:.2e}  C={C:.2e} (n={n_train}) | Train: {train_auroc:.4f} | Valid: {val_auroc:.4f} | Test: {test_auroc:.4f}")

        if val_auroc > best_val_auroc:
            best_val_auroc   = val_auroc
            best_train_auroc = train_auroc
            best_test_auroc  = test_auroc
            best_l2          = l2

    logging.info(
        f"Best L2={best_l2:.2e} → Train: {best_train_auroc:.4f} | "
        f"Valid: {best_val_auroc:.4f} | Test: {best_test_auroc:.4f}"
    )

    # Selection at either end of the grid means the optimum lies outside it and
    # the chosen value is censored, not optimal. This is how the C-conversion
    # bug originally surfaced -- keep the warning so it cannot happen silently.
    nonzero = [v for v in l2_values if v > 0]
    if best_l2 >= max(nonzero) / 1.05:
        logging.warning(
            f"L2 selected at the TOP of the grid ({best_l2:.2e}). The optimum is "
            "likely stronger than the grid allows -- model is under-regularised."
        )
    elif 0 < best_l2 <= min(nonzero) * 1.05:
        logging.warning(
            f"L2 selected at the BOTTOM of the grid ({best_l2:.2e}). The optimum "
            "is likely weaker than the grid allows."
        )

    return {
        "train_auroc": float(best_train_auroc),
        "valid_auroc": float(best_val_auroc),
        "test_auroc":  float(best_test_auroc),
        "l2_strength": float(best_l2),
    }


# ---------------------------------------------------------------------------
# Per-task pipeline
# ---------------------------------------------------------------------------

def run_task(task: str, pid_to_split: dict) -> dict:
    logging.info(f"\n{'='*70}\nStarting MOTOR linear probe for task: {task}\n{'='*70}")

    # Step 1: extract representations
    try:
        reprs, labels, pids, ages, split_indices, database = extract_representations(
            task, pid_to_split
        )
    except Exception as e:
        logging.error(f"Representation extraction failed for {task}: {e}", exc_info=True)
        return {"task": task, "status": "failed (extraction)",
                **{k: None for k in ["train_auroc", "valid_auroc", "test_auroc", "l2_strength"]}}

    # Step 2: train linear probe
    try:
        metrics = run_linear_probe(reprs, labels, split_indices)
    except Exception as e:
        logging.error(f"Linear probe failed for {task}: {e}", exc_info=True)
        return {"task": task, "status": "failed (probe)",
                **{k: None for k in ["train_auroc", "valid_auroc", "test_auroc", "l2_strength"]}}

    # Step 3: save predictions
    output_dir = RESULTS_ROOT / f"{RUN_TIMESTAMP}_{task}"
    output_dir.mkdir(parents=True, exist_ok=True)

    import datetime as dt
    from sklearn.linear_model import LogisticRegression as _LR

    # Re-fit best model to save predictions.
    # Must use the SAME C conversion as the sweep (C = 1/(l2*n)), or the saved
    # predictions come from a differently-regularised model than the reported
    # metrics.
    train_mask = split_indices == 0
    _n_train = int(train_mask.sum())
    best_C = (1.0 / (metrics["l2_strength"] * _n_train)
              if metrics["l2_strength"] > 0 else 1e12)
    clf = _LR(C=best_C, solver="lbfgs", max_iter=1000, fit_intercept=True, tol=1e-4)
    clf.fit(reprs.astype(np.float32)[train_mask], labels[train_mask])
    all_probs = clf.predict_proba(reprs.astype(np.float32))[:, 1]

    prediction_dates = []
    for pid, age in zip(pids, ages):
        birth = dt.datetime.combine(database.get_patient_birth_date(pid), dt.time.min)
        prediction_dates.append(birth + dt.timedelta(minutes=int(age)))

    with open(output_dir / "predictions.pkl", "wb") as f:
        pickle.dump([all_probs, pids, labels, prediction_dates], f)

    logging.info(
        f"[{task}] SUMMARY → Train: {metrics['train_auroc']:.4f} | "
        f"Valid: {metrics['valid_auroc']:.4f} | Test: {metrics['test_auroc']:.4f} | "
        f"L2: {metrics['l2_strength']:.2e}"
    )
    logging.info(f"Results saved to {output_dir}")

    return {"task": task, "status": "ok", **metrics}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run MOTOR linear probe across all INSPECT tasks (cohort-file splits)"
    )
    parser.add_argument("--force-batches", action="store_true",
                        help="Re-create MOTOR batches even if they already exist")
    parser.add_argument(
        "--cohort-file", type=str, default=None,
        help="Path to cohort CSV/TSV with 'split' column. Auto-detected if not given."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if VENV_DIR is None:
        sys.exit("[ERROR] Could not locate venv.")

    clmbr_create_bin = VENV_DIR / "bin" / "clmbr_create_batches"
    if not clmbr_create_bin.exists():
        sys.exit(f"[ERROR] {clmbr_create_bin} not found. Is femr_cuda 0.1.16 installed?")

    if not MODEL_DIR.exists():
        sys.exit(f"[ERROR] Model directory not found: {MODEL_DIR}")
    if not DICTIONARY_DIR.exists():
        sys.exit(f"[ERROR] Dictionary directory not found: {DICTIONARY_DIR}")

    cohort_file = Path(args.cohort_file) if args.cohort_file else COHORT_FILE
    if cohort_file is None or not cohort_file.is_file():
        sys.exit(
            "[ERROR] Cohort file not found. Use --cohort-file /path/to/cohort_0.2.0_master_file_anon.csv"
        )

    pid_to_split = load_cohort_splits(cohort_file)

    logging.info(f"femr venv   : {VENV_DIR}")
    logging.info(f"model dir   : {MODEL_DIR}")
    logging.info(f"dictionary  : {DICTIONARY_DIR}")
    logging.info(f"extract     : {EXTRACT_PATH}")
    logging.info(f"batches     : {BATCHES_ROOT}")
    logging.info(f"results     : {RESULTS_ROOT}")
    logging.info(f"cohort file : {cohort_file}")
    logging.info(f"XLA_FLAGS   : {XLA_FLAGS}\n")

    all_results = []

    for task in TASKS:
        logging.info(f"\n{'='*70}\n  Task: {task}\n{'='*70}")

        # Step 1: ensure batches exist
        if not create_batches(task, clmbr_create_bin, args.force_batches):
            all_results.append({
                "task": task, "status": "failed (batch creation)",
                **{k: None for k in ["train_auroc", "valid_auroc", "test_auroc", "l2_strength"]}
            })
            continue

        # Step 2: extract representations + run linear probe
        result = run_task(task, pid_to_split)
        all_results.append(result)

    # Write CSV
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_ROOT / "motor_results.csv"
    fieldnames = ["task", "status", "train_auroc", "valid_auroc", "test_auroc", "l2_strength"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # Print summary table
    col = {"task": 25, "status": 22, "train": 12, "valid": 12, "test": 12, "l2": 14}
    header = (
        f"{'Task':<{col['task']}} {'Status':<{col['status']}} "
        f"{'Train AUROC':>{col['train']}} {'Valid AUROC':>{col['valid']}} "
        f"{'Test AUROC':>{col['test']}} {'L2 Strength':>{col['l2']}}"
    )

    print(f"\n\n{'='*85}")
    print("  MOTOR LINEAR PROBE — ALL TASKS  (cohort-file splits)")
    print(f"{'='*85}")
    print(header)
    print("-" * 85)

    for r in all_results:
        fmt = lambda v: f"{v:.4f}" if v is not None else "—"
        fmt_l2 = lambda v: f"{v:.2e}" if v is not None else "—"
        print(
            f"{r['task']:<{col['task']}} {r['status']:<{col['status']}} "
            f"{fmt(r['train_auroc']):>{col['train']}} "
            f"{fmt(r['valid_auroc']):>{col['valid']}} "
            f"{fmt(r['test_auroc']):>{col['test']}} "
            f"{fmt_l2(r['l2_strength']):>{col['l2']}}"
        )

    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
