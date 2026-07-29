#!/usr/bin/env python3
"""
run_MOTOR_MLP.py

Runs the MOTOR-t-base foundation model representation extraction pipeline for a specified task,
replaces the linear probe head with a 2-stage Multi-Layer Perceptron (MLP) featuring a 0.5 dropout rate,
trains the MLP classifier, and evaluates Train/Valid/Test AUROC.

Usage:
------
    # Run on a single task:
    python Custom/run_MOTOR_MLP.py --task PE

    # Run on all 8 INSPECT tasks:
    python Custom/run_MOTOR_MLP.py --task all

    # Custom options (e.g. force batch creation, change epochs):
    python Custom/run_MOTOR_MLP.py --task 1_month_mortality --epochs 40 --dropout 0.5 --force-batches
"""

import argparse
import csv
from datetime import datetime, timedelta
import json
import logging
import os
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import sklearn.metrics

# Torch for PyTorch 2-Stage MLP Head
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# JAX & FEMR imports for MOTOR representation extraction
import jax
import jax.numpy as jnp
import haiku as hk
import msgpack

# ---------------------------------------------------------------------------
# Path & Task Configuration
# ---------------------------------------------------------------------------

ALL_TASKS = [
    "PE",
    "1_month_mortality",
    "6_month_mortality",
    "12_month_mortality",
    "1_month_readmission",
    "6_month_readmission",
    "12_month_readmission",
    "12_month_PH",
]

SCRIPT_DIR  = Path(__file__).resolve().parent          # Custom/
PROJECT_DIR = SCRIPT_DIR.parent                         # INSPECT_custom_data_preprocessing/
DATA_RAW    = PROJECT_DIR.parent / "DATA_RAW" / "EHR_FEMR_DB"

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

EXTRACT_PATH   = DATA_RAW / "extract"
FEATURES_ROOT  = DATA_RAW / "features"
BATCHES_ROOT   = DATA_RAW / "MOTOR_batches"
RESULTS_ROOT   = DATA_RAW / "motor_mlp_results"
_motor_candidates = [
    Path.home() / "Documents" / "INSPECT" / "motor-t-base",
    Path.home() / "Documents" / "Internship_INSPECT" / "motor-t-base",
    PROJECT_DIR.parent / "motor-t-base",
    PROJECT_DIR / "motor-t-base",
]
MOTOR_ROOT = next((p for p in _motor_candidates if p.is_dir()), _motor_candidates[0])
MODEL_DIR      = MOTOR_ROOT / "model"
DICTIONARY_DIR = MOTOR_ROOT / "dictionary"

# Cohort file candidates (provides the authoritative train/valid/test split column)
_cohort_candidates = [
    PROJECT_DIR.parent / "DATA_PROCESSED" / "cohort_0.2.0_master_file_anon.csv",
    Path.home() / "Documents" / "Internship_INSPECT" / "DATA_PROCESSED" / "cohort_0.2.0_master_file_anon.csv",
    Path.home() / "Documents" / "INSPECT" / "DATA_PROCESSED" / "cohort_0.2.0_master_file_anon.csv",
    Path.home() / "sravar" / "Documents" / "INSPECT" / "DATA_PROCESSED" / "cohort_0.2.0_master_file_anon.csv",
]
COHORT_FILE = next((p for p in _cohort_candidates if p.is_file()), None)

# Virtual environment candidates
_venv_candidates = [
    PROJECT_DIR.parent / ".venv_legacy",
    PROJECT_DIR / ".venv_legacy",
    PROJECT_DIR / "venv_legacy",
    Path.home() / "Documents" / "INSPECT" / "venv_legacy"
]
VENV_DIR = next((p for p in _venv_candidates if p.is_dir()), None)

# ---------------------------------------------------------------------------
# XLA / JAX Environment Setup for Blackwell GPUs (Must be set before import jax)
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

# Torch for PyTorch 2-Stage MLP Head
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# JAX & FEMR imports for MOTOR representation extraction
import jax
import jax.numpy as jnp
import haiku as hk
import msgpack

# ---------------------------------------------------------------------------
# PyTorch 2-Stage MLP Classifier Head
# ---------------------------------------------------------------------------

class MOTOR_MLP_Head(nn.Module):
    """
    2-Stage MLP Head replacing the linear probe:
    Input (768) -> Linear(hidden_dim) -> LayerNorm -> GELU -> Dropout(0.5)
                -> Linear(hidden_dim // 2) -> LayerNorm -> GELU -> Dropout(0.5)
                -> Linear(1)
    """
    def __init__(self, in_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.5):
        super().__init__()
        stage2_dim = max(64, hidden_dim // 2)
        self.net = nn.Sequential(
            # Stage 1
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            # Stage 2
            nn.Linear(hidden_dim, stage2_dim),
            nn.LayerNorm(stage2_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            # Output Logit
            nn.Linear(stage2_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Batch Creation Helper
# ---------------------------------------------------------------------------

def ensure_batches(task: str, force: bool = False) -> bool:
    """Ensures MOTOR batches exist for the given task via clmbr_create_batches."""
    batches_dir      = BATCHES_ROOT / task
    labeled_patients = FEATURES_ROOT / task / "labeled_patients.csv"

    if not labeled_patients.exists():
        logging.error(f"labeled_patients.csv not found for {task} at {labeled_patients}")
        logging.error("Run 9a_run_baseline_benchmark.py first to generate labels.")
        return False

    if batches_dir.exists():
        if force:
            logging.info(f"Removing existing batches for {task} (--force-batches).")
            shutil.rmtree(batches_dir)
        else:
            logging.info(f"Batches already exist for {task} at {batches_dir}")
            return True

    clmbr_create_bin = VENV_DIR / "bin" / "clmbr_create_batches"
    if not clmbr_create_bin.exists():
        logging.error(f"Executable {clmbr_create_bin} not found.")
        return False

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
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=build_env())
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()

    if proc.returncode != 0:
        logging.error(f"clmbr_create_batches failed for {task} with exit code {proc.returncode}")
        return False

    logging.info(f"Batches successfully created at {batches_dir}")
    return True


# ---------------------------------------------------------------------------
# Representation Extraction using FEMR
# ---------------------------------------------------------------------------

def load_cohort_splits(cohort_file: Path) -> dict:
    """
    Read the 'split' column from the cohort CSV/TSV and return a
    {patient_id (int) -> split_idx (0=train, 1=valid, 2=test)} mapping.
    """
    split_map = {"train": 0, "valid": 1, "test": 2}
    pid_to_split = {}
    with open(cohort_file) as f:
        reader = csv.DictReader(f, delimiter="\t" if cohort_file.suffix == ".tsv" else ",")
        for row in reader:
            pid_col = next((c for c in ("PatientID", "patient_id") if c in row), None)
            if pid_col is None or "split" not in row:
                raise ValueError(f"Cohort file missing 'PatientID'/'patient_id' or 'split' column: {cohort_file}")
            pid = int(row[pid_col])
            s   = row["split"].strip().lower()
            if s in split_map:
                pid_to_split[pid] = split_map[s]
    logging.info(f"Loaded {len(pid_to_split)} patient splits from {cohort_file}")
    return pid_to_split


def extract_representations(task: str, cohort_file: Path):
    """Loads MOTOR-t-base model, extracts patient representations and assigns
    train/valid/test splits from the authoritative cohort file."""
    import femr.datasets
    import femr.extension.dataloader
    import femr.models.transformer

    batches_path = BATCHES_ROOT / task
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

    random_seed = config.get("seed", 42)
    rng = jax.random.PRNGKey(random_seed)

    loader = femr.extension.dataloader.BatchLoader(str(EXTRACT_PATH), str(batch_info_path))
    logging.info(f"Loaded batches: Train={loader.get_number_of_batches('train')}, Dev={loader.get_number_of_batches('dev')}, Test={loader.get_number_of_batches('test')}")

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

    # Load authoritative splits from the cohort file
    pid_to_split = load_cohort_splits(cohort_file)

    reprs_list = []
    ages_list = []
    pids_list = []
    offsets_list = []

    for split in ("train", "dev", "test"):
        num_batches = loader.get_number_of_batches(split)
        logging.info(f"Processing {split} split ({num_batches} batches)...")
        for j in range(num_batches):
            raw_batch = loader.get_batch(split, j)
            batch = jax.tree_map(lambda a: jax.device_put(a, device=jax.devices("gpu")[0]), raw_batch)
            repr = compute_repr(params, rng, config, batch)

            num_indices = batch["num_indices"]
            p_index = (batch["transformer"]["label_indices"] // batch["transformer"]["length"])[:num_indices]

            # integer_ages is per TOKEN across the flattened batch; label_indices
            # gives the token positions where labels sit. Must fancy-index by those
            # positions, not take the first num_indices entries.
            li = np.asarray(raw_batch["transformer"]["label_indices"])[:num_indices]

            reprs_list.append(np.array(repr[:num_indices]))
            ages_list.append(np.asarray(raw_batch["transformer"]["integer_ages"])[li])
            pids_list.append(np.array(raw_batch["patient_ids"][p_index]))
            offsets_list.append(np.array(raw_batch["offsets"][p_index]))

    reprs = np.concatenate(reprs_list, axis=0)
    repr_ages = np.concatenate(ages_list, axis=0)
    repr_pids = np.concatenate(pids_list, axis=0)
    repr_offsets = np.concatenate(offsets_list, axis=0).astype(np.int32)

    # Label alignment
    task_labels = batch_info['config']['task']['labels']
    label_pids = np.array([val[0] for val in task_labels], dtype=np.uint64)
    label_ages = np.array([val[1] for val in task_labels], dtype=np.uint32)
    label_values = np.array([val[2] for val in task_labels], dtype=np.float32)

    sort_indices_label = np.lexsort((label_ages, label_pids))
    label_pids = label_pids[sort_indices_label]
    label_ages = label_ages[sort_indices_label]
    label_values = label_values[sort_indices_label]

    sort_indices_repr = np.lexsort((-repr_offsets, repr_ages, repr_pids))
    repr_offsets = repr_offsets[sort_indices_repr]
    repr_ages = repr_ages[sort_indices_repr]
    repr_pids = repr_pids[sort_indices_repr]

    matching_indices = []
    split_indices = []
    kept_label_idx = []   # which LABELS survived; keeps labels aligned with reprs

    j_idx = 0
    for i_lab, (l_pid, l_age) in enumerate(zip(label_pids, label_ages)):
        while True:
            if j_idx + 1 == len(repr_pids):
                break
            elif repr_pids[j_idx] < l_pid:
                pass
            else:
                next_pid = repr_pids[j_idx + 1]
                next_age = repr_ages[j_idx + 1]
                if next_pid != l_pid or next_age > l_age:
                    break
            j_idx += 1

        assert repr_pids[j_idx] == l_pid
        assert repr_ages[j_idx] <= l_age

        # Assign split from the authoritative cohort file; skip patients not in the cohort
        s = pid_to_split.get(int(l_pid))
        if s is None:
            logging.warning(f"Patient {l_pid} not found in cohort file — skipping.")
            continue
        split_indices.append(s)
        matching_indices.append(j_idx)
        kept_label_idx.append(i_lab)

    kept_label_idx = np.array(kept_label_idx, dtype=np.int64)
    matched_reprs = reprs[sort_indices_repr[matching_indices], :]
    split_indices = np.array(split_indices)

    # Labels must be filtered by LABEL index, not representation index, or they
    # desynchronise from matched_reprs/split_indices whenever a patient is skipped.
    label_values = label_values[kept_label_idx]
    label_pids = label_pids[kept_label_idx]
    label_ages = label_ages[kept_label_idx]

    assert len(matched_reprs) == len(label_values) == len(split_indices), (
        f"alignment mismatch: reprs={len(matched_reprs)} "
        f"labels={len(label_values)} splits={len(split_indices)}"
    )
    logging.info(
        f"Aligned {len(matched_reprs):,} labels to representations "
        f"(Train={(split_indices==0).sum():,} Valid={(split_indices==1).sum():,} "
        f"Test={(split_indices==2).sum():,})"
    )

    return matched_reprs, label_values, label_pids, label_ages, split_indices, database


# ---------------------------------------------------------------------------
# Training PyTorch 2-Stage MLP Classifier Head
# ---------------------------------------------------------------------------

def train_mlp_classifier(
    reprs: np.ndarray,
    labels: np.ndarray,
    split_indices: np.ndarray,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    dropout: float = 0.5,
    hidden_dim: int = 256
):
    """Trains the 2-stage MLP classifier head on extracted MOTOR representations."""
    train_mask = split_indices == 0
    val_mask   = split_indices == 1
    test_mask  = split_indices == 2

    X_train, y_train = reprs[train_mask], labels[train_mask]
    X_val,   y_val   = reprs[val_mask],   labels[val_mask]
    X_test,  y_test  = reprs[test_mask],  labels[test_mask]

    logging.info(f"Dataset split counts -> Train: {len(X_train)} | Valid: {len(X_val)} | Test: {len(X_test)}")
    logging.info(f"Train Prevalence: {y_train.mean():.4f} | Valid Prevalence: {y_val.mean():.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training PyTorch 2-Stage MLP Head on device: {device} (dropout={dropout})")

    model = MOTOR_MLP_Head(in_dim=reprs.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auroc = 0.0
    best_train_auroc = 0.0
    best_test_auroc = 0.0
    best_probs = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)

        scheduler.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            train_logits = model(torch.tensor(X_train, dtype=torch.float32).to(device))
            val_logits   = model(torch.tensor(X_val,   dtype=torch.float32).to(device))
            test_logits  = model(torch.tensor(X_test,  dtype=torch.float32).to(device))
            all_logits   = model(torch.tensor(reprs,   dtype=torch.float32).to(device))

            train_probs = torch.sigmoid(train_logits).cpu().numpy()
            val_probs   = torch.sigmoid(val_logits).cpu().numpy()
            test_probs  = torch.sigmoid(test_logits).cpu().numpy()
            all_probs   = torch.sigmoid(all_logits).cpu().numpy()

            train_auroc = sklearn.metrics.roc_auc_score(y_train, train_probs)
            val_auroc   = sklearn.metrics.roc_auc_score(y_val, val_probs)
            test_auroc  = sklearn.metrics.roc_auc_score(y_test, test_probs)

        logging.info(f"Epoch {epoch:02d}/{epochs:02d} - Loss: {total_loss/len(X_train):.4f} | Train AUROC: {train_auroc:.4f} | Valid AUROC: {val_auroc:.4f} | Test AUROC: {test_auroc:.4f}")

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_train_auroc = train_auroc
            best_test_auroc = test_auroc
            best_probs = all_probs

    metrics = {
        "train_auroc": float(best_train_auroc),
        "valid_auroc": float(best_val_auroc),
        "test_auroc": float(best_test_auroc),
        "dropout": float(dropout),
        "hidden_dim": int(hidden_dim),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr)
    }

    return model, metrics, best_probs


# ---------------------------------------------------------------------------
# Main Execution Function per Task
# ---------------------------------------------------------------------------

def run_task(task: str, args) -> dict:
    logging.info(f"\n{'='*70}\nStarting MOTOR 2-Stage MLP pipeline for task: {task}\n{'='*70}")

    output_dir = RESULTS_ROOT / f"{RUN_TIMESTAMP}_{task}_MLP"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure task file logger
    task_logger = logging.getLogger(f"task_{task}")
    task_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(output_dir / "log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    task_logger.addHandler(fh)

    # Step 1: Ensure batch files exist
    if not ensure_batches(task, force=args.force_batches):
        return {"task": task, "status": "failed (batches)"}

    # Step 2: Extract representations using MOTOR
    cohort_file = Path(args.cohort_file) if args.cohort_file else COHORT_FILE
    if cohort_file is None or not cohort_file.is_file():
        return {"task": task, "status": "failed (cohort file not found — use --cohort-file)"}

    try:
        reprs, labels, pids, ages, split_indices, database = extract_representations(task, cohort_file)
    except Exception as e:
        logging.error(f"Representation extraction failed for {task}: {e}", exc_info=True)
        return {"task": task, "status": "failed (extraction)"}

    # Step 3: Train 2-Stage MLP Head
    model, metrics, probabilities = train_mlp_classifier(
        reprs,
        labels,
        split_indices,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim
    )

    # Step 4: Save Predictions and Metrics
    prediction_dates = []
    for pid, age in zip(pids, ages):
        birth_date = datetime.combine(database.get_patient_birth_date(pid), datetime.min.time())
        prediction_dates.append(birth_date + timedelta(minutes=int(age)))

    with open(output_dir / "predictions.pkl", "wb") as f:
        pickle.dump([probabilities, pids, labels, prediction_dates], f)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logging.info(f"Results saved to {output_dir}")
    logging.info(f"[{task}] SUMMARY -> Train AUROC: {metrics['train_auroc']:.4f} | Valid AUROC: {metrics['valid_auroc']:.4f} | Test AUROC: {metrics['test_auroc']:.4f}")

    return {"task": task, "status": "ok", **metrics, "output_dir": str(output_dir)}


# ---------------------------------------------------------------------------
# Main CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    global MOTOR_ROOT, MODEL_DIR, DICTIONARY_DIR

    parser = argparse.ArgumentParser(
        description="Run MOTOR-t-base representation extraction with a 2-stage MLP classifier head."
    )
    parser.add_argument(
        "--task", "-t", type=str, required=True,
        help=f"Target task name (e.g. 'PE', '1_month_mortality') or 'all' for all tasks. Choices: {ALL_TASKS + ['all']}"
    )
    parser.add_argument("--force-batches", action="store_true", help="Re-create MOTOR batches even if already present.")
    parser.add_argument("--motor-dir", type=str, default=str(MOTOR_ROOT), help="Path to motor-t-base model directory.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of PyTorch MLP training epochs (default: 30).")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for MLP training (default: 256).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for AdamW optimizer (default: 1e-3).")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate for 2-stage MLP head (default: 0.5).")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension for MLP stage 1 (default: 256).")
    parser.add_argument(
        "--cohort-file", type=str, default=None,
        help="Path to cohort CSV/TSV with a 'split' column (train/valid/test). "
             "Auto-detected from standard DATA_PROCESSED locations if not given."
    )

    args = parser.parse_args()

    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    MOTOR_ROOT = Path(args.motor_dir)
    MODEL_DIR = MOTOR_ROOT / "model"
    DICTIONARY_DIR = MOTOR_ROOT / "dictionary"

    if VENV_DIR is None:
        sys.exit("[ERROR] Legacy venv directory not found.")

    if not MODEL_DIR.exists() or not DICTIONARY_DIR.exists():
        logging.warning(f"[WARNING] motor-t-base model/dictionary not found at {MOTOR_ROOT}")
        logging.warning("          Ensure motor-t-base is downloaded or specify --motor-dir /path/to/motor-t-base")

    # Determine tasks to process
    if args.task.lower() == "all":
        tasks_to_run = ALL_TASKS
    elif args.task in ALL_TASKS:
        tasks_to_run = [args.task]
    else:
        # Check if user stripped quotes or passed close match
        clean_task = args.task.strip("\"'")
        if clean_task in ALL_TASKS:
            tasks_to_run = [clean_task]
        else:
            sys.exit(f"[ERROR] Unknown task '{args.task}'. Choose from {ALL_TASKS} or 'all'.")

    # Set environment variables for JAX / Blackwell GPU
    for k, v in build_env().items():
        os.environ[k] = v

    summary_results = []
    for task in tasks_to_run:
        res = run_task(task, args)
        summary_results.append(res)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Task':<25} | {'Status':<15} | {'Train AUROC':<12} | {'Valid AUROC':<12} | {'Test AUROC':<12}")
    print("=" * 80)
    for r in summary_results:
        status = r["status"]
        tr = f"{r['train_auroc']:.4f}" if r.get("train_auroc") is not None else "N/A"
        va = f"{r['valid_auroc']:.4f}" if r.get("valid_auroc") is not None else "N/A"
        te = f"{r['test_auroc']:.4f}" if r.get("test_auroc") is not None else "N/A"
        print(f"{r['task']:<25} | {status:<15} | {tr:<12} | {va:<12} | {te:<12}")
    print("=" * 80)


if __name__ == "__main__":
    main()
