"""
9e_run_all_tasks_motor.py

Runs clmbr_create_batches + clmbr_train_linear_probe (MOTOR foundation model)
across all 8 INSPECT tasks on Blackwell GPUs, captures Train/Valid/Test AUROC
and L2 Strength per task, and writes a results CSV + printed summary table.

Prerequisites
-------------
- femr_cuda 0.1.16 installed in the venv
- CUDA 12.8 ptxas downloaded to ~/cu12_8_ptxas/ (see README.md Step B)
- Labeled patients CSVs generated for each task under:
    DATA_RAW/EHR_FEMR_DB/features/<task>/labeled_patients.csv
  (produced by 9a_run_baseline_benchmark.py or ehr/2_generate_labels_and_features.py)
- motor-t-base model + dictionary at ~/Documents/INSPECT/motor-t-base/
  (model/ and dictionary/ subdirs must both be present)

Usage
-----
    # Activate venv first, then from the project root:
    python Custom/9e_run_all_tasks_motor.py

    # To force-recreate batches even if they already exist:
    python Custom/9e_run_all_tasks_motor.py --force-batches

    # To force-retrain probes even if output dirs already exist:
    python Custom/9e_run_all_tasks_motor.py --force-probe
"""

import argparse
import csv
from datetime import datetime
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

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

# Paths — all resolved relative to this script's location so the script can be
# called from any working directory.
SCRIPT_DIR  = Path(__file__).resolve().parent          # Custom/
PROJECT_DIR = SCRIPT_DIR.parent                         # INSPECT_custom_data_preprocessing/
DATA_RAW    = PROJECT_DIR.parent / "DATA_RAW" / "EHR_FEMR_DB"

RUN_TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")

EXTRACT_PATH   = DATA_RAW / "extract"
FEATURES_ROOT  = DATA_RAW / "features"       # labeled_patients.csv lives here per task
BATCHES_ROOT   = DATA_RAW / "MOTOR_batches"
RESULTS_ROOT   = DATA_RAW / "motor_results"
MOTOR_ROOT     = Path.home() / "Documents" / "INSPECT" / "motor-t-base"
MODEL_DIR      = MOTOR_ROOT / "model"
DICTIONARY_DIR = MOTOR_ROOT / "dictionary"

# Virtual environment — supports both hidden (.venv_legacy) and standard name
_venv_candidates = [PROJECT_DIR / ".venv_legacy", PROJECT_DIR / "venv_legacy",
                    Path.home() / "Documents" / "INSPECT" / "venv_legacy"]
VENV_DIR = next((p for p in _venv_candidates if p.is_dir()), None)

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


def build_env() -> dict:
    env = os.environ.copy()
    env["XLA_FLAGS"] = XLA_FLAGS
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env.pop("JAX_PLATFORMS", None)   # must not be set to 'cpu'
    return env


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

AUROC_RE = re.compile(
    r"\[?(INFO|WARNING)\]?\s+(Train|Valid|Test) AUROC\s+([\d.]+)", re.IGNORECASE
)
L2_RE = re.compile(r"\[?(INFO|WARNING)\]?\s+L2 Strength\s+([\d.eE+\-]+)", re.IGNORECASE)


def parse_metrics(output: str) -> dict:
    metrics = {"train_auroc": None, "valid_auroc": None, "test_auroc": None, "l2_strength": None}
    for match in AUROC_RE.finditer(output):
        split = match.group(2).lower()
        metrics[f"{split}_auroc"] = float(match.group(3))
    l2 = L2_RE.search(output)
    if l2:
        metrics["l2_strength"] = float(l2.group(2))
    return metrics


# ---------------------------------------------------------------------------
# Batch creation
# ---------------------------------------------------------------------------

def create_batches(task: str, clmbr_create_bin: Path, force: bool) -> bool:
    """Run clmbr_create_batches for a single task. Returns True on success."""
    batches_dir      = BATCHES_ROOT / task
    labeled_patients = FEATURES_ROOT / task / "labeled_patients.csv"

    if not labeled_patients.exists():
        print(f"  [ERROR] labeled_patients.csv not found for {task}: {labeled_patients}")
        print(f"          Run 9a_run_baseline_benchmark.py --task {task} first.")
        return False

    if batches_dir.exists():
        if force:
            print(f"  [INFO] Removing existing batches for {task} (--force-batches).")
            shutil.rmtree(batches_dir)
        else:
            print(f"  [INFO] Batches already exist for {task}, skipping creation.")
            return True

    cmd = [
        str(clmbr_create_bin),
        str(batches_dir),
        "--data_path",            str(EXTRACT_PATH),
        "--task",                 "labeled_patients",
        "--labeled_patients_path", str(labeled_patients),
        "--val_start",            "80",
        "--dictionary_path",      str(DICTIONARY_DIR),
        "--is_hierarchical",
    ]

    print(f"  Creating batches: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=build_env())
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()

    if proc.returncode != 0:
        print(f"  [ERROR] clmbr_create_batches failed for {task} (exit {proc.returncode})")
        return False

    print(f"  [OK] Batches created at {batches_dir}")
    return True


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

def run_probe(task: str, clmbr_probe_bin: Path, force: bool) -> dict:
    batches_dir = BATCHES_ROOT / task
    output_dir  = RESULTS_ROOT / f"{RUN_TIMESTAMP}_{task}"

    if not batches_dir.exists():
        return {"task": task, "status": "failed (no batches)",
                **{k: None for k in ["train_auroc", "valid_auroc", "test_auroc", "l2_strength"]}}

    if output_dir.exists():
        print(f"  [INFO] Removing existing results for {task} to allow fresh run.")
        shutil.rmtree(output_dir)

    # Do NOT pre-create output_dir — clmbr_train_linear_probe must create it itself
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(clmbr_probe_bin),
        str(output_dir),
        "--data_path",    str(EXTRACT_PATH),
        "--model_dir",    str(MODEL_DIR),
        "--batches_path", str(batches_dir),
    ]

    print(f"  Running probe: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=build_env())
    captured = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        captured.append(line)
    proc.wait()

    full_output = "".join(captured)

    if proc.returncode != 0:
        return {"task": task, "status": "failed",
                **{k: None for k in ["train_auroc", "valid_auroc", "test_auroc", "l2_strength"]}}

    metrics = parse_metrics(full_output)
    return {"task": task, "status": "ok", **metrics}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run MOTOR linear probe across all INSPECT tasks"
    )
    parser.add_argument("--force-batches", action="store_true",
                        help="Re-create MOTOR batches even if they already exist")
    parser.add_argument("--force-probe", action="store_true",
                        help="Re-run linear probe even if results already exist")
    args = parser.parse_args()

    if VENV_DIR is None:
        sys.exit("[ERROR] Could not locate venv.")

    clmbr_create_bin = VENV_DIR / "bin" / "clmbr_create_batches"
    clmbr_probe_bin  = VENV_DIR / "bin" / "clmbr_train_linear_probe"

    for bin_path in (clmbr_create_bin, clmbr_probe_bin):
        if not bin_path.exists():
            sys.exit(f"[ERROR] {bin_path.name} not found. Is femr_cuda 0.1.16 installed?")

    if not MODEL_DIR.exists():
        sys.exit(f"[ERROR] Model directory not found: {MODEL_DIR}")

    if not DICTIONARY_DIR.exists():
        sys.exit(f"[ERROR] Dictionary directory not found: {DICTIONARY_DIR}\n"
                 "        motor-t-base must contain both model/ and dictionary/ subdirs.")

    print(f"femr venv   : {VENV_DIR}")
    print(f"model dir   : {MODEL_DIR}")
    print(f"dictionary  : {DICTIONARY_DIR}")
    print(f"extract     : {EXTRACT_PATH}")
    print(f"features    : {FEATURES_ROOT}")
    print(f"batches     : {BATCHES_ROOT}")
    print(f"results     : {RESULTS_ROOT}")
    print(f"XLA_FLAGS   : {XLA_FLAGS}\n")

    all_results = []

    for task in TASKS:
        print(f"\n{'='*70}")
        print(f"  Task: {task}")
        print(f"{'='*70}")

        # Step 1: ensure batches exist
        batches_ok = create_batches(task, clmbr_create_bin, args.force_batches)
        if not batches_ok:
            all_results.append({
                "task": task, "status": "failed (batch creation)",
                **{k: None for k in ["train_auroc", "valid_auroc", "test_auroc", "l2_strength"]}
            })
            continue

        # Step 2: train linear probe
        result = run_probe(task, clmbr_probe_bin, args.force_probe)
        all_results.append(result)

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_ROOT / "motor_results.csv"

    fieldnames = ["task", "status", "train_auroc", "valid_auroc", "test_auroc", "l2_strength"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    col = {"task": 25, "status": 22, "train": 12, "valid": 12, "test": 12, "l2": 14}

    header = (f"{'Task':<{col['task']}} {'Status':<{col['status']}} "
              f"{'Train AUROC':>{col['train']}} {'Valid AUROC':>{col['valid']}} "
              f"{'Test AUROC':>{col['test']}} {'L2 Strength':>{col['l2']}}")

    print(f"\n\n{'='*85}")
    print("  MOTOR LINEAR PROBE — ALL TASKS")
    print(f"{'='*85}")
    print(header)
    print("-" * 85)

    for r in all_results:
        fmt = lambda v: f"{v:.4f}" if v is not None else "—"
        print(f"{r['task']:<{col['task']}} {r['status']:<{col['status']}} "
              f"{fmt(r['train_auroc']):>{col['train']}} "
              f"{fmt(r['valid_auroc']):>{col['valid']}} "
              f"{fmt(r['test_auroc']):>{col['test']}} "
              f"{fmt(r['l2_strength']):>{col['l2']}}")

    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
