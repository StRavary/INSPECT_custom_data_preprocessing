"""
9e_run_all_tasks_motor.py

Runs clmbr_train_linear_probe (MOTOR foundation model) across all 8 INSPECT
tasks on Blackwell GPUs, captures Train/Valid/Test AUROC and L2 Strength per
task, and writes a results CSV + printed summary table.

Prerequisites
-------------
- femr_cuda 0.1.16 installed in the venv
- CUDA 12.8 ptxas downloaded to ~/cu12_8_ptxas/ (see README.md Step B)
- MOTOR_batches generated for all tasks under DATA_RAW/EHR_FEMR_DB/MOTOR_batches/<task>
- motor-t-base model weights at ~/Documents/INSPECT/motor-t-base/model
  (or set MODEL_DIR below)

Usage
-----
    # Activate venv first, then:
    python Custom/9e_run_all_tasks_motor.py

    # Skip tasks whose batches dir is missing (instead of erroring):
    python Custom/9e_run_all_tasks_motor.py --skip-missing
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime
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

EXTRACT_PATH  = DATA_RAW / "extract"
BATCHES_ROOT  = DATA_RAW / "MOTOR_batches"
RESULTS_ROOT  = DATA_RAW / "motor_results"
MODEL_DIR     = Path.home() / "Documents" / "INSPECT" / "motor-t-base" / "model"

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AUROC_RE = re.compile(
    r"\[(INFO|WARNING)\]\s+(Train|Valid|Test) AUROC\s+([\d.]+)", re.IGNORECASE
)
L2_RE = re.compile(r"\[(INFO|WARNING)\]\s+L2 Strength\s+([\d.eE+\-]+)", re.IGNORECASE)


def parse_metrics(output: str) -> dict:
    metrics = {"train_auroc": None, "valid_auroc": None, "test_auroc": None, "l2_strength": None}
    for match in AUROC_RE.finditer(output):
        split = match.group(2).lower()
        metrics[f"{split}_auroc"] = float(match.group(3))
    l2 = L2_RE.search(output)
    if l2:
        metrics["l2_strength"] = float(l2.group(2))
    return metrics


def build_env() -> dict:
    env = os.environ.copy()
    env["XLA_FLAGS"] = XLA_FLAGS
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env.pop("JAX_PLATFORMS", None)   # must not be set to 'cpu'
    return env


def run_task(task: str, clmbr_bin: Path, skip_missing: bool) -> dict:
    batches_path = BATCHES_ROOT / task
    output_dir   = RESULTS_ROOT / task

    if not batches_path.exists():
        msg = f"MOTOR_batches directory not found: {batches_path}"
        if skip_missing:
            print(f"  [SKIP] {msg}")
            return {"task": task, "status": "skipped (no batches)", **{k: None for k in
                    ["train_auroc", "valid_auroc", "test_auroc", "l2_strength"]}}
        raise FileNotFoundError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(clmbr_bin),
        str(output_dir),
        "--data_path",    str(EXTRACT_PATH),
        "--model_dir",    str(MODEL_DIR),
        "--batches_path", str(batches_path),
    ]

    print(f"  Command: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=build_env(),
    )

    captured = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        captured.append(line)

    proc.wait()
    full_output = "".join(captured)

    if proc.returncode != 0:
        return {"task": task, "status": "failed", **{k: None for k in
                ["train_auroc", "valid_auroc", "test_auroc", "l2_strength"]}}

    metrics = parse_metrics(full_output)
    return {"task": task, "status": "ok", **metrics}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run MOTOR linear probe across all INSPECT tasks")
    parser.add_argument("--skip-missing", action="store_true",
                        help="Skip tasks whose MOTOR_batches directory is absent")
    args = parser.parse_args()

    # Resolve clmbr binary
    if VENV_DIR is None:
        sys.exit("[ERROR] Could not locate venv. Set VENV_DIR manually.")
    clmbr_bin = VENV_DIR / "bin" / "clmbr_train_linear_probe"
    if not clmbr_bin.exists():
        sys.exit(f"[ERROR] clmbr_train_linear_probe not found at {clmbr_bin}. "
                 "Is femr_cuda 0.1.16 installed?")

    if not MODEL_DIR.exists():
        sys.exit(f"[ERROR] Model directory not found: {MODEL_DIR}. "
                 "Download motor-t-base from StanfordShahLab/motor-t-base on HuggingFace.")

    print(f"femr venv  : {VENV_DIR}")
    print(f"clmbr bin  : {clmbr_bin}")
    print(f"model dir  : {MODEL_DIR}")
    print(f"extract    : {EXTRACT_PATH}")
    print(f"batches    : {BATCHES_ROOT}")
    print(f"results    : {RESULTS_ROOT}")
    print(f"XLA_FLAGS  : {XLA_FLAGS}\n")

    all_results = []

    for task in TASKS:
        print(f"\n{'='*70}")
        print(f"  Task: {task}")
        print(f"{'='*70}")
        result = run_task(task, clmbr_bin, args.skip_missing)
        all_results.append(result)

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_ROOT / f"motor_results_{timestamp}.csv"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    fieldnames = ["task", "status", "train_auroc", "valid_auroc", "test_auroc", "l2_strength"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    col = {"task": 25, "status": 9, "train": 12, "valid": 12, "test": 12, "l2": 14}

    header = (f"{'Task':<{col['task']}} {'Status':<{col['status']}} "
              f"{'Train AUROC':>{col['train']}} {'Valid AUROC':>{col['valid']}} "
              f"{'Test AUROC':>{col['test']}} {'L2 Strength':>{col['l2']}}")

    print(f"\n\n{'='*80}")
    print("  MOTOR LINEAR PROBE — ALL TASKS")
    print(f"{'='*80}")
    print(header)
    print("-" * 80)

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
