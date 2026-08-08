"""
Collect test AUROCs from all classify runs in the checkpoints directory.
Picks the most recent run for each target task.

Usage (from any directory):
    python Custom/collect_aurocs.py
    python Custom/collect_aurocs.py --ckpt_dir /data/processed/INSPECT/checkpoints
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from collections import defaultdict

CKPT_DIR = "/data/processed/INSPECT/checkpoints"

TASKS = [
    "pe_positive_nlp",
    "1_month_mortality",
    "6_month_mortality",
    "12_month_mortality",
    "1_month_readmission",
    "6_month_readmission",
    "12_month_readmission",
    "12_month_PH",
]

PAPER_AUROCS = {
    "pe_positive_nlp":    0.931,
    "1_month_mortality":  0.740,
    "6_month_mortality":  0.721,
    "12_month_mortality": 0.712,
    "1_month_readmission":  0.613,
    "6_month_readmission":  0.619,
    "12_month_readmission": 0.607,
    "12_month_PH":        0.782,
}

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", default=CKPT_DIR)
args = parser.parse_args()

ckpt_dir = Path(args.ckpt_dir)

# Find most recent test_preds.csv for each task
latest = defaultdict(lambda: (None, None))  # task -> (timestamp_str, path)
for run_dir in ckpt_dir.iterdir():
    if not run_dir.is_dir():
        continue
    pred_file = run_dir / "test_preds.csv"
    if not pred_file.exists():
        continue
    name = run_dir.name  # e.g. classify_pe_positive_nlp_2026-08-08_10:00:00
    for task in TASKS:
        if task in name:
            # Extract timestamp suffix for comparison
            ts = name.split(task + "_")[-1] if task in name else ""
            if ts > latest[task][0]:
                latest[task] = (ts, pred_file)

print(f"\n{'Task':<25} {'Our AUROC':>10} {'Paper AUROC':>12} {'Gap':>8}  {'Run'}")
print("-" * 85)

results = {}
for task in TASKS:
    ts, pred_file = latest[task]
    if pred_file is None:
        print(f"{task:<25} {'NOT FOUND':>10}")
        continue
    df = pd.read_csv(pred_file)
    if df["label"].nunique() < 2:
        print(f"{task:<25} {'skip (1 class)':>10}  {pred_file.parent.name}")
        continue
    auroc = roc_auc_score(df["label"], df["prob"])
    paper = PAPER_AUROCS.get(task, float("nan"))
    gap = auroc - paper
    results[task] = auroc
    print(f"{task:<25} {auroc:>10.3f} {paper:>12.3f} {gap:>+8.3f}  {pred_file.parent.name}")

if results:
    avg = sum(results.values()) / len(results)
    paper_avg = sum(PAPER_AUROCS[t] for t in results) / len(results)
    print("-" * 85)
    print(f"{'Mean':<25} {avg:>10.3f} {paper_avg:>12.3f} {avg-paper_avg:>+8.3f}")
