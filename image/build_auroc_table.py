"""
Build an AUROC comparison table from run_classify_all.sh's outputs.

For each of the 8 INSPECT classification targets, finds the most recent
classify_{target}_{timestamp} run directory under CHECKPOINT_DIR, computes
AUROC from its test_preds.csv (label vs prob columns, written by
ClassificationLightningModel.shared_epoch_end), and prints it alongside the
published Image-modality (CT-only) AUROCs from the INSPECT paper.
"""

import glob
import os

import pandas as pd
from sklearn.metrics import roc_auc_score

CHECKPOINT_DIR = "/data/processed/INSPECT/checkpoints"

# (display name, dataset.target string used in run_classify_*.sh, published CT-only AUROC)
TARGETS = [
    ("PE",               "pe_positive_nlp",      0.721),
    ("Mortality 1m",     "1_month_mortality",    0.794),
    ("Mortality 6m",     "6_month_mortality",    0.755),
    ("Mortality 12m",    "12_month_mortality",   0.748),
    ("Readmission 1m",   "1_month_readmission",  0.549),
    ("Readmission 6m",   "6_month_readmission",  0.515),
    ("Readmission 12m",  "12_month_readmission", 0.525),
    ("PH 12m",           "12_month_PH",          0.661),
]


def latest_run_dir(target):
    pattern = os.path.join(CHECKPOINT_DIR, f"classify_{target}_*")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    return matches[-1] if matches else None


def compute_auroc(run_dir):
    preds_path = os.path.join(run_dir, "test_preds.csv")
    if not os.path.exists(preds_path):
        return None, preds_path
    df = pd.read_csv(preds_path)
    return roc_auc_score(df["label"], df["prob"]), preds_path


def main():
    rows = []
    for name, target, published in TARGETS:
        run_dir = latest_run_dir(target)
        if run_dir is None:
            rows.append((name, published, None, None, f"NO RUN DIR matching classify_{target}_*"))
            continue
        auroc, preds_path = compute_auroc(run_dir)
        if auroc is None:
            rows.append((name, published, None, run_dir, f"test_preds.csv missing at {preds_path}"))
            continue
        diff = auroc - published
        rows.append((name, published, auroc, run_dir, f"{diff:+.3f}"))

    print(f"{'Target':<18}{'Published':>10}{'Ours':>10}{'Diff':>10}   Run dir")
    print("-" * 110)
    for name, published, auroc, run_dir, note in rows:
        auroc_str = f"{auroc:.3f}" if auroc is not None else "N/A"
        diff_str = note if auroc is not None else ""
        run_dir_str = run_dir if run_dir else note
        print(f"{name:<18}{published:>10.3f}{auroc_str:>10}{diff_str:>10}   {run_dir_str}")


if __name__ == "__main__":
    main()
