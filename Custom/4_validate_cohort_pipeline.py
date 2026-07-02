"""
INSPECT Cohort Pipeline Validation
===================================
Validates the full cohort creation pipeline across three layers:
  1. cohort_0.2.0_master_file_anon.csv  (output of 2_merge_labels.py)
  2. labeled_patients.csv               (output of 2_generate_labels_and_features.py)
  3. featurized_patients.pkl            (output of 2_generate_labels_and_features.py)

Run from the repo root or any directory — all paths are resolved from $HOME.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import scipy.sparse

COHORT_PATH    = os.path.expanduser("../DATA_PROCESSED/cohort_0.2.0_master_file_anon.csv")
LABELS_CSV     = os.path.expanduser("../DATA_RAW/EHR_FEMR_DB/features/PE/labeled_patients.csv")
FEATURES_PKL   = os.path.expanduser("../DATA_RAW/EHR_FEMR_DB/features/PE/featurized_patients.pkl")

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
WARN = "\033[93m WARN\033[0m"

def check(label, condition, detail="", warn=False):
    status = (WARN if warn else FAIL) if not condition else PASS
    suffix = f"  → {detail}" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return condition


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1: cohort CSV
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("LAYER 1: cohort_0.2.0_master_file_anon.csv")
print("="*70)

if not os.path.exists(COHORT_PATH):
    print(f"  [FAIL] File not found: {COHORT_PATH}")
    sys.exit(1)

df = pd.read_csv(COHORT_PATH)

# Required columns
required_cols = ["PatientID", "StudyTime", "pe_positive_nlp", "split", "impression_id"]
for col in required_cols:
    check(f"Column '{col}' present", col in df.columns)

print(f"\n  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

# Missing values in critical columns
for col in required_cols:
    if col in df.columns:
        n_miss = df[col].isna().sum()
        check(f"No nulls in '{col}'", n_miss == 0, f"{n_miss} nulls" if n_miss else "")

# Label distribution
print("\n  Label distribution (pe_positive_nlp):")
print("  " + df["pe_positive_nlp"].value_counts().to_string().replace("\n", "\n  "))
# Labels may be bool or string depending on how pandas loaded the CSV
raw_vals = set(df["pe_positive_nlp"].dropna().unique())
valid_bool   = raw_vals.issubset({True, False})
valid_string = raw_vals.issubset({"True", "False"})
check("Labels are boolean or 'True'/'False' string values",
      valid_bool or valid_string,
      f"unexpected values: {raw_vals}" if not (valid_bool or valid_string) else "")
# Compute prevalence robustly regardless of dtype
if valid_bool:
    prevalence = df["pe_positive_nlp"].mean()
else:
    prevalence = (df["pe_positive_nlp"] == "True").mean()
check("PE prevalence in expected range (5–30%)", 0.05 <= prevalence <= 0.30,
      f"{prevalence*100:.1f}%")

# Split distribution
print("\n  Split distribution:")
print("  " + df["split"].value_counts().to_string().replace("\n", "\n  "))
expected_splits = {"train", "valid", "test"}
actual_splits = set(df["split"].dropna().unique())
check("Split values are train/valid/test", actual_splits == expected_splits,
      f"got {actual_splits}")

# Patient-level split integrity: no patient in >1 split
pid_splits = df.groupby("PatientID")["split"].nunique()
crossover_pids = pid_splits[pid_splits > 1]
check("No PatientID appears in multiple splits",
      len(crossover_pids) == 0,
      f"{len(crossover_pids)} patients span splits — DATA LEAKAGE" if len(crossover_pids) else "")

# impression_id uniqueness
n_dup = df["impression_id"].duplicated().sum() if "impression_id" in df.columns else 0
check("No duplicate impression_ids", n_dup == 0, f"{n_dup} duplicates" if n_dup else "")
if n_dup > 0:
    dup_ids = df[df["impression_id"].duplicated(keep=False)]["impression_id"].unique()
    dup_rows = df[df["impression_id"].isin(dup_ids)][["impression_id", "PatientID", "split", "StudyTime"]]
    splits_per_imp = dup_rows.groupby("impression_id")["split"].nunique()
    cross_split_dups = (splits_per_imp > 1).sum()
    same_split_dups  = (splits_per_imp == 1).sum()
    check("Duplicate impression_ids are within the same split (not cross-split leakage)",
          cross_split_dups == 0,
          f"{cross_split_dups} duplicates span multiple splits — DATA LEAKAGE" if cross_split_dups else
          f"all {same_split_dups} duplicates are within a single split (double-counting only)")
    print("\n  Duplicate impression_id details:")
    print(dup_rows.sort_values("impression_id").to_string(index=False))

# StudyTime sanity
df["StudyTime_dt"] = pd.to_datetime(df["StudyTime"], errors="coerce")
n_parse_fail = df["StudyTime_dt"].isna().sum()
check("All StudyTimes parse as datetime", n_parse_fail == 0, f"{n_parse_fail} failures")
sentinel_rows = (df["StudyTime_dt"] < pd.Timestamp("1980-01-01")).sum()
future_rows   = (df["StudyTime_dt"] > pd.Timestamp("2025-01-01")).sum()
check("No sentinel/ghost timestamps (<1980)", sentinel_rows == 0,
      f"{sentinel_rows} rows with pre-1980 timestamp", warn=True)
check("No implausible future timestamps (>2025)", future_rows == 0,
      f"{future_rows} rows", warn=True)
print(f"  StudyTime range: {df['StudyTime_dt'].min().date()} → {df['StudyTime_dt'].max().date()}")

# Multiple scans per patient
scans_per_patient = df.groupby("PatientID").size()
multi_scan = (scans_per_patient > 1).sum()
print(f"\n  Patients with >1 scan: {multi_scan:,} / {df['PatientID'].nunique():,} unique patients")


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2: labeled_patients.csv
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("LAYER 2: labeled_patients.csv  (femr output)")
print("="*70)

if not os.path.exists(LABELS_CSV):
    print(f"  [SKIP] File not found: {LABELS_CSV}")
else:
    ldf = pd.read_csv(LABELS_CSV)
    print(f"  Rows: {len(ldf):,}  |  Columns: {list(ldf.columns)}")
    for col in ["patient_id", "prediction_time", "value"]:
        check(f"Column '{col}' present", col in ldf.columns)

    n_dup = ldf.duplicated(subset=["patient_id", "prediction_time"]).sum() if \
            {"patient_id","prediction_time"}.issubset(ldf.columns) else "N/A"
    check("No duplicate (patient_id, prediction_time) pairs",
          n_dup == 0, f"{n_dup} duplicates" if n_dup else "")

    # Label distribution
    print("\n  Label distribution (value):")
    print("  " + ldf["value"].value_counts().to_string().replace("\n", "\n  ") if "value" in ldf.columns else "  N/A")

    # Cross-check prevalence against cohort CSV
    if "value" in ldf.columns:
        femr_prevalence = ldf["value"].mean() if ldf["value"].dtype == bool \
            else (ldf["value"].astype(str).str.strip().str.lower() == "true").mean()
        delta = abs(femr_prevalence - prevalence)
        check("Label prevalence matches cohort CSV (±2%)", delta < 0.02,
              f"cohort={prevalence*100:.1f}%  femr={femr_prevalence*100:.1f}%  Δ={delta*100:.1f}%",
              warn=delta >= 0.01)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3: featurized_patients.pkl
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("LAYER 3: featurized_patients.pkl  (femr output)")
print("="*70)

if not os.path.exists(FEATURES_PKL):
    print(f"  [SKIP] File not found: {FEATURES_PKL}")
else:
    with open(FEATURES_PKL, "rb") as f:
        feature_matrix, patient_ids, label_values, label_times = pickle.load(f)

    n_samples = feature_matrix.shape[0]
    n_features = feature_matrix.shape[1]
    print(f"  Matrix shape: {n_samples:,} × {n_features:,}")

    nnz = feature_matrix.nnz if scipy.sparse.issparse(feature_matrix) else np.count_nonzero(feature_matrix)
    density = nnz / (n_samples * n_features) * 100
    print(f"  Non-zero elements: {nnz:,}  ({density:.3f}% density)")

    # NaN / Inf
    data = feature_matrix.data if scipy.sparse.issparse(feature_matrix) else feature_matrix.ravel()
    check("No NaN values in feature matrix", not np.isnan(data).any())
    check("No Inf values in feature matrix", not np.isinf(data).any())

    # Length alignment
    check("patient_ids length matches matrix rows",
          len(patient_ids) == n_samples, f"{len(patient_ids)} vs {n_samples}")
    check("label_values length matches matrix rows",
          len(label_values) == n_samples, f"{len(label_values)} vs {n_samples}")
    check("label_times length matches matrix rows",
          len(label_times) == n_samples, f"{len(label_times)} vs {n_samples}")

    # Label distribution in pickle
    pkl_prevalence = label_values.mean() if hasattr(label_values, "mean") else np.mean(label_values)
    print(f"\n  PE prevalence in pickle: {pkl_prevalence*100:.1f}%")
    delta = abs(pkl_prevalence - prevalence)
    check("Pickle prevalence matches cohort CSV (±2%)", delta < 0.02,
          f"cohort={prevalence*100:.1f}%  pkl={pkl_prevalence*100:.1f}%  Δ={delta*100:.1f}%",
          warn=delta >= 0.01)

    # ── Cross-pipeline split integrity ────────────────────────────────────────
    print("\n" + "="*70)
    print("LAYER 4: Cross-pipeline split integrity")
    print("="*70)

    cohort_pid_split = dict(zip(df["PatientID"].astype(int), df["split"]))
    pkl_pids = np.array(patient_ids, dtype=int)

    # All pickle PIDs exist in cohort
    missing_from_cohort = set(pkl_pids) - set(cohort_pid_split.keys())
    check("All featurized PatientIDs exist in cohort CSV",
          len(missing_from_cohort) == 0,
          f"{len(missing_from_cohort)} PIDs in pickle but not in cohort")

    # Reconstruct split masks
    pkl_splits = np.array([cohort_pid_split.get(p, "unknown") for p in pkl_pids])
    train_mask = pkl_splits == "train"
    valid_mask = pkl_splits == "valid"
    test_mask  = pkl_splits == "test"
    unknown    = (pkl_splits == "unknown").sum()

    check("No featurized patients with unknown split", unknown == 0,
          f"{unknown} patients have no split assignment")

    print(f"\n  Featurized split counts:")
    print(f"    train : {train_mask.sum():,}  ({label_values[train_mask].mean()*100:.1f}% PE+)")
    print(f"    valid : {valid_mask.sum():,}  ({label_values[valid_mask].mean()*100:.1f}% PE+)")
    print(f"    test  : {test_mask.sum():,}  ({label_values[test_mask].mean()*100:.1f}% PE+)")

    # Check no patient is in multiple splits (at the featurized level too)
    from collections import Counter
    pid_split_count = Counter(cohort_pid_split.get(p, "unknown") for p in pkl_pids)
    # A patient appearing multiple times in pkl_pids is fine (multi-scan) as long
    # as all their entries map to the same split
    pid_to_splits_in_pkl = {}
    for p, s in zip(pkl_pids, pkl_splits):
        pid_to_splits_in_pkl.setdefault(p, set()).add(s)
    crossover = {p: s for p, s in pid_to_splits_in_pkl.items() if len(s) > 1}
    check("No PatientID spans multiple splits in featurized output",
          len(crossover) == 0,
          f"{len(crossover)} patients span splits — DATA LEAKAGE" if crossover else "")

print("\n" + "="*70)
print("Validation complete.")
print("="*70 + "\n")
