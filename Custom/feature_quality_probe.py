"""
Feature quality ceiling probe — image/Custom/feature_quality_probe.py

Fits a logistic regression on mean-pooled CNN embeddings (no LSTM) to establish
the upper bound on what the 1D classifier can achieve given the current features.

If this probe AUROC ≈ LSTM AUROC, the bottleneck is feature quality, not the
sequence model. Re-featurize with a better backbone to raise the ceiling.

Usage:
    python Custom/feature_quality_probe.py
    python Custom/feature_quality_probe.py --target 12_month_mortality
    python Custom/feature_quality_probe.py --hdf5 /path/to/features.hdf5
"""

import argparse
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ── defaults ────────────────────────────────────────────────────────────────
DEFAULT_HDF5   = "/data/processed/INSPECT/CNN_embeddings/features_resnext_uncompressed.hdf5"
DEFAULT_CSV    = "/home/users/steven/INSPECT/DATA_RAW/LABELS/series_metadata_20250611.tsv"
DEFAULT_LABELS = "/home/users/steven/INSPECT/DATA_RAW/LABELS/labels_20250611.tsv"
DEFAULT_SPLITS = "/home/users/steven/INSPECT/DATA_RAW/LABELS/splits_20250611.tsv"
DEFAULT_MAPPING = "/home/users/steven/INSPECT/DATA_RAW/LABELS/study_mapping_20250611.tsv"
DEFAULT_TARGET = "pe_positive_nlp"

CENSORED = {"Censored", "Censor"}


def load_dataframe(csv_path, label_path, split_path, mapping_path, target):
    sep = lambda p: "\t" if str(p).endswith(".tsv") else ","

    df = pd.read_csv(csv_path, sep=sep(csv_path))
    df["patient_datetime"] = df["image_id"].astype(str).str.replace(".nii.gz", "", regex=False)
    df = df.drop_duplicates(subset=["patient_datetime"])

    # merge impression_id
    if "impression_id" not in df.columns:
        map_df = pd.read_csv(mapping_path, sep=sep(mapping_path))
        map_df["clean_image_id"] = map_df["image_id"].astype(str).str.replace(".nii.gz", "", regex=False)
        df["clean_image_id"] = df["patient_datetime"]
        df = df.merge(
            map_df[["clean_image_id", "impression_id"]].drop_duplicates("clean_image_id"),
            on="clean_image_id", how="left",
        ).drop(columns=["clean_image_id"])

    # merge labels
    label_df = pd.read_csv(label_path, sep=sep(label_path))
    df = df.merge(label_df, on="impression_id", how="left")

    # merge splits
    split_df = pd.read_csv(split_path, sep=sep(split_path))
    df = df.merge(split_df, on="impression_id", how="left")

    # drop censored
    df = df[~df[target].astype(str).isin(CENSORED)]

    # binary label
    df["label"] = (df[target].astype(str).str.upper() == "TRUE").astype(int)

    return df


def build_feature_matrix(df, hdf5_path):
    """Mean-pool each scan's slice features into a single vector."""
    with h5py.File(hdf5_path, "r") as f:
        available = set(f.keys())
        rows, labels, splits = [], [], []
        for _, row in df.iterrows():
            key = str(row["patient_datetime"]).replace(".nii.gz", "")
            if key not in available:
                continue
            feat = f[key][:]           # (num_slices, feature_dim)
            rows.append(feat.mean(0))  # mean pool → (feature_dim,)
            labels.append(row["label"])
            splits.append(row["split"])

    X = np.stack(rows, axis=0)
    y = np.array(labels)
    splits = np.array(splits)
    return X, y, splits


def run_probe(X, y, splits):
    train_mask = splits == "train"
    test_mask  = splits == "test"

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask])
    X_test  = scaler.transform(X[test_mask])
    y_train = y[train_mask]
    y_test  = y[test_mask]

    print(f"Train: {train_mask.sum()} samples  (pos={y_train.sum()}, neg={(y_train==0).sum()})")
    print(f"Test:  {test_mask.sum()} samples   (pos={y_test.sum()}, neg={(y_test==0).sum()})")

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, proba)
    print(f"\nLogistic Regression on mean-pooled features  →  AUROC = {auroc:.4f}")
    print("This is the feature quality ceiling — the LSTM cannot exceed it.")
    return auroc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5",    default=DEFAULT_HDF5)
    parser.add_argument("--csv",     default=DEFAULT_CSV)
    parser.add_argument("--labels",  default=DEFAULT_LABELS)
    parser.add_argument("--splits",  default=DEFAULT_SPLITS)
    parser.add_argument("--mapping", default=DEFAULT_MAPPING)
    parser.add_argument("--target",  default=DEFAULT_TARGET)
    args = parser.parse_args()

    print(f"Target: {args.target}")
    print(f"HDF5:   {args.hdf5}\n")

    df = load_dataframe(args.csv, args.labels, args.splits, args.mapping, args.target)
    X, y, splits = build_feature_matrix(df, args.hdf5)
    run_probe(X, y, splits)


if __name__ == "__main__":
    main()
