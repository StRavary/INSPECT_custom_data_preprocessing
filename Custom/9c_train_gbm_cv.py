import argparse
import os
import pickle
import sys
from typing import List, Tuple, Optional
from collections import Counter
import collections

# Add project root and ehr directory to sys.path so relative imports work
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ehr"))

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    LogisticRegressionCV,
)
from sklearn.preprocessing import MaxAbsScaler
from loguru import logger
from utils import load_data, save_data
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, PredefinedSplit, ParameterGrid
from scipy.sparse import issparse
import scipy
import csv
import lightgbm as ltb

import femr
import femr.datasets


XGB_PARAMS = {
    "max_depth": [3, 6, -1],
    "learning_rate": [0.02, 0.1, 0.5],
    "num_leaves": [10, 25, 100],
}

PATIENT_ID_COLUMN = "PatientID"  # "patient_id"
TIME_COLUMN = "StudyTime"  # "procedure_time"


def compute_sens_spec(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sens, spec


def tune_hyperparams(
    X_train, y_train, X_val, y_val, model, params, num_threads: int = 1
):
    # In `test_fold`, -1 indicates that the corresponding sample is used for training, and a value >=0 indicates the test set.
    # We use `PredefinedSplit` to specify our custom validation split
    if issparse(X_train):
        # Need to concatenate sparse matrices differently
        X = scipy.sparse.vstack([X_train, X_val])
    else:
        X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)
    test_fold = -np.ones(X.shape[0])
    test_fold[X_train.shape[0] :] = 1
    clf = GridSearchCV(
        model,
        params,
        n_jobs=6,
        verbose=1,
        cv=PredefinedSplit(test_fold=test_fold),
        refit=False,
    )
    clf.fit(X, y)
    best_model = model.__class__(**clf.best_params_)
    best_model.fit(X_train, y_train)
    return best_model


def main(args):
    # Load all patient IDs from the cohort file
    all_cohort_pids = set()
    with open(args.path_to_cohort) as f:
        if args.path_to_cohort.endswith(".csv"):
            reader = csv.DictReader(f)
        elif args.path_to_cohort.endswith(".tsv"):
            reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            all_cohort_pids.add(int(row[PATIENT_ID_COLUMN]))

    # Load patient labels & features
    with open(
        os.path.join(args.path_to_label_features, "featurized_patients.pkl"), "rb"
    ) as f:
        features, patient_ids, label_values, label_times = pickle.load(f)

    # Filter by cohort patient IDs (leak-proof cohort mask)
    cohort_mask = np.isin(patient_ids, list(all_cohort_pids))
    logger.info(f"Task: {args.path_to_output_dir.split('/')[-1]}")
    logger.info(f"Total cohort samples in features: {sum(cohort_mask)} out of {len(patient_ids)}")

    # Extract unique cohort patient IDs to split into folds
    unique_pids = np.unique(patient_ids[cohort_mask])
    unique_pids = sorted(list(unique_pids))
    
    # Deterministic partition of patients into 5 folds
    rng = np.random.default_rng(42)
    rng.shuffle(unique_pids)
    folds = np.array_split(unique_pids, 5)

    os.makedirs(args.path_to_output_dir, exist_ok=True)

    # Array to collect test probabilities out-of-fold
    oof_proba = np.zeros(features.shape[0])
    
    # Track statistics for each fold
    fold_metrics = []

    for fold_idx in range(5):
        logger.info(f"\n==================== FOLD {fold_idx} ====================")
        test_pids = folds[fold_idx]
        val_pids = folds[(fold_idx + 1) % 5]
        
        # Remaining 3 folds are train
        train_pids = []
        for j in range(5):
            if j != fold_idx and j != (fold_idx + 1) % 5:
                train_pids.extend(folds[j])
                
        # Create masks relative to full features dataset
        train_mask = np.isin(patient_ids, train_pids) & cohort_mask
        valid_mask = np.isin(patient_ids, val_pids) & cohort_mask
        test_mask = np.isin(patient_ids, test_pids) & cohort_mask
        
        logger.info(f"Samples - Train: {sum(train_mask)}, Val: {sum(valid_mask)}, Test: {sum(test_mask)}")
        
        X_train = features[train_mask, :]
        X_valid = features[valid_mask, :]
        X_test = features[test_mask, :]
        
        y_train = label_values[train_mask]
        y_valid = label_values[valid_mask]
        y_test = label_values[test_mask]
        
        model_path = os.path.join(args.path_to_output_dir, f"model_fold_{fold_idx}.pkl")
        
        if not args.test_GBM:
            # Hyperparameter tuning on Train / Val split
            model = tune_hyperparams(
                X_train,
                y_train,
                X_valid,
                y_valid,
                ltb.LGBMClassifier(),
                XGB_PARAMS,
                num_threads=args.num_threads,
            )
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        else:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
                
        # Calculate predicted probabilities on all sets
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_valid_proba = model.predict_proba(X_valid)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        # Store in OOF array
        oof_proba[test_mask] = y_test_proba
        
        # Calculate AUROCs
        fold_train_auroc = metrics.roc_auc_score(y_train, y_train_proba)
        fold_val_auroc = metrics.roc_auc_score(y_valid, y_valid_proba)
        fold_test_auroc = metrics.roc_auc_score(y_test, y_test_proba)
        
        # Sensitivity/Specificity at default 0.5 threshold
        train_sens_05, train_spec_05 = compute_sens_spec(y_train, y_train_proba, threshold=0.5)
        val_sens_05, val_spec_05 = compute_sens_spec(y_valid, y_valid_proba, threshold=0.5)
        test_sens_05, test_spec_05 = compute_sens_spec(y_test, y_test_proba, threshold=0.5)
        
        # Sensitivity/Specificity at validation-optimized threshold (Youden's J)
        fpr, tpr, thresholds = metrics.roc_curve(y_valid, y_valid_proba)
        best_idx = np.argmax(tpr - fpr)
        best_threshold = thresholds[best_idx]
        best_threshold = max(0.0, min(1.0, float(best_threshold)))
        
        train_sens_opt, train_spec_opt = compute_sens_spec(y_train, y_train_proba, threshold=best_threshold)
        val_sens_opt, val_spec_opt = compute_sens_spec(y_valid, y_valid_proba, threshold=best_threshold)
        test_sens_opt, test_spec_opt = compute_sens_spec(y_test, y_test_proba, threshold=best_threshold)
        
        logger.info(f"Fold {fold_idx} - Train AUROC: {fold_train_auroc:.4f} | Val AUROC: {fold_val_auroc:.4f} | Test AUROC: {fold_test_auroc:.4f}")
        logger.info(f"Fold {fold_idx} - Test Sensitivity (threshold=0.5): {test_sens_05:.4f} | Specificity (threshold=0.5): {test_spec_05:.4f}")
        logger.info(f"Fold {fold_idx} - Optimized Threshold (validation Youden's J): {best_threshold:.4f}")
        logger.info(f"Fold {fold_idx} - Test Sensitivity (optimized): {test_sens_opt:.4f} | Specificity (optimized): {test_spec_opt:.4f}")
        
        fold_metrics.append({
            'fold': fold_idx,
            'train_auroc': fold_train_auroc,
            'val_auroc': fold_val_auroc,
            'test_auroc': fold_test_auroc,
            'test_sens_05': test_sens_05,
            'test_spec_05': test_spec_05,
            'best_threshold': best_threshold,
            'test_sens_opt': test_sens_opt,
            'test_spec_opt': test_spec_opt
        })

    # Save all predictions (matches default 3_train_gbm.py output signature)
    if not args.test_GBM:
        with open(os.path.join(args.path_to_output_dir, "predictions.pkl"), "wb") as f:
            # We dump the full oof_proba matching original schema
            pickle.dump([oof_proba, patient_ids, label_values, label_times], f)

    # Compute overall metrics pooled across all cohort samples
    y_cohort_test = label_values[cohort_mask]
    y_cohort_proba = oof_proba[cohort_mask]
    overall_oof_auroc = metrics.roc_auc_score(y_cohort_test, y_cohort_proba)

    # Aggregated metrics (mean ± std) across folds
    test_aurocs = [m['test_auroc'] for m in fold_metrics]
    test_sens_05s = [m['test_sens_05'] for m in fold_metrics]
    test_spec_05s = [m['test_spec_05'] for m in fold_metrics]
    test_sens_opts = [m['test_sens_opt'] for m in fold_metrics]
    test_spec_opts = [m['test_spec_opt'] for m in fold_metrics]
    
    logger.info("\n" + "="*50)
    logger.info("=== 5-FOLD CROSS VALIDATION SUMMARY ===")
    logger.info("="*50)
    logger.info(f"Overall Pooled OOF AUROC: {overall_oof_auroc:.4f}")
    logger.info(f"Average Test AUROC: {np.mean(test_aurocs):.4f} ± {np.std(test_aurocs):.4f}")
    logger.info(f"Average Test Sensitivity (threshold=0.5): {np.mean(test_sens_05s):.4f} ± {np.std(test_sens_05s):.4f}")
    logger.info(f"Average Test Specificity (threshold=0.5): {np.mean(test_spec_05s):.4f} ± {np.std(test_spec_05s):.4f}")
    logger.info(f"Average Test Sensitivity (optimized): {np.mean(test_sens_opts):.4f} ± {np.std(test_sens_opts):.4f}")
    logger.info(f"Average Test Specificity (optimized): {np.mean(test_spec_opts):.4f} ± {np.std(test_spec_opts):.4f}")
    logger.success("DONE!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate CLMBR patient representations using 5-Fold Cross-Validation"
    )
    parser.add_argument(
        "--path_to_cohort", required=True, type=str, help="Path to cohort master file"
    )
    parser.add_argument(
        "--path_to_database", required=True, type=str, help="Path to femr database"
    )
    parser.add_argument(
        "--path_to_label_features",
        required=True,
        type=str,
        help="Path to save labeles and featurizers",
    )
    parser.add_argument(
        "--test_GBM",
        default=False,
        action="store_true",
        help="if you want to run GBM for test split only (evaluates pre-saved fold models)",
    )
    parser.add_argument(
        "--path_to_output_dir",
        required=True,
        type=str,
        help="Path to save labeles and featurizers",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        help="The number of threads to use",
        default=1,
    )

    args = parser.parse_args()
    main(args)
