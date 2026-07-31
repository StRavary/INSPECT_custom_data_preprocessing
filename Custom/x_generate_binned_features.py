"""
generate_binned_features.py
---------------------------
A custom wrapper script that allows generating binned and category-specific
features from the patient database. It mimics the logic of ehr/2_generate_labels_and_features.py
but adds fine-grained control over the time windows (date ranges) for different feature categories.

Usage Example:
  python Custom/generate_binned_features.py \
      --task PE \
      --vitals_labs_bins 2,30 \
      --diag_proc_bins 1825
"""

import argparse
import datetime
import os
import pickle
import json
import csv
import sys
import collections
import numpy as np
from loguru import logger

# Add project root and ehr directory to sys.path so relative imports (like utils) work
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ehr"))

import femr
from femr.datasets import PatientDatabase
from femr.labelers.core import LabeledPatients
from femr.featurizers.core import FeaturizerList
from femr.featurizers.featurizers import AgeFeaturizer, CountFeaturizer
from utils import save_data

def parse_timedelta_bins(bins_str):
    """Converts a comma-separated string of days (e.g. '2,30') into list of timedeltas."""
    if not bins_str or bins_str.lower() == "none":
        return None
    try:
        days_list = [int(d.strip()) for d in bins_str.split(",") if d.strip()]
        return [datetime.timedelta(days=days) for days in days_list]
    except Exception as e:
        raise ValueError(f"Failed to parse time bins string '{bins_str}': {e}")

def main():
    parser = argparse.ArgumentParser(description="Run custom-binned femr featurization")
    parser.add_argument(
        "--path_to_cohort",
        type=str,
        default="../DATA_PROCESSED/cohort_0.2.0_master_file_anon.csv",
        help="Path to cohort dataframe"
    )
    parser.add_argument(
        "--path_to_database",
        type=str,
        default="../DATA_RAW/EHR_FEMR_DB/extract",
        help="Path to femr database"
    )
    parser.add_argument(
        "--path_to_output_dir",
        type=str,
        default=None,
        help="Path to save labels and featurizers. Defaults to DATA_RAW/EHR_FEMR_DB/features/{task}_binned"
    )
    parser.add_argument(
        "--task",
        required=True,
        type=str,
        help="Name of labeling function/task (e.g., PE, 1_month_mortality, etc.)",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        help="The number of threads to use",
        default=4,
    )
    
    # Custom temporal control parameters
    parser.add_argument(
        "--vitals_labs_bins",
        type=str,
        default=None,
        help="Comma-separated day limits for vitals/labs (measurements/observations). E.g., '2,30' (last 48h, and 48h-30d)."
    )
    parser.add_argument(
        "--diag_proc_bins",
        type=str,
        default=None,
        help="Comma-separated day limits for diagnoses/procedures (conditions/procedures/devices). E.g., '1825' (last 5 years)."
    )
    parser.add_argument(
        "--other_bins",
        type=str,
        default=None,
        help="Comma-separated day limits for all other categories (e.g., drug exposures)."
    )

    args = parser.parse_args()

    PATIENT_ID_COLUMN = "PatientID"
    TIME_COLUMN = "StudyTime"

    if args.path_to_output_dir is None:
        args.path_to_output_dir = f"../DATA_RAW/EHR_FEMR_DB/features/{args.task}_binned"

    # Convert paths to absolute to be robust
    args.path_to_cohort = os.path.abspath(args.path_to_cohort)
    args.path_to_database = os.path.abspath(args.path_to_database)
    args.path_to_output_dir = os.path.abspath(args.path_to_output_dir)

    os.makedirs(args.path_to_output_dir, exist_ok=True)

    # Logging setup
    path_to_log_file = os.path.join(args.path_to_output_dir, "info.log")
    if os.path.exists(path_to_log_file):
        os.remove(path_to_log_file)
    logger.add(path_to_log_file, level="INFO")
    
    logger.info(f"Task: {args.task}")
    logger.info(f"Loading patient database from: {args.path_to_database}")
    logger.info(f"Saving output to: {args.path_to_output_dir}")
    
    # Save the arguments used for reproducibility
    with open(os.path.join(args.path_to_output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Load cohort file
    with open(args.path_to_cohort) as f:
        if args.path_to_cohort.endswith(".csv"):
            reader = csv.DictReader(f)
        elif args.path_to_cohort.endswith(".tsv"):
            reader = csv.DictReader(f, delimiter="\t")
        cohort_lines = list(reader)
        for row in cohort_lines:
            row[TIME_COLUMN] = datetime.datetime.fromisoformat(row[TIME_COLUMN])
            row[PATIENT_ID_COLUMN] = int(row[PATIENT_ID_COLUMN])

    database = PatientDatabase(args.path_to_database)
    
    # Reconstruct Labels (same fallback logic as in ehr/2_generate_labels_and_features.py)
    auxiliary_tasks = [
        "1_month_mortality", "6_month_mortality", "12_month_mortality",
        "1_month_readmission", "6_month_readmission", "12_month_readmission",
        "12_month_PH"
    ]
    
    labels = collections.defaultdict(list)
    for row in cohort_lines:
        try:
            # Skip patients not present in database
            _ = database[row[PATIENT_ID_COLUMN]]
        except IndexError:
            continue

        if args.task in auxiliary_tasks:
            label = str(row.get(args.task, "")).strip().upper()
            if label in ["CENSORED", "NAN", ""]:
                continue
            labels[row[PATIENT_ID_COLUMN]].append(
                femr.labelers.Label(
                    time=row[TIME_COLUMN],
                    value=label == "TRUE"
                )
            )
        elif args.task == "PE":
            labels[row[PATIENT_ID_COLUMN]].append(
                femr.labelers.Label(
                    time=row[TIME_COLUMN] - datetime.timedelta(days=1),
                    value=row["pe_positive_nlp"] == "True"
                )
            )
        else:
            raise ValueError(f"Task '{args.task}' not supported.")

    for _, v in labels.items():
        v.sort(key=lambda a: a.time)

    labeled_patients = LabeledPatients(labels, "boolean")
    labeled_patients.save(os.path.join(args.path_to_output_dir, "labeled_patients.csv"))
    logger.info(f"Total labeled patients: {labeled_patients.get_num_patients()}")

    # Setup the custom-binned Featurizers
    logger.info("Initializing custom binned featurizers...")
    featurizers = [AgeFeaturizer()]

    # Parse timedelta bins
    vitals_labs_bins = parse_timedelta_bins(args.vitals_labs_bins)
    diag_proc_bins = parse_timedelta_bins(args.diag_proc_bins)
    other_bins = parse_timedelta_bins(args.other_bins)

    # 1. Vitals & Labs featurizer (measurements/observations)
    if vitals_labs_bins:
        logger.info(f"Vitals & Labs bins (days): {args.vitals_labs_bins}")
        vitals_labs_tables = {"measurement", "observation"}
        featurizers.append(
            CountFeaturizer(
                is_ontology_expansion=True,
                time_bins=vitals_labs_bins,
                excluded_event_filter=lambda ev: ev.omop_table not in vitals_labs_tables
            )
        )
    
    # 2. Diagnoses & Procedures featurizer (conditions/procedures/devices)
    if diag_proc_bins:
        logger.info(f"Diagnoses & Procedures bins (days): {args.diag_proc_bins}")
        diag_proc_tables = {"condition_occurrence", "procedure_occurrence", "device_exposure"}
        featurizers.append(
            CountFeaturizer(
                is_ontology_expansion=True,
                time_bins=diag_proc_bins,
                excluded_event_filter=lambda ev: ev.omop_table not in diag_proc_tables
            )
        )

    # 3. Other clinical events (e.g. drugs/visits)
    if other_bins:
        logger.info(f"Other events bins (days): {args.other_bins}")
        matched_tables = {"measurement", "observation", "condition_occurrence", "procedure_occurrence", "device_exposure"}
        featurizers.append(
            CountFeaturizer(
                is_ontology_expansion=True,
                time_bins=other_bins,
                excluded_event_filter=lambda ev: ev.omop_table in matched_tables
            )
        )

    # Fallback to standard CountFeaturizer if no custom bins were requested
    if not vitals_labs_bins and not diag_proc_bins and not other_bins:
        logger.info("No custom bins specified. Falling back to the default CountFeaturizer (entire history).")
        featurizers.append(CountFeaturizer(is_ontology_expansion=True))

    featurizer_list = FeaturizerList(featurizers)

    # Preprocessing
    logger.info("Starting preprocessing of featurizers...")
    featurizer_list.preprocess_featurizers(args.path_to_database, labeled_patients, args.num_threads)
    save_data(featurizer_list, os.path.join(args.path_to_output_dir, "preprocessed_featurizers.pkl"))
    logger.info("Finished preprocessing.")

    # Featurization
    logger.info("Starting featurization...")
    results = featurizer_list.featurize(args.path_to_database, labeled_patients, args.num_threads)
    save_data(results, os.path.join(args.path_to_output_dir, "featurized_patients.pkl"))
    logger.info("Finished featurization.")

    feature_matrix, patient_ids, label_values, label_times = results
    label_set, counts_per_label = np.unique(label_values, return_counts=True)
    
    logger.info(
        "Stats:\n"
        f"  feature_matrix={repr(feature_matrix)}\n"
        f"  patient_ids count={len(patient_ids)}\n"
        f"  label_set={repr(label_set)}\n"
        f"  counts_per_label={repr(counts_per_label)}"
    )
    logger.info("All done successfully!")

if __name__ == "__main__":
    main()
