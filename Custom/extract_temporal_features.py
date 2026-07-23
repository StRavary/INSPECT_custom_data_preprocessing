"""
extract_temporal_features.py
----------------------------
A standalone temporal feature extraction script for INSPECT / FEMR.
Ingests the femr PatientDatabase (DATA_RAW/EHR_FEMR_DB/extract) and cohort file,
filters clinical events relative to CTPA procedure time (T0) within a specified window
(e.g., -50 days to 0 days), purges noisy historical data outside the window, and
retains exact temporal metadata (days_since_procedure).

Outputs:
1. temporal_events.parquet / .csv: Flat long table of events with days_since_procedure.
2. temporal_sequences.pkl: Patient-indexed dict mapping patient_id -> [(code, days_since_procedure, value), ...].
3. temporal_binned_features.pkl: Time-binned feature matrix + metadata for tabular models (GBM).

Usage Examples:
  # Extract events between -50 days and 0 days for all domains in parquet & sequence format:
  python Custom/extract_temporal_features.py \
      --path_to_cohort ../DATA_PROCESSED/cohort_0.2.0_master_file_anon.csv \
      --path_to_database ../DATA_RAW/EHR_FEMR_DB/extract \
      --min_days -50 \
      --max_days 0 \
      --output_format all
"""

import argparse
import datetime
import math
import os
import pickle
import sys
import collections
import numpy as np
import pandas as pd
from loguru import logger

# Add parent directories so relative imports work
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ehr"))

try:
    import femr
    import femr.datasets
    from femr.datasets import PatientDatabase
except ImportError:
    logger.error("femr package not found. Ensure you are running in the legacy venv (.venv_legacy).")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract temporal features from FEMR database relative to CTPA procedure time."
    )
    parser.add_argument(
        "--path_to_cohort",
        type=str,
        default="../DATA_PROCESSED/cohort_0.2.0_master_file_anon.csv",
        help="Path to cohort master CSV file",
    )
    parser.add_argument(
        "--path_to_database",
        type=str,
        default="../DATA_RAW/EHR_FEMR_DB/extract",
        help="Path to femr extract directory",
    )
    parser.add_argument(
        "--path_to_output_dir",
        type=str,
        default="../DATA_PROCESSED/features_temporal_50d",
        help="Output directory for generated temporal features",
    )
    parser.add_argument(
        "--min_days",
        type=float,
        default=-50.0,
        help="Minimum relative offset in days from procedure_time (e.g. -50.0). Set to -inf for no lower bound.",
    )
    parser.add_argument(
        "--max_days",
        type=float,
        default=0.0,
        help="Maximum relative offset in days from procedure_time (e.g. 0.0, up to procedure time).",
    )
    parser.add_argument(
        "--include_domains",
        type=str,
        default=None,
        help="Comma-separated OMOP domains/concept prefixes to keep (e.g. 'LOINC,CPT4,SNOMED,RxNorm'). Default: all",
    )
    parser.add_argument(
        "--domain_windows",
        type=str,
        default=None,
        help="Custom min/max days per domain format 'DOMAIN:min:max,DOMAIN2:min:max' (e.g., 'LOINC:-2:0,CPT4:-365:0,SNOMED:-1825:0').",
    )
    parser.add_argument(
        "--domain_windows_json",
        type=str,
        default=None,
        help="Path to JSON file specifying custom [min_days, max_days] per domain or code (e.g., {'LOINC': [-2, 0], 'LOINC/718-7': [-7, 0]}).",
    )
    parser.add_argument(
        "--time_bins",
        type=str,
        default="2,7,30,50",
        help="Comma-separated day limits for binned matrix (e.g. '2,7,30,50' creates bins [0, -2], [-2, -7], [-7, -30], [-30, -50]).",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["parquet", "csv", "pkl_sequence", "pkl_matrix", "all"],
        default="all",
        help="Which output file formats to generate.",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=None,
        help="Optional limit on number of cohort rows to process (useful for fast testing).",
    )
    return parser.parse_args()


def get_domain_from_code(code: str) -> str:
    """Extract domain prefix from code (e.g. LOINC/718-7 -> LOINC)."""
    if "/" in code:
        return code.split("/")[0]
    return "Other"


def get_event_window(code: str, domain: str, custom_windows: dict, default_min: float, default_max: float):
    """Retrieve custom (min_days, max_days) window for code or domain, falling back to global defaults."""
    code_upper = code.upper()
    domain_upper = domain.upper()
    
    if code_upper in custom_windows:
        return custom_windows[code_upper]
    if domain_upper in custom_windows:
        return custom_windows[domain_upper]
    return (default_min, default_max)


def main():
    args = parse_args()

    # Resolve absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.abspath(os.path.join(script_dir, ".."))
    project_root = os.path.abspath(os.path.join(workspace_dir, ".."))

    # Helper to check potential candidates for input files
    def resolve_existing_path(path_arg, candidates):
        if os.path.isabs(path_arg) and os.path.exists(path_arg):
            return path_arg
        for cand in candidates:
            if os.path.exists(cand):
                return cand
        return os.path.abspath(os.path.join(script_dir, path_arg))

    cohort_candidates = [
        os.path.abspath(os.path.join(workspace_dir, args.path_to_cohort)),
        os.path.abspath(os.path.join(project_root, "DATA_PROCESSED/cohort_0.2.0_master_file_anon.csv")),
        os.path.abspath(os.path.join(workspace_dir, "DATA_PROCESSED/cohort_0.2.0_master_file_anon.csv")),
        os.path.abspath(os.path.join(project_root, "cohort_0.2.0_master_file_anon.csv")),
    ]
    cohort_path = resolve_existing_path(args.path_to_cohort, cohort_candidates)

    db_candidates = [
        os.path.abspath(os.path.join(workspace_dir, args.path_to_database)),
        os.path.abspath(os.path.join(project_root, "DATA_RAW/EHR_FEMR_DB/extract")),
        os.path.abspath(os.path.join(workspace_dir, "DATA_RAW/EHR_FEMR_DB/extract")),
    ]
    db_path = resolve_existing_path(args.path_to_database, db_candidates)

    output_dir = os.path.abspath(os.path.join(workspace_dir, args.path_to_output_dir))
    os.makedirs(output_dir, exist_ok=True)

    # Setup logger
    log_file = os.path.join(output_dir, "extraction.log")
    if os.path.exists(log_file):
        os.remove(log_file)
    logger.add(log_file, level="INFO")

    logger.info("=== INSPECT Temporal Feature Extraction ===")
    logger.info(f"Cohort File: {cohort_path}")
    logger.info(f"FEMR Database: {db_path}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Time Window: [{args.min_days}, {args.max_days}] days relative to CTPA T0")

    allowed_domains = None
    if args.include_domains:
        allowed_domains = set([d.strip().upper() for d in args.include_domains.split(",") if d.strip()])
        logger.info(f"Filtered Domains: {allowed_domains}")

    # Parse custom domain/code windows
    custom_windows = {}
    if args.domain_windows:
        for item in args.domain_windows.split(","):
            if not item.strip():
                continue
            parts = item.strip().split(":")
            if len(parts) == 3:
                k = parts[0].strip().upper()
                w_min = float(parts[1])
                w_max = float(parts[2])
                custom_windows[k] = (w_min, w_max)

    if args.domain_windows_json:
        json_path = resolve_existing_path(args.domain_windows_json, [
            os.path.abspath(os.path.join(workspace_dir, args.domain_windows_json)),
            os.path.abspath(os.path.join(script_dir, args.domain_windows_json))
        ])
        if os.path.exists(json_path):
            import json
            with open(json_path, "r") as f:
                json_data = json.load(f)
                for k, v in json_data.items():
                    if isinstance(v, (list, tuple)) and len(v) == 2:
                        custom_windows[k.upper()] = (float(v[0]), float(v[1]))

    if custom_windows:
        logger.info(f"Custom Per-Feature Windows: {custom_windows}")

    # Load cohort data
    if not os.path.exists(cohort_path):
        # Fallback check for root cohort
        alt_cohort_path = os.path.abspath(os.path.join(script_dir, "../cohort_0.2.0_master_file_anon.csv"))
        if os.path.exists(alt_cohort_path):
            cohort_path = alt_cohort_path

    df_cohort = pd.read_csv(cohort_path)
    if args.sample_limit:
        df_cohort = df_cohort.head(args.sample_limit)
        logger.info(f"Testing with sample limit: {args.sample_limit} rows")

    logger.info(f"Loaded {len(df_cohort)} cohort records.")

    # Load FEMR PatientDatabase
    logger.info("Loading FEMR PatientDatabase...")
    db = PatientDatabase(db_path)

    # Process events
    flat_records = []
    patient_sequences = collections.defaultdict(list)
    
    # Vocabulary & Binned matrix builders
    bin_limits = [float(b.strip()) for b in args.time_bins.split(",") if b.strip()]
    bin_limits.sort()  # e.g. [2.0, 7.0, 30.0, 50.0]
    
    missing_db_patients = 0
    total_events_extracted = 0

    for idx, row in df_cohort.iterrows():
        patient_id = int(row["patient_id"])
        impression_id = row.get("impression_id", f"{patient_id}_{idx}")

        # Parse procedure anchor timestamp T0
        proc_time_raw = row.get("procedure_DATETIME")
        if pd.isna(proc_time_raw) or str(proc_time_raw).strip() in ["", "nan", "None"]:
            proc_time_raw = row.get("StudyTime")

        if pd.isna(proc_time_raw):
            continue

        try:
            proc_time = datetime.datetime.fromisoformat(str(proc_time_raw).replace(" ", "T"))
        except Exception:
            continue

        try:
            patient = db[patient_id]
        except (KeyError, IndexError):
            missing_db_patients += 1
            continue

        for event in patient.events:
            # Calculate days since procedure (fractional float days)
            delta_days = (event.start - proc_time).total_seconds() / 86400.0

            domain = get_domain_from_code(event.code)
            if allowed_domains and domain.upper() not in allowed_domains:
                continue

            # Retrieve feature window (custom per domain/code or global default)
            win_min, win_max = get_event_window(event.code, domain, custom_windows, args.min_days, args.max_days)

            # Apply relative time window filtering
            if delta_days < win_min or delta_days > win_max:
                continue

            delta_days_rounded = round(delta_days, 4)

            # Store flat record
            if args.output_format in ["parquet", "csv", "all"]:
                flat_records.append(
                    {
                        "patient_id": patient_id,
                        "impression_id": impression_id,
                        "procedure_time": proc_time,
                        "event_time": event.start,
                        "days_since_procedure": delta_days_rounded,
                        "domain": domain,
                        "code": event.code,
                        "value": str(event.value) if event.value is not None else None,
                    }
                )

            # Store sequence representation
            if args.output_format in ["pkl_sequence", "all"]:
                patient_sequences[patient_id].append(
                    (event.code, delta_days_rounded, event.value)
                )

            total_events_extracted += 1

    logger.info(f"Extraction complete.")
    logger.info(f"  Missing patients in DB: {missing_db_patients}")
    logger.info(f"  Total events extracted within window [{args.min_days}, {args.max_days}]: {total_events_extracted}")

    # Export Flat Tabular Dataset (Parquet / CSV)
    if args.output_format in ["parquet", "csv", "all"] and flat_records:
        df_events = pd.DataFrame(flat_records)
        
        if args.output_format in ["parquet", "all"]:
            parquet_path = os.path.join(output_dir, "temporal_events.parquet")
            df_events.to_parquet(parquet_path, index=False)
            logger.info(f"Saved: {parquet_path} ({len(df_events):,} rows)")

        if args.output_format in ["csv", "all"]:
            csv_path = os.path.join(output_dir, "temporal_events.csv")
            df_events.to_csv(csv_path, index=False)
            logger.info(f"Saved: {csv_path} ({len(df_events):,} rows)")

    # Export Sequence Dictionary
    if args.output_format in ["pkl_sequence", "all"]:
        seq_path = os.path.join(output_dir, "temporal_sequences.pkl")
        with open(seq_path, "wb") as f:
            pickle.dump(dict(patient_sequences), f)
        logger.info(f"Saved: {seq_path} ({len(patient_sequences):,} patients)")

    # Export Binned Matrix (if requested)
    if args.output_format in ["pkl_matrix", "all"] and flat_records:
        logger.info("Building time-binned feature matrix...")
        # Create bins: e.g. [0, -2], [-2, -7], [-7, -30], [-30, -50]
        bins = [(0.0, -bin_limits[0])]
        for i in range(len(bin_limits) - 1):
            bins.append((-bin_limits[i], -bin_limits[i+1]))
        
        # Build feature count map
        feature_counts = collections.defaultdict(lambda: collections.defaultdict(int))
        for rec in flat_records:
            pid = rec["patient_id"]
            d = rec["days_since_procedure"]
            code = rec["code"]

            for bin_idx, (b_high, b_low) in enumerate(bins):
                if b_low <= d <= b_high:
                    feat_name = f"{code}_bin{bin_idx}_[{b_low}d,{b_high}d]"
                    feature_counts[pid][feat_name] += 1
                    break

        matrix_path = os.path.join(output_dir, "temporal_binned_features.pkl")
        with open(matrix_path, "wb") as f:
            pickle.dump(dict(feature_counts), f)
        logger.info(f"Saved: {matrix_path} ({len(feature_counts):,} patients featurized)")

    logger.info("Done successfully!")


if __name__ == "__main__":
    main()
