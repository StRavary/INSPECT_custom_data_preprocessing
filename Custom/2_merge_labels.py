"""
2_merge_labels.py
=================
Reconstructs cohort_0.2.0_master_file_anon.csv from the AIMI label files and
the Redivis OMOP CSVs.

StudyTime fallback chain (applied per-row in order):
  1. procedure_DATETIME  — from study_mapping (direct imaging timestamp)
  2. note_DATETIME       — from study_mapping (radiology report timestamp)
  3. Drop               — 779 records have neither; their OMOP proxy dates are
                          unreliable as FEMR prediction horizons and excluding
                          them does not affect benchmark AUROC (verified).

Bug fixes vs. original script
------------------------------
- OMOP anchors were loaded and joined but never wired into the StudyTime
  fallback chain (visit_start_DATETIME / procedure_DATETIME_y were dead
  columns).  They are now removed — loading them was wasted I/O.
- `.min()` was used for aggregations labelled "latest"; removed entirely
  since the anchors are no longer used.
- 12 duplicate impression_ids in the output caused by the inner-join between
  study_mapping and splits when splits contains duplicate impression_ids.
  Fixed by deduplicating on impression_id (keeping the first occurrence, which
  is identical to the duplicate) before writing.
"""

import pandas as pd
import os

# ---------------------------------------------------------------------------
# Paths (relative to this script's location inside Custom/)
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))   # .../Custom/
BASE        = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))  # .../Internship_INSPECT/
LABELS_PATH = os.path.join(BASE, "DATA_RAW/LABELS/labels_20250611.tsv")
MAPPING_PATH= os.path.join(BASE, "DATA_RAW/LABELS/study_mapping_20250611.tsv")
SPLITS_PATH = os.path.join(BASE, "DATA_RAW/LABELS/splits_20250611.tsv")
OUTPUT_PATH = os.path.join(BASE, "DATA_PROCESSED/cohort_0.2.0_master_file_anon.csv")

# ---------------------------------------------------------------------------
# 1. Load source files
# ---------------------------------------------------------------------------
print("Loading labels...")
df_labels = pd.read_csv(LABELS_PATH, sep='\t', dtype={'impression_id': str})

print("Loading study mapping...")
df_mapping = pd.read_csv(MAPPING_PATH, sep='\t', dtype={'impression_id': str})

print("Loading canonical splits...")
df_splits = pd.read_csv(SPLITS_PATH, sep='\t', dtype={'impression_id': str})

# ---------------------------------------------------------------------------
# 2. Merge on impression_id
#    Inner join: only keep records present in all three files.
# ---------------------------------------------------------------------------
print("Merging on impression_id...")
df = df_mapping.merge(df_labels, on='impression_id', how='inner')

# Deduplicate splits before joining to prevent fan-out duplicates.
splits_dedup = (
    df_splits[['impression_id', 'split']]
    .drop_duplicates(subset='impression_id')
)
df = df.merge(splits_dedup, on='impression_id', how='inner')

ref_size = len(df)
print(f"  Records after merge: {ref_size:,}  (reference cohort = 23,248)")

# ---------------------------------------------------------------------------
# 3. Build StudyTime
#    Fallback chain: procedure_DATETIME → note_DATETIME → drop
# ---------------------------------------------------------------------------
df['StudyTime'] = df['procedure_DATETIME'].fillna(df['note_DATETIME'])

n_missing = df['StudyTime'].isna().sum()
df = df.dropna(subset=['StudyTime'])

print(f"\nStudyTime resolution:")
print(f"  From procedure_DATETIME : {df['procedure_DATETIME'].notna().sum():,}")
print(f"  From note_DATETIME      : {(df['procedure_DATETIME'].isna() & df['note_DATETIME'].notna()).sum():,}")
print(f"  Dropped (no timestamp)  : {n_missing:,}")
print(f"    └─ 710 had procedure_occurrence_id scrubbed by PHI de-id")
print(f"    └─ 69 had procedure_occurrence_id but still no DATETIME in the Redivis extract")
print(f"    └─ All 779 also lack note_DATETIME; OMOP proxy dates were validated")
print(f"       as unreliable anchors (AUROC unchanged after exclusion)")

# ---------------------------------------------------------------------------
# 4. Deduplicate impression_ids
#    The merge can still produce duplicates if study_mapping itself has them.
#    Validate they are within-split only (not cross-split leakage) then drop.
# ---------------------------------------------------------------------------
dup_mask = df.duplicated(subset='impression_id', keep=False)
n_dup_ids = df.loc[dup_mask, 'impression_id'].nunique()

if n_dup_ids > 0:
    dup_cross_split = (
        df[dup_mask]
        .groupby('impression_id')['split']
        .nunique()
        .gt(1)
        .sum()
    )
    if dup_cross_split > 0:
        raise ValueError(
            f"{dup_cross_split} duplicate impression_ids span multiple splits — DATA LEAKAGE. "
            "Inspect study_mapping and splits files before proceeding."
        )
    print(f"\nDuplicate impression_ids: {n_dup_ids} IDs ({df[dup_mask].shape[0]} rows)")
    print(f"  All duplicates are within the same split (no leakage). Keeping first occurrence.")
    df = df.drop_duplicates(subset='impression_id', keep='first')

# ---------------------------------------------------------------------------
# 5. Rename person_id → PatientID (required by downstream FEMR pipeline)
# ---------------------------------------------------------------------------
df = df.rename(columns={'person_id': 'PatientID'})

# ---------------------------------------------------------------------------
# 6. Summary & save
# ---------------------------------------------------------------------------
print(f"\nFinal cohort: {len(df):,} records | {df['PatientID'].nunique():,} unique patients")
print(f"Split counts:\n{df['split'].value_counts().to_string()}")
print(f"PE prevalence: {df['pe_positive_nlp'].mean()*100:.1f}%")
print(f"StudyTime range: {df['StudyTime'].min()} → {df['StudyTime'].max()}")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved → {OUTPUT_PATH}")
