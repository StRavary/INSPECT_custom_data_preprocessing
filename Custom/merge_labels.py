import pandas as pd
import os

# 1. Paths
labels_path = os.path.expanduser("~/Documents/Internship_INSPECT/DATA_RAW/LABELS/labels_20250611.tsv")
mapping_path = os.path.expanduser("~/Documents/Internship_INSPECT/DATA_RAW/LABELS/study_mapping_20250611.tsv")
output_path = os.path.expanduser("~/Documents/Internship_INSPECT/DATA_PROCESSED/cohort_0.2.0_master_file_anon.csv")

print("Loading true labels...")
df_labels = pd.read_csv(labels_path, sep='\t')

print("Loading official study mapping...")
df_mapping = pd.read_csv(mapping_path, sep='\t')

# Ensure impression_id is string in both to prevent merge issues
df_labels['impression_id'] = df_labels['impression_id'].astype(str)
df_mapping['impression_id'] = df_mapping['impression_id'].astype(str)

print("Merging datasets on impression_id...")
df_master = pd.merge(df_mapping, df_labels, on='impression_id', how='inner')

# Combine datetimes into a single StudyTime column
# Use procedure_DATETIME first, fallback to note_DATETIME
df_master['StudyTime'] = df_master['procedure_DATETIME'].fillna(df_master['note_DATETIME'])

# Drop any rows that STILL have missing StudyTime
missing_before = len(df_master)
df_master = df_master.dropna(subset=['StudyTime'])
missing_after = len(df_master)
print(f"Dropped {missing_before - missing_after} records due to missing timestamps.")

# Rename columns to strictly match what the Stanford benchmark pipeline expects
# 'person_id' -> 'PatientID'
df_master = df_master.rename(columns={
    'person_id': 'PatientID'
})

print(f"Master cohort generated with {len(df_master)} valid timestamped records.")

# Save to the specific filename required by the benchmark pipeline
df_master.to_csv(output_path, index=False)
print(f"Saved complete master cohort to: {output_path}")
