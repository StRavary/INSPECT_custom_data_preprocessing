import pandas as pd
import pickle
import numpy as np

features_dir = "/home/steven/Documents/Internship_INSPECT/DATA_RAW/EHR_FEMR_DB/features/PE"

# 1. Check CSV
print("Loading CSV...")
df = pd.read_csv(f"{features_dir}/labeled_patients.csv")
print(f"Any NaNs in CSV? {df.isna().any().any()}")
print(f"Duplicate prediction times for a patient? {df.duplicated(subset=['patient_id', 'prediction_time']).any()}")

# 2. Check Pickle
print("\nLoading feature tuples...")
with open(f"{features_dir}/featurized_patients.pkl", "rb") as f:
    featurized_data = pickle.load(f)

# Extract components from the femr tuple
feature_matrix = featurized_data[0]
patient_ids = featurized_data[1]
label_values = featurized_data[2]
label_times = featurized_data[3]

print(f"Matrix shape: {feature_matrix.shape}")
print(f"Number of stored elements: {feature_matrix.nnz}")
print(f"Density: {(feature_matrix.nnz / (feature_matrix.shape[0] * feature_matrix.shape[1])) * 100:.2f}%")
print(f"Any NaNs in matrix? {np.isnan(feature_matrix.data).any()}")
print(f"Any Infs in matrix? {np.isinf(feature_matrix.data).any()}")

# 3. Check alignment between CSV and Pickle
print("\nChecking alignment...")
csv_labels = df['value'].values
pkl_labels = label_values

# Note: CSV string matching vs femr boolean array might need slight type alignment
if len(csv_labels) == len(pkl_labels):
    print("Length matches!")
else:
    print(f"Length mismatch: CSV has {len(csv_labels)}, PKL has {len(pkl_labels)}")