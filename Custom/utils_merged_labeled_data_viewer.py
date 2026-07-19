"""
To run this data viewer, execute the following command in your terminal:

../.venv_legacy/bin/python -m streamlit run ../INSPECT_custom_data_preprocessing/Custom/utils_merged_labeled_data_viewer.py
"""

import streamlit as st
import pandas as pd
import os
import glob

# Set page config for a wider layout
st.set_page_config(page_title="INSPECT Data Viewer", layout="wide")

st.title("INSPECT Data Viewer")
st.sidebar.header("Dataset Selection")

# Define data directories
processed_dir = "/home/steven/Documents/Internship_INSPECT/DATA_PROCESSED"
features_base_dir = "/home/steven/Documents/Internship_INSPECT/DATA_RAW/EHR_FEMR_DB/features"

# Find all CSV files in relevant directories
csv_files = glob.glob(os.path.join(processed_dir, "*.csv")) + glob.glob(os.path.join(features_base_dir, "**/*.csv"), recursive=True)
# Find all PKL files in features directory
pkl_files = glob.glob(os.path.join(features_base_dir, "**/*.pkl"), recursive=True)

all_files = sorted(csv_files + pkl_files)

if not all_files:
    st.error("No CSV or PKL files found in the defined directories.")
    st.stop()

# Helper function to format dropdown options to display parent directory context
def format_file_label(filepath):
    parts = filepath.split(os.sep)
    if len(parts) >= 3:
        return os.path.join(*parts[-3:])
    return os.path.basename(filepath)

# Select a dataset
selected_file = st.sidebar.selectbox("Choose a dataset to view:", all_files, format_func=format_file_label)

# Cache data loading so the app is fast when interacting with widgets
@st.cache_data
def load_csv_data(filepath):
    return pd.read_csv(filepath)

@st.cache_resource
def load_pkl_data(filepath):
    import pickle
    import scipy.sparse
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    
    # Check if this matches the featurized_patients.pkl format (features, patient_ids, label_values, label_times)
    if isinstance(data, tuple) and len(data) == 4:
        features, patient_ids, label_values, label_times = data
        df = pd.DataFrame({
            "PatientID": patient_ids,
            "label": label_values,
            "label_time": label_times,
        })
        
        feature_info = {
            "type": type(features).__name__,
            "shape": features.shape,
            "is_sparse": scipy.sparse.issparse(features),
        }
        if feature_info["is_sparse"]:
            feature_info["nnz"] = features.nnz
            feature_info["sparsity"] = features.nnz / (features.shape[0] * features.shape[1]) * 100
        
        # Load featurizer if it exists in the same folder to resolve feature column names
        featurizer_path = os.path.join(os.path.dirname(filepath), "preprocessed_featurizers.pkl")
        featurizer = None
        if os.path.exists(featurizer_path):
            try:
                with open(featurizer_path, "rb") as f2:
                    featurizer = pickle.load(f2)
            except Exception as e:
                pass
            
        return df, feature_info, features, featurizer
    else:
        # Generic pickle viewer
        if isinstance(data, pd.DataFrame):
            return data, None, None, None
        elif isinstance(data, pd.Series):
            return pd.DataFrame(data), None, None, None
        else:
            df = pd.DataFrame([{"Data": str(data)}])
            return df, {"type": type(data).__name__, "raw_data": data}, None, None

st.write(f"### Viewing: `{os.path.basename(selected_file)}`")
st.caption(f"Path: `{selected_file}`")

feature_info = None
features = None
featurizer = None
try:
    if selected_file.endswith(".pkl"):
        df, feature_info, features, featurizer = load_pkl_data(selected_file)
    else:
        df = load_csv_data(selected_file)
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# If feature matrix properties exist, render them at the top
if feature_info:
    st.subheader("🧬 Feature Matrix Properties")
    if "shape" in feature_info:
        col1, col2, col3 = st.columns(3)
        col1.metric("Matrix Type", feature_info["type"])
        col2.metric("Matrix Shape", f"{feature_info['shape'][0]} × {feature_info['shape'][1]}")
        if feature_info.get("is_sparse") and "sparsity" in feature_info:
            col3.metric("Sparsity (Non-zero %)", f"{feature_info['sparsity']:.4f}%")
        else:
            col3.metric("Density", "100.00%")
    else:
        st.metric("Object Type", feature_info["type"])
        
    if "raw_data" in feature_info:
        st.subheader("Raw Pickle Content")
        st.write(feature_info["raw_data"])

# Sidebar: Data Summary
st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Summary")
st.sidebar.write(f"**Total Rows:** {df.shape[0]:,}")
st.sidebar.write(f"**Total Columns:** {df.shape[1]:,}")

# Attempt to identify the patient ID column for filtering/stats (case-insensitive)
patient_col = None
for col in df.columns:
    if col.lower() in ('patientid', 'patient_id'):
        patient_col = col
        break

# Sidebar Debug Info
with st.sidebar.expander("🔍 Debug Info"):
    st.write("**File:**", os.path.basename(selected_file))
    st.write("**Columns:**", list(df.columns))
    st.write("**Identified ID Col:**", str(patient_col))
    st.write("**Has Feature Matrix?**", features is not None)

if patient_col:
    st.sidebar.write(f"**Unique Patients:** {df[patient_col].nunique():,}")

    # Main area: Filtering
    st.subheader("Filter Data")
    search_id = st.text_input(f"🔍 Search by {patient_col} (leave blank for all records):")
    if search_id:
        try:
            # Convert to numeric if possible to match types, else keep as string
            numeric_id = int(search_id)
            filtered_df = df[df[patient_col] == numeric_id]
        except ValueError:
            filtered_df = df[df[patient_col].astype(str) == search_id]
        
        st.write(f"Found **{len(filtered_df)}** records for Patient **{search_id}**")
        st.dataframe(filtered_df, use_container_width=True)
        
        # If we have feature data (X), show the non-zero features for this patient
        if features is not None and len(filtered_df) > 0:
            st.markdown("---")
            st.subheader("🔬 Patient Feature Vector (X)")
            
            # Allow selecting which patient to view if multiple records match
            if len(filtered_df) > 1:
                selected_pid = st.selectbox("Select Patient ID to inspect features:", filtered_df[patient_col].unique())
            else:
                selected_pid = filtered_df[patient_col].iloc[0]
                
            try:
                # Find the row index corresponding to this patient in the df
                matching_indices = df[df[patient_col] == selected_pid].index.tolist()
                if matching_indices:
                    idx = matching_indices[0]
                    row_features = features[idx, :]
                    
                    import numpy as np
                    import scipy.sparse
                    if scipy.sparse.issparse(row_features):
                        coo = row_features.tocoo()
                        cols = coo.col
                        vals = coo.data
                    else:
                        cols = np.nonzero(row_features)[0]
                        vals = row_features[cols]
                        
                    feature_names = []
                    for col in cols:
                        if featurizer:
                            try:
                                feature_names.append(featurizer.get_column_name(col))
                            except Exception:
                                feature_names.append(f"Column {col}")
                        else:
                            feature_names.append(f"Column {col}")
                            
                    patient_feat_df = pd.DataFrame({
                        "Feature Index": cols,
                        "Feature Description": feature_names,
                        "Value": vals
                    }).sort_values(by="Value", ascending=False).reset_index(drop=True)
                    
                    st.write(f"Showing non-zero features (X) for Patient **{selected_pid}**:")
                    st.dataframe(patient_feat_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error extracting features for patient: {e}")
    else:
        st.dataframe(df, use_container_width=True)
else:
    st.dataframe(df, use_container_width=True)

# Data Quality & Distribution Section
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    # Plot label distribution if a label column exists
    label_col = None
    if 'value' in df.columns:
        label_col = 'value'
    elif 'label' in df.columns:
        label_col = 'label'
        
    if label_col:
        st.subheader("📊 Label Distribution")
        val_counts = df[label_col].value_counts().reset_index()
        val_counts.columns = [label_col, 'Count']
        st.bar_chart(val_counts.set_index(label_col))
    else:
        st.info("No label column ('value' or 'label') detected for class distribution.")

with col2:
    st.subheader("⚠️ Missing Values")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        missing_df = pd.DataFrame({"Missing Count": missing})
        missing_df["% Missing"] = (missing_df["Missing Count"] / len(df)) * 100
        st.dataframe(missing_df.style.format({"% Missing": "{:.2f}%"}))
    else:
        st.success("No missing values (NaNs) found in this dataset!")
