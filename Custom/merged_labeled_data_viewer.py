"""
To run this data viewer, execute the following command in your terminal:

~/Documents/Internship_INSPECT/.venv_legacy/bin/python -m streamlit run ~/Documents/Internship_INSPECT/INSPECT_custom_data_preprocessing/Custom/merged_labeled_data_viewer.py
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
features_dir = "/home/steven/Documents/Internship_INSPECT/DATA_RAW/EHR_FEMR_DB/features/PE"

# Find all CSV files in relevant directories
csv_files = glob.glob(os.path.join(processed_dir, "*.csv")) + glob.glob(os.path.join(features_dir, "*.csv"))

if not csv_files:
    st.error("No CSV files found in the defined directories.")
    st.stop()

# Select a dataset
selected_file = st.sidebar.selectbox("Choose a dataset to view:", sorted(csv_files), format_func=lambda x: os.path.basename(x))

# Cache data loading so the app is fast when interacting with widgets
@st.cache_data
def load_data(filepath):
    return pd.read_csv(filepath)

st.write(f"### Viewing: `{os.path.basename(selected_file)}`")
st.caption(f"Path: `{selected_file}`")

try:
    df = load_data(selected_file)
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# Sidebar: Data Summary
st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Summary")
st.sidebar.write(f"**Total Rows:** {df.shape[0]:,}")
st.sidebar.write(f"**Total Columns:** {df.shape[1]:,}")

# Attempt to identify the patient ID column for filtering/stats
patient_col = None
if 'PatientID' in df.columns:
    patient_col = 'PatientID'
elif 'patient_id' in df.columns:
    patient_col = 'patient_id'

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
