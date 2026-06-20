import os
import glob
import pandas as pd
import streamlit as st

st.set_page_config(page_title="INSPECT EHR Data Viewer", layout="wide")
st.title("📊 INSPECT EHR Dataset Explorer")

DATA_DIR = os.path.expanduser("~/Documents/Internship_INSPECT/DATA_RAW/EHR")

subfolders = sorted(glob.glob(os.path.join(DATA_DIR, "*/")))
table_options = {}

for folder in subfolders:
    parquet_files = glob.glob(os.path.join(folder, "*.parquet"))
    if parquet_files:
        table_name = os.path.basename(os.path.dirname(folder))
        table_options[table_name] = folder

if not table_options:
    st.warning(f"No table folders containing .parquet files found in {DATA_DIR}.")
else:
    
    selected_table = st.selectbox("📂 Select an EHR Table to Inspect:", list(table_options.keys()))
    folder_path = table_options[selected_table]
    
    
    @st.cache_data
    def load_data(path):
        return pd.read_parquet(path)
    
    with st.spinner(f"Loading all parts for {selected_table}..."):
        try:
            df = load_data(folder_path)
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            st.stop()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows (All Shards combined)", f"{df.shape[0]:,}")
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Combined Memory Usage", f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    
    search_term = st.text_input("🔍 Filter rows by a search keyword (optional):")
    if search_term:
        mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)
        display_df = df[mask]
    else:
        display_df = df

    st.subheader(f"Data Preview: {selected_table}")
    num_rows = st.slider("Rows to display:", min_value=5, max_value=500, value=50)
    st.dataframe(display_df.head(num_rows), width="stretch")
    
    st.subheader("📋 Missing Values & Data Types")
    
    missing_info = pd.DataFrame({
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum(),
        "Missing %": (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.table(missing_info)