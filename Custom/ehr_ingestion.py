import os
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.compute as pc
import pandas as pd
import numpy as np

class OMOPIngestionEngine:
    """
    High-Performance EHR Ingestion Engine using PyArrow.
    Extracts structured clinical data, constructs dense feature matrices,
    and generates binary missingness masks for self-supervised imputation.
    """
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define target features based on OMOP measurement_concept_id
        # These are placeholders for common PE-related measurements (Vitals, D-dimer, Troponin, etc.)
        self.target_measurements = {
            3027018: "Heart_Rate",
            3012888: "Diastolic_BP",
            3023314: "Hematocrit",
            # Add more specific concept IDs here based on clinician input
        }

    def ingest_measurements(self):
        print("Starting PyArrow ingestion of measurement.csv...")
        measurement_path = os.path.join(self.data_dir, "measurement.csv")
        
        if not os.path.exists(measurement_path):
            raise FileNotFoundError(f"Cannot find {measurement_path}")

        # Stream the CSV to avoid OOM errors
        # We only need specific columns to build the clinical feature vectors
        convert_options = pv.ConvertOptions(
            include_columns=["person_id", "measurement_concept_id", "value_as_number", "measurement_datetime"]
        )
        
        # Read table efficiently
        table = pv.read_csv(measurement_path, convert_options=convert_options)
        
        # Filter for our target measurement concept IDs
        target_ids = list(self.target_measurements.keys())
        condition = pc.is_in(table["measurement_concept_id"], value_set=pa.array(target_ids))
        filtered_table = table.filter(condition)
        
        print(f"Extracted {filtered_table.num_rows} relevant measurement events.")
        return filtered_table

    def build_dense_matrix(self, table: pa.Table):
        """
        Pivots the long-format OMOP data into a wide dense matrix and
        generates the binary missingness mask.
        """
        print("Densifying matrix and generating missingness masks...")
        
        # Convert to pandas for complex pivoting
        df = table.to_pandas()
        
        # Drop rows where value is missing
        df = df.dropna(subset=['value_as_number'])
        
        # Map concept IDs to readable names
        df['feature_name'] = df['measurement_concept_id'].map(self.target_measurements)
        
        # For simplicity, if a patient has multiple measurements of the same type, we take the mean.
        # In a real-time scenario, we would use a specific temporal window (e.g., 24h prior to scan).
        pivoted = df.pivot_table(
            index='person_id', 
            columns='feature_name', 
            values='value_as_number', 
            aggfunc='mean'
        )
        
        # The resulting DataFrame contains NaNs where data is missing
        dense_matrix = pivoted.astype(float)
        
        # Generate the Binary Missingness Mask
        # 1 = value is present, 0 = value is missing (NaN)
        binary_mask = (~dense_matrix.isna()).astype(int)
        
        return dense_matrix, binary_mask

    def run_pipeline(self):
        table = self.ingest_measurements()
        dense_matrix, binary_mask = self.build_dense_matrix(table)
        
        print("\n--- Pipeline Results ---")
        print(f"Processed {len(dense_matrix)} unique patients.")
        print(f"Matrix shape: {dense_matrix.shape}")
        
        # Save artifacts
        matrix_path = os.path.join(self.output_dir, "dense_clinical_matrix.parquet")
        mask_path = os.path.join(self.output_dir, "binary_missingness_mask.parquet")
        
        dense_matrix.to_parquet(matrix_path)
        binary_mask.to_parquet(mask_path)
        
        print(f"Saved artifacts to {self.output_dir}")

if __name__ == "__main__":
    DATA_DIR = os.path.expanduser("~/Documents/Internship_INSPECT/DATA_RAW/EHR_CSV")
    OUT_DIR = os.path.expanduser("~/Documents/Internship_INSPECT/DATA_PROCESSED/upstream_tensors")
    
    engine = OMOPIngestionEngine(data_dir=DATA_DIR, output_dir=OUT_DIR)
    engine.run_pipeline()
