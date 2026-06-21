import os
import glob
import pandas as pd
from pathlib import Path

def convert_all_parquet_to_csv():
    # 1. Define paths
    input_dir = os.path.expanduser("~/Documents/Internship_INSPECT/DATA_RAW/EHR")
    output_dir = os.path.expanduser("~/Documents/Internship_INSPECT/DATA_RAW/EHR_CSV")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Find all table folders in the OMOP EHR raw data
    table_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    
    if not table_folders:
        print(f"No table folders found in {input_dir}. Are you sure the data is there?")
        return

    print(f"Found {len(table_folders)} OMOP tables. Starting conversion to CSV...")
    
    # 3. Process each table
    for table_name in table_folders:
        folder_path = os.path.join(input_dir, table_name)
        parquet_files = glob.glob(os.path.join(folder_path, "*.parquet"))
        
        if not parquet_files:
            continue
            
        print(f" -> Converting {table_name} ({len(parquet_files)} parquet parts)...")
        output_csv = os.path.join(output_dir, f"{table_name}.csv")
        
        # 4. Iteratively read each parquet part and append to the CSV 
        # (This prevents running out of memory if the combined table is too large)
        first_shard = True
        for p_file in parquet_files:
            df = pd.read_parquet(p_file)
            
            # Write to CSV (mode='w' for the first shard to overwrite/create, mode='a' to append the rest)
            df.to_csv(
                output_csv, 
                mode='a' if not first_shard else 'w', 
                header=first_shard, 
                index=False
            )
            first_shard = False
            
    print(f"\n🎉 Success! All OMOP tables are now saved as CSV files in:\n{output_dir}")
    print("\nNext step: Update your --path_to_input to point to this new EHR_CSV directory.")

if __name__ == "__main__":
    convert_all_parquet_to_csv()
