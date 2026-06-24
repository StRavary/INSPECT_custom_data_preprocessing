import os
import sys
import subprocess
import argparse

def run_baseline_pipeline(task):
    print(f"Starting {task} Baseline Label & Feature Generation...")
    
    # Anchor paths to the project root
    base_dir = os.path.expanduser("~/Documents/Internship_INSPECT/INSPECT_custom_data_preprocessing")
    
    # Explicitly use the python executable from inside .venv_legacy
    venv_python = os.path.expanduser("~/Documents/Internship_INSPECT/.venv_legacy/bin/python")
    target_script = os.path.join(base_dir, "ehr", "2_generate_labels_and_features.py")
    
    cohort_path = os.path.expanduser("~/Documents/Internship_INSPECT/DATA_PROCESSED/cohort_0.2.0_master_file_anon.csv")
    db_path = os.path.expanduser("~/Documents/Internship_INSPECT/DATA_RAW/EHR_FEMR_DB/extract")
    output_dir = os.path.expanduser(f"~/Documents/Internship_INSPECT/DATA_RAW/EHR_FEMR_DB/features/{task}")
    
    # Construct the command array
    command = [
        venv_python,
        target_script,
        "--path_to_cohort", cohort_path,
        "--path_to_database", db_path,
        "--path_to_output_dir", output_dir,
        "--labeling_function", task,
        "--num_threads", "14"
    ]
    
    print(f"Executing target script using Python environment: {venv_python}\n")
    
    try:
        # Run the command and stream output directly to the console
        process = subprocess.Popen(command, cwd=base_dir, stdout=sys.stdout, stderr=sys.stderr)
        process.wait()
        
        if process.returncode == 0:
            print("\n==============================================================================")
            print(f"Pipeline complete! Features are saved in: DATA_RAW/EHR_FEMR_DB/features/{task}")
            print("==============================================================================")
        else:
            print(f"\n[!] Pipeline failed with exit code {process.returncode}")
            
    except Exception as e:
        print(f"Failed to execute pipeline: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='PE', help='Task to generate features for')
    args = parser.parse_args()
    run_baseline_pipeline(args.task)
