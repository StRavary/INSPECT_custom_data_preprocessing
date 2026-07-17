import os
import subprocess
import sys

TASKS = [
    "PE",
    "1_month_mortality",
    "6_month_mortality",
    "12_month_mortality",
    "1_month_readmission",
    "6_month_readmission",
    "12_month_readmission",
    "12_month_PH"
]

def main():
    base_dir = os.path.expanduser("../INSPECT_custom_data_preprocessing")
    venv_dir = "../.venv_legacy" if os.path.isdir(os.path.expanduser("../.venv_legacy")) else "../venv_legacy"
    venv_python = os.path.expanduser(f"{venv_dir}/bin/python")
    
    benchmark_script = os.path.join(base_dir, "Custom", "9a_run_baseline_benchmark.py")
    train_script = os.path.join(base_dir, "ehr", "3_train_gbm.py")
    cohort_path = os.path.join(base_dir, "..", "DATA_PROCESSED", "cohort_0.2.0_master_file_anon.csv")
    db_path = os.path.join(base_dir, "..", "DATA_RAW", "EHR_FEMR_DB", "extract")
    
    results = {}
    
    for task in TASKS:
        print(f"\n{'='*80}\nRunning task: {task}\n{'='*80}")
        
        # 1. Generate features
        print("Generating features...")
        feat_cmd = [venv_python, benchmark_script, "--task", task]
        feat_proc = subprocess.Popen(feat_cmd, cwd=base_dir)
        feat_proc.wait()
        
        if feat_proc.returncode != 0:
            print(f"[!] Feature generation failed for {task}")
            results[task] = "Failed (Featurization)"
            continue
            
        # 2. Train GBM
        print("\nTraining GBM...")
        out_dir = os.path.join(base_dir, "..", "DATA_RAW", "EHR_FEMR_DB", "gbm_model_results", task)
        feat_dir = os.path.join(base_dir, "..", "DATA_RAW", "EHR_FEMR_DB", "features", task)
        
        train_cmd = [
            venv_python, train_script,
            "--path_to_cohort", cohort_path,
            "--path_to_database", db_path,
            "--path_to_label_features", feat_dir,
            "--path_to_output_dir", out_dir,
            "--num_threads", "14"
        ]
        
        # We need to capture output to get the AUROC score
        train_proc = subprocess.Popen(train_cmd, cwd=base_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        test_auroc = None
        for line in train_proc.stdout:
            print(line, end='')  # stream to console
            if "Test AUROC:" in line:
                test_auroc = line.split("Test AUROC:")[1].strip()
                
        train_proc.wait()
        
        if train_proc.returncode != 0:
            print(f"[!] Training failed for {task}")
            results[task] = "Failed (Training)"
        else:
            results[task] = test_auroc if test_auroc else "Unknown"

    print("\n\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    for task, auroc in results.items():
        print(f"{task.ljust(25)}: {auroc}")

if __name__ == "__main__":
    main()
