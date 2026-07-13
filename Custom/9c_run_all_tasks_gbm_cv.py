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
    venv_python = os.path.expanduser("../.venv_legacy/bin/python")
    
    benchmark_script = os.path.join(base_dir, "Custom", "9a_run_baseline_benchmark.py")
    train_script = os.path.join(base_dir, "Custom", "3_train_gbm_cv.py")
    cohort_path = os.path.join(base_dir, "..", "DATA_PROCESSED", "cohort_0.2.0_master_file_anon.csv")
    db_path = os.path.join(base_dir, "..", "DATA_RAW", "EHR_FEMR_DB", "extract")
    
    results = {}
    
    for task in TASKS:
        print(f"\n{'='*80}\nRunning task (5-Fold CV): {task}\n{'='*80}")
        
        # 1. Generate features
        print("Generating features...")
        feat_cmd = [venv_python, benchmark_script, "--task", task]
        feat_proc = subprocess.Popen(feat_cmd, cwd=base_dir)
        feat_proc.wait()
        
        if feat_proc.returncode != 0:
            print(f"[!] Feature generation failed for {task}")
            results[task] = {
                "status": "Failed (Featurization)",
                "oof_auroc": "N/A",
                "avg_auroc": "N/A",
                "avg_sens": "N/A",
                "avg_spec": "N/A"
            }
            continue
            
        # 2. Train GBM with 5-Fold Cross-Validation
        print("\nTraining GBM with 5-Fold CV...")
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
        
        # Stream outputs and extract metrics
        train_proc = subprocess.Popen(train_cmd, cwd=base_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        oof_auroc = None
        avg_auroc = None
        avg_sens = None
        avg_spec = None
        
        for line in train_proc.stdout:
            print(line, end='')  # stream to console
            
            if "Overall Pooled OOF AUROC:" in line:
                oof_auroc = line.split("Overall Pooled OOF AUROC:")[1].strip()
            elif "Average Test AUROC:" in line:
                avg_auroc = line.split("Average Test AUROC:")[1].strip()
            elif "Average Test Sensitivity (optimized):" in line:
                avg_sens = line.split("Average Test Sensitivity (optimized):")[1].strip()
            elif "Average Test Specificity (optimized):" in line:
                avg_spec = line.split("Average Test Specificity (optimized):")[1].strip()
                
        train_proc.wait()
        
        if train_proc.returncode != 0:
            print(f"[!] Training failed for {task}")
            results[task] = {
                "status": "Failed (Training)",
                "oof_auroc": "N/A",
                "avg_auroc": "N/A",
                "avg_sens": "N/A",
                "avg_spec": "N/A"
            }
        else:
            results[task] = {
                "status": "Success",
                "oof_auroc": oof_auroc if oof_auroc else "Unknown",
                "avg_auroc": avg_auroc if avg_auroc else "Unknown",
                "avg_sens": avg_sens if avg_sens else "Unknown",
                "avg_spec": avg_spec if avg_spec else "Unknown"
            }

    print("\n\n" + "="*110)
    print("FINAL 5-FOLD CROSS-VALIDATION BENCHMARK RESULTS")
    print("="*110)
    print(f"{'Task'.ljust(22)} | {'Status'.ljust(8)} | {'Overall OOF AUROC'.ljust(18)} | {'Avg Test AUROC'.ljust(20)} | {'Avg Test Sens (opt)'.ljust(20)} | {'Avg Test Spec (opt)'}")
    print("-" * 110)
    for task, res in results.items():
        if res["status"] == "Success":
            print(f"{task.ljust(22)} | {'Success'.ljust(8)} | {res['oof_auroc'].ljust(18)} | {res['avg_auroc'].ljust(20)} | {res['avg_sens'].ljust(20)} | {res['avg_spec']}")
        else:
            status_str = res["status"]
            print(f"{task.ljust(22)} | {status_str.ljust(8)} | {'N/A'.ljust(18)} | {'N/A'.ljust(20)} | {'N/A'.ljust(20)} | {'N/A'}")
    print("="*110)

if __name__ == "__main__":
    main()
