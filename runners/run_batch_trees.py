"""
Batch Runner for GBDT Experiments
=================================

Description:
    Automates the execution of `runner/run_trees.py` across all datasets and models.
    It can be switched between two modes:
    1. 'tune': Runs Hyperparameter Optimization (Optuna) for all dataset-model pairs.
    2. 'eval': Runs final evaluation (15 seeds) using the optimized hyperparameters.

Usage:
    python runner/run_batch_trees.py
"""

import subprocess
import time

# ====================================================
# Configuration: Datasets and Task Types
# ====================================================
DATASET_INFO = {
    "adult": "classification",
    "california": "regression",
    "higgs_small": "classification",
    "allstate": "regression",     
    "covertype": "classification",
    "year_prediction": "regression",
  
    "creditcard": "classification",
    "bnp": "classification",
    "nhanes": "classification",
    "mimic3": "classification",
    "mimic4": "classification", 
}

MODELS = ["xgboost", "catboost"]

# 🔥 [Select Mode]
# "tune": Run HPO to find best hyperparameters.
# "eval": Run final evaluation with found hyperparameters.
MODE = "eval" 

TRIALS = 50  # Number of trials for 'tune' mode

def run_command(cmd):
    """Executes a shell command and handles errors."""
    print(f"\n🚀 [Running] {cmd}")
    try:
        subprocess.run(cmd, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"🚨 [Error] Failed: {cmd}")

def main():
    start_total = time.time()
    
    print("="*60)
    print(f"🔥 Batch Runner Start (Mode: {MODE} | Datasets: {len(DATASET_INFO)})")
    print("="*60)

    for dataset, task_type in DATASET_INFO.items():
        for model in MODELS:
            print(f"\n>> 🛠️ Processing {model} on {dataset} ({task_type})...")
            
            # Construct Command based on Mode
            if MODE == "tune":
                # HPO Phase
                cmd = f"python runner/run_trees.py --dataset {dataset} --model {model} --tune --trials {TRIALS} --task_type {task_type}"
            else:
                # Evaluation Phase
                cmd = f"python runner/run_trees.py --dataset {dataset} --model {model} --eval --task_type {task_type}"
            
            run_command(cmd)
            
            # Cooldown to prevent overheating
            time.sleep(3)

    elapsed = time.time() - start_total
    print("\n" + "="*60)
    print(f"✅ All Jobs Finished! Total Time: {elapsed/60:.2f} min")
    print("="*60)

if __name__ == "__main__":
    main()