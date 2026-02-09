"""
Ablation Study Runner
=====================

Description:
    Automates the training of DAF-MoE variants for the Ablation Study (Section 5.1).
    This script iterates over specified datasets, seeds, and architectural modifications
    to verify the contribution of each component.

    Ablation Scenarios (Variants):
    1. wo_raw_path: Disables the 'Preservation Path' to test raw value retention.
    2. wo_dist_token: Removes 'Distribution-Embedded Tokenization' (Quantiles/Skewness).
    3. wo_aux_loss: Trains without auxiliary losses (Specialization & Repulsion), reducing to a standard MoE.

    Outputs:
    - JSON result files are saved to `results/ablation/` with corresponding suffixes.

Usage:
    python runner/run_ablation.py
"""

import os
import time

# ====================================================
# 🔥 [Configuration] Datasets & Seeds
# ====================================================
# Select datasets to run ablation on (e.g., California Housing, Adult)
DATASETS = ["california"]

GPU_ID = "0"
SEEDS = [43]  # List of seeds for reproducibility
RESULT_DIR = "results/ablation"

# Create output directory if it doesn't exist
os.makedirs(RESULT_DIR, exist_ok=True)

# ====================================================
# 🧪 [Configuration] Ablation Variants
# ====================================================
# Dictionary defining the variants and their corresponding command-line flags.
VARIANTS = {
    # ------------------------------------------------
    # Set 1: Structural Ablation (Architecture)
    # ------------------------------------------------
    # "full_model": {},  # Skipped: Assuming full model baseline is already trained.
    
    "wo_raw_path":      {"--use_raw_path": "False"},   # Key: Disables outlier preservation path
    "wo_dist_token":    {"--use_dist_token": "False"}, # Removes statistical metadata from input

    # ------------------------------------------------
    # Set 2: Loss Ablation (Optimization Objective)
    # ------------------------------------------------
    # Disables both Specialization and Repulsion losses -> Standard MoE behavior
    "wo_aux_loss":      {"--lambda_spec": "0.0", "--lambda_repel": "0.0"} 
}

def run_experiment(dataset, variant_name, flags, seed):
    """Executes a single training run for a specific variant."""
    print(f"\n🚀 [Start] {dataset} | {variant_name} | Seed {seed}")
    
    # Load optimal hyperparameters from the Full Model config
    config_path = f"configs/experiments/{dataset}_daf_moe_best.yaml"
    
    if not os.path.exists(config_path):
        print(f"🚨 Config not found: {config_path}")
        return

    # Construct the training command
    # --verbose: Enables logging
    # --result_dir: Saves results to the ablation folder
    cmd = (f"python train.py --config {config_path} --gpu_ids {GPU_ID} "
           f"--seed {seed} --verbose --result_dir {RESULT_DIR}")
    
    # Append ablation-specific flags
    for k, v in flags.items():
        cmd += f" {k} {v}"
        
    print(f"   Command: {cmd}")
    
    # Execute
    exit_code = os.system(cmd)
    
    if exit_code != 0:
        print(f"❌ Error in {variant_name} (Seed {seed})")
    else:
        print(f"✅ Done {variant_name} (Seed {seed})")
    
    # Pause briefly to prevent file I/O conflicts or overheating
    time.sleep(1)

def main():
    total_runs = len(DATASETS) * len(VARIANTS) * len(SEEDS)
    print(f"🔥 Starting Full Ablation Study")
    print(f"   - Datasets: {DATASETS}")
    print(f"   - Variants: {list(VARIANTS.keys())}")
    print(f"   - Seeds: {SEEDS}")
    print(f"   - Total Runs: {total_runs}")
    print(f"   - Save Directory: {RESULT_DIR}")
    
    for dataset in DATASETS:
        print(f"\n{'='*60}")
        print(f"📂 Processing Dataset: {dataset}")
        print(f"{'='*60}")
        
        for name, flags in VARIANTS.items():
            for seed in SEEDS:
                run_experiment(dataset, name, flags, seed)

if __name__ == "__main__":
    main()