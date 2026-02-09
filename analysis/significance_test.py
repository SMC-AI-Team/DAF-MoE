"""
Statistical Significance Test
=============================

Description:
    Performs a statistical comparison between DAF-MoE and Gradient Boosted Decision Trees 
    (XGBoost, CatBoost) based on the overlap of 95% Confidence Intervals (CIs).
    
    This script is used to generate the statistical parity analysis presented in
    Table 2 (Comparison with GBDTs).

Usage:
    python analysis/significance_test.py
"""

import os
import json
import numpy as np
import pandas as pd
import glob
from scipy import stats
from collections import defaultdict

# ==========================================
# 1. Configuration
# ==========================================
SCORE_DIR = "results/scores"

# Target datasets for analysis
TARGET_DATASETS = [
    "California Housing",
    "Adult Census Income",
    "Higgs small",
    "Covertype",
    "Allstate",
    "BNP Paribas",
    "NHANES",
    "MIMIC-III Mortality",
    "MIMIC-IV Mortality"
]

# Metric Mapping (Metric Name, Is 'Lower Better'?)
METRIC_RULES = {
    "adult": ("acc", False),
    "higgs": ("acc", False),
    "covertype": ("acc", False),
    "bnp": ("acc", False),
    "nhanes": ("acc", False),
    "credit": ("auprc", False),
    "mimic-iii": ("auprc", False),
    "mimic-iv": ("auprc", False),
    "california": ("rmse", True),
    "yearprediction": ("rmse", True),
    "allstate": ("rmse", True)
}

def get_metric_rule(dataset_name):
    """Retrieves the metric name and direction (lower/higher is better) for a dataset."""
    name_lower = dataset_name.lower()
    for key, (metric, is_lower_better) in METRIC_RULES.items():
        if key in name_lower:
            return metric, is_lower_better
    return 'acc', False # Default

def calculate_ci(scores, confidence=0.95):
    """Calculates Mean, Std, and 95% Confidence Interval."""
    n = len(scores)
    if n < 2: return np.mean(scores), 0, (0, 0)
    
    mean = np.mean(scores)
    std = np.std(scores, ddof=1) # Sample standard deviation
    se = std / np.sqrt(n)        # Standard Error
    
    # t-distribution critical value
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n-1)
    margin = t_crit * se
    
    return mean, std, (mean - margin, mean + margin)

def main():
    files = glob.glob(os.path.join(SCORE_DIR, "*.json"))
    if not files:
        print(f"❌ No files found in {SCORE_DIR}")
        return
    
    # Compare DAF-MoE against GBDTs only
    TARGET_DL_KEY = 'daf_moe' 
    GBDT_MODELS = ['xgboost', 'catboost']
    
    data_map = defaultdict(lambda: defaultdict(list))
    
    print(f"📂 Parsing {len(files)} files (Filtering for target datasets)...")

    # 1. Parse Data
    for fpath in files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            dataset_raw = data.get('dataset', 'Unknown')
            if dataset_raw not in TARGET_DATASETS:
                continue

            model_raw = data.get('model', 'Unknown').lower()
            
            # Map model names
            if "daf" in model_raw: 
                model_key = TARGET_DL_KEY
            else: 
                model_key = model_raw
            
            target_metric, _ = get_metric_rule(dataset_raw)
            metrics_data = data.get('metrics', {})
            
            score = None
            if isinstance(metrics_data, list) and metrics_data:
                score = metrics_data[0].get(target_metric)
            elif isinstance(metrics_data, dict):
                score = metrics_data.get(target_metric)
                
            if score is not None:
                data_map[dataset_raw][model_key].append(score)
                
        except Exception as e:
            print(f"⚠️ Error parsing {fpath}: {e}")

    # 2. Run Statistical Analysis (CI Overlap Test)
    results = []
    print(f"\n🔍 Running 95% CI Overlap Tests ({TARGET_DL_KEY} vs GBDTs)...")

    for dataset, models_data in data_map.items():
        metric_name, is_lower_better = get_metric_rule(dataset)
        
        if TARGET_DL_KEY not in models_data:
            continue
            
        scores_dl = models_data[TARGET_DL_KEY]
        if len(scores_dl) < 2: continue
        
        mean_dl, std_dl, ci_dl = calculate_ci(scores_dl)
        
        gbdt_candidates = [m for m in models_data if m in GBDT_MODELS]
        
        for gbdt_model in gbdt_candidates:
            scores_gbdt = models_data[gbdt_model]
            if len(scores_gbdt) < 2: continue
            
            mean_gbdt, std_gbdt, ci_gbdt = calculate_ci(scores_gbdt)
            
            # CI Overlap Logic: max(lower1, lower2) < min(upper1, upper2) indicates overlap
            overlap = max(ci_dl[0], ci_gbdt[0]) < min(ci_dl[1], ci_gbdt[1])
            
            if overlap:
                verdict = "Tie"
            else:
                if is_lower_better: # e.g. RMSE
                    verdict = "Win (DAF)" if mean_dl < mean_gbdt else "Loss (GBDT)"
                else: # e.g. Accuracy
                    verdict = "Win (DAF)" if mean_dl > mean_gbdt else "Loss (GBDT)"
            
            results.append({
                "Dataset": dataset,
                "Metric": metric_name.upper(),
                "My Model": "DAF-MoE",
                "GBDT Model": gbdt_model.upper(),
                "Verdict": verdict,
                "Mean (DAF)": mean_dl,
                "CI (DAF)": f"[{ci_dl[0]:.4f}, {ci_dl[1]:.4f}]",
                "Mean (GBDT)": mean_gbdt,
                "CI (GBDT)": f"[{ci_gbdt[0]:.4f}, {ci_gbdt[1]:.4f}]"
            })

    # 3. Output Results
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by=["Dataset", "GBDT Model"])
        
        print("\n" + "="*110)
        print(f"📊 Analysis Result (DAF-MoE vs GBDT)")
        print("="*110)
        
        cols = ["Dataset", "Metric", "My Model", "GBDT Model", "Verdict", "Mean (DAF)", "Mean (GBDT)"]
        print(df[cols].to_string(index=False))
        
        save_path = "results/summarize_performance/daf_vs_gbdt_ci_overlap.csv"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        df.to_csv(save_path, index=False)
        print(f"\n💾 Full details saved to: {save_path}")
        
        print("\n📈 [Summary Table]")
        summary = df.groupby(["GBDT Model", "Verdict"]).size().unstack(fill_value=0)
        print(summary)
            
    else:
        print("⚠️ No valid comparison pairs found for DAF-MoE.")

if __name__ == "__main__":
    main()