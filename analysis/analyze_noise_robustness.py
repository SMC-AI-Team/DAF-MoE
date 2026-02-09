"""
Noise Robustness Analysis
=========================

Description:
    Evaluates the robustness of DAF-MoE and baselines against Gaussian noise
    injected into numerical features during inference (Figure 3 in the paper).
    
    It injects noise with standard deviations ranging from 0.0 to 0.3
    and plots the performance degradation curves.

Usage:
    python analysis/analyze_noise_robustness.py
"""

import os
import re
import torch
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import glob
import xgboost as xgb
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy

# Project Modules
from src.data.loader import get_dataloaders
from src.models.factory import create_model
from src.configs.default_config import DAFConfig

# ==========================================
# Configuration
# ==========================================
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
TARGET_DATASETS = ["california", "adult"]
OUTPUT_DIR = "results/analysis_robustness"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style Settings
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 12,
    'lines.linewidth': 2.5,
    'lines.markersize': 8
})

CUSTOM_PALETTE = {'XGBoost': '#2ca02c', 'FT-Transformer': '#7f7f7f', 'DAF-MoE': '#d62728'}
MARKERS = {'XGBoost': 's', 'FT-Transformer': 'o', 'DAF-MoE': '*'}

# Drawing order: Draw DAF-MoE last to keep it on top
DRAW_ORDER = ['FT-Transformer', 'XGBoost', 'DAF-MoE'] 
# Legend order: DAF-MoE first for emphasis
LEGEND_ORDER = ['DAF-MoE', 'XGBoost', 'FT-Transformer']

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# ------------------------------------------------------------
# Data & Evaluation Logic
# ------------------------------------------------------------
def impute_data(X_train, X_val, X_test, cat_cols):
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    if num_cols:
        medians = X_train[num_cols].median()
        X_train[num_cols] = X_train[num_cols].fillna(medians)
        if X_val is not None: X_val[num_cols] = X_val[num_cols].fillna(medians)
        if X_test is not None: X_test[num_cols] = X_test[num_cols].fillna(medians)
    return X_train, X_val, X_test

def get_tree_data(dataset, seed, task_type):
    """Prepares data for GBDT models."""
    data_cfg = load_yaml(f"configs/datasets/{dataset}.yaml")
    csv_path = data_cfg.get('csv_path', f"data/{dataset}.csv")
    if not os.path.exists(csv_path): csv_path = f"data/{os.path.basename(csv_path)}"
    
    df = pd.read_csv(csv_path, skipinitialspace=True)
    target_col = data_cfg.get('target_col', 'target')
    cat_cols = data_cfg.get('cat_cols', [])
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if cat_cols:
        X[cat_cols] = X[cat_cols].fillna('Unknown').astype(str)
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            
    if y.dtype == 'object' or y.dtype.name == 'category':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        
    stratify_param = y if task_type == 'classification' else None
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=stratify_param, random_state=seed
    )
    stratify_temp = y_temp if task_type == 'classification' else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=stratify_temp, random_state=seed
    )
    X_train, X_val, X_test = impute_data(X_train, X_val, X_test, cat_cols)
    return X_train, y_train, X_val, y_val, X_test, y_test, cat_cols

def add_noise_tensor(tensor_data, noise_std):
    """Injects Gaussian noise into the feature tensor."""
    if noise_std == 0.0: return tensor_data
    return tensor_data + torch.randn_like(tensor_data) * noise_std

def calculate_metric(y_true, y_pred, metric_name, out_dim=1, is_proba=True):
    if metric_name == 'RMSE':
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif metric_name == 'ACC':
        if out_dim > 1 and is_proba: pred_labels = np.argmax(y_pred, axis=1)
        elif is_proba: pred_labels = (y_pred > 0.5).astype(int)
        else: pred_labels = y_pred
        return accuracy_score(y_true, pred_labels)
    return 0.0

def evaluate_dl_model(model, loader, noise_std, task_type, out_dim, metric_name, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            x_num = inputs['x_numerical'].to(device)
            x_cat = inputs['x_categorical_idx'].to(device)
            x_meta = inputs['x_categorical_meta'].to(device)
            
            # Inject noise only during inference
            x_num_noisy = add_noise_tensor(x_num, noise_std)
            
            try: output = model(x_num_noisy, x_cat, x_meta)
            except TypeError: output = model(x_num_noisy, x_cat)
            
            if isinstance(output, dict): preds = output['logits']
            else: preds = output 
            
            if task_type == 'classification':
                if out_dim == 1: preds = torch.sigmoid(preds)
                else: preds = torch.softmax(preds, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())
            
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    return calculate_metric(y_true, y_pred, metric_name, out_dim)

def get_available_seeds(dataset_real_name):
    pattern = f"checkpoints/{dataset_real_name}_daf_moe_seed*_best.pth"
    files = glob.glob(pattern)
    seeds = []
    for f in files:
        match = re.search(r'seed(\d+)', f)
        if match: seeds.append(int(match.group(1)))
    return sorted(list(set(seeds)))

def collect_results_multiseed(dataset_name, device):
    print(f"\n🚀 Processing: {dataset_name} (Multi-seed Analysis)")
    data_config_path = f"configs/datasets/{dataset_name}.yaml"
    data_cfg = load_yaml(data_config_path)
    real_dataset_name = data_cfg.get('dataset_name', dataset_name)
    seeds = get_available_seeds(real_dataset_name)
    if not seeds:
        if os.path.exists(f"checkpoints/{real_dataset_name}_daf_moe_best.pth"): seeds = [42]
        else: return pd.DataFrame()
    
    results = []
    if dataset_name == 'california':
        metric = 'RMSE'; config_task = 'regression'; config_out = 1
    elif dataset_name == 'adult':
        metric = 'ACC'; config_task = 'classification'; config_out = 1
    else: return pd.DataFrame()

    exp_config_path = f"configs/experiments/{dataset_name}_daf_moe_best.yaml"
    base_exp_cfg = load_yaml(exp_config_path)
    
    for seed in tqdm(seeds, desc=f"Seeds ({dataset_name})"):
        np.random.seed(seed)
        torch.manual_seed(seed)
        config = DAFConfig()
        config.seed = seed
        config.task_type = config_task
        config.out_dim = config_out
        for k, v in base_exp_cfg.items():
            if hasattr(config, k): setattr(config, k, None if v == 'None' else v)

        # 1. XGBoost
        X_train, y_train, X_val, y_val, X_test, y_test, cat_cols = get_tree_data(dataset_name, seed, config.task_type)
        num_cols = [c for c in X_train.columns if c not in cat_cols]
        xgb_path = f"configs/experiments/{dataset_name}_xgboost_best.yaml"
        xgb_params = load_yaml(xgb_path) if os.path.exists(xgb_path) else {'n_estimators': 1000}
        
        if config.task_type == 'regression':
            xgb_model = xgb.XGBRegressor(**xgb_params, tree_method='hist', device='cuda' if torch.cuda.is_available() else 'cpu', n_jobs=-1, early_stopping_rounds=50, enable_categorical=False)
        else:
            xgb_model = xgb.XGBClassifier(**xgb_params, tree_method='hist', device='cuda' if torch.cuda.is_available() else 'cpu', n_jobs=-1, early_stopping_rounds=50, enable_categorical=False)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        feat_stds = X_train[num_cols].std().values
        
        for sigma in NOISE_LEVELS:
            X_noisy = X_test.copy()
            if sigma > 0:
                noise = np.random.normal(0, sigma, X_noisy[num_cols].shape) * feat_stds
                X_noisy[num_cols] += noise
            preds = xgb_model.predict(X_noisy)
            score = calculate_metric(y_test, preds, metric, config.out_dim, is_proba=False)
            results.append({'Model': 'XGBoost', 'Sigma': sigma, 'Score': score, 'Seed': seed})

        # 2. DAF-MoE
        _, _, test_loader = get_dataloaders(config, data_cfg)
        model_daf = create_model(config).to(device)
        ckpt_daf = f"checkpoints/{real_dataset_name}_daf_moe_seed{seed}_best.pth"
        if os.path.exists(ckpt_daf):
            model_daf.load_state_dict(torch.load(ckpt_daf, map_location=device))
            for sigma in NOISE_LEVELS:
                score = evaluate_dl_model(model_daf, test_loader, sigma, config.task_type, config.out_dim, metric, device)
                results.append({'Model': 'DAF-MoE', 'Sigma': sigma, 'Score': score, 'Seed': seed})
        
        # 3. FT-Transformer
        config_ft = deepcopy(config); config_ft.model_type = 'ft_transformer'
        ft_path = f"configs/experiments/{dataset_name}_ft_transformer_best.yaml"
        if os.path.exists(ft_path):
            ft_p = load_yaml(ft_path)
            for k, v in ft_p.items():
                if hasattr(config_ft, k): setattr(config_ft, k, None if v == 'None' else v)
        model_ft = create_model(config_ft).to(device)
        ckpt_ft = f"checkpoints/{real_dataset_name}_ft_transformer_seed{seed}_best.pth"
        if os.path.exists(ckpt_ft):
            model_ft.load_state_dict(torch.load(ckpt_ft, map_location=device))
            for sigma in NOISE_LEVELS:
                score = evaluate_dl_model(model_ft, test_loader, sigma, config.task_type, config.out_dim, metric, device)
                results.append({'Model': 'FT-Transformer', 'Sigma': sigma, 'Score': score, 'Seed': seed})

    return pd.DataFrame(results)

# ------------------------------------------------------------
# Main Visualization
# ------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df_cal = collect_results_multiseed("california", device)
    df_adult = collect_results_multiseed("adult", device)
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.0)) 
    
    # (a) California Housing
    if not df_cal.empty:
        sns.lineplot(data=df_cal, x='Sigma', y='Score', hue='Model', style='Model', 
                     markers=MARKERS, dashes=True, linewidth=2.5, palette=CUSTOM_PALETTE, 
                     hue_order=DRAW_ORDER, style_order=DRAW_ORDER, 
                     markersize=9, errorbar=None, ax=axes[0], legend=False)
        axes[0].set_title('(a) California Housing', fontweight='bold', pad=10)
        axes[0].set_ylabel(f'RMSE (Lower is Better)')
        axes[0].set_xlabel(r'Gaussian Noise Level ($\sigma$)')
        
        means = df_cal.groupby(['Model', 'Sigma'])['Score'].mean()
        min_y, max_y = means.min(), means.max()
        axes[0].set_ylim(min_y * 0.9, max_y * 1.1)
        axes[0].grid(True, linestyle='--', alpha=0.6)

    # (b) Adult
    if not df_adult.empty:
        sns.lineplot(data=df_adult, x='Sigma', y='Score', hue='Model', style='Model', 
                     markers=MARKERS, dashes=True, linewidth=2.5, palette=CUSTOM_PALETTE, 
                     hue_order=DRAW_ORDER, style_order=DRAW_ORDER, 
                     markersize=9, errorbar=None, ax=axes[1], legend=True)
        axes[1].set_title('(b) Adult', fontweight='bold', pad=10)
        axes[1].set_xlabel(r'Gaussian Noise Level ($\sigma$)')
        axes[1].set_ylabel(f'Accuracy (Higher is Better)')
        
        means = df_adult.groupby(['Model', 'Sigma'])['Score'].mean()
        min_y = means.min()
        axes[1].set_ylim(min_y - 0.05, 1.01)
        axes[1].grid(True, linestyle='--', alpha=0.6)

    # Rearrange Legend: Use 'LEGEND_ORDER' to prioritize DAF-MoE
    if axes[1].get_legend():
        handles, labels = axes[1].get_legend_handles_labels()
        axes[1].get_legend().remove()
        
        hl_dict = {label: handle for label, handle in zip(labels, handles)}
        sorted_handles = [hl_dict[m] for m in LEGEND_ORDER if m in hl_dict]
        sorted_labels = [m for m in LEGEND_ORDER if m in hl_dict]

        # Place legend at the bottom
        fig.legend(sorted_handles, sorted_labels, loc='lower center', 
                   bbox_to_anchor=(0.5, 0.02), 
                   ncol=3, frameon=True, edgecolor='black', fancybox=False)

    plt.subplots_adjust(bottom=0.28, wspace=0.25, left=0.08, right=0.98, top=0.88)
    
    save_path = os.path.join(OUTPUT_DIR, "noise_robustness.png")
    plt.savefig(save_path, dpi=600)
    print(f"\n✅ Final Horizontal Figure saved to: {save_path}")

    plt.savefig(save_path.replace(".png", ".pdf"), format='pdf')
    print(f"✅ PDF saved to: {save_path.replace('.png', '.pdf')}")

if __name__ == "__main__":
    main()