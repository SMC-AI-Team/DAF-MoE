"""
Expert Specialization Analysis
==============================

Description:
    Analyzes and visualizes the gating behavior of DAF-MoE (Figure 4 in the paper).
    It loads a trained model, runs inference on the test set, and captures the 
    distribution of input features assigned to each expert.

    Outputs:
    - Heatmap: Average gating probability per feature distribution quantile.
    - Density Plot: Expert coverage overlaid on the global data distribution.

Usage:
    python analysis/analyze_expert.py --dataset california
"""

import os
import re
import torch
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import glob
from matplotlib.gridspec import GridSpec

# Project Modules
from src.data.loader import get_dataloaders
from src.models.factory import create_model
from src.configs.default_config import DAFConfig

# ==========================================
# Configuration
# ==========================================
DEFAULT_DATASET = "california"
OUTPUT_DIR = "results/analysis_expert"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Experts to highlight in the density plot
TARGET_EXPERTS = [1, 3, 4]
EXPERT_COLORS = ['#ff7f0e', '#2ca02c', '#d62728'] 

# Style Settings
sns.set_style("whitegrid", {'axes.grid': False})
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16
})

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_data_and_model(args):
    """Loads configuration, data loaders, and the pre-trained model."""
    exp_config_path = f"configs/experiments/{args.dataset}_daf_moe_best.yaml"
    data_config_path = f"configs/datasets/{args.dataset}.yaml"
    
    exp_cfg = load_yaml(exp_config_path)
    data_cfg = load_yaml(data_config_path)
    real_dataset_name = data_cfg.get('dataset_name', args.dataset)
    
    config = DAFConfig()
    for k, v in exp_cfg.items():
        if hasattr(config, k):
            if v == 'None': v = None
            setattr(config, k, v)
            
    if args.dataset in ['california', 'year_prediction', 'allstate', 'diamond']:
        config.task_type = 'regression'; config.out_dim = 1
    else:
        config.task_type = 'classification'; config.out_dim = 1

    # Locate Checkpoint
    ckpt_pattern = f"checkpoints/{real_dataset_name}_daf_moe_seed*_best.pth"
    found = glob.glob(ckpt_pattern)
    if found: 
        found.sort()
        config.checkpoint_path = found[0]
        match = re.search(r'seed(\d+)', found[0])
        config.seed = int(match.group(1)) if match else 42
    else:
        config.checkpoint_path = f"checkpoints/{real_dataset_name}_daf_moe_best.pth"
        config.seed = 42

    _, _, test_loader = get_dataloaders(config, data_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(config).to(device)
    
    if os.path.exists(config.checkpoint_path):
        model.load_state_dict(torch.load(config.checkpoint_path, map_location=device))
        print(f"✅ Loaded: {config.checkpoint_path}")
    
    return model, test_loader, device, config

# ------------------------------------------------------------
# Main Analysis Function
# ------------------------------------------------------------
def analyze_aligned(args):
    model, loader, device, config = get_data_and_model(args)
    model.eval()
    
    all_z, all_gates = [], []
    final_mus = None
    
    print("Running Inference...")
    with torch.no_grad():
        for inputs, _ in tqdm(loader):
            for k, v in inputs.items(): inputs[k] = v.to(device)
            output = model(inputs['x_numerical'], inputs['x_categorical_idx'], inputs['x_categorical_meta'])
            
            # Extract gating weights and centroids from the first layer
            block_info = output['history'][0]
            if final_mus is None: final_mus = block_info['mu'].detach().cpu().numpy()
            
            n_num = config.n_numerical
            z = inputs['x_numerical'][:, :n_num, 0].cpu().numpy().flatten()
            gates = block_info['weights'][:, :n_num, :].cpu().numpy().reshape(-1, config.n_experts)
            
            all_z.append(z); all_gates.append(gates)
            if len(all_z) * len(z) > 100000: break 

    flat_z = np.concatenate(all_z)
    flat_gate = np.concatenate(all_gates, axis=0)
    centroids = 1 / (1 + np.exp(-final_mus))

    # 📌 Layout: 2x2 Grid (Heatmap top, Density bottom, Colorbar right)
    fig = plt.figure(figsize=(8, 10)) 
    gs = GridSpec(2, 2, height_ratios=[1, 0.8], width_ratios=[1, 0.05], 
                  hspace=0.3, wspace=0.05, figure=fig)

    # ==========================
    # (a) Heatmap Plotting
    # ==========================
    ax_heat = fig.add_subplot(gs[0, 0])
    cbar_ax = fig.add_subplot(gs[0, 1])
    
    bins = [-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf]
    bin_labels = [
        "Low Tail\n(z < -1.5)", 
        "Low Range\n(-1.5 ~ -0.5)", 
        "Central\n(-0.5 ~ 0.5)", 
        "High Range\n(0.5 ~ 1.5)", 
        "High Tail\n(z > 1.5)"
    ]
    categories = pd.cut(flat_z, bins=bins, labels=bin_labels)
    
    df_analysis = pd.DataFrame({'Zone': categories})
    expert_labels = []
    for i in range(config.n_experts):
        col = f'E{i}'
        df_analysis[col] = flat_gate[:, i]
        expert_labels.append(f"E{i} ($\sigma(\mu)$={centroids[i]:.2f})")
        
    df_heatmap = df_analysis.groupby('Zone', observed=False).mean().T
    
    # Heatmap with a separate colorbar axis
    sns.heatmap(df_heatmap, annot=True, fmt=".2f", cmap="YlGnBu", 
                linewidths=.5, cbar_ax=cbar_ax, 
                cbar_kws={'label': 'Mean Gating Probability'}, ax=ax_heat)
    
    ax_heat.set_title('(a) Gating Distribution Map', fontweight='bold', pad=10)
    ax_heat.set_xlabel('')
    ax_heat.set_yticklabels(expert_labels, rotation=0, fontsize=11)
    ax_heat.set_xticklabels(bin_labels, rotation=0, fontsize=10)

    # ==========================
    # (b) Expert Coverage Overlay
    # ==========================
    ax_dens = fig.add_subplot(gs[1, 0])
    
    # Plot Global Data Distribution
    df_plot = pd.DataFrame({'z': flat_z})
    sns.kdeplot(data=df_plot, x='z', color='gray', fill=True, alpha=0.15, linewidth=0, 
                label='Global Data', ax=ax_dens)
    
    # Plot Weighted Expert Coverage
    for idx, color in zip(TARGET_EXPERTS, EXPERT_COLORS):
        label = f"Expert {idx} ($\sigma(\mu)$={centroids[idx]:.2f})"
        sns.kdeplot(data=df_plot, x='z', weights=flat_gate[:, idx], 
                    fill=True, color=color, alpha=0.3, linewidth=2, 
                    label=label, ax=ax_dens)

    ax_dens.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=1, label='Center (z=0)')
    
    ax_dens.set_xlim(-3.5, 3.5)
    ax_dens.set_title('(b) Expert Coverage Comparison', fontweight='bold', pad=10)
    ax_dens.set_xlabel('Feature Value (z-score)')
    ax_dens.set_ylabel('Density (Weighted)')
    ax_dens.set_yticks([]) 
    
    ax_dens.legend(loc='upper right', frameon=True, fontsize=11, 
                   fancybox=True, framealpha=0.9, edgecolor='gray')

    plt.subplots_adjust(left=0.20, right=0.92, top=0.95, bottom=0.08)

    save_path = os.path.join(OUTPUT_DIR, "analysis_expert.pdf")
    plt.savefig(save_path, format='pdf')
    print(f"\n✅ Aligned Perfect Figure saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    args = parser.parse_args()
    analyze_aligned(args)