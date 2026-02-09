Here is the updated **README.md**. I have refined the **Introduction**, **Data Preparation**, and **Analysis & Evaluation** sections to perfectly match your 5 new analysis scripts and their corresponding Tables/Figures in the paper.

---

# DAF-MoE: Distribution-Aware Feature-level Mixture of Experts

> **Official PyTorch Implementation**
> *This repository contains the code for the paper "DAF-MoE: Robust Tabular Deep Learning via Distribution-Aware Routing".*

## 📌 Introduction

**DAF-MoE** is a novel deep learning architecture designed for heterogeneous tabular data. Unlike standard Transformers that treat all features uniformly, DAF-MoE employs a **Distribution-Guided Gating (DGG)** mechanism to route features to specialized experts based on their statistical characteristics (e.g., outliers vs. common values).

This repository provides:

* Complete implementation of **DAF-MoE** and baselines (FT-Transformer, ResNet, MLP).
* Scripts for **Hyperparameter Optimization (HPO)** using Optuna.
* Automated scripts to **reproduce the main results** (15 random seeds).
* **Comprehensive Analysis Suite:** Tools for expert visualization, noise robustness tests, statistical significance checks, and ablation summaries.

---

## 🛠️ 1. Installation

Since this repository is provided for anonymous review, please **download the source code as a ZIP file** and extract it.

### Prerequisites

* Python 3.10
* CUDA (Recommended for Deep Learning models)

### Setup

1. Open your terminal and navigate to the extracted project root:

```bash
cd DAF-MoE-FBA4 

```

### Option A: Conda (Recommended)

Use this option to automatically set up the environment using `environment.yml`.

```bash
# 1. Create Conda environment
conda env create -f environment.yml

# 2. Activate environment
conda activate daf_moe

# 3. Install the package in editable mode (Important!)
pip install -e .

```

### Option B: Pip (Standard)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install the package in editable mode (Important!)
pip install -e .

```

---

## 📂 2. Data Preparation

Due to licensing and size constraints, raw data files are **not included** in this repository.
Please download the datasets and place them in the `data/` directory.

### Directory Structure

```text
data/
├── adult.csv
├── california_housing.csv
├── covertype.csv
├── higgs_small.csv
...

```

* **Note:** The schema for each dataset is defined in `configs/datasets/*.yaml`.

---

## 🚀 3. Reproduction (Main Results)

We provide the **best hyperparameter configurations** in `configs/experiments/*_best.yaml`. You can reproduce the results reported in the paper (Table 2) using the following scripts.

### A. Deep Learning Models (15 Seeds)

```bash
# Syntax: bash scripts/reproduce_results.sh <CONFIG_PATH> [GPU_ID]

# Example: Reproduce DAF-MoE on Adult dataset
bash scripts/reproduce_results.sh configs/experiments/adult_daf_moe_best.yaml 0

```

### B. Tree-Based Models (XGBoost / CatBoost)

```bash
# Run evaluation for XGBoost on Adult dataset
python runners/run_trees.py --dataset adult --model xgboost --eval

```

---

## ⚡ 4. Hyperparameter Optimization (Optional)

To re-tune hyperparameters from scratch using **Optuna**:

```bash
# Syntax: bash scripts/run_hpo.sh <BASE_CONFIG> <HPO_SEARCH_SPACE> <METRIC> <TRIALS> <GPU_ID>

# Example: Tune DAF-MoE on Higgs dataset
bash scripts/run_hpo.sh \
    configs/experiments/higgs_small_daf_moe.yaml \
    configs/hpo/daf_moe.yaml \
    acc 50 0

```

---

## 🔬 5. Ablation Studies

To validate the contribution of each component (e.g., removal of the Preservation Path or Distribution Tokens):

```bash
python runners/run_ablation.py

```

* **Output:** Results are saved in `results/ablation/`.

---

## 📊 6. Analysis & Evaluation

We provide a comprehensive suite of scripts to generate the tables and figures presented in the paper.

### A. Main Performance Summary (Table 2 & 4)

Aggregates metrics across all datasets and models, and generates the performance ranking chart.

```bash
python analysis/summarize_results.py

```

### B. Expert Visualization (Figure 4)

Visualizes how the Distribution-Guided Gating mechanism routes features based on their statistical properties (e.g., Mode vs. Tail).

```bash
# Default dataset: California Housing
python analysis/analyze_expert.py

```

### C. Noise Robustness Analysis (Figure 3)

Evaluates model stability by injecting Gaussian noise into input features and comparing performance degradation against baselines.

```bash
python analysis/analyze_noise_robustness.py

```

### D. Statistical Significance Test (Table 2 Support)

Performs 95% Confidence Interval (CI) overlap tests to determine statistical parity between DAF-MoE and GBDTs.

```bash
python analysis/significance_test.py

```

### E. Ablation Study Summary (Table 5)

Summarizes the results from `runners/run_ablation.py` to quantify the impact of specific architectural components.

```bash
python analysis/summarize_ablation.py

```

---

## 📁 Project Structure

```text
DAF-MoE/
├── src/                    # Source code (Models, Loss, Data Loader)
├── configs/                # Configuration files (YAML)
├── scripts/                # Shell scripts for execution
├── runners/                # Python runners for Trees & Ablation
├── analysis/               # Analysis & Evaluation scripts
├── results/                # Output logs
├── train.py                # Training entry point
├── tune.py                 # HPO entry point
└── setup.py                # Package installation script

```

## 📜 License

This project is licensed under the MIT License.