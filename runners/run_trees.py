"""
GBDT Training & Evaluation Runner
=================================

Description:
    Manages the lifecycle of Gradient Boosted Decision Tree (GBDT) models:
    XGBoost and CatBoost.

    Key Features:
    1. Hyperparameter Optimization (HPO): Uses Optuna to find the best hyperparameters
       for a given dataset and model.
    2. Final Evaluation: Trains the model using the optimal hyperparameters on
       15 different random seeds to ensure statistical robustness.
    3. Metrics: Supports RMSE (Regression), Accuracy (Classification), and
       AUPRC (for imbalanced datasets like MIMIC and Credit Card).

Usage:
    # Tune Hyperparameters
    python runner/run_trees.py --dataset adult --model xgboost --tune --trials 50

    # Evaluate using Best Params
    python runner/run_trees.py --dataset adult --model xgboost --eval
"""

import argparse
import os
import gc
import yaml
import json
import time
import pandas as pd
import numpy as np
import optuna
from tqdm import tqdm

# Force CPU Parallelism (16 Cores)
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

# Create output directories
os.makedirs("configs/experiments", exist_ok=True)
os.makedirs("results/scores", exist_ok=True)

# Datasets requiring AUPRC optimization (Imbalanced)
AUPRC_DATASETS = ['creditcard', 'mimic3', 'mimic4']

def load_data_config(dataset_name):
    """Loads the dataset-specific configuration YAML."""
    config_path = f"configs/datasets/{dataset_name}.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    return data_cfg

def get_data(dataset_name, data_cfg, model_type):
    """Loads and preprocesses data for GBDT models."""
    csv_path = data_cfg.get('csv_path', f"data/{dataset_name}.csv")
    if not os.path.exists(csv_path):
        csv_path = f"data/{os.path.basename(csv_path)}"
    
    print(f"📂 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, skipinitialspace=True)
    
    target_col = data_cfg.get('target_col', 'target')
    cat_cols = data_cfg.get('cat_cols', [])

    # Separate Features and Target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Pre-fill Categorical NaNs (Numerical NaNs handled after split)
    if cat_cols:
        X[cat_cols] = X[cat_cols].fillna('Unknown').astype(str)

    # Encode Target (if categorical)
    if y.dtype == 'object' or y.dtype.name == 'category':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)

    # Label Encode Categorical Features
    if cat_cols:
        for col in cat_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                
    return X, y, cat_cols

def get_model(model_type, task_type, params, cat_cols=None, seed=42):
    """Initializes the GBDT model with specified parameters."""
    params = params.copy()

    if model_type == 'xgboost':
        ModelClass = XGBRegressor if task_type == 'regression' else XGBClassifier
        model = ModelClass(
            **params,
            random_state=seed,
            tree_method='hist', 
            device='cpu',        # CPU Training
            enable_categorical=False,
            n_jobs=-1,
            early_stopping_rounds=50
        )
    elif model_type == 'catboost':
        if 'colsample_bytree' in params:
            params['colsample_bylevel'] = params.pop('colsample_bytree')
            
        ModelClass = CatBoostRegressor if task_type == 'regression' else CatBoostClassifier
        model = ModelClass(
            **params,
            random_seed=seed,
            task_type="CPU",      # CPU Training
            thread_count=-1,
            verbose=0,
            early_stopping_rounds=50,
            bootstrap_type='Bernoulli'
        )
    return model

# ---------------------------------------------------------
# Helper: Fair Imputation
# ---------------------------------------------------------
def impute_data(X_train, X_val, X_test, cat_cols):
    """Imputes missing numerical values using Training set medians only."""
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    
    if num_cols:
        medians = X_train[num_cols].median()
        
        X_train[num_cols] = X_train[num_cols].fillna(medians)
        X_val[num_cols] = X_val[num_cols].fillna(medians)
        
        if X_test is not None and not X_test.empty:
            X_test[num_cols] = X_test[num_cols].fillna(medians)
        
    return X_train, X_val, X_test

# =========================================================
# 2. Hyperparameter Optimization (HPO) Logic
# =========================================================
def objective(trial, dataset_name, model_type, task_type, X, y, cat_cols):
    """Optuna objective function for tuning hyperparameters."""
    stratify_param = y if task_type == 'classification' else None

    # Split: Train (80%) / Val (10%) / Test (10%) - Test unused here
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=stratify_param, random_state=42
    )
    
    stratify_temp = y_temp if task_type == 'classification' else None
    X_val, _, y_val, _ = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=stratify_temp, random_state=42
    )
    
    # Impute after split to prevent leakage
    X_train, X_val, _ = impute_data(X_train.copy(), X_val.copy(), None, cat_cols)

    # Search Space
    if model_type == 'xgboost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        }
    elif model_type == 'catboost':
        params = {
            'boosting_type': 'Plain', 
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
        }

    model = get_model(model_type, task_type, params, cat_cols, seed=42)
    
    # Training with Early Stopping
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Evaluation Metric
    if task_type == 'regression':
        preds = model.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        return np.sqrt(mse) # RMSE
    
    else: # Classification
        n_classes = len(np.unique(y))
        
        if n_classes > 2: # Multiclass
            preds = model.predict(X_val)
            return accuracy_score(y_val, preds)
        
        else: # Binary
            if dataset_name in AUPRC_DATASETS:
                preds_proba = model.predict_proba(X_val)[:, 1]
                try:
                    score = average_precision_score(y_val, preds_proba)
                except:
                    score = 0.0
                return score
            else:
                preds = model.predict(X_val)
                return accuracy_score(y_val, preds)

def run_hpo(args):
    """Executes the Hyperparameter Optimization process."""
    print(f"🚀 Start HPO for {args.model} on {args.dataset}")
    data_cfg = load_data_config(args.dataset)
    
    if args.task_type:
        task_type = args.task_type
    else:
        task_type = data_cfg.get('task_type', 'classification')
    
    direction = 'minimize' if task_type == 'regression' else 'maximize'
    
    print("⏳ Pre-loading data into memory...")
    X, y, cat_cols = get_data(args.dataset, data_cfg, args.model)
    
    study = optuna.create_study(direction=direction, study_name=f"{args.dataset}_{args.model}")
    study.optimize(lambda trial: objective(trial, args.dataset, args.model, task_type, X, y, cat_cols), n_trials=args.trials)
    
    print(f"🏆 Best Params: {study.best_params}")
    
    save_path = f"configs/experiments/{args.dataset}_{args.model}_best.yaml"
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(study.best_params, f)
    print(f"💾 Config saved to {save_path}")
    
    del study, X, y
    gc.collect()

# =========================================================
# 3. Final Evaluation
# =========================================================
def run_eval(args):
    """Runs the final evaluation on 15 random seeds using best hyperparameters."""
    print(f"📊 Start Final Evaluation for {args.model} on {args.dataset} (15 Seeds)")
    
    config_path = f"configs/experiments/{args.dataset}_{args.model}_best.yaml"
    if not os.path.exists(config_path):
        print(f"🚨 Config not found! Run --tune first.")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        best_params = yaml.safe_load(f)
        
    data_cfg = load_data_config(args.dataset)
    real_dataset_name = data_cfg.get('dataset_name', args.dataset)
    if args.task_type:
        task_type = args.task_type
    else:
        task_type = data_cfg.get('task_type', 'classification')

    X, y, cat_cols = get_data(args.dataset, data_cfg, args.model)
    
    n_classes = len(np.unique(y))
    
    # Use seeds 43 to 57
    seeds = range(43, 43 + 15)
    scores = []
    
    for seed in tqdm(seeds, desc="Running Seeds"):
        stratify_param = y if task_type == 'classification' else None
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, stratify=stratify_param, random_state=seed
        )
        
        stratify_temp = y_temp if task_type == 'classification' else None
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=stratify_temp, random_state=seed
        )
        
        # Impute
        X_train, X_val, X_test = impute_data(X_train.copy(), X_val.copy(), X_test.copy(), cat_cols)
        
        model = get_model(args.model, task_type, best_params, cat_cols, seed=seed)
        
        start_train = time.time()
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        train_time = time.time() - start_train
        
        start_inf = time.time()
        
        metric_dict = {}
        
        if task_type == 'regression':
            preds = model.predict(X_test)
            inf_time = time.time() - start_inf
            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)
            metric_dict['rmse'] = rmse
            main_score = rmse
            
        else: # Classification
            preds = model.predict(X_test)
            inf_time = time.time() - start_inf
            
            acc = accuracy_score(y_test, preds)
            metric_dict['acc'] = acc
            
            if n_classes > 2:
                auprc = 0.0
                auroc = 0.0
            else:
                preds_proba = model.predict_proba(X_test)[:, 1]
                auprc = average_precision_score(y_test, preds_proba)
                try:
                    auroc = roc_auc_score(y_test, preds_proba)
                except:
                    auroc = 0.5
            
            metric_dict['auprc'] = auprc
            metric_dict['auroc'] = auroc
            
            # Determine Main Score
            if args.dataset in AUPRC_DATASETS:
                main_score = auprc
            else:
                main_score = acc
            
        metric_dict['train_time'] = train_time
        metric_dict['inference_time'] = inf_time
        scores.append(main_score)
        
        result_json = {
            "dataset": real_dataset_name,
            "model": args.model,
            "seed": seed,
            "metrics": metric_dict,
            "config": best_params
        }

        save_path = f"results/scores/{real_dataset_name}_{args.model}_seed{seed}.json"

        with open(save_path, "w", encoding='utf-8') as f:
            json.dump(result_json, f, indent=4)

    print(f"\n✅ Final Results ({real_dataset_name}): {np.mean(scores):.4f} ± {np.std(scores):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--task_type', type=str, default=None, choices=['regression', 'classification'])
    
    args = parser.parse_args()
    
    if args.tune: run_hpo(args)
    if args.eval: run_eval(args)