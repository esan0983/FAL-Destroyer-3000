# main/ml.py
# Runs three machine learning algortihms: Random Forest, XGBoost, and a PyTorch NN
# Most preprocessing protocols were already done in main/feature_engineering.py and utils/ml_utils.py

import os
os.environ["SCIPY_ARRAY_API"] = "1" 

import sklearn
sklearn.set_config(array_api_dispatch=True)

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    KFold,
    train_test_split
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, 
    ndcg_score
)
from scipy.stats import spearmanr


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import copy
import json
import shap
import torch

import random

import optuna
optuna.logging.set_verbosity(optuna.logging.INFO)

import warnings
warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")
warnings.filterwarnings("ignore", message=".*will be ignored.*")

import joblib

from pprint import pprint

from utils import (
    PytorchNN,
    pytorch_train_processing,
    pytorch_preprocessing,
    multivalue_preprocessing,
    encode_features
)

# Separate NDCG@K metrics for the RF model
def rf_ndcg(train_df, test_df, target):
    metrics = ['score', 'wc', 'favorites', 'dropped', 'forum']
    unwanted_metrics = [metric for metric in metrics if metric != target]

    train_df = train_df.drop(columns=unwanted_metrics, errors='ignore')
    test_df = test_df.drop(columns=unwanted_metrics, errors='ignore')

    X_train_full = train_df.drop(columns=[target], errors='ignore')
    
    X_test_full = test_df.drop(columns=[target], errors='ignore')

    rf_model = joblib.load(f"data/ml_predictions/models/rf_{target}_model.joblib")

    X_train_full_prep, X_test_full_prep = multivalue_preprocessing(X_train_full, X_test_full)
    X_train_full_proc, X_test_full_proc = encode_features(X_train_full_prep, X_test_full_prep)

    predicted = rf_model.predict(X_test_full_proc)
    actual = test_df[target]

    # Turn into 2D for NDCG scoring
    predicted_2d = [predicted]
    actual_2d = [actual.to_numpy()]

    score10 = ndcg_score(actual_2d, predicted_2d, k=10)
    score50 = ndcg_score(actual_2d, predicted_2d, k=50)

    print(f"NCDG@10: {score10:.4f}")
    print(f"NCDG@50: {score50:.4f}")


# Feature importance for the RF model
def rf_feature_importance(train_df, test_df, target):
    metrics = ['score', 'wc', 'favorites', 'dropped', 'forum']
    unwanted_metrics = [metric for metric in metrics if metric != target]

    train_df = train_df.drop(columns=unwanted_metrics, errors='ignore')
    test_df = test_df.drop(columns=unwanted_metrics, errors='ignore')

    X_train_full = train_df.drop(columns=[target], errors='ignore')
    
    X_test_full = test_df.drop(columns=[target], errors='ignore')

    rf_model = joblib.load(f"data/ml_predictions/models/rf_{target}_model.joblib")

    X_train_full_prep, X_test_full_prep = multivalue_preprocessing(X_train_full, X_test_full)
    X_train_full_proc, X_test_full_proc = encode_features(X_train_full_prep, X_test_full_prep)

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer(X_test_full_proc)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_full_proc, show=False)
    plt.title(f'Individual Feature Importance (SHAP) - Target: {target}', fontsize=14, pad=12)
    plt.tight_layout()

    output_dir = "data/ml_predictions/graphs"

    indiv_save_path = os.path.join(output_dir, f"rf_features_{target}.png") if len(X_test_full_proc) > 1 else os.path.join(output_dir, f"rf_features_{target}_specific.png")
    plt.savefig(indiv_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Individual feature plot saved to {indiv_save_path}")

# Using inference data and a specified metric to load the RF model and save predictions as JSON
def rf_inference(train_df, inference_df, target):
    metrics = ['score', 'wc', 'favorites', 'dropped', 'forum']
    unwanted_metrics = [metric for metric in metrics if metric != target]

    rf_model = joblib.load(f"data/ml_predictions/models/rf_{target}_model.joblib")

    train_df = train_df.drop(columns=unwanted_metrics)
    X_inf_full = inference_df.drop(columns=['title'])
    titles = inference_df['title']
    X_train_full = train_df.drop(columns=[target])

    X_train_full_prep, X_inf_full_prep = multivalue_preprocessing(X_train_full, inference_df)
    X_train_full_proc, X_inf_full_proc = encode_features(X_train_full_prep, X_inf_full_prep)

    inferences = rf_model.predict(X_inf_full_proc)

    inference_dict = {}
    
    for title in titles:
        inference_dict[title] = 0

    inference_dict = {title: inferences[idx].item() for idx, title in enumerate(titles)}
    inference_dict = dict(sorted(inference_dict.items(), key=lambda item: item[1], reverse=True))

    output_dir = "data/ml_predictions/roster"

    file_path = os.path.join(output_dir, f"{target}_predictions.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(inference_dict, f, indent=4)

    print(f"Predictions for {target} saved!")

# Runs a random forest model and returns test MSE
def random_forest(train_df, test_df, target, seed):
    metrics = ['score', 'wc', 'favorites', 'dropped', 'forum']
    unwanted_metrics = [metric for metric in metrics if metric != target]

    train_df = train_df.drop(columns=unwanted_metrics)
    test_df = test_df.drop(columns=unwanted_metrics)

    X_train_full = train_df.drop(columns=[target])
    y_train_full = train_df[target]
    
    X_test_full = test_df.drop(columns=[target])
    y_test_full = test_df[target]

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    cached_folds = []

    print("Caching folds...")
    for train_idx, val_idx in kf.split(X_train_full, y_train_full):
        X_tr, X_va = X_train_full.iloc[train_idx].copy(), X_train_full.iloc[val_idx].copy()
        y_tr, y_va = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        X_tr_proc, X_va_proc = multivalue_preprocessing(X_tr, X_va)
        X_tr_proc, X_va_proc = encode_features(X_tr_proc, X_va_proc)

        cached_folds.append((
            X_tr_proc.to_numpy(),
            y_tr.to_numpy(),
            X_va_proc.to_numpy(),
            y_va.to_numpy()
        ))

    def objective(trial):

        max_features_type = trial.suggest_categorical('max_features_type', ['sqrt', 'float'])

        if max_features_type == 'sqrt':
            max_features = 'sqrt'
        else:
            max_features = trial.suggest_float('max_features', 0.1, 0.5)
            
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 400),
            'max_features': max_features,
            'max_depth': trial.suggest_int('max_depth', 5, 25),
            'min_samples_split': trial.suggest_int('min_samples_split', 4, 8),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 6),
            'criterion': 'squared_error', # Scikit-learn optimizes squared error directly
            'random_state': seed,
            'n_jobs': -1
        }

        fold_rmses = []

        for fold_idx, (X_tr_proc, y_tr, X_va_proc, y_va) in enumerate(cached_folds):
            rf_model = RandomForestRegressor(**params)
            rf_model.fit(X_tr_proc, y_tr)

            preds = rf_model.predict(X_va_proc)
            rmse = np.sqrt(mean_squared_error(y_va, preds))

            fold_rmses.append(rmse)

            trial.report(np.mean(fold_rmses), fold_idx)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(fold_rmses)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=10, seed=seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=0     
        )
    )
    study.optimize(objective, n_trials=50, n_jobs=1)

    best_params = study.best_params
    print(f"Best CV RMSE: {study.best_value:.4f}")
    print("Best Hyperparameters:", best_params)

    X_train_full_prep, X_test_full_prep = multivalue_preprocessing(X_train_full, X_test_full)
    X_train_full_proc, X_test_proc = encode_features(X_train_full_prep, X_test_full_prep)

    X_train_final_proc, X_val_final_proc, y_train_final, y_val_final = train_test_split(
        X_train_full_proc, y_train_full, test_size=0.1875, random_state=seed
    )

    max_trees = best_params.pop("n_estimators", 750) 

    best_rf_model = RandomForestRegressor(
        **best_params,
        n_estimators=0,
        warm_start=True,
        random_state=seed,
        n_jobs=-1
    )

    patience = 10
    best_val_rmse = float('inf')
    no_improvement_count = 0
    step_size = 5 
    final_model_checkpoint = None

    print("Training final model with early stopping...")
    for n_trees in range(step_size, max_trees + 1, step_size):
        best_rf_model.n_estimators = n_trees
        best_rf_model.fit(X_train_final_proc, y_train_final)
        
        val_preds = best_rf_model.predict(X_val_final_proc)
        val_rmse = np.sqrt(mean_squared_error(y_val_final, val_preds))
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            no_improvement_count = 0
            final_model_checkpoint = copy.deepcopy(best_rf_model)
        else:
            no_improvement_count += 1
            
        if no_improvement_count >= patience:
            print(f"Early stopping triggered at {n_trees} trees.")
            break
    else:
        print(f"Completed training up to maximum limit of {max_trees} trees.")
        if final_model_checkpoint is None:
            final_model_checkpoint = best_rf_model

    print("Saving model...")
    joblib.dump(final_model_checkpoint, f"data/ml_predictions/models/rf_{target}_model.joblib", compress=3)

    y_pred = final_model_checkpoint.predict(X_test_proc)
    mse = mean_squared_error(y_test_full, y_pred)
    spearman, _ = spearmanr(y_test_full, y_pred)
    spearman = spearman if not np.isnan(spearman) else 0.0

    print(f"Target {target} -> Test MSE: {mse:.4f} | Test Spearman: {spearman:.4f}")
    return mse, spearman

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df = pd.read_parquet("data/ml_data/train_df.parquet")
    test_df = pd.read_parquet("data/ml_data/test_df.parquet")
    inference_df = pd.read_parquet("data/ml_data/inference_df.parquet")

    rf_mses = {}
    rf_spearman = {}
    metrics = ['score', 'wc', 'favorites', 'dropped', 'forum']

    rf_mse_path = Path("data/ml_predictions/models/rf_mses.json")
    rf_spearman_path = Path("data/ml_predictions/models/rf_spearman.json")

    for metric in metrics:
        rf_spearman[metric] = []
        rf_mses[metric] = []

    for seed in [42]:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Current seed: {seed}")

        for metric in metrics:
            print(f"Metric: {metric}")
            # mse, spearman = random_forest(train_df, test_df, metric, seed)
            # rf_mses[metric].append(mse)
            # rf_spearman[metric].append(spearman)

            # rf_feature_importance(train_df, test_df, metric)
            # rf_inference(train_df, inference_df, metric)

            rf_ndcg(train_df, test_df, metric)

        # pprint(rf_mses, indent=4)
        # pprint(rf_spearman, indent=4)


