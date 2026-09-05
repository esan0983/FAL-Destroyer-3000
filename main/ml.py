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
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    ClassNamePrefixFeaturesOutMixin
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    KFold,
    cross_val_score,
    train_test_split
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, 
    r2_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import copy
from pprint import pprint
import json
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBRegressor
import shap
import seaborn as sns

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from pprint import pprint

import random

import optuna
optuna.logging.set_verbosity(optuna.logging.INFO)

import warnings
warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")
warnings.filterwarnings("ignore", message=".*will be ignored.*")

import joblib

from utils import (
    PytorchNN,
    pytorch_train_processing,
    pytorch_preprocessing,
    multivalue_preprocessing,
    encode_features
)


# Creates both individual and group feature importance charts
def feature_importance(test_df, target):
    metrics = ['score', 'wc', 'favorites', 'dropped', 'forum']
    unwanted_metrics = [metric for metric in metrics if metric != target]   

    target_df = test_df.drop(columns=unwanted_metrics)

    X_test = target_df.drop(columns=[target])

    output_dir = "data/ml_predictions/models"
    model_path = os.path.join(output_dir, f"xgb_{target}.ubj")

    test_model = xgb.XGBRegressor()
    test_model.load_model(model_path)

    explainer = shap.TreeExplainer(test_model)
    shap_values = explainer.shap_values(X_test)

    df_shap = pd.DataFrame(abs(shap_values), columns=X_test.columns)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f'Individual Feature Importance (SHAP) - Target: {target}', fontsize=14, pad=12)
    plt.tight_layout()

    output_dir = "data/ml_predictions/graphs"

    indiv_save_path = os.path.join(output_dir, f"feature_importance_individual_{target}.png")
    plt.savefig(indiv_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Individual feature plot saved to {indiv_save_path}")

    group_dict = {
        "source": ["source"],
        "rating": ["rating"],
        "prequel_data": ["prequel_score", "prequel_wc", "prequel_favorites", "prequel_dropped", "prequel_forum",
                        "prequel_year", "prequel_season"],
        "adaptation_data": ["adaptation_score", "adaptation_members"],
        "season": ["season"],
        "year": ["year"],
        "genre": [f"genre_svd_{i}" for i in range(5)],
        "theme": [f"theme_svd_{i}" for i in range(7)],
        "demographic": ["Josei", "Shounen", "Shoujo", "Seinen", "Kids"]
    }

    grouped_shap = pd.DataFrame()
    for group_name, features in group_dict.items():
        grouped_shap[group_name] = df_shap[features].sum(axis=1)

    mean_group_importance = grouped_shap.mean().reset_index()
    mean_group_importance.columns = ['Group', 'SHAP_Importance']
    mean_group_importance = mean_group_importance.sort_values(by='SHAP_Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=mean_group_importance,
        x='SHAP_Importance',
        y='Group',
        palette='viridis'
    )
    plt.title(f'Grouped Feature Importance (SHAP) - Metric: {target}', fontsize=14, pad=12)
    plt.xlabel('Mean |SHAP Value|', fontsize=12)
    plt.ylabel('Feature Group', fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"feature_importance_{target}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

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
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 400),
            'max_features': trial.suggest_float('max_features', 0.1, 0.5),
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
    r2 = r2_score(y_test_full, y_pred)

    print(f"Target {target} -> Test MSE: {mse:.4f} | Test R^2: {r2:.4f}")
    return mse, r2

# Uses inference data and a specified metric to load the XGBoost model and save predictions as JSON
def xgboost_infer(inference_df, target):
    X_infer = inference_df.drop(columns=['title'])
    titles = inference_df['title']

    output_dir = "data/ml_predictions/models"
    model_path = os.path.join(output_dir, f"xgb_{target}.ubj")

    inference_model = xgb.XGBRegressor()
    inference_model.load_model(model_path)

    inferences = inference_model.predict(X_infer)

    inference_dict = {}

    for title in titles:
        inference_dict[title] = 0

    inference_dict = {title: inferences[idx].item() for idx, title in enumerate(titles)}
    inference_dict = dict(sorted(inference_dict.items(), key=lambda item: item[1], reverse=True))

    file_path = os.path.join(output_dir, f"{target}_predictions.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(inference_dict, f, indent=4)

    print(f"Predictions for {target} saved!")

# Takes loaded XGB model and returns R^2 score from test data without having to run XGBoost again
def xgboost_r2(test_df, target):
    metrics = ['score', 'wc', 'favorites', 'dropped', 'forum']
    unwanted_metrics = [metric for metric in metrics if metric != target]   

    target_df = test_df.drop(columns=unwanted_metrics)

    X_test = target_df.drop(columns=[target])
    y_test = target_df[target]

    output_dir = "data/ml_predictions/models"
    model_path = os.path.join(output_dir, f"xgb_{target}.ubj")

    test_model = xgb.XGBRegressor()
    test_model.load_model(model_path)

    y_pred = test_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"Correlation score for {target}: {r2}")

# Runs an XGBoost model, saves the best model parameters, and returns test MSE
def xgboost(train_df, test_df, target, seed):
    metrics = ['score', 'wc', 'favorites', 'dropped', 'forum']
    unwanted_metrics = [metric for metric in metrics if metric != target]

    train_df = train_df.drop(columns=unwanted_metrics)
    test_df = test_df.drop(columns=unwanted_metrics)

    X_train_full = train_df.drop(columns=[target])
    y_train_full = train_df[target]
    
    X_test_full = test_df.drop(columns=[target])
    y_test_full = test_df[target]

    cat_cols = ['source', 'rating', 'season', 'prequel_season']
    for col in cat_cols:
        if col in X_train_full.columns:
            X_train_full[col] = X_train_full[col].astype('category')
        if col in X_test_full.columns:
            X_test_full[col] = X_test_full[col].astype('category')

    print("Pre-processing and caching CV folds...")
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    cached_folds = []

    for train_idx, val_idx in kf.split(X_train_full, y_train_full):
        X_tr, X_va = X_train_full.iloc[train_idx].copy(), X_train_full.iloc[val_idx].copy()
        y_tr, y_va = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        X_tr_proc, X_va_proc = multivalue_preprocessing(X_tr, X_va)
        cached_folds.append((X_tr_proc, y_tr, X_va_proc, y_va))

    def objective(trial):
        params = {
            'n_estimators': 3000,
            'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int("max_depth", 3, 8),
            'min_child_weight': trial.suggest_int("min_child_weight", 1, 20),
            'subsample': trial.suggest_float("subsample", 0.5, 1.0),
            'colsample_bytree': trial.suggest_float("colsample_bytree", 0.4, 1.0),
            'gamma': trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'tree_method': "hist",
            'eval_metric': "rmse",
            'device': "cuda",
            'enable_categorical': True,
            'random_state': seed
        }

        fold_scores = []

        for fold_idx, (X_tr, y_tr, X_va, y_va) in enumerate(cached_folds):
            model = XGBRegressor(**params, early_stopping_rounds=50)

            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False
            )

            preds = model.predict(X_va)
            fold_rmse = np.sqrt(mean_squared_error(y_va, preds))
            fold_scores.append(fold_rmse)
            current_mean_rmse = np.mean(fold_scores)
            trial.report(current_mean_rmse, step=fold_idx)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(fold_scores)

    print("Starting Optuna hyperparameter search...")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=10, seed=seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=0     
        )
    )
    study.optimize(objective, n_trials=50)

    print(f"\nBest Fold RMSE: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    best_params = study.best_params
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    X_train_full_proc, X_test_proc = multivalue_preprocessing(X_train_full, X_test_full)

    X_train_final_proc, X_val_final_proc, y_train_final, y_val_final = train_test_split(
        X_train_full_proc, y_train_full, test_size=0.1875, random_state=seed
    )

    final_model_params = {
        **best_params,
        'tree_method': "hist",
        'device': "cuda",
        'enable_categorical': True,
        'random_state': seed,
        'eval_metric': "rmse"
    }

    best_xgb_model = XGBRegressor(**final_model_params, early_stopping_rounds=50)
    best_xgb_model.fit(
        X_train_final_proc,
        y_train_final,
        eval_set=[(X_val_final_proc, y_val_final)],
        verbose=100
    )

    X_test_proc = X_test_proc[X_train_final_proc.columns]

    y_pred = best_xgb_model.predict(X_test_proc)
    mse = mean_squared_error(y_test_full, y_pred)
    r2 = r2_score(y_test_full, y_pred)

    output_dir = "data/ml_predictions"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"xgb_{target}.ubj")
    best_xgb_model.save_model(model_path)

    print(f"\nTarget {target} -> Final Test MSE: {mse:.4f} | Final Test R^2: {r2:.4f}")
    return mse, r2

# Runs a custom PyTorch model and returns test MSE
# All data went through preprocessing
# During cross-validation, training and validation data were processed to avoid data leakage. See utils/ml_utils.py for more details
def pytorch_nn(train_df, val_df, test_df, target, seed):
    # NOTE: I WILL ONLY PASS AVERAGE VALIDATION LOSS FOR NOW, BUT THE PLAN IS TO RETRAIN THE ENTIRE MODEL ON THE AVERAGE
    # NUMBER OF EPOCHS AND THEN TEST IT
    metrics = ['score', 'wc', 'favorites', 'dropped', 'forum']
    unwanted_metrics = [metric for metric in metrics if metric != target]

    train_df = train_df.drop(columns=unwanted_metrics)
    val_df = val_df.drop(columns=unwanted_metrics)
    test_df = test_df.drop(columns=unwanted_metrics)

    train_df, val_df, test_df = pytorch_preprocessing(train_df, val_df, test_df)
    train_combined = pd.concat([train_df, val_df], axis=0)

    X_train = train_combined.drop(columns=[target])
    y_train = train_combined[target].to_numpy()
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target].to_numpy()

    num_epochs = 1_000_000 # rely on early stopping
    kfold = KFold(n_splits=8, shuffle=True, random_state=seed)
    batch_size = 64

    best_fold_loss = float("inf")
    best_params = {}

    for fold, (train_ids, val_ids) in enumerate(kfold.split(X_train)):
        print(f"FOLD {fold + 1}")

        param_dist = {
            "hidden_dim": random.choice([128, 256, 512]),
            "lr": random.choice([5e-5, 3e-5, 1e-5]),
            "weight_decay": random.choice([1e-3, 1e-4, 1e-5])
        }

        X_cv_train = X_train.iloc[train_ids].copy()
        X_cv_val = X_train.iloc[val_ids].copy()

        y_cv_train = y_train[train_ids]
        y_cv_val = y_train[val_ids]

        X_cv_train, X_cv_val = pytorch_train_processing(X_cv_train, X_cv_val)

        X_cv_train = torch.tensor(X_cv_train.to_numpy(), dtype=torch.float32).to(device)
        X_cv_val = torch.tensor(X_cv_val.to_numpy(), dtype=torch.float32).to(device)
        y_cv_train = torch.tensor(y_cv_train, dtype=torch.float32).unsqueeze(1).to(device)
        y_cv_val = torch.tensor(y_cv_val, dtype=torch.float32).unsqueeze(1).to(device)

        train_dataset = TensorDataset(X_cv_train, y_cv_train)
        val_dataset = TensorDataset(X_cv_val, y_cv_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model = PytorchNN(X_cv_train.shape[1], param_dist.get("hidden_dim", 128)).to(device)
        criterion = nn.MSELoss()
        optimizer = AdamW(
            model.parameters(),
            lr=param_dist.get("lr", 5e-5),
            weight_decay=param_dist.get("weight_decay", 1e-4)
        )

        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0)

        best_val_loss = float('inf')
        best_epoch = num_epochs
        patience_counter = 0
        patience = 300
        val_loss_arr = np.full(10, np.nan)

        for epoch in range(num_epochs):
            model.train()
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item() * inputs.size(0)

            val_loss /= len(val_loader.dataset)

            if epoch < 10:
                val_loss_arr[epoch] = val_loss
            else:
                for i in range(9):
                    val_loss_arr[i] = val_loss_arr[i + 1]
                val_loss_arr[9] = val_loss

            moving_avg_val_loss = np.nanmean(val_loss_arr)

            if moving_avg_val_loss <= best_val_loss:
                best_val_loss = moving_avg_val_loss
                print(f"Epoch {epoch} moving average validation loss: {moving_avg_val_loss}")
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    print(f"Best moving average validation loss: {best_val_loss}")
                    print(f"Best epoch: {best_epoch}")

                    if best_val_loss < best_fold_loss:
                        best_fold_loss = best_val_loss
                        best_params = param_dist

                    break

    print(f"Best fold validation loss: {best_fold_loss}")
    print("Best fold parameters:")
    pprint(best_params, indent=4)

    y_train_final = train_df[target]
    y_val_final = val_df[target]
    y_test_final = test_df[target]

    X_train_final, X_val_final = pytorch_train_processing(train_df.drop(columns=[target]), val_df.drop(columns=[target]))
    X_train_final, X_test_final = pytorch_train_processing(train_df.drop(columns=[target]), test_df.drop(columns=[target]))

    X_train_final = torch.tensor(X_train_final.to_numpy(), dtype=torch.float32).to(device)
    y_train_final = torch.tensor(y_train_final.to_numpy(), dtype=torch.float32).unsqueeze(1).to(device)
    X_val_final = torch.tensor(X_val_final.to_numpy(), dtype=torch.float32).to(device)
    y_val_final = torch.tensor(y_val_final.to_numpy(), dtype=torch.float32).unsqueeze(1).to(device)
    X_test_final = torch.tensor(X_test_final.to_numpy(), dtype=torch.float32).to(device)
    y_test_final = torch.tensor(y_test_final.to_numpy(), dtype=torch.float32).unsqueeze(1).to(device)

    train_dataset = TensorDataset(X_train_final, y_train_final)
    val_dataset = TensorDataset(X_val_final, y_val_final)
    test_dataset = TensorDataset(X_test_final, y_test_final)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    final_model = PytorchNN(X_train_final.shape[1], best_params.get("hidden_dim", 128)).to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(
        final_model.parameters(),
        lr=best_params.get("lr", 5e-5),
        weight_decay=best_params.get("weight_decay", 1e-4)
    )

    MODEL_PATH = f"data/ml_predictions/final_model_{target}.pth"

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0)

    best_val_loss = float('inf')
    best_epoch = num_epochs
    patience_counter = 0
    patience = 300
    val_loss_arr = np.full(10, np.nan)

    for epoch in range(num_epochs):
        final_model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = final_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

        final_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = final_model(inputs)
                val_loss += criterion(outputs, targets).item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)

        if epoch < 10:
            val_loss_arr[epoch] = val_loss
        else:
            for i in range(9):
                val_loss_arr[i] = val_loss_arr[i + 1]
            val_loss_arr[9] = val_loss

        moving_avg_val_loss = np.nanmean(val_loss_arr)

        if moving_avg_val_loss <= best_val_loss:
            best_val_loss = moving_avg_val_loss
            print(f"Epoch {epoch} moving average validation loss: {moving_avg_val_loss}")
            best_epoch = epoch
            patience_counter = 0
            torch.save(final_model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"Best moving average validation loss: {best_val_loss}")
                print(f"Best epoch: {best_epoch}")
                break

    test_model = PytorchNN(X_train_final.shape[1], best_params.get("hidden_dim", 128)).to(device)
    state_dict = torch.load(MODEL_PATH, weights_only=True, map_location=device)
    test_model.load_state_dict(state_dict)
    test_model.eval()
    test_mse = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = test_model(inputs)
            loss = criterion(outputs, targets)
            test_mse += loss.item() * inputs.size(0)

    test_mse /= len(test_loader.dataset)
    print(f"Test MSE: {test_mse}")

    return test_mse


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df = pd.read_parquet("data/ml_data/train_df.parquet")
    test_df = pd.read_parquet("data/ml_data/test_df.parquet")
    inference_df = pd.read_parquet("data/ml_data/inference_df.parquet")

    rf_mses = {}
    rf_r2s = {}
    xgb_mses = {}
    xgb_r2s = {}

    metrics = ['score', 'wc', 'favorites', 'dropped', 'forum']

    xgb_mse_path = Path("data/ml_predictions/models/xgb_mses.json")
    xgb_r2_path = Path("data/ml_predictions/models/xgb_r2s.json")
    rf_mse_path = Path("data/ml_predictions/models/rf_mses.json")
    rf_r2_path = Path("data/ml_predictions/models/rf_r2s.json")

    for metric in metrics:
        xgb_r2s[metric] = []
        xgb_mses[metric] = []
        rf_r2s[metric] = []
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
            # mse, r2 = random_forest(train_df, test_df, metric, seed)
            # print("Calculating feature importance...")
            # rf_feature_importance(train_df, test_df, metric)
            # print("Saving predictions...")
            # rf_inference(train_df, inference_df, metric)

            psyren_df = inference_df[inference_df['title'] == "Psyren"].drop(columns=['title'])
            rf_feature_importance(train_df, psyren_df, metric)

            # with open(rf_mse_path, "w", encoding="utf-8") as f:
            #     json.dump(rf_mses, f, indent=4)

            # with open(rf_r2_path, "w", encoding="utf-8") as f:
            #     json.dump(rf_r2s, f, indent=4)

            # mse, r2 = xgboost(train_df, test_df, metric, seed)
            # xgb_mses[metric].append(mse)
            # xgb_r2s[metric].append(r2)

            # with open(xgb_mse_path, "w", encoding="utf-8") as f:
            #     json.dump(xgb_mses, f, indent=4)

            # with open(xgb_r2_path, "w", encoding="utf-8") as f:
            #     json.dump(xgb_r2s, f, indent=4)

