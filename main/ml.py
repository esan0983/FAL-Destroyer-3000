import os
os.environ["SCIPY_ARRAY_API"] = "1" 

import sklearn
sklearn.set_config(array_api_dispatch=True)

import cupy as cp
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    PredefinedSplit,
    RandomizedSearchCV
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, 
    r2_score
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from scipy.stats import (
    uniform, 
    randint,
    loguniform
)
import matplotlib.pyplot as plt
import copy
from pprint import pprint
import json
from statsmodels.stats.outliers_influence import variance_inflation_factor
import torch
from xgboost import XGBRegressor

mses = []
r2s = []

def safe_boolean_vif(df):
    zero_variance_cols = df.columns[df.nunique() <= 1].tolist()
    if zero_variance_cols:
        print(f"Dropped zero-variance columns immediately: {zero_variance_cols}")
        df = df.drop(columns=zero_variance_cols)
        
    X = df.copy()
    X['constant'] = 1.0
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    
    vifs = []
    for i in range(len(X.columns)):
        try:
            val = variance_inflation_factor(X.values, i)
            vifs.append(np.inf if np.isinf(val) or np.isnan(val) else val)
        except ZeroDivisionError:
            vifs.append(np.inf)
            
    vif_data["VIF"] = vifs
    
    return vif_data[vif_data["feature"] != 'constant'].reset_index(drop=True)

def random_forest(train_df, val_df, test_df, target):
    metrics = ['score_z', 'wc_z', 'favorites_z', 'dropped_z', 'forum_z']
    unwanted_metrics = [metric for metric in metrics if metric != target]
    unwanted_genres = [f"genre_{metric}_mean" for metric in unwanted_metrics] + [f"genre_{metric}_min" for metric in unwanted_metrics] + [f"genre_{metric}_max" for metric in unwanted_metrics]
    unwanted_themes = [f"theme_{metric}_mean" for metric in unwanted_metrics] + [f"theme_{metric}_min" for metric in unwanted_metrics] + [f"theme_{metric}_max" for metric in unwanted_metrics]

    train_df = train_df.drop(columns=unwanted_metrics+unwanted_genres+unwanted_themes)
    val_df = val_df.drop(columns=unwanted_metrics+unwanted_genres+unwanted_themes)
    test_df = test_df.drop(columns=unwanted_metrics+unwanted_genres+unwanted_themes)

    full_train_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    full_train_df = pd.get_dummies(full_train_df, columns=['rating', 'source'])
    test_df = pd.get_dummies(test_df, columns=['rating', 'source'])

    param_distributions = {
        'n_estimators':[150, 250, 400],
        'max_features': ['sqrt', 0.2, 0.3, 0.4],            
        'max_depth': [10, 15, 20, 25, None],                 
        'min_samples_split':[4, 6, 8],                  
        'min_samples_leaf': [2, 3, 4, 6]                     
    }

    rf_random_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions=param_distributions,
        n_iter=50,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    X_train = full_train_df.drop(columns=[target])
    X_test = test_df.drop(columns=[target])
    y_train = full_train_df[target]
    y_test = test_df[target]

    print("\nStarting hyperparameter tuning...")
    rf_random_search.fit(X_train, y_train)

    print("--- Tuning Complete ---")
    print("Best Hyperparameters Found:", rf_random_search.best_params_)
    best_rf_model = rf_random_search.best_estimator_
    y_pred = best_rf_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Target {target} -> MSE: {mse:.4f} | R² Score: {r2:.4f}")
    mses.append(mse)
    r2s.append(r2)

def xgboost(train_df, val_df, test_df, target):
    metrics = ['score_z', 'wc_z', 'favorites_z', 'dropped_z', 'forum_z']
    unwanted_metrics = [metric for metric in metrics if metric != target]
    unwanted_genres = [f"genre_{metric}_mean" for metric in unwanted_metrics] + [f"genre_{metric}_min" for metric in unwanted_metrics] + [f"genre_{metric}_max" for metric in unwanted_metrics]
    unwanted_themes = [f"theme_{metric}_mean" for metric in unwanted_metrics] + [f"theme_{metric}_min" for metric in unwanted_metrics] + [f"theme_{metric}_max" for metric in unwanted_metrics]

    drop_cols = unwanted_metrics + unwanted_genres + unwanted_themes
    train_df = train_df.drop(columns=drop_cols)
    val_df = val_df.drop(columns=drop_cols)
    test_df = test_df.drop(columns=drop_cols)

    X_train = train_df.drop(columns=[target])
    X_val = val_df.drop(columns=[target])
    X_test = test_df.drop(columns=[target])

    y_train = train_df[target]
    y_val = val_df[target]
    y_test = test_df[target] 

    X_search = pd.concat([X_train, X_val], axis=0)
    y_search = pd.concat([y_train, y_val], axis=0)

    # Contatenation flattens categories to strings
    for col in ['source', 'rating']:
        if col in X_search.columns:
            X_search[col] = X_search[col].astype('category')
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('category')
        if col in X_val.columns:
            X_val[col] = X_val[col].astype('category')
        if col in X_test.columns:
            X_test[col] = X_test[col].astype('category')

    param_distributions = {
        'n_estimators':[500, 1000, 1500, 2000, 3000],
        'learning_rate': [0.01, 0.02, 0.05, 0.1],
        'max_depth': [2, 3, 4, 5, 6, 8],
        'min_child_weight': [1, 3, 5, 7, 10],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.01, 0.1, 0.5, 1],
        'reg_alpha': [0, 0.01, 0.1, 1, 10],
        'reg_lambda': [0.1, 1, 5, 10, 20],
        'tree_method': ['hist'],
    }

    xgb_random_search = RandomizedSearchCV(
        estimator=XGBRegressor(
            device="cuda",
            random_state=42,
            eval_metric='rmse',
            enable_categorical=True 
        ),
        param_distributions=param_distributions,
        n_iter=50,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=2,
        random_state=42,
        n_jobs=1
    )

    xgb_random_search.fit(X_search, y_search)

    best_params = xgb_random_search.best_params_
    print("\n--- Tuning Complete ---")
    print("Best Hyperparameters Found:", best_params)

    best_xgb_model = XGBRegressor(
        **best_params,
        device="cuda",
        random_state=42,
        eval_metric='rmse',
        enable_categorical=True,  
        early_stopping_rounds=15
    )

    best_xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    y_pred = best_xgb_model.predict(X_test)

    mse_per_target = mean_squared_error(y_test, y_pred)
    r2_per_target = r2_score(y_test, y_pred)

    print(f"Target {target} -> MSE: {mse_per_target:.4f} | R² Score: {r2_per_target:.4f}")
    mses.append(mse)
    r2s.append(r2)
    

if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        print("torch cuda is available!")
        torch.cuda.manual_seed_all(SEED)

    train_df = pd.read_parquet("data/ml_data/train_df.parquet")
    val_df = pd.read_parquet("data/ml_data/val_df.parquet")
    test_df = pd.read_parquet("data/ml_data/test_df.parquet")
    inference_df = pd.read_parquet("data/ml_data/inference_df.parquet")

    metrics = ['score_z', 'wc_z', 'favorites_z', 'dropped_z', 'forum_z']
    for metric in metrics:
        random_forest(train_df, val_df, test_df, metric)
        xgboost(train_df, val_df, test_df, metric)

    print("MSEs:")
    print(mses)
    print("R2s:")
    print(r2s)