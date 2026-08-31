# main/ml.py
# Runs two machine learning algortihms: Random Forest and XGBoost. Most preprocessing protocols were already down in main/feature_engineering.py.

import os
os.environ["SCIPY_ARRAY_API"] = "1" 

import sklearn
sklearn.set_config(array_api_dispatch=True)

import pandas as pd
import numpy as np
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    ClassNamePrefixFeaturesOutMixin
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    KFold
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, 
    r2_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import matplotlib.pyplot as plt
import copy
from pprint import pprint
import json
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBRegressor

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.optim import AdamW

from utils import (
    PytorchNN,
    pytorch_train_processing,
    pytorch_preprocessing
)


# REDO
def random_forest(train_df, val_df, test_df, target):
    metrics = ['score', 'wc', 'favorites', 'dropped', 'forum']
    unwanted_metrics = [metric for metric in metrics if metric != target]

    train_df = train_df.drop(columns=unwanted_metrics)
    val_df = val_df.drop(columns=unwanted_metrics)
    test_df = test_df.drop(columns=unwanted_metrics)

    full_train_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

    combined = pd.concat([full_train_df, test_df], axis=0, ignore_index=True)
    combined = pd.get_dummies(combined, columns=['rating', 'source', 'season', 'prequel_season'], dtype=int)

    full_train_df = combined.iloc[:len(full_train_df)].copy()
    test_df = combined.iloc[len(full_train_df):].copy()

    param_distributions = {
        'n_estimators':[150, 250, 400],
        'max_features': ['sqrt', 0.2, 0.3, 0.4],            
        'max_depth': [10, 15, 20, 25, None],                 
        'min_samples_split':[4, 6, 8],                  
        'min_samples_leaf': [2, 3, 4, 6]                     
    }

    pipe = Pipeline([ # WILL ADD MORE IF NEEDED
        ("model", RandomForestRegressor(random_state=42))
    ])

    rf_random_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions={f"model__{k}": v for k, v in param_distributions.items()},
        n_iter=50,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    X_train = full_train_df.drop(columns=[target])
    X_test = test_df.drop(columns=[target])
    y_train = full_train_df[target]
    y_test = test_df[target]

    print("\nStarting hyperparameter tuning...")
    rf_random_search.fit(X_train, y_train)

    best_params = rf_random_search.best_params_
    print("--- Tuning Complete ---")
    print("Best Hyperparameters Found:", best_params)

    best_rf_model = rf_random_search.best_estimator_
    y_pred = best_rf_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Target {target} -> MSE: {mse:.4f} | R² Score: {r2:.4f}")
    return mse, r2

def xgboost(train_df, val_df, test_df, target):
    metrics = ['score', 'wc', 'favorites', 'dropped', 'forum']
    unwanted_metrics = [metric for metric in metrics if metric != target]

    drop_cols = unwanted_metrics
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
    # Note that this is not as robust since there could be something in the test data that's not in training data.
    # For now, this works as is. In the future, try to make this more robust.
    for col in ['source', 'rating', 'season', 'prequel_season']:
        if col in X_search.columns:
            X_search[col] = X_search[col].astype('category')
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('category')
        if col in X_val.columns:
            X_val[col] = X_val[col].astype('category')
        if col in X_test.columns:
            X_test[col] = X_test[col].astype('category')

    pipe = Pipeline([ # WILL ADD MORE IF NEEDED
        ("model", XGBRegressor(
            device="cuda",
            random_state=42,
            eval_metric='rmse',
            enable_categorical=True 
        ))
    ])

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
        estimator=pipe,
        param_distributions={f"model__{k}": v for k, v in param_distributions.items()},
        n_iter=50,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=3,
        random_state=42,
        n_jobs=1
    )

    xgb_random_search.fit(X_search, y_search)

    best_params = {k.split("__")[1]: v for k, v in xgb_random_search.best_params_.items()}
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

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Target {target} -> MSE: {mse:.4f} | R² Score: {r2:.4f}")

    return mse, r2

def pytorch_nn(train_df, val_df, test_df, target):
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

    num_epochs = 300
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    batch_size = 32
    cv_scores = np.zeros(5)

    for fold, (train_ids, val_ids) in enumerate(kfold.split(X_train)):
        print(f"FOLD {fold + 1}")

        X_cv_train = X_train.iloc[train_ids].copy()
        X_cv_val = X_train.iloc[val_ids].copy()

        y_cv_train = y_train[train_ids]
        y_cv_val = y_train[val_ids]

        X_cv_train, X_cv_val = pytorch_train_processing(X_cv_train, X_cv_val)

        X_cv_train = torch.tensor(X_cv_train.to_numpy(), dtype=torch.float32)
        X_cv_val = torch.tensor(X_cv_val.to_numpy(), dtype=torch.float32)
        y_cv_train = torch.tensor(y_cv_train, dtype=torch.float32).unsqueeze(1)
        y_cv_val = torch.tensor(y_cv_val, dtype=torch.float32).unsqueeze(1)

        train_dataset = TensorDataset(X_cv_train, y_cv_train)
        val_dataset = TensorDataset(X_cv_val, y_cv_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model = PytorchNN(X_cv_train.shape[1])
        criterion = nn.MSELoss()
        optimizer = AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-2
        )

        best_val_loss = float('inf')
        best_model_weights = None
        patience_counter = 0
        patience = 20

        for epoch in range(num_epochs):
            model.train()
            for inputs, targets in train_loader:
                targets = targets
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item() * inputs.size(0)

            val_loss /= len(val_loader.dataset)

            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                print(f"Epoch {epoch} validation loss: {val_loss}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    print(f"Best validation loss: {best_val_loss}")
                    cv_scores[fold] = best_val_loss
                    break

    return np.mean(cv_scores)


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

    rf_mses = []
    rf_r2s = []

    xgb_mses = []
    xgb_r2s = []

    pt_mses = []

    metrics = ['score', 'wc', 'favorites', 'dropped', 'forum']
    for metric in metrics:
        mse, r2 = random_forest(train_df, val_df, test_df, metric)
        rf_mses.append(mse)

        mse, r2 = xgboost(train_df, val_df, test_df, metric)
        xgb_mses.append(mse)

        mse = pytorch_nn(train_df, val_df, test_df, metric)
        pt_mses.append(mse)

    print("Random Forest MSEs:")
    print(rf_mses)
    print("XGBoost MSEs:")
    print(xgb_mses)
    print("PyTorch MSEs:")
    print(pt_mses)