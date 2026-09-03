# utils/ml_utils.py
# Functions that are used in the machine learning phase but are not the models themselves

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import shap
import os
import xgboost as xgb

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline 

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Not used
# Performs VIF test for multicollinearity
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

# Not used
# Class used for target encoding whenever cross-validation splits the PyTorch data
class MultiLabelTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, list_col, smoothing=10):
        self.list_col = list_col
        self.smoothing = smoothing

    def fit(self, X, y):
        exploded = pd.DataFrame({self.list_col: X[self.list_col], "_y": np.asarray(y)}).explode(self.list_col)
        stats = exploded.groupby(self.list_col)["_y"].agg(["mean", "count"])
        self.global_mean_ = float(np.mean(y))

        stats["smoothed"] = (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_) / (stats["count"] + self.smoothing)
        self.category_map_ = stats["smoothed"].to_dict()
        return self

    def transform(self, X):
        def agg(lst):
            if len(lst) == 0:
                return (self.global_mean_,) * 3
            vals = [self.category_map_.get(v, self.global_mean_) for v in lst]
            return (np.mean(vals), np.min(vals), np.max(vals))
        out = X[self.list_col].apply(agg)
        return pd.DataFrame(out.tolist(), columns=[f"{self.list_col}_mean", f"{self.list_col}_min", f"{self.list_col}_max"], index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{self.list_col}_mean", f"{self.list_col}_min", f"{self.list_col}_max"], dtype=object)

# Preprocesses the entire dataset to follow PyTorch NN assumptions
def pytorch_preprocessing(train_df, val_df, test_df):
    categoricals = ['source', 'rating', 'season', 'prequel_season']
    for df in [train_df, val_df, test_df]:
        # removing category type
        df[categoricals] = str(df[categoricals])

        # deal with adaptation NaNs
        df['has_adaptation_score'] = df['adaptation_score'].notna().astype(int)
        df['has_adaptation_members'] = df['adaptation_members'].notna().astype(int)

        # deal with prequel NaNs
        df['has_prequel_score'] = df['prequel_score'].notna().astype(int)
        df['has_prequel_wc'] = df['prequel_wc'].notna().astype(int)
        df['has_prequel_favorites'] = df['prequel_favorites'].notna().astype(int)
        df['has_prequel_dropped'] = df['prequel_dropped'].notna().astype(int)
        df['has_prequel_forum'] = df['prequel_forum'].notna().astype(int)
        df['has_prequel_season'] = df['prequel_season'].notna().astype(int)
        df['has_prequel_year'] = df['prequel_year'].notna().astype(int)
        df = df.fillna({'prequel_season':""})

    # categoricals
    combined = pd.concat([train_df, val_df, test_df], axis=0)
    combined = pd.get_dummies(combined, columns=categoricals, dtype=int)

    train_df = combined[:len(train_df)]
    val_df = combined[len(train_df):len(train_df)+len(val_df)]
    test_df = combined[len(train_df)+len(val_df):]

    return train_df, val_df, test_df

# Processes training and validation data every CV fold to avoid data leakage
def pytorch_train_processing(train_df, val_df):
    # adaptation scaling
    metrics = ['score', 'members']
    for metric in metrics:
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler())
        ])

        train_df[[f"adaptation_{metric}"]] = pipeline.fit_transform(train_df[[f"adaptation_{metric}"]])
        val_df[[f"adaptation_{metric}"]] = pipeline.transform(val_df[[f"adaptation_{metric}"]])

    # prequel scaling
    metrics = ['score', 'wc', 'favorites', 'dropped', 'forum', 'year']
    for metric in metrics:
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler())
        ])

        train_df[[f"prequel_{metric}"]] = pipeline.fit_transform(train_df[[f"prequel_{metric}"]])
        val_df[[f"prequel_{metric}"]] = pipeline.transform(val_df[[f"prequel_{metric}"]])

    return train_df, val_df

# Custom PyTorch NN architecture
class PytorchNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PytorchNN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        out = self.layers(x)
        return out