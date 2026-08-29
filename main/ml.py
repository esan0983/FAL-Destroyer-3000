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
    # NOTE FOR FUTURE ME: DEPENDING ON THE TARGET METRIC, YOU'LL REMOVE A LOT OF COLUMNS THAT WERE MADE FROM TARGET ENCODING

if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)