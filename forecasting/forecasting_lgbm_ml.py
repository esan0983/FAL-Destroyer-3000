import os
os.environ["SCIPY_ARRAY_API"] = "1"

from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.metrics import mean_squared_error, ndcg_score
from scipy.stats import spearmanr

import json
import shap
import random

import optuna
optuna.logging.set_verbosity(optuna.logging.INFO)

import warnings
warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")
warnings.filterwarnings("ignore", message=".*will be ignored.*")

import joblib

from utils import (
    multivalue_preprocessing,
    encode_features
)


# ---------------------------------------------------------------------------
# Precision@k: literal "did my predicted top-k match the true top-k".
# Sign-agnostic, so it's fine with negative points.
# ---------------------------------------------------------------------------
def precision_at_k(y_true, y_pred, k):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    k = min(k, len(y_true))

    pred_top_k = set(np.argsort(-y_pred)[:k])
    true_top_k = set(np.argsort(-y_true)[:k])

    return len(pred_top_k & true_top_k) / k


def min_shift(y):
    y = np.asarray(y, dtype=float)
    return y - y.min()


# ---------------------------------------------------------------------------
# Sample weights that upweight the true top-k rows within a fold, so a
# plain regression objective still concentrates its error budget on
# getting the top right -- rather than switching to a ranking loss, which
# gives back a relevance score instead of a points estimate.
# ---------------------------------------------------------------------------
def top_weights(y, top_frac=0.1, top_weight=8.0):
    y = np.asarray(y, dtype=float)
    n = len(y)
    k = max(1, int(np.ceil(n * top_frac)))
    top_idx = set(np.argsort(-y)[:k])
    return np.array([top_weight if i in top_idx else 1.0 for i in range(n)])


def lgbm_ndcg_precision(train_df, test_df, week):
    X_train_full = train_df.drop(columns=['points'])
    X_test_full = test_df.drop(columns=['points'])

    lgbm_model = joblib.load(f"data/forecasting/ml_data/week{week}/lgbm_model.joblib")

    X_train_full_prep, X_test_full_prep = multivalue_preprocessing(X_train_full, X_test_full)
    X_train_full_proc, X_test_full_proc = encode_features(X_train_full_prep, X_test_full_prep)

    predicted = lgbm_model.predict(X_test_full_proc)
    actual = test_df['points'].to_numpy()

    # ndcg_score needs non-negative true relevance -- shift for this
    # metric only, never touches the predictions themselves
    actual_shifted = min_shift(actual)

    ndcg8 = ndcg_score([actual_shifted], [predicted], k=min(8, len(actual)))
    ndcg15 = ndcg_score([actual_shifted], [predicted], k=min(15, len(actual)))

    prec8 = precision_at_k(actual, predicted, k=8)
    prec15 = precision_at_k(actual, predicted, k=15)

    spearman, _ = spearmanr(actual, predicted)
    spearman = spearman if not np.isnan(spearman) else 0.0

    print(f"Week {week}:")
    print(f"NDCG@8: {ndcg8:.4f} | NDCG@15: {ndcg15:.4f}")
    print(f"Precision@8: {prec8:.4f} | Precision@15: {prec15:.4f}")
    print(f"Spearman: {spearman:.4f}")

    return {
        "ndcg@8": ndcg8, "ndcg@15": ndcg15,
        "precision@8": prec8, "precision@15": prec15,
        "spearman": spearman
    }


def lgbm_feature_importance(train_df, test_df, week):
    X_train_full = train_df.drop(columns=['points'])
    X_test_full = test_df.drop(columns=['points'])

    lgbm_model = joblib.load(f"data/forecasting/ml_data/week{week}/lgbm_model.joblib")

    X_train_full_prep, X_test_full_prep = multivalue_preprocessing(X_train_full, X_test_full)
    X_train_full_proc, X_test_full_proc = encode_features(X_train_full_prep, X_test_full_prep)

    explainer = shap.TreeExplainer(lgbm_model)
    shap_values = explainer(X_test_full_proc)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_full_proc, show=False)
    plt.title('Individual Feature Importance (SHAP)', fontsize=14, pad=12)
    plt.tight_layout()

    output_dir = f"data/forecasting/ml_data/week{week}"
    indiv_save_path = os.path.join(output_dir, "lgbm_features.png") if len(X_test_full_proc) > 1 else os.path.join(output_dir, "lgbm_features_specific.png")
    plt.savefig(indiv_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Individual feature plot saved to {indiv_save_path}")


def lgbm_inference(train_df, inference_df, week):
    lgbm_model = joblib.load(f"data/forecasting/ml_data/week{week}/lgbm_model.joblib")

    X_train_full = train_df.drop(columns=['points'])
    X_inf_full = inference_df.drop(columns=['anime', 'points'], errors='ignore')
    titles = inference_df['anime']

    X_train_full_prep, X_inf_full_prep = multivalue_preprocessing(X_train_full, inference_df)
    X_train_full_proc, X_inf_full_proc = encode_features(X_train_full_prep, X_inf_full_prep)

    # actual predicted points -- no shifting, no grading, no rescaling
    inferences = lgbm_model.predict(X_inf_full_proc)

    inference_dict = {title: inferences[idx].item() for idx, title in enumerate(titles)}
    inference_dict = dict(sorted(inference_dict.items(), key=lambda item: item[1], reverse=True))

    output_dir = f"data/forecasting/ml_data/week{week}"
    file_path = os.path.join(output_dir, "lgbm_predictions.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(inference_dict, f, indent=4)

    print("Predictions saved!")


def lightgbm_ranker(train_df, test_df, week, seed, top_frac=0.2, top_weight=5.0):
    X_train_full = train_df.drop(columns=['points'])
    y_train_full = train_df['points']

    X_test_full = test_df.drop(columns=['points'])
    y_test_full = test_df['points']

    # --- 1. RepeatedKFold caching, same pattern as random_forest() ---
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=seed)
    cached_folds = []

    print("Caching RepeatedKFold splits...")
    for train_idx, val_idx in rkf.split(X_train_full, y_train_full):
        X_tr, X_va = X_train_full.iloc[train_idx].copy(), X_train_full.iloc[val_idx].copy()
        y_tr, y_va = y_train_full.iloc[train_idx].to_numpy(), y_train_full.iloc[val_idx].to_numpy()

        X_tr_proc, X_va_proc = multivalue_preprocessing(X_tr, X_va)
        X_tr_proc, X_va_proc = encode_features(X_tr_proc, X_va_proc)

        w_tr = top_weights(y_tr, top_frac=top_frac, top_weight=top_weight)

        cached_folds.append((
            X_tr_proc.to_numpy(),
            y_tr,
            w_tr,
            X_va_proc.to_numpy(),
            y_va,
        ))

    # --- 2. Optuna objective: plain regression, but selection criterion
    #        is precision@4 (did we get the top right), not RMSE ---
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': trial.suggest_int('num_leaves', 7, 31),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 40),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': 1,
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
            'verbosity': -1,
            'seed': seed,
        }

        fold_precisions = []

        for fold_idx, (X_tr_proc, y_tr, w_tr, X_va_proc, y_va) in enumerate(cached_folds):
            train_set = lgb.Dataset(X_tr_proc, label=y_tr, weight=w_tr)
            val_set = lgb.Dataset(X_va_proc, label=y_va, reference=train_set)

            booster = lgb.train(
                params,
                train_set,
                num_boost_round=400,
                valid_sets=[val_set],
                callbacks=[lgb.early_stopping(30, verbose=False)]
            )

            preds = booster.predict(X_va_proc, num_iteration=booster.best_iteration)
            prec8 = precision_at_k(y_va, preds, k=8)
            fold_precisions.append(prec8)

            trial.report(np.mean(fold_precisions), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # maximize precision@4 -> Optuna minimizes by default
        return -np.mean(fold_precisions)

    # --- 3. Run Optuna ---
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(n_startup_trials=10, seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3),
    )
    study.optimize(objective, n_trials=50, n_jobs=1)

    best_params = study.best_params.copy()
    print(f"Best CV Precision@4: {-study.best_value:.4f}")
    print("Best Hyperparameters:", best_params)

    best_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'bagging_freq': 1,
        'verbosity': -1,
        'seed': seed,
    })

    # --- 4. Final processing & training with early stopping ---
    X_train_full_prep, X_test_full_prep = multivalue_preprocessing(X_train_full, X_test_full)
    X_train_full_proc, X_test_proc = encode_features(X_train_full_prep, X_test_full_prep)

    X_train_final_proc, X_val_final_proc, y_train_final, y_val_final = train_test_split(
        X_train_full_proc, y_train_full, test_size=0.2, random_state=seed
    )

    y_train_final_arr = y_train_final.to_numpy()
    w_train_final = top_weights(y_train_final_arr, top_frac=top_frac, top_weight=top_weight)

    train_set = lgb.Dataset(X_train_final_proc.to_numpy(), label=y_train_final_arr, weight=w_train_final)
    val_set = lgb.Dataset(X_val_final_proc.to_numpy(), label=y_val_final.to_numpy(), reference=train_set)

    print("Training final model with early stopping...")
    final_booster = lgb.train(
        best_params,
        train_set,
        num_boost_round=2000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(50, verbose=True)]
    )

    print("Saving model...")
    output_dir = Path(f"data/forecasting/ml_data/week{week}")
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_booster, output_dir / "lgbm_model.joblib", compress=3)

    y_pred = final_booster.predict(X_test_proc, num_iteration=final_booster.best_iteration)
    y_test_arr = y_test_full.to_numpy()

    mse = mean_squared_error(y_test_arr, y_pred)
    spearman, _ = spearmanr(y_test_arr, y_pred)
    spearman = spearman if not np.isnan(spearman) else 0.0
    prec8 = precision_at_k(y_test_arr, y_pred, k=8)
    prec15 = precision_at_k(y_test_arr, y_pred, k=15)

    print(f"Points -> Test MSE: {mse:.4f} | Spearman: {spearman:.4f} | P@8: {prec8:.4f} | P@15: {prec15:.4f}")

    return mse, spearman, prec8, prec15


def main():
    for i in range(13):
        train_df = pd.read_parquet(f"data/forecasting/ml_data/week{i+1}/train_df.parquet")
        test_df = pd.read_parquet(f"data/forecasting/ml_data/week{i+1}/test_df.parquet")
        inference_df = pd.read_parquet(f"data/forecasting/ml_data/week{i+1}/inference_df.parquet")

        lgbm_mses = {}
        lgbm_spearman = {}
        lgbm_precision = {}

        lgbm_mse_path = Path(f"data/forecasting/ml_data/week{i+1}/lgbm_mses.json")
        lgbm_spearman_path = Path(f"data/forecasting/ml_data/week{i+1}/lgbm_spearman.json")
        lgbm_precision_path = Path(f"data/forecasting/ml_data/week{i+1}/lgbm_precision.json")

        for seed in [42]:
            random.seed(seed)
            np.random.seed(seed)
            print(f"Current seed: {seed}")

            mse, spearman, prec8, prec15 = lightgbm_ranker(train_df, test_df, i+1, seed)
            lgbm_mses['points'] = mse
            lgbm_spearman['points'] = spearman
            lgbm_precision['points'] = {"p@8": prec8, "p@15": prec15}

            lgbm_feature_importance(train_df, test_df, i+1)
            lgbm_inference(train_df, inference_df, i+1)
            lgbm_ndcg_precision(train_df, test_df, i+1)

        with open(lgbm_mse_path, "w", encoding="utf-8") as f:
            json.dump(lgbm_mses, f, indent=4)
        with open(lgbm_spearman_path, "w", encoding="utf-8") as f:
            json.dump(lgbm_spearman, f, indent=4)
        with open(lgbm_precision_path, "w", encoding="utf-8") as f:
            json.dump(lgbm_precision, f, indent=4)


if __name__ == "__main__":
    main()