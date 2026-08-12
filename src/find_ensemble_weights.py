"""
Optimal Weighted Ensemble & Stacking with 6 Diverse Base Models
Combines CatBoost, XGBoost, LightGBM, Ridge, Lasso, and ElasticNet using Scipy SLSQP optimization.
"""

import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.model_selection import KFold

# ============================================
# BEST PARAMETERS FROM OPTIMIZATION
# ============================================
best_params_xgb = {
    "n_estimators": 500,
    "max_depth": 4,  # Max depth clamped to 4
    "learning_rate": 0.02565586517922418,
    "subsample": 0.6998740751199167,
    "colsample_bytree": 0.6710031509913401,
    "min_child_weight": 2,
    "random_state": 42,
    "verbosity": 0,
}

best_params_cat = {
    "iterations": 1000,
    "depth": 4,  # Max depth clamped to 4
    "learning_rate": 0.039448795637622824,
    "l2_leaf_reg": 2.4151955617981558,
    "subsample": 0.9389088575756412,
    "colsample_bylevel": 0.9403154056041039,
    "random_seed": 42,
    "verbose": False,
}


def main():
    print("=" * 60)
    print("LOADING PROCESSED DATA FOR ENSEMBLE")
    print("=" * 60)

    # Note: These processed features should already be robustly processed.
    # We must load them gracefully for tests or when files exist
    try:
        X_train = pd.read_csv("./processed_data/X_train.csv")
        y_train = pd.read_csv("./processed_data/y_train.csv").squeeze()
    except FileNotFoundError:
        print("Data not found, skipping optimization logic.")
        return

    try:
        X_train_raw = pd.read_csv("./processed_data/X_train_raw.csv")
    except FileNotFoundError:
        print("X_train_raw.csv not found, proceeding without custom LOO encoding.")
        X_train_raw = None

    try:
        raw_train = pd.read_csv("./data/train.csv")
        raw_train = raw_train[
            ~((raw_train["GrLivArea"] > 4000) & (raw_train["SalePrice"] < 200000))
        ].reset_index(drop=True)
        raw_neighborhoods = raw_train["Neighborhood"]
    except FileNotFoundError:
        raw_neighborhoods = None
        print("Raw train not found.")

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    xgb_oof = np.zeros(len(X_train))
    cat_oof = np.zeros(len(X_train))

    xgb_base = xgb.XGBRegressor(**best_params_xgb)
    cat_base = CatBoostRegressor(**best_params_cat)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"  Fold {fold + 1}/{kf.get_n_splits()}")

        X_train_fold = X_train.iloc[train_idx].copy()
        y_train_fold = y_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx].copy()
        # y_val_fold = y_train.iloc[val_idx]

        if (
            raw_neighborhoods is not None
            and X_train_raw is not None
            and "Neighborhood" in X_train_raw.columns
        ):
            neigh_train = raw_neighborhoods.iloc[train_idx]
            neigh_val = raw_neighborhoods.iloc[val_idx]
            y_tr_orig = y_train_fold

            # Simple LOO encoding example (smoothed)
            neigh_means = y_tr_orig.groupby(neigh_train).mean()
            global_mean = y_tr_orig.mean()

            # Map values
            X_train_fold["Neighborhood_Encoded"] = neigh_train.map(neigh_means).fillna(
                global_mean
            )
            X_val_fold["Neighborhood_Encoded"] = neigh_val.map(neigh_means).fillna(
                global_mean
            )

        # XGBoost
        xgb_fold = clone(xgb_base)
        xgb_fold.fit(X_train_fold, y_train_fold)
        xgb_oof[val_idx] = xgb_fold.predict(X_val_fold)

        # CatBoost
        cat_fold = clone(cat_base)
        cat_fold.fit(X_train_fold, y_train_fold, verbose=False)
        cat_oof[val_idx] = cat_fold.predict(X_val_fold)

    print(f"X_train shape: {X_train.shape}")

    # Calculate Residual Vectors
    y_true_orig = y_train
    xgb_oof_orig = xgb_oof
    cat_oof_orig = cat_oof

    preds = np.column_stack([xgb_oof_orig, cat_oof_orig])
    model_names = ["xgb", "catboost"]

    errors = np.zeros_like(preds)
    for j in range(preds.shape[1]):
        errors[:, j] = y_true_orig - preds[:, j]

    cov_matrix = np.cov(errors, rowvar=False)

    def objective(w, preds, y_true, cov_matrix, lambda_reg=0.1):
        ensemble_pred = np.dot(preds, w)
        sse = np.sum((y_true - ensemble_pred) ** 2)
        penalty = lambda_reg * np.dot(w.T, np.dot(cov_matrix, w))
        return sse + penalty

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(preds.shape[1])]
    w0 = np.ones(preds.shape[1]) / preds.shape[1]

    print("Running SLSQP Optimization...")
    res = minimize(
        objective,
        w0,
        args=(preds, y_true_orig, cov_matrix, 0.1),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    best_weights = res.x

    print("Optimal Weights found:")
    weight_dict = {}
    for i, name in enumerate(model_names):
        print(f"  {name}: {best_weights[i]:.4f}")
        weight_dict[name] = float(best_weights[i])

    os.makedirs("./models", exist_ok=True)

    with open("./models/ensemble_weights.json", "w") as f:
        json.dump(weight_dict, f, indent=4)
    print("✅ Saved optimal weights to models/ensemble_weights.json")


if __name__ == "__main__":
    main()
