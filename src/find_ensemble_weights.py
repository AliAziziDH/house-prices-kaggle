import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.model_selection import KFold

best_params_xgb = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.02565586517922418,
    "subsample": 0.6998740751199167,
    "colsample_bytree": 0.6710031509913401,
    "min_child_weight": 2,
    "random_state": 42,
    "verbosity": 0,
}

best_params_cat = {
    "iterations": 1000,
    "depth": 4,
    "learning_rate": 0.039448795637622824,
    "l2_leaf_reg": 2.4151955617981558,
    "subsample": 0.9389088575756412,
    "colsample_bylevel": 0.9403154056041039,
    "random_seed": 42,
    "verbose": False,
}


def main():
    print("=" * 60)
    print("LOADING FULL DATA FOR ENSEMBLE SLSQP")
    print("=" * 60)

    try:
        X_train_full = pd.read_csv("./processed_data/X_train_full.csv")
        y_train_full = pd.read_csv("./processed_data/y_train_full.csv").squeeze()
        X_train_full_raw = pd.read_csv("./processed_data/X_train_full_raw.csv")
    except FileNotFoundError:
        print("Data not found, skipping optimization logic.")
        return

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    xgb_oof = np.zeros(len(X_train_full))
    cat_oof = np.zeros(len(X_train_full))

    xgb_base = xgb.XGBRegressor(**best_params_xgb)
    cat_base = CatBoostRegressor(**best_params_cat)

    # We need PowerTransformer to box-cox transform the targets locally for the OOF fit
    from sklearn.preprocessing import PowerTransformer

    pt = PowerTransformer(method="box-cox")
    y_full_transformed = pt.fit_transform(y_train_full.values.reshape(-1, 1)).flatten()

    import joblib

    cat_features = joblib.load("./models/cat_features.pkl")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
        print(f"  Fold {fold + 1}/{kf.get_n_splits()}")

        X_train_fold_xgb = X_train_full.iloc[train_idx].copy()
        X_val_fold_xgb = X_train_full.iloc[val_idx].copy()

        y_train_fold_trans = y_full_transformed[train_idx]

        X_train_fold_cat = X_train_full_raw.iloc[train_idx].copy()
        X_val_fold_cat = X_train_full_raw.iloc[val_idx].copy()

        for col in cat_features:
            X_train_fold_cat[col] = X_train_fold_cat[col].fillna("Missing").astype(str)
            X_val_fold_cat[col] = X_val_fold_cat[col].fillna("Missing").astype(str)

        # XGBoost
        xgb_fold = clone(xgb_base)
        xgb_fold.fit(X_train_fold_xgb, y_train_fold_trans)
        y_pred_xgb_trans = xgb_fold.predict(X_val_fold_xgb)
        xgb_oof[val_idx] = pt.inverse_transform(y_pred_xgb_trans.reshape(-1, 1)).flatten()

        # CatBoost
        cat_fold = clone(cat_base)
        cat_fold.fit(
            X_train_fold_cat,
            y_train_fold_trans,
            cat_features=cat_features,
            verbose=False,
        )
        y_pred_cat_trans = cat_fold.predict(X_val_fold_cat)
        cat_oof[val_idx] = pt.inverse_transform(y_pred_cat_trans.reshape(-1, 1)).flatten()

    y_true_orig = y_train_full.values
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
