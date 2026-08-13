import json
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.model_selection import KFold


def load_best_params(model_name):
    if model_name == "xgb":
        return {
            "n_estimators": 500,
            "max_depth": 4,
            "learning_rate": 0.025,
            "subsample": 0.7,
            "colsample_bytree": 0.67,
            "min_child_weight": 2,
            "random_state": 42,
            "verbosity": 0,
        }
    return {
        "iterations": 1000,
        "depth": 4,
        "learning_rate": 0.039,
        "l2_leaf_reg": 2.4,
        "subsample": 0.93,
        "colsample_bylevel": 0.94,
        "random_seed": 42,
        "verbose": False,
    }


def main():
    print("=" * 60)
    print("LOADING FULL DATA FOR ENSEMBLE SLSQP (LOG-SPACE STACKING)")
    print("=" * 60)

    try:
        X_train_full = pd.read_csv("./processed_data/X_train_full.csv")
        y_train_full = pd.read_csv("./processed_data/y_train_full.csv").squeeze()
        X_train_full_raw = pd.read_csv("./processed_data/X_train_full_raw.csv")
    except FileNotFoundError:
        print("Data not found, skipping optimization logic.")
        return

    y_full_transformed = np.log1p(y_train_full.values)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    xgb_oof = np.zeros(len(X_train_full))
    cat_oof = np.zeros(len(X_train_full))

    best_params_xgb = load_best_params("xgb")
    best_params_cat = load_best_params("cat")

    xgb_base = xgb.XGBRegressor(n_jobs=-1, **best_params_xgb)

    best_params_cat_clean = best_params_cat.copy()
    if "random_seed" in best_params_cat_clean:
        del best_params_cat_clean["random_seed"]
    cat_base = CatBoostRegressor(thread_count=-1, random_seed=42, **best_params_cat_clean)

    cat_features = joblib.load("./models/cat_features.pkl")

    oof_xgb_path = "./processed_data/oof_xgboost.csv"
    oof_cat_path = "./processed_data/oof_catboost.csv"

    if os.path.exists(oof_xgb_path) and os.path.exists(oof_cat_path):
        print("Loading pre-calculated OOF predictions for XGBoost and CatBoost...")
        xgb_oof_df = pd.read_csv(oof_xgb_path)
        cat_oof_df = pd.read_csv(oof_cat_path)

        train_df = pd.read_csv("./data/train.csv")
        outlier_mask = (train_df["GrLivArea"] > 4000) & (train_df["SalePrice"] < 200000)
        train_full_ids = train_df[~outlier_mask]["Id"]

        xgb_oof_df = xgb_oof_df.set_index("Id").loc[train_full_ids]
        cat_oof_df = cat_oof_df.set_index("Id").loc[train_full_ids]

        if "OOF_SalePrice" in xgb_oof_df.columns:
            xgb_oof = np.log1p(np.clip(xgb_oof_df["OOF_SalePrice"].values, 1, None))
            cat_oof = np.log1p(np.clip(cat_oof_df["OOF_SalePrice"].values, 1, None))
        else:
            xgb_oof = np.log1p(np.clip(xgb_oof_df["SalePrice"].values, 1, None))
            cat_oof = np.log1p(np.clip(cat_oof_df["SalePrice"].values, 1, None))
    else:
        print("Generating OOF predictions for XGBoost and CatBoost (Log Space)...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
            X_train_fold_xgb = X_train_full.iloc[train_idx].copy()
            X_val_fold_xgb = X_train_full.iloc[val_idx].copy()
            y_train_fold_trans = y_full_transformed[train_idx]
            X_train_fold_cat = X_train_full_raw.iloc[train_idx].copy()
            X_val_fold_cat = X_train_full_raw.iloc[val_idx].copy()
            for col in cat_features:
                X_train_fold_cat[col] = X_train_fold_cat[col].fillna("Missing").astype(str)
                X_val_fold_cat[col] = X_val_fold_cat[col].fillna("Missing").astype(str)
            xgb_fold = clone(xgb_base)
            xgb_fold.fit(X_train_fold_xgb, y_train_fold_trans)
            xgb_oof[val_idx] = xgb_fold.predict(X_val_fold_xgb)
            cat_fold = clone(cat_base)
            cat_fold.fit(X_train_fold_cat, y_train_fold_trans, cat_features=cat_features, verbose=False)
            cat_oof[val_idx] = cat_fold.predict(X_val_fold_cat)

        # Save OOF predictions to disk so they can be reused
        # We save them as dollars for consistency with linear models
        train_df = pd.read_csv("./data/train.csv")
        outlier_mask = (train_df["GrLivArea"] > 4000) & (train_df["SalePrice"] < 200000)
        train_full_ids = train_df[~outlier_mask]["Id"]

        xgb_df = pd.DataFrame({"Id": train_full_ids, "OOF_SalePrice": np.expm1(xgb_oof)})
        cat_df = pd.DataFrame({"Id": train_full_ids, "OOF_SalePrice": np.expm1(cat_oof)})
        xgb_df.to_csv(oof_xgb_path, index=False)
        cat_df.to_csv(oof_cat_path, index=False)

    print("Loading pre-calculated OOF predictions for remaining models...")
    oof_ridge_phys = pd.read_csv("./processed_data/oof_ridge.csv").squeeze().values
    oof_ridge = np.log1p(np.clip(oof_ridge_phys, 1, None))

    preds = np.column_stack([xgb_oof, cat_oof, oof_ridge])
    model_names = ["xgb", "catboost", "ridge"]

    errors = np.zeros_like(preds)
    for j in range(preds.shape[1]):
        errors[:, j] = y_full_transformed - preds[:, j]

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
        args=(preds, y_full_transformed, cov_matrix, 0.1),
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
