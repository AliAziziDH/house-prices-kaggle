"""
Optimal Weighted Ensemble & Stacking with 6 Diverse Base Models
Combines CatBoost, XGBoost, LightGBM, Ridge, Lasso, and ElasticNet using Scipy SLSQP optimization.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor

from src.metrics import rmsle


# ============================================
# BEST PARAMETERS FROM OPTIMIZATION
# ============================================
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
    "depth": 7,
    "learning_rate": 0.039448795637622824,
    "l2_leaf_reg": 2.4151955617981558,
    "subsample": 0.9389088575756412,
    "colsample_bylevel": 0.9403154056041039,
    "random_seed": 42,
    "verbose": False,
}


# ============================================
# LOAD DATA
# ============================================
def main():
    print("=" * 60)
    print("LOADING PROCESSED DATA FOR ENSEMBLE")
    print("=" * 60)


X_train = pd.read_csv("./processed_data/X_train.csv")
y_train = pd.read_csv("./processed_data/y_train.csv").squeeze()
X_test = pd.read_csv("./processed_data/X_test.csv")
test_ids = pd.read_csv("./data/test.csv")["Id"]

X_train_raw = pd.read_csv("./processed_data/X_train_raw.csv")
X_test_raw = pd.read_csv("./processed_data/X_test_raw.csv")

# Load original raw train data to prevent target leakage in Neighborhood encoding
raw_train = pd.read_csv("./data/train.csv")
raw_train = raw_train[
    ~((raw_train["GrLivArea"] > 4000) & (raw_train["SalePrice"] < 200000))
].reset_index(drop=True)
raw_neighborhoods = raw_train["Neighborhood"]

cat_features = X_train_raw.select_dtypes(include=["object"]).columns.tolist()
for col in cat_features:
    X_train_raw[col] = X_train_raw[col].fillna("Missing").astype(str)
    X_test_raw[col] = X_test_raw[col].fillna("Missing").astype(str)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"  Fold {fold + 1}/5")

    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # XGBoost
    xgb_fold = xgb.XGBRegressor(**best_params_xgb)
    xgb_fold.fit(X_train_fold, y_train_fold)
    xgb_oof[val_idx] = xgb_fold.predict(X_val_fold)

    # CatBoost
    cat_fold = CatBoostRegressor(**best_params_cat)
    cat_fold.fit(X_train_fold, y_train_fold, verbose=False)
    cat_oof[val_idx] = cat_fold.predict(X_val_fold)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_train_raw shape: {X_train_raw.shape}")
    print(f"y_train_log shape: {y_train_log.shape}")

best_weight = 0.5
best_rmsle = float("inf")
results = []

for w in np.arange(0.0, 1.01, 0.01):
    ensemble_pred = w * xgb_oof + (1 - w) * cat_oof
    score = rmsle(y_train, ensemble_pred)
    results.append({"weight_xgb": w, "rmsle": score})

    if score < best_rmsle:
        best_rmsle = score
        best_weight = w

print(
    f"\n✅ Best weight: XGBoost = {best_weight:.2f}, CatBoost = {1 - best_weight:.2f}"
)
print(f"✅ Best OOF RMSLE: {best_rmsle:.6f}")

final_pred = best_weight * xgb_test + (1 - best_weight) * cat_test

submission = pd.DataFrame({"Id": test_ids, "SalePrice": final_pred})

submission.to_csv("./submissions/submission_ensemble_rmsle_final.csv", index=False)

print("\n" + "=" * 60)
print("ENSEMBLE WEIGHT OPTIMIZATION COMPLETED")
print("=" * 60)
