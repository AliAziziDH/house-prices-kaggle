"""
Generate final submissions for Kaggle using the best models.
Two versions:
1. CatBoost only (best CV RMSLE)
2. Weighted Ensemble (XGB 0.14 + Cat 0.86) – based on OOF optimization
"""

import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor

# ============================================
# LOAD DATA
# ============================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

X_test = pd.read_csv("./processed_data/X_test.csv")
test_ids = pd.read_csv("./data/test.csv")["Id"]
print(f"X_test shape: {X_test.shape}")

# ============================================
# BEST PARAMETERS
# ============================================
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

# ============================================
# TRAIN CATBOOST ON FULL DATA
# ============================================
print("\n" + "=" * 60)
print("TRAINING CATBOOST ON FULL DATA")
print("=" * 60)

X_train = pd.read_csv("./processed_data/X_train.csv")
y_train = pd.read_csv("./processed_data/y_train.csv").squeeze()

cat_final = CatBoostRegressor(**best_params_cat)
cat_final.fit(X_train, y_train, verbose=False)
print("✅ CatBoost trained on full data.")

# ============================================
# TRAIN XGBOOST ON FULL DATA (for ensemble)
# ============================================
print("\n" + "=" * 60)
print("TRAINING XGBOOST ON FULL DATA")
print("=" * 60)

xgb_final = xgb.XGBRegressor(**best_params_xgb)
xgb_final.fit(X_train, y_train)
print("✅ XGBoost trained on full data.")

# ============================================
# GENERATE PREDICTIONS
# ============================================
print("\n" + "=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

cat_pred = cat_final.predict(X_test)
xgb_pred = xgb_final.predict(X_test)

# Ensemble with best weights (0.14 XGB, 0.86 Cat)
ensemble_pred = 0.14 * xgb_pred + 0.86 * cat_pred

# ============================================
# CREATE SUBMISSIONS
# ============================================
print("\n" + "=" * 60)
print("CREATING SUBMISSION FILES")
print("=" * 60)

# 1. CatBoost only
import os

os.makedirs("./submissions", exist_ok=True)

submission_cat = pd.DataFrame({"Id": test_ids, "SalePrice": cat_pred})
submission_cat.to_csv("./submissions/submission_catboost_final.csv", index=False)
print("✅ CatBoost submission saved to ./submissions/submission_catboost_final.csv")

# 2. Ensemble
submission_ensemble = pd.DataFrame({"Id": test_ids, "SalePrice": ensemble_pred})
submission_ensemble.to_csv("./submissions/submission_ensemble_final.csv", index=False)
print("✅ Ensemble submission saved to ./submissions/submission_ensemble_final.csv")

print("\n   Shape of each: (1459, 2)")
print("   First 5 rows of CatBoost:")
print(submission_cat.head())
print("\n   First 5 rows of Ensemble:")
print(submission_ensemble.head())

print("\n" + "=" * 60)
print("ALL SUBMISSIONS GENERATED")
print("=" * 60)
