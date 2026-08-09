"""
Lasso & ElasticNet Regularized Linear Models with RobustScaler
Uses 5-Fold Cross-Validation on y_train_log to tune alpha and l1_ratio,
generating clean OOF and Test predictions.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

# ============================================
# CONFIGURATION
# ============================================
RANDOM_STATE: int = 42
N_FOLDS: int = 5


def main():

    # ============================================
    # LOAD DATA
    # ============================================
    print("=" * 60)
    print("LOADING PROCESSED DATA FOR LINEAR MODELS")
    print("=" * 60)

    X_train = pd.read_csv("./processed_data/X_train.csv")
    X_test = pd.read_csv("./processed_data/X_test.csv")
    y_train_log = pd.read_csv("./processed_data/y_train_log.csv").squeeze()

    # Load original raw train data to prevent target leakage in Neighborhood encoding
    raw_train = pd.read_csv("./data/train.csv")
    raw_train = raw_train[
        ~((raw_train["GrLivArea"] > 4000) & (raw_train["SalePrice"] < 300000))
    ].reset_index(drop=True)
    raw_neighborhoods = raw_train["Neighborhood"]

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train_log shape: {y_train_log.shape}")

    # ============================================
    # 5-FOLD CROSS-VALIDATION OOF PREDICTIONS
    # ============================================
    print("\n" + "=" * 60)
    print("TRAINING LASSO & ELASTICNET WITH ROBUSTSCALER (5-FOLD CV)")
    print("=" * 60)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_lasso = np.zeros(len(X_train))
    oof_elasticnet = np.zeros(len(X_train))

    test_preds_lasso = np.zeros(len(X_test))
    test_preds_elasticnet = np.zeros(len(X_test))

    alphas_lasso = np.logspace(-5, 1, 100)
    alphas_elasticnet = np.logspace(-5, 1, 100)
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"  Fold {fold + 1}/{N_FOLDS}...")
        X_tr, X_va = X_train.iloc[train_idx].copy(), X_train.iloc[val_idx].copy()
        y_tr = y_train_log.iloc[train_idx]

        # Leakage-Free Fold-by-Fold Neighborhood Target Rank Mapping
        fold_raw_train = raw_train.iloc[train_idx].copy()
        fold_raw_train["TotalSF"] = (
            fold_raw_train["TotalBsmtSF"]
            + fold_raw_train["1stFlrSF"]
            + fold_raw_train["2ndFlrSF"]
        )
        fold_raw_train["PricePerSF"] = (
            fold_raw_train["SalePrice"] / fold_raw_train["TotalSF"]
        )

        neigh_order = (
            fold_raw_train.groupby("Neighborhood")["PricePerSF"]
            .median()
            .sort_values()
            .index
        )
        neigh_map = {n: i + 1 for i, n in enumerate(neigh_order)}

        X_tr["Neighborhood"] = (
            raw_neighborhoods.iloc[train_idx].map(neigh_map).fillna(13).astype(int)
        )
        X_va["Neighborhood"] = (
            raw_neighborhoods.iloc[val_idx].map(neigh_map).fillna(13).astype(int)
        )

        # 1. Lasso Pipeline
        model_lasso = make_pipeline(
            RobustScaler(),
            LassoCV(
                alphas=alphas_lasso, cv=5, random_state=RANDOM_STATE, max_iter=10000
            ),
        )
        model_lasso.fit(X_tr, y_tr)
        oof_lasso[val_idx] = model_lasso.predict(X_va)
        test_preds_lasso += model_lasso.predict(X_test) / N_FOLDS

        # 2. ElasticNet Pipeline
        model_elasticnet = make_pipeline(
            RobustScaler(),
            ElasticNetCV(
                alphas=alphas_elasticnet,
                l1_ratio=l1_ratios,
                cv=5,
                random_state=RANDOM_STATE,
                max_iter=10000,
            ),
        )
        model_elasticnet.fit(X_tr, y_tr)
        oof_elasticnet[val_idx] = model_elasticnet.predict(X_va)
        test_preds_elasticnet += model_elasticnet.predict(X_test) / N_FOLDS

    # ============================================
    # EVALUATE OOF SCORES
    # ============================================
    rmsle_lasso = np.sqrt(mean_squared_error(y_train_log, oof_lasso))
    rmsle_elasticnet = np.sqrt(mean_squared_error(y_train_log, oof_elasticnet))

    print("\n" + "-" * 40)
    print("LINEAR MODELS OOF RMSLE SCORES:")
    print("-" * 40)
    print(f"  Lasso OOF RMSLE:      {rmsle_lasso:.6f}")
    print(f"  ElasticNet OOF RMSLE: {rmsle_elasticnet:.6f}")

    # ============================================
    # SAVE ARTIFACTS
    # ============================================
    os.makedirs("./models", exist_ok=True)

    joblib.dump(oof_lasso, "./models/oof_lasso.pkl")
    joblib.dump(oof_elasticnet, "./models/oof_elasticnet.pkl")
    joblib.dump(test_preds_lasso, "./models/test_preds_lasso.pkl")
    joblib.dump(test_preds_elasticnet, "./models/test_preds_elasticnet.pkl")

    print("\n✅ OOF and Test predictions saved successfully to ./models/")


if __name__ == "__main__":
    main()
