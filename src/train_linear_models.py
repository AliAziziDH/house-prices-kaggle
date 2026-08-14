"""
Lasso & ElasticNet Regularized Linear Models with RobustScaler
Uses 5-Fold Cross-Validation on y_train_log to tune alpha and l1_ratio,
generating clean OOF and Test predictions.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42
N_FOLDS = 5


def main():
    print("=" * 60)
    print("LOADING PROCESSED DATA FOR LINEAR MODELS")
    print("=" * 60)

    # 100% data
    X_train_full = pd.read_csv("./processed_data/X_train_full_linear.csv")
    y_train_full = pd.read_csv("./processed_data/y_train_full.csv").squeeze()

    # 90% data
    X_train_90 = pd.read_csv("./processed_data/X_train_linear.csv")
    y_train_90 = pd.read_csv("./processed_data/y_train.csv").squeeze()

    X_test_linear = pd.read_csv("./processed_data/X_test_linear.csv")
    test_neighborhoods = pd.read_csv("./processed_data/X_test_raw.csv")["Neighborhood"]

    raw_train = pd.read_csv("./data/train.csv")
    raw_train = raw_train[~((raw_train["GrLivArea"] > 4000) & (raw_train["SalePrice"] < 200000))].reset_index(drop=True)
    raw_neighborhoods = raw_train["Neighborhood"]

    print("\n" + "=" * 60)
    print("TRAINING LASSO, RIDGE & ELASTICNET WITH ROBUSTSCALER (5-FOLD CV)")
    print("=" * 60)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_lasso = np.zeros(len(X_train_full))
    oof_ridge = np.zeros(len(X_train_full))
    oof_elasticnet = np.zeros(len(X_train_full))

    alphas_lasso = np.logspace(-5, 1, 50)
    alphas_ridge = np.logspace(-3, 3, 50)
    alphas_elasticnet = np.logspace(-5, 1, 50)
    l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
        X_tr, X_va = X_train_full.iloc[train_idx].copy(), X_train_full.iloc[val_idx].copy()
        y_tr = y_train_full.iloc[train_idx]

        # Calculate LOO fold-local encoding to prevent target leakage
        train_neighborhoods = raw_neighborhoods.iloc[train_idx]
        val_neighborhoods = raw_neighborhoods.iloc[val_idx]

        global_mean = y_tr.mean()
        neigh_sums = y_tr.groupby(train_neighborhoods).sum()
        neigh_counts = train_neighborhoods.value_counts()
        m = 20

        loo_encodings = []
        for n, y_i in zip(train_neighborhoods, y_tr):
            sum_c = neigh_sums.get(n, 0)
            n_c = neigh_counts.get(n, 0)
            enc = (sum_c - y_i + m * global_mean) / (n_c - 1 + m) if (n_c - 1 + m) > 0 else global_mean
            loo_encodings.append(enc)
        X_tr["Neighborhood"] = loo_encodings

        val_encodings = []
        for n in val_neighborhoods:
            sum_c = neigh_sums.get(n, 0)
            n_c = neigh_counts.get(n, 0)
            enc = (sum_c + m * global_mean) / (n_c + m) if (n_c + m) > 0 else global_mean
            val_encodings.append(enc)
        X_va["Neighborhood"] = val_encodings

        base_lasso = make_pipeline(
            RobustScaler(), LassoCV(alphas=alphas_lasso, cv=5, random_state=RANDOM_STATE, max_iter=2000, n_jobs=-1)
        )
        model_lasso = TransformedTargetRegressor(regressor=base_lasso, func=np.log1p, inverse_func=np.expm1)
        model_lasso.fit(X_tr, y_tr)
        oof_lasso[val_idx] = model_lasso.predict(X_va)

        base_ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_ridge, cv=5))
        model_ridge = TransformedTargetRegressor(regressor=base_ridge, func=np.log1p, inverse_func=np.expm1)
        model_ridge.fit(X_tr, y_tr)
        oof_ridge[val_idx] = model_ridge.predict(X_va)

        base_elasticnet = make_pipeline(
            RobustScaler(),
            ElasticNetCV(
                alphas=alphas_elasticnet, l1_ratio=l1_ratios, cv=5, random_state=RANDOM_STATE, max_iter=2000, n_jobs=-1
            ),
        )
        model_elasticnet = TransformedTargetRegressor(regressor=base_elasticnet, func=np.log1p, inverse_func=np.expm1)
        model_elasticnet.fit(X_tr, y_tr)
        oof_elasticnet[val_idx] = model_elasticnet.predict(X_va)


    print("\nTRAINING 100% LINEAR MODELS...")
    global_mean_full = y_train_full.mean()
    neigh_sums_full = y_train_full.groupby(raw_neighborhoods).sum()
    neigh_counts_full = raw_neighborhoods.value_counts()

    X_train_full_enc = X_train_full.copy()
    full_encodings = []
    for n, y_i in zip(raw_neighborhoods, y_train_full):
        sum_c = neigh_sums_full.get(n, 0)
        n_c = neigh_counts_full.get(n, 0)
        enc = (sum_c - y_i + m * global_mean_full) / (n_c - 1 + m) if (n_c - 1 + m) > 0 else global_mean_full
        full_encodings.append(enc)
    X_train_full_enc["Neighborhood"] = full_encodings

    model_lasso_full = TransformedTargetRegressor(
        regressor=make_pipeline(
            RobustScaler(), LassoCV(alphas=alphas_lasso, cv=5, random_state=RANDOM_STATE, max_iter=2000, n_jobs=-1)
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    model_lasso_full.fit(X_train_full_enc, y_train_full)

    model_ridge_full = TransformedTargetRegressor(
        regressor=make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_ridge, cv=5)), func=np.log1p, inverse_func=np.expm1
    )
    model_ridge_full.fit(X_train_full_enc, y_train_full)

    model_elasticnet_full = TransformedTargetRegressor(
        regressor=make_pipeline(
            RobustScaler(),
            ElasticNetCV(
                alphas=alphas_elasticnet, l1_ratio=l1_ratios, cv=5, random_state=RANDOM_STATE, max_iter=2000, n_jobs=-1
            ),
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    model_elasticnet_full.fit(X_train_full_enc, y_train_full)

    print("TRAINING 90% LINEAR MODELS...")
    from sklearn.model_selection import train_test_split

    raw_neighborhoods_proper, _ = train_test_split(raw_neighborhoods, test_size=0.1, random_state=42)

    global_mean_90 = y_train_90.mean()
    neigh_sums_90 = y_train_90.groupby(raw_neighborhoods_proper).sum()
    neigh_counts_90 = raw_neighborhoods_proper.value_counts()

    X_train_90_enc = X_train_90.copy()
    encodings_90 = []
    for n, y_i in zip(raw_neighborhoods_proper, y_train_90):
        sum_c = neigh_sums_90.get(n, 0)
        n_c = neigh_counts_90.get(n, 0)
        enc = (sum_c - y_i + m * global_mean_90) / (n_c - 1 + m) if (n_c - 1 + m) > 0 else global_mean_90
        encodings_90.append(enc)
    X_train_90_enc["Neighborhood"] = encodings_90

    model_lasso_90 = TransformedTargetRegressor(
        regressor=make_pipeline(
            RobustScaler(), LassoCV(alphas=alphas_lasso, cv=5, random_state=RANDOM_STATE, max_iter=2000, n_jobs=-1)
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    model_lasso_90.fit(X_train_90_enc, y_train_90)

    model_ridge_90 = TransformedTargetRegressor(
        regressor=make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_ridge, cv=5)), func=np.log1p, inverse_func=np.expm1
    )
    model_ridge_90.fit(X_train_90_enc, y_train_90)

    model_elasticnet_90 = TransformedTargetRegressor(
        regressor=make_pipeline(
            RobustScaler(),
            ElasticNetCV(
                alphas=alphas_elasticnet, l1_ratio=l1_ratios, cv=5, random_state=RANDOM_STATE, max_iter=2000, n_jobs=-1
            ),
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    model_elasticnet_90.fit(X_train_90_enc, y_train_90)



    # Save test encodings
    X_test_linear_encoded = X_test_linear.copy()
    test_encodings = []
    for n in test_neighborhoods:
        sum_c = neigh_sums_full.get(n, 0)
        n_c = neigh_counts_full.get(n, 0)
        enc = (sum_c + m * global_mean_full) / (n_c + m) if (n_c + m) > 0 else global_mean_full
        test_encodings.append(enc)
    X_test_linear_encoded["Neighborhood"] = test_encodings

    X_calib_linear = pd.read_csv("./processed_data/X_calib_linear.csv")
    calib_neighborhoods = pd.read_csv("./processed_data/X_calib_raw.csv")["Neighborhood"]

    X_calib_linear_encoded = X_calib_linear.copy()
    calib_encodings = []
    for n in calib_neighborhoods:
        sum_c = neigh_sums_90.get(n, 0)
        n_c = neigh_counts_90.get(n, 0)
        enc = (sum_c + m * global_mean_90) / (n_c + m) if (n_c + m) > 0 else global_mean_90
        calib_encodings.append(enc)
    X_calib_linear_encoded["Neighborhood"] = calib_encodings

    X_test_linear_encoded.to_csv("./processed_data/X_test_linear.csv", index=False)
    X_calib_linear_encoded.to_csv("./processed_data/X_calib_linear.csv", index=False)



    from src.metrics import rmsle

    rmsle(y_train_full, oof_lasso)
    rmsle(y_train_full, oof_ridge)
    rmsle(y_train_full, oof_elasticnet)

    os.makedirs("./models", exist_ok=True)
    os.makedirs("./processed_data", exist_ok=True)

    pd.Series(oof_lasso).to_csv("./processed_data/oof_lasso.csv", index=False)
    pd.Series(oof_ridge).to_csv("./processed_data/oof_ridge.csv", index=False)
    pd.Series(oof_elasticnet).to_csv("./processed_data/oof_elasticnet.csv", index=False)

    joblib.dump(model_lasso_full, "./models/lasso_best_rmsle.pkl")
    joblib.dump(model_ridge_full, "./models/ridge_best_rmsle.pkl")
    joblib.dump(model_elasticnet_full, "./models/elasticnet_best_rmsle.pkl")

    joblib.dump(model_lasso_90, "./models/lasso_90.pkl")
    joblib.dump(model_ridge_90, "./models/ridge_90.pkl")
    joblib.dump(model_elasticnet_90, "./models/elasticnet_90.pkl")

    print("\n✅ OOF and models saved successfully.")


if __name__ == "__main__":
    main()
