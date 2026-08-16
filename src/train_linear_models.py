import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from src.metrics import rmsle

RANDOM_STATE = 42
N_FOLDS = 5

ALPHAS_LASSO = np.logspace(-5, 1, 50)
ALPHAS_RIDGE = np.logspace(-3, 3, 50)
ALPHAS_ELASTICNET = np.logspace(-5, 1, 50)
L1_RATIOS = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]


def vectorized_target_encode(
    neighborhoods, y=None, neigh_sums=None, neigh_counts=None, global_mean=None, m=20, leave_one_out=False
):
    import numpy as np

    mapped_sums = neighborhoods.map(neigh_sums).fillna(0)
    mapped_counts = neighborhoods.map(neigh_counts).fillna(0)

    if leave_one_out:
        denom = mapped_counts - 1 + m
        return np.where(denom > 0, (mapped_sums - y + m * global_mean) / denom, global_mean)
    else:
        denom = mapped_counts + m
        return np.where(denom > 0, (mapped_sums + m * global_mean) / denom, global_mean)


def generate_oof_predictions(X_train_full, y_train_full, raw_neighborhoods, m=20):
    print("\n" + "=" * 60)
    print("TRAINING LASSO, RIDGE & ELASTICNET WITH ROBUSTSCALER (5-FOLD CV)")
    print("=" * 60)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_lasso = np.zeros(len(X_train_full))
    oof_ridge = np.zeros(len(X_train_full))
    oof_elasticnet = np.zeros(len(X_train_full))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
        X_tr, X_va = X_train_full.iloc[train_idx].copy(), X_train_full.iloc[val_idx].copy()
        y_tr = y_train_full.iloc[train_idx]

        train_neighborhoods = raw_neighborhoods.iloc[train_idx]
        val_neighborhoods = raw_neighborhoods.iloc[val_idx]

        global_mean = y_tr.mean()
        neigh_sums = y_tr.groupby(train_neighborhoods).sum()
        neigh_counts = train_neighborhoods.value_counts()

        X_tr["Neighborhood"] = vectorized_target_encode(
            train_neighborhoods, y_tr, neigh_sums, neigh_counts, global_mean, m, leave_one_out=True
        )

        X_va["Neighborhood"] = vectorized_target_encode(
            val_neighborhoods, None, neigh_sums, neigh_counts, global_mean, m, leave_one_out=False
        )

        base_lasso = make_pipeline(
            RobustScaler(), LassoCV(alphas=ALPHAS_LASSO, cv=5, random_state=RANDOM_STATE, max_iter=2000, n_jobs=-1)
        )
        model_lasso = TransformedTargetRegressor(regressor=base_lasso, func=np.log1p, inverse_func=np.expm1)
        model_lasso.fit(X_tr, y_tr)
        oof_lasso[val_idx] = model_lasso.predict(X_va)

        base_ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ALPHAS_RIDGE, cv=5))
        model_ridge = TransformedTargetRegressor(regressor=base_ridge, func=np.log1p, inverse_func=np.expm1)
        model_ridge.fit(X_tr, y_tr)
        oof_ridge[val_idx] = model_ridge.predict(X_va)

        base_elasticnet = make_pipeline(
            RobustScaler(),
            ElasticNetCV(
                alphas=ALPHAS_ELASTICNET, l1_ratio=L1_RATIOS, cv=5, random_state=RANDOM_STATE, max_iter=2000, n_jobs=-1
            ),
        )
        model_elasticnet = TransformedTargetRegressor(regressor=base_elasticnet, func=np.log1p, inverse_func=np.expm1)
        model_elasticnet.fit(X_tr, y_tr)
        oof_elasticnet[val_idx] = model_elasticnet.predict(X_va)

    return oof_lasso, oof_ridge, oof_elasticnet


def train_all_models(X_train, y_train, raw_neighborhoods, m=20):
    global_mean = y_train.mean()
    neigh_sums = y_train.groupby(raw_neighborhoods).sum()
    neigh_counts = raw_neighborhoods.value_counts()

    X_train_enc = X_train.copy()
    X_train_enc["Neighborhood"] = vectorized_target_encode(
        raw_neighborhoods, y_train, neigh_sums, neigh_counts, global_mean, m, leave_one_out=True
    )

    model_lasso = TransformedTargetRegressor(
        regressor=make_pipeline(
            RobustScaler(), LassoCV(alphas=ALPHAS_LASSO, cv=5, random_state=RANDOM_STATE, max_iter=2000, n_jobs=-1)
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    model_lasso.fit(X_train_enc, y_train)

    model_ridge = TransformedTargetRegressor(
        regressor=make_pipeline(RobustScaler(), RidgeCV(alphas=ALPHAS_RIDGE, cv=5)), func=np.log1p, inverse_func=np.expm1
    )
    model_ridge.fit(X_train_enc, y_train)

    model_elasticnet = TransformedTargetRegressor(
        regressor=make_pipeline(
            RobustScaler(),
            ElasticNetCV(
                alphas=ALPHAS_ELASTICNET, l1_ratio=L1_RATIOS, cv=5, random_state=RANDOM_STATE, max_iter=2000, n_jobs=-1
            ),
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    model_elasticnet.fit(X_train_enc, y_train)

    return {
        "lasso": model_lasso,
        "ridge": model_ridge,
        "elasticnet": model_elasticnet,
        "neigh_sums": neigh_sums,
        "neigh_counts": neigh_counts,
        "global_mean": global_mean,
    }


def save_results(
    oof_lasso,
    oof_ridge,
    oof_elasticnet,
    models_full,
    models_90,
    X_test_linear,
    test_neighborhoods,
    X_calib_linear,
    calib_neighborhoods,
    m=20,
):
    # Save test encodings
    X_test_linear_encoded = X_test_linear.copy()
    X_test_linear_encoded["Neighborhood"] = vectorized_target_encode(
        test_neighborhoods,
        None,
        models_full["neigh_sums"],
        models_full["neigh_counts"],
        models_full["global_mean"],
        m,
        leave_one_out=False,
    )

    X_calib_linear_encoded = X_calib_linear.copy()
    X_calib_linear_encoded["Neighborhood"] = vectorized_target_encode(
        calib_neighborhoods,
        None,
        models_90["neigh_sums"],
        models_90["neigh_counts"],
        models_90["global_mean"],
        m,
        leave_one_out=False,
    )

    os.makedirs("./models", exist_ok=True)
    os.makedirs("./processed_data", exist_ok=True)

    X_test_linear_encoded.to_csv("./processed_data/X_test_linear.csv", index=False)
    X_calib_linear_encoded.to_csv("./processed_data/X_calib_linear.csv", index=False)

    pd.Series(oof_lasso).to_csv("./processed_data/oof_lasso.csv", index=False)
    pd.Series(oof_ridge).to_csv("./processed_data/oof_ridge.csv", index=False)
    pd.Series(oof_elasticnet).to_csv("./processed_data/oof_elasticnet.csv", index=False)

    joblib.dump(models_full["lasso"], "./models/lasso_best_rmsle.pkl")
    joblib.dump(models_full["ridge"], "./models/ridge_best_rmsle.pkl")
    joblib.dump(models_full["elasticnet"], "./models/elasticnet_best_rmsle.pkl")

    joblib.dump(models_90["lasso"], "./models/lasso_90.pkl")
    joblib.dump(models_90["ridge"], "./models/ridge_90.pkl")
    joblib.dump(models_90["elasticnet"], "./models/elasticnet_90.pkl")

    print("\n✅ OOF and models saved successfully.")


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

    X_calib_linear = pd.read_csv("./processed_data/X_calib_linear.csv")
    calib_neighborhoods = pd.read_csv("./processed_data/X_calib_raw.csv")["Neighborhood"]

    oof_lasso, oof_ridge, oof_elasticnet = generate_oof_predictions(X_train_full, y_train_full, raw_neighborhoods)

    rmsle(y_train_full, oof_lasso)
    rmsle(y_train_full, oof_ridge)
    rmsle(y_train_full, oof_elasticnet)

    print("\nTRAINING 100% LINEAR MODELS...")
    models_full = train_all_models(X_train_full, y_train_full, raw_neighborhoods)

    print("TRAINING 90% LINEAR MODELS...")
    if os.path.exists("./processed_data/X_train_raw.csv"):
        raw_neighborhoods_proper = pd.read_csv("./processed_data/X_train_raw.csv")["Neighborhood"]
    else:
        raw_neighborhoods_proper = raw_neighborhoods.iloc[: len(y_train_90)].reset_index(drop=True)
    models_90 = train_all_models(X_train_90, y_train_90, raw_neighborhoods_proper)

    save_results(
        oof_lasso,
        oof_ridge,
        oof_elasticnet,
        models_full,
        models_90,
        X_test_linear,
        test_neighborhoods,
        X_calib_linear,
        calib_neighborhoods,
    )


if __name__ == "__main__":
    main()
