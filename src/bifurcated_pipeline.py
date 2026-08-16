import os

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, RobustScaler

from src.conformal import compute_empirical_quantile, compute_non_conformity_scores, compute_prediction_intervals
from src.metrics import rmsle
from src.preprocess import AmesDataTransformer

RANDOM_STATE = 42
N_FOLDS = 10
N_TRIALS = 50


def get_neighborhood_ranks(df_train, df_transform):
    """
    Computes fold-local neighborhood ranks based on median SalePrice / TotalSF.
    Ranks are from 1 to 25.
    Returns the transformed dataframe with 'Neighborhood' as integer rank.
    """
    df_train = df_train.copy()
    df_transform = df_transform.copy()

    # Calculate TotalSF safely
    gr_liv_area_tr = df_train["GrLivArea"] if "GrLivArea" in df_train.columns else 0
    total_bsmt_sf_tr = df_train["TotalBsmtSF"].fillna(0) if "TotalBsmtSF" in df_train.columns else 0
    df_train["TotalSF_temp"] = gr_liv_area_tr + total_bsmt_sf_tr

    gr_liv_area_tf = df_transform["GrLivArea"] if "GrLivArea" in df_transform.columns else 0
    total_bsmt_sf_tf = df_transform["TotalBsmtSF"].fillna(0) if "TotalBsmtSF" in df_transform.columns else 0
    df_transform["TotalSF_temp"] = gr_liv_area_tf + total_bsmt_sf_tf

    # Calculate price per sqft
    df_train["PricePerSF"] = df_train["SalePrice"] / df_train["TotalSF_temp"].replace(0, np.nan)

    medians = df_train.groupby("Neighborhood")["PricePerSF"].median().sort_values()
    rank_map = {neigh: i + 1 for i, neigh in enumerate(medians.index)}

    # Default to 13 (middle rank out of 25) for unseen
    df_transform["Neighborhood"] = df_transform["Neighborhood"].map(rank_map).fillna(13).astype(int)
    df_train["Neighborhood"] = df_train["Neighborhood"].map(rank_map).fillna(13).astype(int)

    df_transform = df_transform.drop(columns=["TotalSF_temp"], errors="ignore")
    return df_transform


def objective_xgb(trial, train_full):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 1, 3),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 3.0, 10.0),
        "random_state": RANDOM_STATE,
        "verbosity": 0,
    }
    model = xgb.XGBRegressor(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rmsle_scores = []
    for tr_idx, va_idx in kf.split(train_full):
        tr = train_full.iloc[tr_idx].copy()
        va = train_full.iloc[va_idx].copy()

        va = get_neighborhood_ranks(tr, va)
        tr = get_neighborhood_ranks(tr, tr)

        transformer = AmesDataTransformer(vif_prune=False)
        X_tr = transformer.fit_transform(tr.drop(columns=["Id", "SalePrice"]))
        X_va = transformer.transform(va.drop(columns=["Id", "SalePrice"]))

        y_tr = np.log1p(tr["SalePrice"])
        y_va = np.log1p(va["SalePrice"])

        model.fit(X_tr, y_tr)
        preds = np.expm1(model.predict(X_va))
        rmsle_scores.append(rmsle(np.expm1(y_va), preds))

    return np.mean(rmsle_scores)


def objective_cat(trial, train_full):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000, step=100),
        "depth": trial.suggest_int("depth", 1, 3),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 3.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        "random_seed": RANDOM_STATE,
        "verbose": False,
    }

    model = CatBoostRegressor(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rmsle_scores = []
    for tr_idx, va_idx in kf.split(train_full):
        tr = train_full.iloc[tr_idx].copy()
        va = train_full.iloc[va_idx].copy()

        va = get_neighborhood_ranks(tr, va)
        tr = get_neighborhood_ranks(tr, tr)

        transformer = AmesDataTransformer(vif_prune=False)
        X_tr = transformer.fit_transform(tr.drop(columns=["Id", "SalePrice"]))
        X_va = transformer.transform(va.drop(columns=["Id", "SalePrice"]))

        y_tr = np.log1p(tr["SalePrice"])
        y_va = np.log1p(va["SalePrice"])

        # Identify categorical features
        cat_features = X_tr.select_dtypes(include=["object", "str", "category"]).columns.tolist()
        for col in cat_features:
            X_tr[col] = X_tr[col].fillna("Missing").astype(str)
            X_va[col] = X_va[col].fillna("Missing").astype(str)

        model.fit(X_tr, y_tr, cat_features=cat_features, verbose=False)
        preds = np.expm1(model.predict(X_va))
        rmsle_scores.append(rmsle(np.expm1(y_va), preds))

    return np.mean(rmsle_scores)


def load_and_split_data():
    print("=" * 60)
    print("STARTING BIFURCATED MULTI-STREAM ENSEMBLE PIPELINE")
    print("=" * 60)

    # 1. Load Data
    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")

    outlier_mask = (train["GrLivArea"] > 4000) & (train["SalePrice"] < 200000)
    train_full = train[~outlier_mask].reset_index(drop=True)

    # 2. Setup 90/10 split for Conformal Bounds
    train_90, calib_10 = train_test_split(train_full, test_size=0.1, random_state=RANDOM_STATE)
    train_90 = train_90.reset_index(drop=True)
    calib_10 = calib_10.reset_index(drop=True)

    return train_full, train_90, calib_10, test


def tune_models(train_full):
    from src.utils.optuna_helper import setup_optuna_study

    print("\n--- Tuning XGBoost ---")
    study_xgb = setup_optuna_study(study_name="xgb_bifurcated")
    study_xgb.optimize(lambda trial: objective_xgb(trial, train_full), n_trials=N_TRIALS, n_jobs=-1)
    best_xgb_params = study_xgb.best_params
    best_xgb_params.update({"random_state": RANDOM_STATE, "verbosity": 0})

    print("\n--- Tuning CatBoost ---")
    study_cat = setup_optuna_study(study_name="cat_bifurcated")
    study_cat.optimize(lambda trial: objective_cat(trial, train_full), n_trials=N_TRIALS, n_jobs=-1)
    best_cat_params = study_cat.best_params
    best_cat_params.update({"random_seed": RANDOM_STATE, "verbose": False})

    return best_xgb_params, best_cat_params


def generate_oof_predictions(train_full, xgb_base, cat_base, ridge_model):
    print("\n--- Generating 10-Fold OOF Predictions ---")
    kf10 = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    oof_xgb = np.zeros(len(train_full))
    oof_cat = np.zeros(len(train_full))
    oof_ridge = np.zeros(len(train_full))

    y_full = train_full["SalePrice"].values
    np.log1p(y_full)

    for fold, (tr_idx, va_idx) in enumerate(kf10.split(train_full)):
        print(f"Fold {fold + 1}/10...")
        tr = train_full.iloc[tr_idx].copy()
        va = train_full.iloc[va_idx].copy()

        # 1. Target Encoding Streams
        # Ranking (Trees)
        va_tree = get_neighborhood_ranks(tr, va)
        tr_tree = get_neighborhood_ranks(tr, tr)

        # Linear models use a different target encoding with smoothing parameter m=20
        m = 20
        global_mean = tr["SalePrice"].mean()
        neigh_sums = tr.groupby("Neighborhood")["SalePrice"].sum()
        neigh_counts = tr.groupby("Neighborhood")["SalePrice"].count()

        tr_linear = tr.copy()
        n_c_tr = tr["Neighborhood"].map(neigh_counts).fillna(0)
        sum_c_tr = tr["Neighborhood"].map(neigh_sums).fillna(0)
        denominator_tr = n_c_tr - 1 + m
        numerator_tr = sum_c_tr - tr["SalePrice"] + m * global_mean
        tr_linear["Neighborhood"] = np.where(denominator_tr > 0, numerator_tr / denominator_tr, global_mean)

        va_linear = va.copy()
        n_c_va = va["Neighborhood"].map(neigh_counts).fillna(0)
        sum_c_va = va["Neighborhood"].map(neigh_sums).fillna(0)
        denominator_va = n_c_va + m
        numerator_va = sum_c_va + m * global_mean
        va_linear["Neighborhood"] = np.where(denominator_va > 0, numerator_va / denominator_va, global_mean)

        # 2. Preprocessing Streams
        # Tree Stream (No VIF Prune)
        tree_tf = AmesDataTransformer(vif_prune=False)
        X_tr_tree = tree_tf.fit_transform(tr_tree.drop(columns=["Id", "SalePrice"]))
        X_va_tree = tree_tf.transform(va_tree.drop(columns=["Id", "SalePrice"]))

        cat_features = X_tr_tree.select_dtypes(include=["object", "str"]).columns.tolist()
        for col in cat_features:
            X_tr_tree[col] = X_tr_tree[col].fillna("Missing").astype(str)
            X_va_tree[col] = X_va_tree[col].fillna("Missing").astype(str)

        # Linear Stream (VIF Prune)
        linear_tf = AmesDataTransformer(vif_prune=True)
        X_tr_linear = linear_tf.fit_transform(tr_linear.drop(columns=["Id", "SalePrice"]))
        X_va_linear = linear_tf.transform(va_linear.drop(columns=["Id", "SalePrice"]))

        y_tr = tr["SalePrice"].values
        y_tr_log = np.log1p(y_tr)

        # 3. Fitting & Predicting
        xgb_fold = clone(xgb_base)
        xgb_fold.fit(X_tr_tree, y_tr_log)
        oof_xgb[va_idx] = xgb_fold.predict(X_va_tree)

        cat_fold = clone(cat_base)
        cat_fold.fit(X_tr_tree, y_tr_log, cat_features=cat_features, verbose=False)
        oof_cat[va_idx] = cat_fold.predict(X_va_tree)

        ridge_fold = clone(ridge_model)
        # Ridge is wrapped in TransformedTargetRegressor with QuantileTransformer,
        # so we feed it the RAW physical targets (y_tr) and it handles the transformation internally.
        ridge_fold.fit(X_tr_linear, y_tr)

        # The Ridge predictions are physically scaled dollars, so we must log1p them to blend!
        oof_ridge[va_idx] = np.log1p(np.clip(ridge_fold.predict(X_va_linear), 1, None))

    return oof_xgb, oof_cat, oof_ridge


def calculate_ensemble_weights(preds, y_true):
    print("\n--- Running SLSQP Meta-Learner ---")
    errors = np.zeros_like(preds)
    for j in range(preds.shape[1]):
        errors[:, j] = y_true - preds[:, j]

    cov_matrix = np.cov(errors, rowvar=False)

    def objective_slsqp(w, preds_inner, y_true_inner, cov_matrix_inner, lambda_reg=0.1):
        ensemble_pred = np.dot(preds_inner, w)
        sse = np.sum((y_true_inner - ensemble_pred) ** 2)
        penalty = lambda_reg * np.dot(w.T, np.dot(cov_matrix_inner, w))
        return sse + penalty

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(preds.shape[1])]
    w0 = np.ones(preds.shape[1]) / preds.shape[1]

    res = minimize(
        objective_slsqp,
        w0,
        args=(preds, y_true, cov_matrix, 0.1),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    best_weights = res.x
    model_names = ["xgb", "catboost", "ridge"]
    print("Optimal Weights:")
    for i, name in enumerate(model_names):
        print(f"  {name}: {best_weights[i]:.4f}")

    return best_weights


def calculate_conformal_bounds(train_90, calib_10, best_weights, xgb_base, cat_base, ridge_model):
    print("\n--- Calculating Conformal Bounds (90/10 Split) ---")
    m = 20
    # 1. 90% Fold-local Neighborhood Ranking
    calib_tree = get_neighborhood_ranks(train_90, calib_10)
    train_90_tree = get_neighborhood_ranks(train_90, train_90)

    # Linear Encoding for 90/10
    global_mean_90 = train_90["SalePrice"].mean()
    neigh_sums_90 = train_90.groupby("Neighborhood")["SalePrice"].sum()
    neigh_counts_90 = train_90.groupby("Neighborhood")["SalePrice"].count()

    train_90_linear = train_90.copy()

    n_c_tr90 = train_90["Neighborhood"].map(neigh_counts_90).fillna(0)
    sum_c_tr90 = train_90["Neighborhood"].map(neigh_sums_90).fillna(0)

    denominator_tr90 = n_c_tr90 - 1 + m
    numerator_tr90 = sum_c_tr90 - train_90["SalePrice"] + m * global_mean_90

    train_90_linear["Neighborhood"] = np.where(denominator_tr90 > 0, numerator_tr90 / denominator_tr90, global_mean_90)

    calib_linear = calib_10.copy()

    n_c_ca = calib_10["Neighborhood"].map(neigh_counts_90).fillna(0)
    sum_c_ca = calib_10["Neighborhood"].map(neigh_sums_90).fillna(0)

    denominator_ca = n_c_ca + m
    numerator_ca = sum_c_ca + m * global_mean_90

    calib_linear["Neighborhood"] = np.where(denominator_ca > 0, numerator_ca / denominator_ca, global_mean_90)

    # 2. Transform 90/10
    tree_tf_90 = AmesDataTransformer(vif_prune=False)
    X_tr90_tree = tree_tf_90.fit_transform(train_90_tree.drop(columns=["Id", "SalePrice"]))
    X_ca_tree = tree_tf_90.transform(calib_tree.drop(columns=["Id", "SalePrice"]))

    cat_features = X_tr90_tree.select_dtypes(include=["object", "str"]).columns.tolist()
    for col in cat_features:
        X_tr90_tree[col] = X_tr90_tree[col].fillna("Missing").astype(str)
        X_ca_tree[col] = X_ca_tree[col].fillna("Missing").astype(str)

    linear_tf_90 = AmesDataTransformer(vif_prune=True)
    X_tr90_linear = linear_tf_90.fit_transform(train_90_linear.drop(columns=["Id", "SalePrice"]))
    X_ca_linear = linear_tf_90.transform(calib_linear.drop(columns=["Id", "SalePrice"]))

    y_tr90 = train_90["SalePrice"].values
    y_tr90_log = np.log1p(y_tr90)

    xgb_90 = clone(xgb_base)
    xgb_90.fit(X_tr90_tree, y_tr90_log)
    calib_xgb_log = xgb_90.predict(X_ca_tree)

    cat_90 = clone(cat_base)
    cat_90.fit(X_tr90_tree, y_tr90_log, cat_features=cat_features, verbose=False)
    calib_cat_log = cat_90.predict(X_ca_tree)

    ridge_90 = clone(ridge_model)
    ridge_90.fit(X_tr90_linear, y_tr90)
    calib_ridge_log = np.log1p(np.clip(ridge_90.predict(X_ca_linear), 1, None))

    calib_ensemble_log = (
        best_weights[0] * calib_xgb_log + best_weights[1] * calib_cat_log + best_weights[2] * calib_ridge_log
    )

    y_calib_log = np.log1p(calib_10["SalePrice"].values)
    residuals = compute_non_conformity_scores(y_calib_log, calib_ensemble_log)
    q = compute_empirical_quantile(residuals, alpha=0.05)

    return q


def train_final_models_and_predict(train_full, test, best_weights, q, xgb_base, cat_base, ridge_model):
    print("\n--- Training 100% Final Models & Predicting Test ---")
    m = 20
    # 1. 100% Fold-local Neighborhood Ranking (Trees)
    test_tree = get_neighborhood_ranks(train_full, test)
    train_full_tree = get_neighborhood_ranks(train_full, train_full)

    # 100% Linear Encoding
    global_mean_100 = train_full["SalePrice"].mean()
    neigh_sums_100 = train_full.groupby("Neighborhood")["SalePrice"].sum()
    neigh_counts_100 = train_full.groupby("Neighborhood")["SalePrice"].count()

    train_full_linear = train_full.copy()

    n_c_tr100 = train_full["Neighborhood"].map(neigh_counts_100).fillna(0)
    sum_c_tr100 = train_full["Neighborhood"].map(neigh_sums_100).fillna(0)

    denominator_tr100 = n_c_tr100 - 1 + m
    numerator_tr100 = sum_c_tr100 - train_full["SalePrice"] + m * global_mean_100

    train_full_linear["Neighborhood"] = np.where(denominator_tr100 > 0, numerator_tr100 / denominator_tr100, global_mean_100)

    test_linear = test.copy()

    n_c_te = test["Neighborhood"].map(neigh_counts_100).fillna(0)
    sum_c_te = test["Neighborhood"].map(neigh_sums_100).fillna(0)

    denominator_te = n_c_te + m
    numerator_te = sum_c_te + m * global_mean_100

    test_linear["Neighborhood"] = np.where(denominator_te > 0, numerator_te / denominator_te, global_mean_100)

    # 2. Transform 100%
    tree_tf_100 = AmesDataTransformer(vif_prune=False)
    X_tr100_tree = tree_tf_100.fit_transform(train_full_tree.drop(columns=["Id", "SalePrice"]))
    X_te_tree = tree_tf_100.transform(test_tree.drop(columns=["Id"]))

    cat_features = X_tr100_tree.select_dtypes(include=["object", "str"]).columns.tolist()
    for col in cat_features:
        X_tr100_tree[col] = X_tr100_tree[col].fillna("Missing").astype(str)
        X_te_tree[col] = X_te_tree[col].fillna("Missing").astype(str)

    linear_tf_100 = AmesDataTransformer(vif_prune=True)
    X_tr100_linear = linear_tf_100.fit_transform(train_full_linear.drop(columns=["Id", "SalePrice"]))
    X_te_linear = linear_tf_100.transform(test_linear.drop(columns=["Id"]))

    y_full = train_full["SalePrice"].values
    y_full_log = np.log1p(y_full)

    # 3. Fit 100%
    xgb_full = clone(xgb_base)
    xgb_full.fit(X_tr100_tree, y_full_log)
    test_xgb_log = xgb_full.predict(X_te_tree)

    cat_full = clone(cat_base)
    cat_full.fit(X_tr100_tree, y_full_log, cat_features=cat_features, verbose=False)
    test_cat_log = cat_full.predict(X_te_tree)

    ridge_full = clone(ridge_model)
    ridge_full.fit(X_tr100_linear, y_full)
    test_ridge_log = np.log1p(np.clip(ridge_full.predict(X_te_linear), 1, None))

    test_ensemble_log = best_weights[0] * test_xgb_log + best_weights[1] * test_cat_log + best_weights[2] * test_ridge_log

    y_pred_point, lower_bound, upper_bound = compute_prediction_intervals(
        test_ensemble_log, q, min_physical_price=42000.0, max_physical_price=525000.0
    )

    return y_pred_point, lower_bound, upper_bound


def save_submissions(test_ids, y_pred_point, lower_bound, upper_bound):
    os.makedirs("./submissions", exist_ok=True)

    sub = pd.DataFrame({"Id": test_ids, "SalePrice": y_pred_point})
    sub.to_csv("submissions/submission_bifurcated.csv", index=False)

    sub_int = pd.DataFrame(
        {"Id": test_ids, "SalePrice": y_pred_point, "SalePrice_Lower": lower_bound, "SalePrice_Upper": upper_bound}
    )
    sub_int.to_csv("submissions/submission_with_intervals_bifurcated.csv", index=False)

    print("\n✅ Bifurcated Pipeline Complete! Submissions saved.")


def main():
    train_full, train_90, calib_10, test = load_and_split_data()
    best_xgb_params, best_cat_params = tune_models(train_full)

    y_full = train_full["SalePrice"].values
    y_full_log = np.log1p(y_full)

    xgb_base = xgb.XGBRegressor(**best_xgb_params)
    cat_base = CatBoostRegressor(**best_cat_params)

    alphas_ridge = np.logspace(-3, 3, 50)
    ridge_base = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_ridge, cv=5))
    ridge_model = TransformedTargetRegressor(
        regressor=ridge_base, transformer=QuantileTransformer(n_quantiles=900, output_distribution="normal")
    )

    oof_xgb, oof_cat, oof_ridge = generate_oof_predictions(train_full, xgb_base, cat_base, ridge_model)

    preds = np.column_stack([oof_xgb, oof_cat, oof_ridge])
    best_weights = calculate_ensemble_weights(preds, y_full_log)

    q = calculate_conformal_bounds(train_90, calib_10, best_weights, xgb_base, cat_base, ridge_model)

    y_pred_point, lower_bound, upper_bound = train_final_models_and_predict(
        train_full, test, best_weights, q, xgb_base, cat_base, ridge_model
    )

    save_submissions(test["Id"], y_pred_point, lower_bound, upper_bound)


if __name__ == "__main__":
    main()
