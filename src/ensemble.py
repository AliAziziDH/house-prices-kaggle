import json
import os

import joblib
import numpy as np
import pandas as pd

from src.conformal import compute_empirical_quantile, compute_non_conformity_scores, compute_prediction_intervals


def main():
    print("=" * 60)
    print("LOADING MODELS AND DATA")
    print("=" * 60)

    X_test = pd.read_csv("./processed_data/X_test.csv")
    test_ids = pd.read_csv("./data/test.csv")["Id"]
    X_calib = pd.read_csv("./processed_data/X_calib.csv")
    y_calib = pd.read_csv("./processed_data/y_calib.csv").squeeze()
    X_calib_raw = pd.read_csv("./processed_data/X_calib_raw.csv")
    X_test_raw = pd.read_csv("./processed_data/X_test_raw.csv")

    cat_features = joblib.load("./models/cat_features.pkl")
    for col in cat_features:
        X_calib_raw[col] = X_calib_raw[col].fillna("Missing").astype(str)
        X_test_raw[col] = X_test_raw[col].fillna("Missing").astype(str)

    xgb_model_full = joblib.load("./models/xgboost_best_rmsle.pkl")
    catboost_model_full = joblib.load("./models/catboost_best_rmsle.pkl")
    lightgbm_model_full = joblib.load("./models/lightgbm_best_rmsle.pkl")
    lasso_model_full = joblib.load("./models/lasso_best_rmsle.pkl")
    ridge_model_full = joblib.load("./models/ridge_best_rmsle.pkl")
    elasticnet_model_full = joblib.load("./models/elasticnet_best_rmsle.pkl")

    xgb_model_90 = joblib.load("./models/xgboost_90.pkl")
    catboost_model_90 = joblib.load("./models/catboost_90.pkl")
    lightgbm_model_90 = joblib.load("./models/lightgbm_90.pkl")
    lasso_model_90 = joblib.load("./models/lasso_90.pkl")
    ridge_model_90 = joblib.load("./models/ridge_90.pkl")
    elasticnet_model_90 = joblib.load("./models/elasticnet_90.pkl")

    with open("./models/ensemble_weights.json", "r") as f:
        weight_dict = json.load(f)

    weight_xgb = weight_dict.get("xgb", 0.0)
    weight_catboost = weight_dict.get("catboost", 0.0)
    weight_lightgbm = weight_dict.get("lightgbm", 0.0)
    weight_lasso = weight_dict.get("lasso", 0.0)
    weight_ridge = weight_dict.get("ridge", 0.0)
    weight_elasticnet = weight_dict.get("elasticnet", 0.0)

    print("✅ Models loaded successfully.")

    print("\n" + "=" * 60)
    print("GENERATING CALIBRATION PREDICTIONS (90% MODELS)")
    print("=" * 60)

    raw_train = pd.read_csv("./data/train.csv")
    raw_train = raw_train[~((raw_train["GrLivArea"] > 4000) & (raw_train["SalePrice"] < 200000))].reset_index(drop=True)
    global_mean_full = raw_train["SalePrice"].mean()
    neigh_sums_full = raw_train.groupby("Neighborhood")["SalePrice"].sum()
    neigh_counts_full = raw_train["Neighborhood"].value_counts()
    from sklearn.model_selection import train_test_split

    raw_train_90, _ = train_test_split(raw_train, test_size=0.1, random_state=42)
    global_mean_90 = raw_train_90["SalePrice"].mean()
    neigh_sums_90 = raw_train_90.groupby("Neighborhood")["SalePrice"].sum()
    neigh_counts_90 = raw_train_90["Neighborhood"].value_counts()
    m = 20
    X_calib_linear = X_calib.copy()
    encodings_calib = []
    for n in X_calib_raw["Neighborhood"]:
        sum_c = neigh_sums_90.get(n, 0)
        n_c = neigh_counts_90.get(n, 0)
        enc = (sum_c + m * global_mean_90) / (n_c + m) if (n_c + m) > 0 else global_mean_90
        encodings_calib.append(enc)
    X_calib_linear["Neighborhood"] = encodings_calib
    X_test_linear = X_test.copy()
    encodings_test = []
    for n in X_test_raw["Neighborhood"]:
        sum_c = neigh_sums_full.get(n, 0)
        n_c = neigh_counts_full.get(n, 0)
        enc = (sum_c + m * global_mean_full) / (n_c + m) if (n_c + m) > 0 else global_mean_full
        encodings_test.append(enc)
    X_test_linear["Neighborhood"] = encodings_test

    xgb_calib_log = xgb_model_90.predict(X_calib)
    catboost_calib_log = catboost_model_90.predict(X_calib_raw)
    lightgbm_calib_log = lightgbm_model_90.predict(X_calib)

    lasso_calib_log = np.log1p(np.clip(lasso_model_90.predict(X_calib_linear), 1, None))
    ridge_calib_log = np.log1p(np.clip(ridge_model_90.predict(X_calib_linear), 1, None))
    elasticnet_calib_log = np.log1p(np.clip(elasticnet_model_90.predict(X_calib_linear), 1, None))

    ensemble_calib_log = (
        weight_xgb * xgb_calib_log
        + weight_catboost * catboost_calib_log
        + weight_lightgbm * lightgbm_calib_log
        + weight_lasso * lasso_calib_log
        + weight_ridge * ridge_calib_log
        + weight_elasticnet * elasticnet_calib_log
    )

    y_calib_log = np.log1p(y_calib)
    residuals = compute_non_conformity_scores(y_calib_log, ensemble_calib_log)
    q = compute_empirical_quantile(residuals, alpha=0.05)
    print(f"✅ Empirical quantile 'q' (95% coverage) in log-space: {q:.6f}")

    print("\n" + "=" * 60)
    print("GENERATING TEST PREDICTIONS (100% MODELS)")
    print("=" * 60)

    xgb_test_log = xgb_model_full.predict(X_test)
    catboost_test_log = catboost_model_full.predict(X_test_raw)
    lightgbm_test_log = lightgbm_model_full.predict(X_test)

    lasso_test_log = np.log1p(np.clip(lasso_model_full.predict(X_test_linear), 1, None))
    ridge_test_log = np.log1p(np.clip(ridge_model_full.predict(X_test_linear), 1, None))
    elasticnet_test_log = np.log1p(np.clip(elasticnet_model_full.predict(X_test_linear), 1, None))

    ensemble_pred_log = (
        weight_xgb * xgb_test_log
        + weight_catboost * catboost_test_log
        + weight_lightgbm * lightgbm_test_log
        + weight_lasso * lasso_test_log
        + weight_ridge * ridge_test_log
        + weight_elasticnet * elasticnet_test_log
    )

    y_pred_point, lower_bound, upper_bound = compute_prediction_intervals(
        ensemble_pred_log, q, min_physical_price=42000.0, max_physical_price=525000.0
    )

    print("\n" + "=" * 60)
    print("SAVING CONFORMAL PREDICTION INTERVALS")
    print("=" * 60)

    submission = pd.DataFrame({"Id": test_ids, "SalePrice": y_pred_point})
    submission_intervals = pd.DataFrame(
        {
            "Id": test_ids,
            "SalePrice": y_pred_point,
            "SalePrice_Lower": lower_bound,
            "SalePrice_Upper": upper_bound,
        }
    )

    os.makedirs("./submissions", exist_ok=True)
    submission.to_csv("submissions/submission.csv", index=False)
    submission_intervals.to_csv("submissions/submission_with_intervals.csv", index=False)

    print("✅ Submission files saved.")
    print(f"   submission.csv: {submission.shape}")
    print(f"   submission_with_intervals.csv: {submission_intervals.shape}")


if __name__ == "__main__":
    main()
