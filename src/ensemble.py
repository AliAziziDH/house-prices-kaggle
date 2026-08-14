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
    ridge_model_full = joblib.load("./models/ridge_best_rmsle.pkl")

    xgb_model_90 = joblib.load("./models/xgboost_90.pkl")
    catboost_model_90 = joblib.load("./models/catboost_90.pkl")
    ridge_model_90 = joblib.load("./models/ridge_90.pkl")

    with open("./models/ensemble_weights.json", "r") as f:
        weight_dict = json.load(f)

    weight_xgb = weight_dict.get("xgb", 0.0)
    weight_catboost = weight_dict.get("catboost", 0.0)
    weight_ridge = weight_dict.get("ridge", 0.0)

    print("✅ Models loaded successfully.")

    print("\n" + "=" * 60)
    print("GENERATING CALIBRATION PREDICTIONS (90% MODELS)")
    print("=" * 60)

    X_calib_linear = pd.read_csv("./processed_data/X_calib_linear.csv")

    xgb_calib_log = xgb_model_90.predict(X_calib)
    catboost_calib_log = catboost_model_90.predict(X_calib_raw)

    # Neighborhood encoding handled by transformer now, X_calib_linear is ready
    ridge_calib_log = np.log1p(np.clip(ridge_model_90.predict(X_calib_linear), 1, None))

    ensemble_calib_log = (
        weight_xgb * xgb_calib_log
        + weight_catboost * catboost_calib_log
        + weight_ridge * ridge_calib_log
    )

    y_calib_log = np.log1p(y_calib)
    residuals = compute_non_conformity_scores(y_calib_log, ensemble_calib_log)
    q = compute_empirical_quantile(residuals, alpha=0.05)
    print(f"✅ Empirical quantile 'q' (95% coverage) in log-space: {q:.6f}")

    print("\n" + "=" * 60)
    print("GENERATING TEST PREDICTIONS (100% MODELS)")
    print("=" * 60)

    X_test_linear = pd.read_csv("./processed_data/X_test_linear.csv")

    xgb_test_log = xgb_model_full.predict(X_test)
    catboost_test_log = catboost_model_full.predict(X_test_raw)

    ridge_test_log = np.log1p(np.clip(ridge_model_full.predict(X_test_linear), 1, None))

    ensemble_pred_log = (
        weight_xgb * xgb_test_log
        + weight_catboost * catboost_test_log
        + weight_ridge * ridge_test_log
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
            "LowerBound": lower_bound,
            "UpperBound": upper_bound,
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
