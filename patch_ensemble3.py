main_code = """
import joblib
import pandas as pd
import numpy as np
import os
import json

from src.conformal import (
    compute_non_conformity_scores,
    compute_empirical_quantile,
    compute_prediction_intervals
)

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

    # Load 100% Models and 90% Models
    xgb_model_full = joblib.load("./models/xgboost_best_rmsle.pkl")
    catboost_model_full = joblib.load("./models/catboost_best_rmsle.pkl")
    pt_full = joblib.load("./models/boxcox_transformer.pkl")

    xgb_model_90 = joblib.load("./models/xgboost_90.pkl")
    catboost_model_90 = joblib.load("./models/catboost_90.pkl")
    pt_90 = joblib.load("./models/boxcox_transformer_90.pkl")

    with open("./models/ensemble_weights.json", "r") as f:
        weight_dict = json.load(f)

    weight_xgb = weight_dict.get("xgb", 0.5)
    weight_catboost = weight_dict.get("catboost", 0.5)

    print("✅ Models and transformer loaded successfully.")

    # ============================================
    # GENERATE CALIBRATION PREDICTIONS (Using 90% Models)
    # ============================================
    print("\\n" + "=" * 60)
    print("GENERATING CALIBRATION PREDICTIONS (90% MODELS)")
    print("=" * 60)

    xgb_calib_transformed = xgb_model_90.predict(X_calib)
    catboost_calib_transformed = catboost_model_90.predict(X_calib_raw)

    xgb_calib_original = pt_90.inverse_transform(xgb_calib_transformed.reshape(-1, 1)).flatten()
    catboost_calib_original = pt_90.inverse_transform(catboost_calib_transformed.reshape(-1, 1)).flatten()

    ensemble_calib_original = (weight_xgb * xgb_calib_original + weight_catboost * catboost_calib_original)

    y_calib_log = np.log1p(y_calib)
    ensemble_calib_log = np.log1p(ensemble_calib_original)

    residuals = compute_non_conformity_scores(y_calib_log, ensemble_calib_log)
    q = compute_empirical_quantile(residuals, alpha=0.05)
    print(f"✅ Empirical quantile 'q' (95% coverage) in log-space: {q:.6f}")

    # ============================================
    # GENERATE TEST PREDICTIONS (Using 100% Models)
    # ============================================
    print("\\n" + "=" * 60)
    print("GENERATING TEST PREDICTIONS (100% MODELS)")
    print("=" * 60)

    xgb_pred_transformed = xgb_model_full.predict(X_test)
    catboost_pred_transformed = catboost_model_full.predict(X_test_raw)

    xgb_pred_original = pt_full.inverse_transform(xgb_pred_transformed.reshape(-1, 1)).flatten()
    catboost_pred_original = pt_full.inverse_transform(catboost_pred_transformed.reshape(-1, 1)).flatten()

    ensemble_pred_original = (weight_xgb * xgb_pred_original + weight_catboost * catboost_pred_original)
    ensemble_pred_log = np.log1p(ensemble_pred_original)

    y_pred_point, lower_bound, upper_bound = compute_prediction_intervals(
        ensemble_pred_log, q, min_physical_price=42000.0, max_physical_price=525000.0
    )

    # ============================================
    # INDUCTIVE CONFORMAL PREDICTION (ICP)
    # ============================================
    print("\\n" + "=" * 60)
    print("SAVING CONFORMAL PREDICTION INTERVALS")
    print("=" * 60)

    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": y_pred_point,
    })

    submission_intervals = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": y_pred_point,
        "SalePrice_Lower": lower_bound,
        "SalePrice_Upper": upper_bound
    })

    os.makedirs("./submissions", exist_ok=True)

    submission.to_csv("submissions/submission.csv", index=False)
    submission_intervals.to_csv("submissions/submission_with_intervals.csv", index=False)

    print("✅ Submission files saved.")
    print(f"   submission.csv: {submission.shape}")
    print(f"   submission_with_intervals.csv: {submission_intervals.shape}")

if __name__ == "__main__":
    main()
"""
with open('src/ensemble.py', 'w') as f:
    f.write(main_code)
