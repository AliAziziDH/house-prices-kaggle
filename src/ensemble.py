import joblib
import pandas as pd

# ============================================
# LOAD MODELS AND DATA
# ============================================
print("=" * 60)
print("LOADING MODELS AND DATA")
print("=" * 60)

# Load test data
X_test = pd.read_csv("./processed_data/X_test.csv")
test_ids = pd.read_csv("./data/test.csv")["Id"]

# Load trained models (we only use the best ones)
xgb_model = joblib.load("./models/xgboost_best_rmsle.pkl")
catboost_model = joblib.load("./models/catboost_best_rmsle.pkl")

# Load the transformer (Box-Cox)
pt = joblib.load("./models/boxcox_transformer.pkl")

print("✅ Models and transformer loaded successfully.")

# ============================================
# GENERATE PREDICTIONS
# ============================================
print("\n" + "=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

# Predict from each model (in transformed scale)
xgb_pred_transformed = xgb_model.predict(X_test)
catboost_pred_transformed = catboost_model.predict(X_test)

# Inverse transform to original scale (dollars)
xgb_pred_original = pt.inverse_transform(xgb_pred_transformed.reshape(-1, 1)).flatten()
catboost_pred_original = pt.inverse_transform(
    catboost_pred_transformed.reshape(-1, 1)
).flatten()

print("✅ Predictions generated for both models.")

# ============================================
# WEIGHTED AVERAGE ENSEMBLE
# ============================================
print("\n" + "=" * 60)
print("ENSEMBLE: WEIGHTED AVERAGE")
print("=" * 60)

# Best weights from optimization (aligned with README)
weight_xgb = 0.64
weight_catboost = 0.36

# Calculate weighted average
ensemble_pred = (
    weight_xgb * xgb_pred_original + weight_catboost * catboost_pred_original
)

print(f"Weights: XGBoost = {weight_xgb:.2f}, CatBoost = {weight_catboost:.2f}")

# ============================================
# CREATE SUBMISSION
# ============================================
print("\n" + "=" * 60)
print("CREATING SUBMISSION FILE")
print("=" * 60)

submission = pd.DataFrame({"Id": test_ids, "SalePrice": ensemble_pred})

import os

os.makedirs("./submissions", exist_ok=True)

submission.to_csv("./submissions/submission_ensemble_final.csv", index=False)
print("✅ Submission file saved as './submissions/submission_ensemble_final.csv'")
print(f"   Shape: {submission.shape}")
print("   First 5 rows:")
print(submission.head())
