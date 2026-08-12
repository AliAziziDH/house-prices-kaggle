import joblib
import pandas as pd

# ============================================
# LOAD MODELS AND TRANSFORMER
# ============================================
print("Loading models...")
xgb_model = joblib.load("./models/xgboost_best_rmsle.pkl")
catboost_model = joblib.load("./models/catboost_best_rmsle.pkl")
pt = joblib.load("./models/boxcox_transformer.pkl")
print("✅ Models loaded.")

# ============================================
# LOAD TEST DATA (ONE-HOT ENCODED)
# ============================================
print("Loading test data...")
X_test = pd.read_csv("./processed_data/X_test.csv")  # One-hot encoded data
test_ids = pd.read_csv("./data/test.csv")["Id"]
print("✅ Test data loaded.")

# ============================================
# PREDICT
# ============================================
print("Making predictions...")
xgb_pred = xgb_model.predict(X_test)
catboost_pred = catboost_model.predict(X_test)  # Using the same one-hot encoded data

# Inverse transform to original scale
xgb_pred_orig = pt.inverse_transform(xgb_pred.reshape(-1, 1)).flatten()
catboost_pred_orig = pt.inverse_transform(catboost_pred.reshape(-1, 1)).flatten()
print("✅ Predictions made and inverse transformed.")

# ============================================
# ENSEMBLE
# ============================================
weight_xgb = 0.64
weight_catboost = 0.36
final_pred = weight_xgb * xgb_pred_orig + weight_catboost * catboost_pred_orig

import numpy as np
final_pred = np.clip(final_pred, 42000, 525000)

import os

os.makedirs("./submissions", exist_ok=True)

submission = pd.DataFrame({"Id": test_ids, "SalePrice": final_pred})
submission.to_csv("./submissions/submission_ensemble_final.csv", index=False)

print("✅ Final submission saved to submissions/submission_ensemble_final.csv")
print(f"   Shape: {submission.shape}")
print("   First 5 rows:")
print(submission.head())
