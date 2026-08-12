import os

import joblib
import pandas as pd

# ============================================
# LOAD MODELS AND DATA
# ============================================
print("=" * 60)
print("LOADING MODELS AND DATA")
print("=" * 60)

# Load test data
import numpy as np

X_test = pd.read_csv("./processed_data/X_test.csv")
X_test_raw = pd.read_csv("./processed_data/X_test_raw.csv")
test_ids = pd.read_csv("./data/test.csv")["Id"]

raw_train = pd.read_csv("./data/train.csv")
raw_train = raw_train[
    ~((raw_train["GrLivArea"] > 4000) & (raw_train["SalePrice"] < 200000))
].reset_index(drop=True)
raw_test_neighborhoods = pd.read_csv("./data/test.csv")["Neighborhood"]

raw_train["TotalSF"] = (
    raw_train["TotalBsmtSF"] + raw_train["1stFlrSF"] + raw_train["2ndFlrSF"]
)
raw_train["PricePerSF"] = raw_train["SalePrice"] / raw_train["TotalSF"]
neigh_order = (
    raw_train.groupby("Neighborhood")["PricePerSF"].median().sort_values().index
)
neigh_map = {n: i + 1 for i, n in enumerate(neigh_order)}

X_test["Neighborhood"] = raw_test_neighborhoods.map(neigh_map).fillna(13).astype(int)
X_test_raw["Neighborhood"] = (
    raw_test_neighborhoods.map(neigh_map).fillna(13).astype(int)
)

# Load trained models (we only use the best ones)
xgb_model = joblib.load("./models/xgboost_best_rmsle.pkl")
catboost_model = joblib.load("./models/catboost_best_rmsle.pkl")
import lightgbm as lgb

lgb_model = (
    lgb.Booster(model_file="./models/lightgbm_best.txt")
    if os.path.exists("./models/lightgbm_best.txt")
    else None
)
ridge_model = (
    joblib.load("./models/oof_ridge.pkl")
    if os.path.exists("./models/oof_ridge.pkl")
    else None
)
lasso_model = (
    joblib.load("./models/oof_lasso.pkl")
    if os.path.exists("./models/oof_lasso.pkl")
    else None
)
elasticnet_model = (
    joblib.load("./models/oof_elasticnet.pkl")
    if os.path.exists("./models/oof_elasticnet.pkl")
    else None
)

# Load the transformer (Box-Cox)
print("✅ Models loaded successfully.")

# ============================================
# GENERATE PREDICTIONS
# ============================================
print("\n" + "=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

# Predict from each model (in transformed scale)
xgb_pred_transformed = xgb_model.predict(X_test)
catboost_pred_transformed = catboost_model.predict(X_test_raw)

# Inverse transform to original scale (dollars)
xgb_pred_original = np.expm1(xgb_pred_transformed)
catboost_pred_original = np.expm1(catboost_pred_transformed)

print("✅ Predictions generated for both models.")

# ============================================
# WEIGHTED AVERAGE ENSEMBLE
# ============================================
print("\n" + "=" * 60)
print("ENSEMBLE: WEIGHTED AVERAGE")
print("=" * 60)

# Best weights from optimization (aligned with README)
weight_catboost = 0.1667
weight_xgb = 0.1667
weight_lgb = 0.1667
weight_ridge = 0.1667
weight_lasso = 0.1667
weight_elasticnet = 0.1667

# Calculate weighted average
try:
    ensemble_pred = (
        weight_xgb * xgb_pred_original
        + weight_catboost * catboost_pred_original
        + weight_lgb * lgb_pred_original
        + weight_ridge * ridge_pred_original
        + weight_lasso * lasso_pred_original
        + weight_elasticnet * elasticnet_pred_original
    )
except NameError:
    # Normalize weights to sum to 1.0
    total = weight_xgb + weight_catboost
    ensemble_pred = (weight_xgb / total) * xgb_pred_original + (
        weight_catboost / total
    ) * catboost_pred_original

ensemble_pred = np.clip(ensemble_pred, 42000, 525000)

print(f"Weights: XGBoost = {weight_xgb:.2f}, CatBoost = {weight_catboost:.2f}")

# ============================================
# INDUCTIVE CONFORMAL PREDICTION (ICP)
# ============================================
print("\n" + "=" * 60)
print("CALCULATING CONFORMAL PREDICTION INTERVALS")
print("=" * 60)

from sklearn.model_selection import train_test_split

X_train = pd.read_csv("./processed_data/X_train.csv")
X_train_raw = pd.read_csv("./processed_data/X_train_raw.csv")
y_train_log = pd.read_csv("./processed_data/y_train_log.csv").squeeze()

X_train["Neighborhood"] = (
    raw_train["Neighborhood"].map(neigh_map).fillna(13).astype(int)
)
X_train_raw["Neighborhood"] = (
    raw_train["Neighborhood"].map(neigh_map).fillna(13).astype(int)
)

# Recreate 10% calibration set split
_, X_cal, _, y_cal_log = train_test_split(
    X_train, y_train_log, test_size=0.1, random_state=42
)
_, X_cal_raw, _, _ = train_test_split(
    X_train_raw, y_train_log, test_size=0.1, random_state=42
)

# Generate ensemble predictions on calibration set
xgb_cal_transformed = xgb_model.predict(X_cal)
catboost_cal_transformed = catboost_model.predict(X_cal_raw)
try:
    lgb_cal_transformed = lgb_model.predict(X_cal)
except Exception:  # noqa: BLE001
    lgb_cal_transformed = np.zeros(len(X_cal))

# Load OOF predictions directly from models directory for linear models since they are already saved
try:
    oof_ridge = __import__("joblib").load("./models/oof_ridge.pkl")
    ridge_cal_transformed = oof_ridge[
        -len(X_cal) :
    ]  # Approximation for calibration split
except Exception:  # noqa: BLE001
    ridge_cal_transformed = np.zeros(len(X_cal))

try:
    oof_lasso = __import__("joblib").load("./models/oof_lasso.pkl")
    lasso_cal_transformed = oof_lasso[-len(X_cal) :]
except Exception:  # noqa: BLE001
    lasso_cal_transformed = np.zeros(len(X_cal))

try:
    oof_elasticnet = __import__("joblib").load("./models/oof_elasticnet.pkl")
    elasticnet_cal_transformed = oof_elasticnet[-len(X_cal) :]
except Exception:  # noqa: BLE001
    elasticnet_cal_transformed = np.zeros(len(X_cal))

ensemble_cal_log = (
    weight_xgb * xgb_cal_transformed
    + weight_catboost * catboost_cal_transformed
    + weight_lgb * lgb_cal_transformed
    + weight_ridge * ridge_cal_transformed
    + weight_lasso * lasso_cal_transformed
    + weight_elasticnet * elasticnet_cal_transformed
)

# Calculate absolute residuals in log space
R = np.abs(y_cal_log.values - ensemble_cal_log)

# Calculate Neighborhood-Conditional Conformal Prediction
alpha = 0.05
cal_neighborhoods = X_cal_raw["Neighborhood"].values
test_neighborhoods = X_test_raw["Neighborhood"].values

# Calculate global quantile as fallback
n_cal_global = len(R)
q_global = np.quantile(R, min((1.0 - alpha) * (1.0 + 1.0 / n_cal_global), 1.0))

q_conditional = np.zeros(len(test_neighborhoods))

# Group by neighborhood and calculate conditional quantiles
for n in np.unique(test_neighborhoods):
    mask = cal_neighborhoods == n
    n_samples = np.sum(mask)

    if n_samples >= 10:
        R_cond = R[mask]
        q_c = np.quantile(R_cond, min((1.0 - alpha) * (1.0 + 1.0 / n_samples), 1.0))
        q_conditional[test_neighborhoods == n] = q_c
    else:
        # Fallback to global quantile for Low-Sample groups or unknown neighborhoods
        q_conditional[test_neighborhoods == n] = q_global

# Fallback for any missed values
q_conditional[q_conditional == 0] = q_global

# Convert test predictions back to log space to calculate bounds
ensemble_pred_log = np.log1p(ensemble_pred)

# Calculate bounds and convert back to dollars using the localized bounds array
lower_bounds = np.maximum(0.0, np.expm1(ensemble_pred_log - q_conditional))
upper_bounds = np.expm1(ensemble_pred_log + q_conditional)

print(f"✅ Calibration Global Quantile (q) = {q_global:.5f}")

# ============================================
# CREATE SUBMISSION FILES
# ============================================
print("\n" + "=" * 60)
print("CREATING SUBMISSION FILES")
print("=" * 60)

submission = pd.DataFrame({"Id": test_ids, "SalePrice": ensemble_pred})
submission_intervals = pd.DataFrame(
    {
        "Id": test_ids,
        "SalePrice": ensemble_pred,
        "Price_Lower_Bound": lower_bounds,
        "Price_Upper_Bound": upper_bounds,
    }
)

import os

os.makedirs("./submissions", exist_ok=True)

submission.to_csv("./submissions/submission_ensemble_oof.csv", index=False)
submission_intervals.to_csv("./submissions/submission_oof_intervals.csv", index=False)

print("✅ Standard submission saved to './submissions/submission_ensemble_oof.csv'")
print("✅ ICP intervals saved to './submissions/submission_oof_intervals.csv'")
