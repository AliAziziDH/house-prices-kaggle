"""
Optimal Weighted Ensemble & Stacking with 6 Diverse Base Models
Combines CatBoost, XGBoost, LightGBM, Ridge, Lasso, and ElasticNet using Scipy SLSQP optimization.
"""

import os
import warnings
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import minimize, minimize_scalar
from sklearn.linear_model import ElasticNetCV, LassoCV, Ridge
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ============================================
# CONFIGURATION
# ============================================
RANDOM_STATE: int = 42
N_FOLDS: int = 5

# ============================================
# LOAD DATA
# ============================================
print("=" * 60)
print("LOADING PROCESSED DATA FOR ENSEMBLE")
print("=" * 60)

X_train = pd.read_csv("./processed_data/X_train.csv")
X_test = pd.read_csv("./processed_data/X_test.csv")
y_train_log = pd.read_csv("./processed_data/y_train_log.csv").squeeze()

X_train_raw = pd.read_csv("./processed_data/X_train_raw.csv")
X_test_raw = pd.read_csv("./processed_data/X_test_raw.csv")

# Load original raw train data to prevent target leakage in Neighborhood encoding
raw_train = pd.read_csv("./data/train.csv")
raw_train = raw_train[
    ~((raw_train["GrLivArea"] > 4000) & (raw_train["SalePrice"] < 300000))
].reset_index(drop=True)
raw_neighborhoods = raw_train["Neighborhood"]

cat_features = X_train_raw.select_dtypes(include=["object"]).columns.tolist()
for col in cat_features:
    X_train_raw[col] = X_train_raw[col].fillna("Missing").astype(str)
    X_test_raw[col] = X_test_raw[col].fillna("Missing").astype(str)

test_ids = pd.read_csv("./data/test.csv")["Id"]

print(f"X_train shape: {X_train.shape}")
print(f"X_train_raw shape: {X_train_raw.shape}")
print(f"y_train_log shape: {y_train_log.shape}")

# ============================================
# LOAD BEST PARAMETERS FROM OPTUNA STUDIES
# ============================================
print("\n" + "=" * 60)
print("LOADING OPTIMAL HYPERPARAMETERS")
print("=" * 60)

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "max_depth": 4,  # Clamped to 4
    "learning_rate": 0.010110231750868986,
    "subsample": 0.5034728283142227,
    "colsample_bytree": 0.504544868823299,
    "min_child_weight": 2,
    "gamma": 0.0018151575742466645,
    "reg_alpha": 0.01606690038615389,
    "reg_lambda": 0.019342719287003877,
}

DEFAULT_LGB_PARAMS: dict[str, Any] = {
    "max_depth": 4,
    "num_leaves": 63,
    "learning_rate": 0.021194390248830346,
    "subsample": 0.6126066923442381,
    "colsample_bytree": 0.40036174914879236,
    "min_child_samples": 16,
    "reg_alpha": 9.381553163911187e-06,
    "reg_lambda": 1.7472249548155823e-06,
}

DEFAULT_CAT_PARAMS: dict[str, Any] = {
    "depth": 4,  # Clamped to 4
    "learning_rate": 0.020978632623778505,
    "l2_leaf_reg": 0.012607414383039797,
    "subsample": 0.6441998268583774,
    "random_strength": 2.4414129060841185,
}

try:
    xgb_study = optuna.load_study(
        study_name="xgboost_optimization_log_target",
        storage=f"sqlite:///{os.path.abspath('./experiments/xgboost_study_log.db')}",
    )
    best_params_xgb = xgb_study.best_params
except Exception:  # noqa: BLE001
    best_params_xgb = DEFAULT_XGB_PARAMS

try:
    lgb_study = optuna.load_study(
        study_name="lightgbm_optimization_log_target",
        storage=f"sqlite:///{os.path.abspath('./experiments/lightgbm_study_log.db')}",
    )
    best_params_lgb = lgb_study.best_params
except Exception:  # noqa: BLE001
    best_params_lgb = DEFAULT_LGB_PARAMS

try:
    cat_study = optuna.load_study(
        study_name="catboost_optimization_log_target",
        storage=f"sqlite:///{os.path.abspath('./experiments/catboost_study_log.db')}",
    )
    best_params_cat = cat_study.best_params
except Exception:  # noqa: BLE001
    best_params_cat = DEFAULT_CAT_PARAMS

# Defensive tree depth clamping
best_params_xgb["max_depth"] = min(best_params_xgb.get("max_depth", 4), 4)
best_params_cat["depth"] = min(best_params_cat.get("depth", 4), 4)

print("✅ XGBoost Best Params:", best_params_xgb)
print("✅ LightGBM Best Params:", best_params_lgb)
print("✅ CatBoost Best Params:", best_params_cat)

# ============================================
# GENERATE OUT-OF-FOLD (OOF) PREDICTIONS FOR 6 MODELS
# ============================================
print("\n" + "=" * 60)
print("GENERATING OUT-OF-FOLD (OOF) PREDICTIONS (6 MODELS)")
print("=" * 60)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

oof_cat = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))
oof_lgb = np.zeros(len(X_train))
oof_ridge = np.zeros(len(X_train))
oof_lasso = np.zeros(len(X_train))
oof_elasticnet = np.zeros(len(X_train))

test_preds_cat = np.zeros(len(X_test_raw))
test_preds_xgb = np.zeros(len(X_test))
test_preds_lgb = np.zeros(len(X_test))
test_preds_ridge = np.zeros(len(X_test))
test_preds_lasso = np.zeros(len(X_test))
test_preds_elasticnet = np.zeros(len(X_test))

alphas_lasso = np.logspace(-5, 1, 100)
alphas_elasticnet = np.logspace(-5, 1, 100)
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]

# Define base models outside the loop
cat_params = best_params_cat.copy()
cat_params.update(
    {
        "iterations": 2000,
        "random_seed": RANDOM_STATE,
        "verbose": False,
        "early_stopping_rounds": 50,
    }
)
base_model_cat = CatBoostRegressor(**cat_params)

xgb_params = best_params_xgb.copy()
xgb_params.update(
    {
        "n_estimators": 2000,
        "random_state": RANDOM_STATE,
        "verbosity": 0,
        "early_stopping_rounds": 50,
    }
)
base_model_xgb = XGBRegressor(**xgb_params)

lgb_params = best_params_lgb.copy()
lgb_params.update({"n_estimators": 2000, "random_state": RANDOM_STATE, "verbosity": -1})
base_model_lgb = lgb.LGBMRegressor(**lgb_params)

base_model_ridge = Ridge(alpha=15.0, random_state=RANDOM_STATE)

base_model_lasso = make_pipeline(
    RobustScaler(),
    LassoCV(alphas=alphas_lasso, cv=5, random_state=RANDOM_STATE, max_iter=10000),
)

base_model_elasticnet = make_pipeline(
    RobustScaler(),
    ElasticNetCV(
        alphas=alphas_elasticnet,
        l1_ratio=l1_ratios,
        cv=5,
        random_state=RANDOM_STATE,
        max_iter=10000,
    ),
)

# Get the raw, unencoded Neighborhood string labels for the test set
raw_test_neighborhoods = X_test_raw["Neighborhood"]

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"  Fold {fold + 1}/{N_FOLDS}...")

    # Fold data splits
    X_tr, X_va = X_train.iloc[train_idx].copy(), X_train.iloc[val_idx].copy()
    y_tr, y_va = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]

    X_tr_raw, X_va_raw = (
        X_train_raw.iloc[train_idx].copy(),
        X_train_raw.iloc[val_idx].copy(),
    )

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

    # Map Neighborhood column fold-by-fold
    X_tr["Neighborhood"] = (
        raw_neighborhoods.iloc[train_idx].map(neigh_map).fillna(13).astype(int)
    )
    X_va["Neighborhood"] = (
        raw_neighborhoods.iloc[val_idx].map(neigh_map).fillna(13).astype(int)
    )

    X_tr_raw["Neighborhood"] = (
        raw_neighborhoods.iloc[train_idx].map(neigh_map).fillna(13).astype(int)
    )
    X_va_raw["Neighborhood"] = (
        raw_neighborhoods.iloc[val_idx].map(neigh_map).fillna(13).astype(int)
    )

    # Create fold-local copies of test features and map them fold-by-fold
    X_te_fold = X_test.copy()
    X_te_raw_fold = X_test_raw.copy()

    X_te_fold["Neighborhood"] = (
        raw_test_neighborhoods.map(neigh_map).fillna(13).astype(int)
    )
    X_te_raw_fold["Neighborhood"] = (
        raw_test_neighborhoods.map(neigh_map).fillna(13).astype(int)
    )

    # Fold Scaler (prevents std/mean leakage from val fold)
    fold_scaler = StandardScaler()
    X_tr_sc = fold_scaler.fit_transform(X_tr)
    X_va_sc = fold_scaler.transform(X_va)
    X_te_sc_fold = fold_scaler.transform(X_te_fold)

    # 1. CatBoost
    model_cat = clone(base_model_cat)
    model_cat.fit(
        X_tr_raw,
        y_tr,
        eval_set=(X_va_raw, y_va),
        cat_features=cat_features,
        verbose=False,
    )
    oof_cat[val_idx] = model_cat.predict(X_va_raw)
    test_preds_cat += model_cat.predict(X_te_raw_fold) / N_FOLDS

    # 2. XGBoost
    model_xgb = clone(base_model_xgb)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_va)
    test_preds_xgb += model_xgb.predict(X_te_fold) / N_FOLDS

    # 3. LightGBM
    model_lgb = clone(base_model_lgb)
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_va)
    test_preds_lgb += model_lgb.predict(X_te_fold) / N_FOLDS

    # 4. Ridge Regression (scaled cleanly inside fold)
    model_ridge = clone(base_model_ridge)
    model_ridge.fit(X_tr_sc, y_tr)
    oof_ridge[val_idx] = model_ridge.predict(X_va_sc)
    test_preds_ridge += model_ridge.predict(X_te_sc_fold) / N_FOLDS

    # 5. Lasso Regression (with RobustScaler inside fold)
    model_lasso = clone(base_model_lasso)
    model_lasso.fit(X_tr, y_tr)
    oof_lasso[val_idx] = model_lasso.predict(X_va)
    test_preds_lasso += model_lasso.predict(X_te_fold) / N_FOLDS

    # 6. ElasticNet Regression (with RobustScaler inside fold)
    model_elasticnet = clone(base_model_elasticnet)
    model_elasticnet.fit(X_tr, y_tr)
    oof_elasticnet[val_idx] = model_elasticnet.predict(X_va)
    test_preds_elasticnet += model_elasticnet.predict(X_te_fold) / N_FOLDS

# Print individual OOF RMSLE scores
print("\n" + "-" * 40)
print("INDIVIDUAL 6-MODEL OOF RMSLE SCORES:")
print("-" * 40)
print(
    f"  CatBoost OOF RMSLE:     {np.sqrt(mean_squared_error(y_train_log, oof_cat)):.6f}"
)
print(
    f"  XGBoost OOF RMSLE:      {np.sqrt(mean_squared_error(y_train_log, oof_xgb)):.6f}"
)
print(
    f"  LightGBM OOF RMSLE:     {np.sqrt(mean_squared_error(y_train_log, oof_lgb)):.6f}"
)
print(
    f"  Ridge Regression OOF:   {np.sqrt(mean_squared_error(y_train_log, oof_ridge)):.6f}"
)
print(
    f"  Lasso Regression OOF:   {np.sqrt(mean_squared_error(y_train_log, oof_lasso)):.6f}"
)
print(
    f"  ElasticNet OOF RMSLE:   {np.sqrt(mean_squared_error(y_train_log, oof_elasticnet)):.6f}"
)

# ============================================
# OPTIMIZE ENSEMBLE WEIGHTS WITH SCIPY (SLSQP)
# ============================================
print("\n" + "=" * 60)
print("OPTIMIZING ENSEMBLE WEIGHTS (SLSQP CONSTRAINED)")
print("=" * 60)

oof_matrix = np.column_stack(
    [oof_cat, oof_xgb, oof_lgb, oof_ridge, oof_lasso, oof_elasticnet]
)
test_matrix = np.column_stack(
    [
        test_preds_cat,
        test_preds_xgb,
        test_preds_lgb,
        test_preds_ridge,
        test_preds_lasso,
        test_preds_elasticnet,
    ]
)


def loss_func(weights: np.ndarray) -> float:
    w = np.array(weights)
    pred = oof_matrix @ w
    return np.sqrt(mean_squared_error(y_train_log, pred))


init_weights: list[float] = [1.0 / 6.0] * 6
bounds: list[tuple[float, float]] = [(0.0, 1.0)] * 6
constraints: dict[str, Any] = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

res = minimize(
    loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
)

best_weights = res.x
best_ensemble_rmsle = res.fun

print("✅ Optimal 6-Model Weights:")
print(f"   CatBoost:    {best_weights[0]:.4f}")
print(f"   XGBoost:     {best_weights[1]:.4f}")
print(f"   LightGBM:    {best_weights[2]:.4f}")
print(f"   Ridge:       {best_weights[3]:.4f}")
print(f"   Lasso:       {best_weights[4]:.4f}")
print(f"   ElasticNet:  {best_weights[5]:.4f}")
print(f"\n🚀 OPTIMAL WEIGHTED ENSEMBLE OOF RMSLE: {best_ensemble_rmsle:.6f}")

# ============================================
# COMPARE WITH STACKING META-MODEL
# ============================================
print("\n" + "=" * 60)
print("COMPARING WITH STACKING META-MODEL (LASSO META-LEARNER)")
print("=" * 60)

meta_model = LassoCV(cv=5, positive=True, random_state=RANDOM_STATE)
meta_model.fit(oof_matrix, y_train_log)

oof_stacking = meta_model.predict(oof_matrix)
stacking_rmsle = np.sqrt(mean_squared_error(y_train_log, oof_stacking))

print(f"  Stacking OOF RMSLE (Positive Constraints): {stacking_rmsle:.6f}")
print(f"  Meta-model Non-Zero Coefficients: {meta_model.coef_.round(4)}")

if best_ensemble_rmsle <= stacking_rmsle:
    print("✅ Constrained Weighted Average is the winning strategy!")
    oof_winning = oof_matrix @ best_weights
    final_test_log_preds = test_matrix @ best_weights
else:
    print("✅ Stacking Meta-model is the winning strategy!")
    oof_winning = oof_stacking
    final_test_log_preds = meta_model.predict(test_matrix)

# ============================================
# POST-PROCESSING: LOG-BIAS CORRECTION & OPTIMAL MULTIPLIER (JENSEN'S INEQUALITY FIX)
# ============================================
print("\n" + "=" * 60)
print("POST-PROCESSING: LOG-BIAS CORRECTION & OPTIMAL MULTIPLIER SEARCH")
print("=" * 60)

raw_y_train = pd.read_csv("./processed_data/y_train.csv").squeeze()


def rmsle_dollars(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.maximum(y_true, 0.0)
    y_pred = np.maximum(y_pred, 0.0)
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))


# 1. Log-normal expectation correction: E[y] = exp(mu + sigma^2 / 2) - 1
oof_mse = mean_squared_error(y_train_log, oof_winning)
variance_correction = oof_mse / 2.0
print(f"  Calculated OOF Log Variance (sigma^2): {oof_mse:.6f}")
print(f"  Variance Correction Factor (+sigma^2 / 2): +{variance_correction:.6f}")

oof_log_corrected = oof_winning + variance_correction
oof_pred_corrected_dollars = np.expm1(oof_log_corrected)

rmsle_uncorrected = rmsle_dollars(raw_y_train, np.expm1(oof_winning))
rmsle_var_corrected = rmsle_dollars(raw_y_train, oof_pred_corrected_dollars)

print(f"  OOF Dollar RMSLE (Uncorrected expm1):       {rmsle_uncorrected:.6f}")
print(f"  OOF Dollar RMSLE (Variance-Corrected):      {rmsle_var_corrected:.6f}")


# 2. Optimal Scalar Multiplier Search (Constrained c >= 1.000 to offset underestimation penalty)
def multiplier_loss(c: float) -> float:
    preds = c * oof_pred_corrected_dollars
    return rmsle_dollars(raw_y_train, preds)


res_multiplier = minimize_scalar(
    multiplier_loss, bounds=(1.000, 1.010), method="bounded"
)
best_c = res_multiplier.x
best_postprocessed_rmsle = res_multiplier.fun

print(f"\n✅ Optimal Scalar Multiplier c*: {best_c:.5f}")
print(f"🚀 FINAL POST-PROCESSED OOF DOLLAR RMSLE: {best_postprocessed_rmsle:.6f}")

# Apply to test predictions
final_test_log_corrected = final_test_log_preds + variance_correction
raw_test_dollars = best_c * np.expm1(final_test_log_corrected)

# Safe quantile boundary clipping [$42,000, $525,000] based on train 0.1% and 99.7% quantiles
final_test_dollars = np.clip(raw_test_dollars, 42000.0, 525000.0)
clipped_count = np.sum(raw_test_dollars != final_test_dollars)
print(
    f"  Applied safe quantile clipping [$42,000, $525,000]: {clipped_count} test prices clipped."
)

# ============================================
# CREATE SUBMISSION
# ============================================
print("\n" + "=" * 60)
print("GENERATING FINAL KAGGLE SUBMISSION (POST-PROCESSED)")
print("=" * 60)

os.makedirs("./submissions", exist_ok=True)
submission = pd.DataFrame({"Id": test_ids, "SalePrice": final_test_dollars})

submission.to_csv("./submissions/submission_ensemble_final.csv", index=False)

print(
    "✅ Calibrated submission successfully saved to './submissions/submission_ensemble_final.csv'"
)
print(f"   Submission shape: {submission.shape}")
print("\n   First 5 rows of calibrated final submission:")
print(submission.head())

print("\n" + "=" * 60)
print("ENSEMBLE & POST-PROCESSING PIPELINE COMPLETED")
print("=" * 60)
