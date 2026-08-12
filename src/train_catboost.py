"""
CatBoost Optimization with Optuna (Native Categoricals & Early Stopping)
Trained on y_train (np.log1p(SalePrice)) directly matching Kaggle RMSLE.
"""

import os

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import PowerTransformer

from src.metrics import rmsle

# ============================================
# CONFIGURATION
# ============================================
RANDOM_STATE = 42
N_FOLDS = 5
N_TRIALS = 15  # Optimized for speed in the sandbox environment


# ============================================
# LOAD DATA (RAW VERSION)
# ============================================
print("=" * 60)
print("LOADING RAW DATA FOR CATBOOST")
print("=" * 60)

X_train_raw = pd.read_csv("./processed_data/X_train_raw.csv")
y_train = pd.read_csv("./processed_data/y_train.csv").squeeze()

# Identify categorical features
cat_features = X_train_raw.select_dtypes(include=["object", "str"]).columns.tolist()

# Clean NaN values in categorical columns
for col in cat_features:
    X_train_raw[col] = X_train_raw[col].fillna("Missing").astype(str)

print(f"X_train_raw shape: {X_train_raw.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"Categorical features count: {len(cat_features)}")

# ============================================
# BOX-COX TRANSFORMATION
# ============================================
print("\n" + "=" * 60)
print("APPLYING BOX-COX TRANSFORMATION")
print("=" * 60)

pt = PowerTransformer(method="box-cox")
y_transformed = pt.fit_transform(y_train.values.reshape(-1, 1)).flatten()
print(f"Skewness after Box-Cox: {pd.Series(y_transformed).skew():.4f}")


# ============================================
# OPTUNA OBJECTIVE FUNCTION
# ============================================
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000, step=100),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        "random_seed": RANDOM_STATE,
        "verbose": False,
    }

    model = CatBoostRegressor(**params)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rmsle_scores = []

    for train_idx, val_idx in kf.split(X_train_raw):
        X_train_fold = X_train_raw.iloc[train_idx].copy()
        X_val_fold = X_train_raw.iloc[val_idx].copy()

        # Clean NaN in categorical columns for each fold
        for col in cat_features:
            X_train_fold[col] = X_train_fold[col].fillna("Missing").astype(str)
            X_val_fold[col] = X_val_fold[col].fillna("Missing").astype(str)

        y_train_fold = y_transformed[train_idx]
        y_val_fold = y_transformed[val_idx]

        model.fit(X_train_fold, y_train_fold, cat_features=cat_features, verbose=False)

        y_pred_transformed = model.predict(X_val_fold)

        y_pred_dollars = pt.inverse_transform(
            y_pred_transformed.reshape(-1, 1)
        ).flatten()
        y_val_original = pt.inverse_transform(y_val_fold.reshape(-1, 1)).flatten()

        rmsle_score = rmsle(y_val_original, y_pred_dollars)
        rmsle_scores.append(rmsle_score)

    return np.mean(rmsle_scores)


# ============================================
# RUN OPTIMIZATION
# ============================================
print("\n" + "=" * 60)
print("STARTING CATBOOST OPTIMIZATION WITH EARLY STOPPING")
print("=" * 60)

os.makedirs("./experiments", exist_ok=True)
os.makedirs("./models", exist_ok=True)

study = optuna.create_study(
    direction="minimize",
    study_name="catboost_raw_optimization_rmsle",
    storage=f"sqlite:///{os.path.abspath('./experiments/catboost_raw_study_rmsle.db')}",
    load_if_exists=True,
)

study.optimize(objective, n_trials=N_TRIALS)

best_params = study.best_params
# Clean training data before final fitting
X_train_clean = X_train_raw.copy()
for col in cat_features:
    X_train_clean[col] = X_train_clean[col].fillna("Missing").astype(str)

# Avoid passing random_seed twice; ensure explicit seed is applied here
best_params_no_rs = {k: v for k, v in best_params.items() if k != "random_seed"}
best_model = CatBoostRegressor(
    **best_params_no_rs, random_seed=RANDOM_STATE, verbose=False
)
best_model.fit(X_train_clean, y_transformed, cat_features=cat_features)

# Save model using consistent RMSLE suffix
joblib.dump(best_model, "./models/catboost_best_rmsle.pkl")
joblib.dump(pt, "./models/boxcox_transformer.pkl")

trials_df = study.trials_dataframe()
trials_df.to_csv("./experiments/catboost_raw_trials_rmsle.csv", index=False)

# Save cat_features list alongside model for downstream ensembling
joblib.dump(cat_features, "./models/cat_features.pkl")

# ============================================
# GENERATE SUBMISSION
# ============================================
print("\n" + "=" * 60)
print("GENERATING SUBMISSION")
print("=" * 60)

X_test_raw = pd.read_csv("./processed_data/X_test_raw.csv")
test_ids = pd.read_csv("./data/test.csv")["Id"]

# Clean test data as well
for col in cat_features:
    X_test_raw[col] = X_test_raw[col].fillna("Missing").astype(str)

y_pred_log = best_model.predict(X_test_raw)
y_pred_dollars = np.expm1(y_pred_log)

submission = pd.DataFrame({"Id": test_ids, "SalePrice": y_pred_dollars})
import os

os.makedirs("./submissions", exist_ok=True)
submission.to_csv("./submissions/submission_catboost_raw.csv", index=False)

print("✅ Submission saved to './submissions/submission_catboost_raw.csv'")
print(f"   Shape: {submission.shape}")
print("   First 5 rows:")
print(submission.head())

print("\n" + "=" * 60)
print("CATBOOST RAW OPTIMIZATION COMPLETED")
print("=" * 60)
