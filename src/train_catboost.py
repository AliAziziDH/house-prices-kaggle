"""
CatBoost Optimization with Optuna (Native Categoricals & Early Stopping)
Trained on y_train_log (np.log1p(SalePrice)) directly matching Kaggle RMSLE.
"""

import os

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================
# CONFIGURATION
# ============================================
RANDOM_STATE = 42
N_FOLDS = 5
N_TRIALS = 15  # Optimized for speed in the sandbox environment

# ============================================
# LOAD DATA (RAW VERSION FOR CATBOOST)
# ============================================
print("=" * 60)
print("LOADING RAW DATA FOR CATBOOST")
print("=" * 60)

X_train_raw = pd.read_csv("./processed_data/X_train_raw.csv")
y_train_log = pd.read_csv("./processed_data/y_train_log.csv").squeeze()

# Load original raw train data to prevent target leakage in Neighborhood encoding
raw_train = pd.read_csv("./data/train.csv")
raw_train = raw_train[
    ~((raw_train["GrLivArea"] > 4000) & (raw_train["SalePrice"] < 300000))
].reset_index(drop=True)
raw_neighborhoods = raw_train["Neighborhood"]

# Identify categorical features
cat_features = X_train_raw.select_dtypes(include=["object"]).columns.tolist()

X_train_raw[cat_features] = X_train_raw[cat_features].fillna("Missing").astype(str)

print(f"X_train_raw shape: {X_train_raw.shape}")
print(f"y_train_log shape: {y_train_log.shape}")
print(f"Categorical features count: {len(cat_features)}")


# ============================================
# OPTUNA OBJECTIVE FUNCTION
# ============================================
def objective(trial):
    params = {
        "loss_function": "Huber:delta=1.0",
        "depth": trial.suggest_int(
            "depth", 3, 4
        ),  # Clamped to [3, 4] for low-variance generalization
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 0.95),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "iterations": 1000,  # Optimized for speed in the sandbox
        "random_seed": RANDOM_STATE,
        "verbose": False,
        "early_stopping_rounds": 50,
    }

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rmse_scores = []

    for train_idx, val_idx in kf.split(X_train_raw):
        X_train_fold = X_train_raw.iloc[train_idx].copy()
        X_val_fold = X_train_raw.iloc[val_idx].copy()
        y_train_fold, y_val_fold = (
            y_train_log.iloc[train_idx],
            y_train_log.iloc[val_idx],
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

        X_train_fold["Neighborhood"] = (
            raw_neighborhoods.iloc[train_idx].map(neigh_map).fillna(13).astype(int)
        )
        X_val_fold["Neighborhood"] = (
            raw_neighborhoods.iloc[val_idx].map(neigh_map).fillna(13).astype(int)
        )

        model = CatBoostRegressor(**params)
        model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=(X_val_fold, y_val_fold),
            cat_features=cat_features,
            verbose=False,
        )

        preds = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)


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
    study_name="catboost_optimization_log_target",
    storage=f"sqlite:///{os.path.abspath('./experiments/catboost_study_log.db')}",
    load_if_exists=True,
)

study.optimize(objective, n_trials=N_TRIALS)

best_params = study.best_params
print(f"\n✅ Best RMSLE (log-RMSE): {study.best_value:.6f}")
print(f"✅ Best parameters: {best_params}")

# ============================================
# TRAIN FINAL MODEL ON FULL DATA
# ============================================
print("\n" + "=" * 60)
print("TRAINING FINAL CATBOOST MODEL ON FULL DATA")
print("=" * 60)

final_params = best_params.copy()
final_params.update(
    {
        "iterations": 1000,
        "random_seed": RANDOM_STATE,
        "verbose": False,
        "early_stopping_rounds": 50,
    }
)

tr_idx, val_idx = train_test_split(
    np.arange(len(X_train_raw)), test_size=0.1, random_state=RANDOM_STATE
)

fold_raw_train = raw_train.iloc[tr_idx].copy()
fold_raw_train["TotalSF"] = (
    fold_raw_train["TotalBsmtSF"]
    + fold_raw_train["1stFlrSF"]
    + fold_raw_train["2ndFlrSF"]
)
fold_raw_train["PricePerSF"] = fold_raw_train["SalePrice"] / fold_raw_train["TotalSF"]

neigh_order = (
    fold_raw_train.groupby("Neighborhood")["PricePerSF"].median().sort_values().index
)
neigh_map = {n: i + 1 for i, n in enumerate(neigh_order)}

X_tr = X_train_raw.iloc[tr_idx].copy()
X_val = X_train_raw.iloc[val_idx].copy()
y_tr = y_train_log.iloc[tr_idx]
y_val = y_train_log.iloc[val_idx]

X_tr["Neighborhood"] = (
    raw_neighborhoods.iloc[tr_idx].map(neigh_map).fillna(13).astype(int)
)
X_val["Neighborhood"] = (
    raw_neighborhoods.iloc[val_idx].map(neigh_map).fillna(13).astype(int)
)

best_model = CatBoostRegressor(**final_params)
best_model.fit(
    X_tr, y_tr, eval_set=(X_val, y_val), cat_features=cat_features, verbose=False
)

# Save both original naming and best rmsle versions for compatibility
joblib.dump(best_model, "./models/catboost_best.pkl")
joblib.dump(best_model, "./models/catboost_best_rmsle.pkl")

trials_df = study.trials_dataframe()
trials_df.to_csv("./experiments/catboost_trials_log.csv", index=False)

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

X_test_raw[cat_features] = X_test_raw[cat_features].fillna("Missing").astype(str)

y_pred_log = best_model.predict(X_test_raw)
y_pred_dollars = np.expm1(y_pred_log)

os.makedirs("./submissions", exist_ok=True)
submission = pd.DataFrame({"Id": test_ids, "SalePrice": y_pred_dollars})
submission.to_csv("./submissions/submission_catboost_log.csv", index=False)

print("✅ Submission saved to './submissions/submission_catboost_log.csv'")
