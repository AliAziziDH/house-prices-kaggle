"""
XGBoost Hyperparameter Optimization with Optuna & Early Stopping
Trained on y_train directly matching Kaggle RMSLE.
"""

import os

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import PowerTransformer
from xgboost import XGBRegressor

from src.metrics import rmsle

# ============================================
# CONFIGURATION
# ============================================
RANDOM_STATE = 42
N_FOLDS = 5
N_TRIALS = 50


# ============================================
# LOAD DATA
# ============================================
print("=" * 60)
print("LOADING PROCESSED DATA FOR XGBOOST")
print("=" * 60)

X_train = pd.read_csv("./processed_data/X_train.csv")
y_train = pd.read_csv("./processed_data/y_train.csv").squeeze()

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")


# ============================================
# BOX-COX TRANSFORMATION (still used for model input)
# ============================================
print("\n" + "=" * 60)
print("APPLYING BOX-COX TRANSFORMATION")
print("=" * 60)

pt = PowerTransformer(method="box-cox")
y_transformed = pt.fit_transform(y_train.values.reshape(-1, 1)).flatten()
print(f"Skewness after Box-Cox: {pd.Series(y_transformed).skew():.4f}")


# ============================================
# OPTUNA OBJECTIVE FUNCTION (with RMSLE)
# ============================================
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": RANDOM_STATE,
        "verbosity": 0,
    }

    # Create model
    model = XGBRegressor(**params)

    # Cross-validation
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rmsle_scores = []

    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_transformed[train_idx], y_transformed[val_idx]

        # Train model on transformed target
        model.fit(X_train_fold, y_train_fold)

        # Predict on validation fold
        y_pred_transformed = model.predict(X_val_fold)

        # Inverse transform to original scale
        y_pred_original = pt.inverse_transform(
            y_pred_transformed.reshape(-1, 1)
        ).flatten()
        y_val_original = pt.inverse_transform(y_val_fold.reshape(-1, 1)).flatten()

        # Calculate RMSLE on original scale
        rmsle_score = rmsle(y_val_original, y_pred_original)
        rmsle_scores.append(rmsle_score)

    avg_rmsle = np.mean(rmsle_scores)
    return avg_rmsle


def main():
    # ============================================
    # RUN OPTIMIZATION
    # ============================================
    print("\n" + "=" * 60)
    print("STARTING XGBOOST OPTIMIZATION WITH EARLY STOPPING")
    print("=" * 60)

    # Create directories
    os.makedirs("./experiments", exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    study = optuna.create_study(
        direction="minimize",
        study_name="xgboost_optimization_rmsle",
        storage=f"sqlite:///{os.path.abspath('./experiments/xgboost_study_rmsle.db')}",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=N_TRIALS)

    best_params = study.best_params
    best_params_no_rs = {k: v for k, v in best_params.items() if k != "random_state"}
    best_model = XGBRegressor(**best_params_no_rs, random_state=RANDOM_STATE, verbosity=0)
    best_model.fit(X_train, y_transformed)

    # Save model and transformer
    joblib.dump(best_model, "./models/xgboost_best_rmsle.pkl")
    joblib.dump(pt, "./models/boxcox_transformer.pkl")

    # Save all trial results
    trials_df = study.trials_dataframe()
    trials_df.to_csv("./experiments/xgboost_trials_rmsle.csv", index=False)

    print(f"✅ Best RMSLE: {study.best_value:.6f}")
    print(f"✅ Best parameters: {best_params}")

    # ============================================
    # TRAIN FINAL MODEL ON FULL DATA
    # ============================================
    print("\n" + "=" * 60)
    print("TRAINING FINAL XGBOOST MODEL ON FULL DATA")
    print("=" * 60)

    final_params = best_params.copy()
    final_params.update(
        {
            "n_estimators": 2000,
            "random_state": RANDOM_STATE,
            "verbosity": 0,
            "early_stopping_rounds": 50,
        }
    )

    tr_idx, val_idx = train_test_split(
        np.arange(len(X_train)), test_size=0.1, random_state=RANDOM_STATE
    )

    raw_train = pd.read_csv("./data/train.csv")
    raw_neighborhoods = raw_train["Neighborhood"]

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

    X_tr = X_train.iloc[tr_idx].copy()
    X_val = X_train.iloc[val_idx].copy()
    y_tr = y_train.iloc[tr_idx]
    y_val = y_train.iloc[val_idx]

    X_tr["Neighborhood"] = (
        raw_neighborhoods.iloc[tr_idx].map(neigh_map).fillna(13).astype(int)
    )
    X_val["Neighborhood"] = (
        raw_neighborhoods.iloc[val_idx].map(neigh_map).fillna(13).astype(int)
    )

    best_model = XGBRegressor(**final_params)
    best_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    # Save both original naming and best rmsle versions for compatibility
    joblib.dump(best_model, "./models/xgboost_best.pkl")
    joblib.dump(best_model, "./models/xgboost_best_rmsle.pkl")

    trials_df = study.trials_dataframe()
    trials_df.to_csv("./experiments/xgboost_trials_log.csv", index=False)

    # ============================================
    # GENERATE SUBMISSION
    # ============================================
    print("\n" + "=" * 60)
    print("GENERATING SUBMISSION")
    print("=" * 60)

    X_test = pd.read_csv("./processed_data/X_test.csv")
    test_ids = pd.read_csv("./data/test.csv")["Id"]

    y_pred_log = best_model.predict(X_test)
    y_pred_dollars = np.expm1(y_pred_log)

    submission = pd.DataFrame({"Id": test_ids, "SalePrice": y_pred_dollars})

    os.makedirs("./submissions", exist_ok=True)

    submission.to_csv("./submissions/submission_xgboost_rmsle.csv", index=False)

    print("✅ Submission saved to './submissions/submission_xgboost_rmsle.csv'")
    print(f"   Shape: {submission.shape}")
    print("   First 5 rows:")
    print(submission.head())

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETED")
    print("=" * 60)

if __name__ == '__main__':
    main()
