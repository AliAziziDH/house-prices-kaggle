import os

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

from src.metrics import rmsle

RANDOM_STATE = 42
N_FOLDS = 5
N_TRIALS = 50

print("=" * 60)
print("LOADING PROCESSED DATA FOR XGBOOST")
print("=" * 60)

X_train_full = pd.read_csv("./processed_data/X_train_full.csv")
y_train_full = pd.read_csv("./processed_data/y_train_full.csv").squeeze()
y_full_transformed = np.log1p(y_train_full.values)

X_train_90 = pd.read_csv("./processed_data/X_train.csv")
y_train_90 = pd.read_csv("./processed_data/y_train.csv").squeeze()
y_90_transformed = np.log1p(y_train_90.values)


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 1, 5),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),

        "random_state": RANDOM_STATE,
        "verbosity": 0,
    }
    model = XGBRegressor(**params)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rmsle_scores = []
    for train_idx, val_idx in kf.split(X_train_full):
        X_train_fold, X_val_fold = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_train_fold, y_val_fold = y_full_transformed[train_idx], y_full_transformed[val_idx]
        model.fit(X_train_fold, y_train_fold)
        y_pred_transformed = model.predict(X_val_fold)
        y_pred_original = np.expm1(y_pred_transformed)
        y_val_original = np.expm1(y_val_fold)
        rmsle_score = rmsle(y_val_original, y_pred_original)
        rmsle_scores.append(rmsle_score)
    return np.mean(rmsle_scores)


def main():
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
    final_params = best_params.copy()
    final_params.update({"n_estimators": 2000, "random_state": RANDOM_STATE, "verbosity": 0, "early_stopping_rounds": 50})

    print("\nTRAINING 100% XGBOOST MODEL")
    tr_idx, val_idx = train_test_split(np.arange(len(X_train_full)), test_size=0.1, random_state=RANDOM_STATE)
    X_tr = X_train_full.iloc[tr_idx].copy()
    X_val = X_train_full.iloc[val_idx].copy()
    y_tr = y_full_transformed[tr_idx]
    y_val = y_full_transformed[val_idx]
    best_model_full = XGBRegressor(**final_params)
    best_model_full.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    joblib.dump(best_model_full, "./models/xgboost_best_rmsle.pkl")

    print("\nTRAINING 90% XGBOOST MODEL")
    tr_idx_90, val_idx_90 = train_test_split(np.arange(len(X_train_90)), test_size=0.1, random_state=RANDOM_STATE)
    X_tr_90 = X_train_90.iloc[tr_idx_90].copy()
    X_val_90 = X_train_90.iloc[val_idx_90].copy()
    y_tr_90 = y_90_transformed[tr_idx_90]
    y_val_90 = y_90_transformed[val_idx_90]
    best_model_90 = XGBRegressor(**final_params)
    best_model_90.fit(X_tr_90, y_tr_90, eval_set=[(X_val_90, y_val_90)], verbose=False)
    joblib.dump(best_model_90, "./models/xgboost_90.pkl")

    trials_df = study.trials_dataframe()
    trials_df.to_csv("./experiments/xgboost_trials_log.csv", index=False)


if __name__ == "__main__":
    main()
