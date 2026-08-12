import os

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import PowerTransformer
from xgboost import XGBRegressor

from src.metrics import rmsle

RANDOM_STATE = 42
N_FOLDS = 5
N_TRIALS = 1

print("=" * 60)
print("LOADING PROCESSED DATA FOR XGBOOST")
print("=" * 60)

X_train = pd.read_csv("./processed_data/X_train.csv")
y_train = pd.read_csv("./processed_data/y_train.csv").squeeze()

pt = PowerTransformer(method="box-cox")
y_transformed = pt.fit_transform(y_train.values.reshape(-1, 1)).flatten()


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
    model = XGBRegressor(**params)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rmsle_scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_transformed[train_idx], y_transformed[val_idx]
        model.fit(X_train_fold, y_train_fold)
        y_pred_transformed = model.predict(X_val_fold)
        y_pred_original = pt.inverse_transform(
            y_pred_transformed.reshape(-1, 1)
        ).flatten()
        y_val_original = pt.inverse_transform(y_val_fold.reshape(-1, 1)).flatten()
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
    best_params_no_rs = {k: v for k, v in best_params.items() if k != "random_state"}
    best_model = XGBRegressor(
        **best_params_no_rs, random_state=RANDOM_STATE, verbosity=0
    )
    best_model.fit(X_train, y_transformed)

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
    X_tr = X_train.iloc[tr_idx].copy()
    X_val = X_train.iloc[val_idx].copy()
    y_tr = y_transformed[tr_idx]
    y_val = y_transformed[val_idx]

    best_model = XGBRegressor(**final_params)
    best_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    joblib.dump(best_model, "./models/xgboost_best.pkl")
    joblib.dump(best_model, "./models/xgboost_best_rmsle.pkl")
    joblib.dump(pt, "./models/boxcox_transformer.pkl")

    trials_df = study.trials_dataframe()
    trials_df.to_csv("./experiments/xgboost_trials_log.csv", index=False)


if __name__ == "__main__":
    main()
