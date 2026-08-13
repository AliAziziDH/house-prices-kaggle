import re

main_code = """
import os
import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import PowerTransformer

from src.metrics import rmsle

RANDOM_STATE = 42
N_FOLDS = 5
N_TRIALS = 1

print("=" * 60)
print("LOADING RAW DATA FOR CATBOOST")
print("=" * 60)

X_train_full_raw = pd.read_csv("./processed_data/X_train_full_raw.csv")
y_train_full = pd.read_csv("./processed_data/y_train_full.csv").squeeze()

cat_features = X_train_full_raw.select_dtypes(include=["object", "str"]).columns.tolist()

for col in cat_features:
    X_train_full_raw[col] = X_train_full_raw[col].fillna("Missing").astype(str)

pt_full = PowerTransformer(method="box-cox")
y_full_transformed = pt_full.fit_transform(y_train_full.values.reshape(-1, 1)).flatten()

# 90% data
X_train_90_raw = pd.read_csv("./processed_data/X_train_raw.csv")
y_train_90 = pd.read_csv("./processed_data/y_train.csv").squeeze()

for col in cat_features:
    X_train_90_raw[col] = X_train_90_raw[col].fillna("Missing").astype(str)

pt_90 = PowerTransformer(method="box-cox")
y_90_transformed = pt_90.fit_transform(y_train_90.values.reshape(-1, 1)).flatten()

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

    for train_idx, val_idx in kf.split(X_train_full_raw):
        X_train_fold = X_train_full_raw.iloc[train_idx].copy()
        X_val_fold = X_train_full_raw.iloc[val_idx].copy()

        for col in cat_features:
            X_train_fold[col] = X_train_fold[col].fillna("Missing").astype(str)
            X_val_fold[col] = X_val_fold[col].fillna("Missing").astype(str)

        y_train_fold = y_full_transformed[train_idx]
        y_val_fold = y_full_transformed[val_idx]

        model.fit(X_train_fold, y_train_fold, cat_features=cat_features, verbose=False)
        y_pred_transformed = model.predict(X_val_fold)
        y_pred_dollars = pt_full.inverse_transform(y_pred_transformed.reshape(-1, 1)).flatten()
        y_val_original = pt_full.inverse_transform(y_val_fold.reshape(-1, 1)).flatten()

        rmsle_score = rmsle(y_val_original, y_pred_dollars)
        rmsle_scores.append(rmsle_score)

    return np.mean(rmsle_scores)


def main():
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
    X_train_full_clean = X_train_full_raw.copy()
    for col in cat_features:
        X_train_full_clean[col] = X_train_full_clean[col].fillna("Missing").astype(str)

    best_params_no_rs = {k: v for k, v in best_params.items() if k != "random_seed"}

    # 100% MODEL
    best_model_full = CatBoostRegressor(**best_params_no_rs, random_seed=RANDOM_STATE, verbose=False)
    best_model_full.fit(X_train_full_clean, y_full_transformed, cat_features=cat_features)

    joblib.dump(best_model_full, "./models/catboost_best_rmsle.pkl")

    # 90% MODEL
    X_train_90_clean = X_train_90_raw.copy()
    for col in cat_features:
        X_train_90_clean[col] = X_train_90_clean[col].fillna("Missing").astype(str)

    best_model_90 = CatBoostRegressor(**best_params_no_rs, random_seed=RANDOM_STATE, verbose=False)
    best_model_90.fit(X_train_90_clean, y_90_transformed, cat_features=cat_features)

    joblib.dump(best_model_90, "./models/catboost_90.pkl")

    trials_df = study.trials_dataframe()
    trials_df.to_csv("./experiments/catboost_raw_trials_rmsle.csv", index=False)

    # Save cat_features list alongside model for downstream ensembling
    joblib.dump(cat_features, "./models/cat_features.pkl")

if __name__ == "__main__":
    main()
"""

with open('src/train_catboost.py', 'w') as f:
    f.write(main_code)
