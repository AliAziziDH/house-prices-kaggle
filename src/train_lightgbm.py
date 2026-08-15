import os

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

from src.metrics import rmsle
from src.utils import load_processed_data

RANDOM_STATE = 42
N_FOLDS = 5


def main():
    print("=" * 60)
    print("LOADING PROCESSED DATA FOR LIGHTGBM")
    print("=" * 60)

    X_train_full, y_train_full, y_full_transformed, X_train_90, y_train_90, y_90_transformed = load_processed_data()

    params = {
        "n_estimators": 1000,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "verbosity": -1,
    }

    model = LGBMRegressor(**params)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_preds = np.zeros(len(X_train_full))
    rmsle_scores = []

    print("Generating OOF predictions using LightGBM...")
    for train_idx, val_idx in kf.split(X_train_full):
        X_train_fold, X_val_fold = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_train_fold, y_val_fold = y_full_transformed[train_idx], y_full_transformed[val_idx]

        model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)])
        y_pred_transformed = model.predict(X_val_fold)

        # Save OOF predictions in physical USD space so find_ensemble_weights.py can log1p them uniformly
        oof_preds[val_idx] = np.expm1(y_pred_transformed)

        rmsle_score = rmsle(np.expm1(y_val_fold), oof_preds[val_idx])
        rmsle_scores.append(rmsle_score)

    print(f"LightGBM Mean OOF RMSLE: {np.mean(rmsle_scores):.6f}")

    os.makedirs("./processed_data", exist_ok=True)
    pd.Series(oof_preds).to_csv("./processed_data/oof_lightgbm.csv", index=False)

    os.makedirs("./models", exist_ok=True)

    print("Training 100% LightGBM Model...")
    best_model_full = LGBMRegressor(**params)
    best_model_full.fit(X_train_full, y_full_transformed)
    joblib.dump(best_model_full, "./models/lightgbm_best_rmsle.pkl")

    print("Training 90% LightGBM Model...")
    best_model_90 = LGBMRegressor(**params)
    best_model_90.fit(X_train_90, y_90_transformed)
    joblib.dump(best_model_90, "./models/lightgbm_90.pkl")

    print("✅ LightGBM complete.")


if __name__ == "__main__":
    main()
