import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

def main():
    print("=" * 60)
    print("VERIFYING LOCAL 10-FOLD CV FOR NEW ENSEMBLE BLEND")
    print("=" * 60)

    try:
        y_train_full = pd.read_csv("./processed_data/y_train_full.csv").squeeze()
        oof_xgb_df = pd.read_csv("./processed_data/oof_xgboost.csv")
        oof_catboost_df = pd.read_csv("./processed_data/oof_catboost.csv")
        oof_ridge = pd.read_csv("./processed_data/oof_ridge.csv").squeeze()

        # Extract predictions for xgb/catboost
        oof_xgb = oof_xgb_df["OOF_SalePrice"] if "OOF_SalePrice" in oof_xgb_df.columns else oof_xgb_df["SalePrice"]
        oof_catboost = oof_catboost_df["OOF_SalePrice"] if "OOF_SalePrice" in oof_catboost_df.columns else \
            oof_catboost_df["SalePrice"]
    except FileNotFoundError:
        print("Required OOF files not found! Make sure to run all model training scripts first.")
        return

    with open("./models/ensemble_weights.json", "r") as f:
        weight_dict = json.load(f)

    weight_xgb = weight_dict.get("xgb", 0.0)
    weight_catboost = weight_dict.get("catboost", 0.0)
    weight_ridge = weight_dict.get("ridge", 0.0)

    total = weight_xgb + weight_catboost + weight_ridge
    weight_xgb /= total
    weight_catboost /= total
    weight_ridge /= total

    xgb_log = np.log1p(np.clip(oof_xgb, 1, None))
    cat_log = np.log1p(np.clip(oof_catboost, 1, None))
    ridge_log = np.log1p(np.clip(oof_ridge, 1, None))

    ensemble_oof_log = (
        weight_xgb * xgb_log
        + weight_catboost * cat_log
        + weight_ridge * ridge_log
    )

    ensemble_oof_dollars = np.clip(np.expm1(ensemble_oof_log), 42000, 525000)
    score = rmsle(y_train_full, ensemble_oof_dollars)

    print(f"✅ Local CV RMSLE: {score:.5f}")
    if score < 0.113:
        print("🎯 Success! Score is below the 0.113 threshold.")
    else:
        print("⚠️ Warning: Score is above the 0.113 threshold.")

if __name__ == "__main__":
    main()
