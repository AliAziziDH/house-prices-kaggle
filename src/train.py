import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from src.models.base import save_oof_predictions
from src.models.catboost_model import train_catboost_cv
from src.models.linear_model import train_linear_cv
from src.models.xgboost_model import train_xgboost_cv


def main():
    parser = argparse.ArgumentParser(description="Unified Model Training Runner")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["xgboost", "catboost", "linear", "all"],
        help="Model to train (default: all)",
    )
    args = parser.parse_args()

    train_path = "./data/train.csv"
    if not os.path.exists(train_path):
        print("❌ Error: data/train.csv not found.")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    train_ids = train_df["Id"]
    y_raw = train_df["SalePrice"]
    X_raw = train_df.drop(columns=["Id", "SalePrice"])

    print("=" * 60)
    print("STARTING UNIFIED MODEL TRAINING RUN")
    print("=" * 60)
    print(f"Data shape: {X_raw.shape}")

    if args.model in ["xgboost", "all"]:
        print("\n--- Training XGBoost ---")
        xgb_res = train_xgboost_cv(X_raw, y_raw)
        print(f"XGBoost Overall OOF RMSLE: {xgb_res['overall_rmsle']:.6f}")
        save_oof_predictions("xgboost", xgb_res["oof_preds"], train_ids)

    if args.model in ["catboost", "all"]:
        print("\n--- Training CatBoost ---")
        cat_res = train_catboost_cv(X_raw, y_raw)
        print(f"CatBoost Overall OOF RMSLE: {cat_res['overall_rmsle']:.6f}")
        save_oof_predictions("catboost", cat_res["oof_preds"], train_ids)

    if args.model in ["linear", "all"]:
        print("\n--- Training Linear (Lasso) ---")
        linear_res = train_linear_cv(X_raw, y_raw, model_type="lasso")
        print(f"Linear (Lasso) Overall OOF RMSLE: {linear_res['overall_rmsle']:.6f}")
        save_oof_predictions("linear", linear_res["oof_preds"], train_ids)

    print("\n" + "=" * 60)
    print("TRAINING RUN COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
