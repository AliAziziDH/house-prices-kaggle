import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from src.preprocess import AmesDataTransformer


def diagnose():
    print("=" * 60)
    print("PIPELINE DIAGNOSTICS")
    print("=" * 60)

    train_path = "data/train.csv"
    test_path = "data/test.csv"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("❌ Raw data files missing in data/")
        return

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print(f"Raw Train shape: {train.shape}")
    print(f"Raw Test shape:  {test.shape}")

    X_train_raw = train.drop(["Id", "SalePrice"], axis=1)
    X_test_raw = test.drop(["Id"], axis=1)

    transformer = AmesDataTransformer()
    transformer.fit(X_train_raw, train["SalePrice"])
    X_train = transformer.transform(X_train_raw)
    X_test = transformer.transform(X_test_raw)

    print(f"Processed Train shape: {X_train.shape}")
    print(f"Processed Test shape:  {X_test.shape}")

    train_nans = X_train.isna().sum().sum()
    test_nans = X_test.isna().sum().sum()

    print(f"NaNs in Processed Train: {train_nans}")
    print(f"NaNs in Processed Test:  {test_nans}")

    if train_nans == 0 and test_nans == 0:
        print("✅ Pipeline diagnostics passed: No missing values found.")
    else:
        print("⚠️ Warning: Missing values remain in processed data.")


if __name__ == "__main__":
    diagnose()
