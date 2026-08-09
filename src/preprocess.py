"""
Top 1% Target Preprocessing Engine for Kaggle House Prices (1,460 rows).
Implements Ordinal Neighborhood Encoding, Quality-SF Interactions, and Age Metrics.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew

QUALITY_MAP = {
    "Ex": 5,
    "Gd": 4,
    "TA": 3,
    "Fa": 2,
    "Po": 1,
    "None": 0,
    "No Garage": 0,
    "No Basement": 0,
}
BSMT_FIN_MAP = {
    "GLQ": 6,
    "ALQ": 5,
    "BLQ": 4,
    "Rec": 3,
    "LwQ": 2,
    "Unf": 1,
    "None": 0,
    "No Basement": 0,
}
EXPOSURE_MAP = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0, "No Basement": 0}


def preprocess_house_prices_data(data_dir: str = "./data") -> tuple:
    dir_path = Path(data_dir)
    train_path = dir_path / "train.csv"
    test_path = dir_path / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # 1. Outlier Removal (GrLivArea > 4000 & SalePrice < 300,000 + Outliers)
    train = train[
        ~((train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000))
    ].reset_index(drop=True)
    y_train_log = np.log1p(train["SalePrice"].values)

    # 2. Ordinal Target Encoding for Neighborhood based on Median Price per SF
    train["TotalSF"] = train["TotalBsmtSF"] + train["1stFlrSF"] + train["2ndFlrSF"]
    train["PricePerSF"] = train["SalePrice"] / train["TotalSF"]
    neigh_order = (
        train.groupby("Neighborhood")["PricePerSF"].median().sort_values().index
    )
    neigh_map = {n: i + 1 for i, n in enumerate(neigh_order)}

    X_train = train.drop(columns=["Id", "SalePrice", "TotalSF", "PricePerSF"])
    X_test = test.drop(columns=["Id"])

    combined = pd.concat([X_train, X_test], ignore_index=True)
    combined["Neighborhood"] = (
        combined["Neighborhood"].map(neigh_map).fillna(13).astype(int)
    )

    # 3. Ordinal Quality Mappings
    ord_cols = [
        "ExterQual",
        "ExterCond",
        "BsmtQual",
        "BsmtCond",
        "HeatingQC",
        "KitchenQual",
        "FireplaceQu",
        "GarageQual",
        "GarageCond",
        "PoolQC",
    ]
    for col in ord_cols:
        combined[col] = (
            combined[col].fillna("None").map(QUALITY_MAP).fillna(0).astype(int)
        )

    combined["BsmtFinType1"] = (
        combined["BsmtFinType1"].fillna("None").map(BSMT_FIN_MAP).fillna(0).astype(int)
    )
    combined["BsmtFinType2"] = (
        combined["BsmtFinType2"].fillna("None").map(BSMT_FIN_MAP).fillna(0).astype(int)
    )
    combined["BsmtExposure"] = (
        combined["BsmtExposure"].fillna("None").map(EXPOSURE_MAP).fillna(0).astype(int)
    )

    # 4. Handle Categoricals & Numeric Imputation
    cat_cols = [c for c in combined.select_dtypes(include=["object"]).columns]
    for col in cat_cols:
        combined[col] = combined[col].fillna("None")

    num_cols = [c for c in combined.select_dtypes(include=[np.number]).columns]
    for col in num_cols:
        combined[col] = combined[col].fillna(combined[col].median())

    # 5. High-Impact Interaction & Age Features
    combined["TotalSF"] = (
        combined["TotalBsmtSF"] + combined["1stFlrSF"] + combined["2ndFlrSF"]
    )
    combined["TotalBath"] = (
        combined["FullBath"]
        + (0.5 * combined["HalfBath"])
        + combined["BsmtFullBath"]
        + (0.5 * combined["BsmtHalfBath"])
    )
    combined["TotalPorch"] = (
        combined["OpenPorchSF"]
        + combined["3SsnPorch"]
        + combined["EnclosedPorch"]
        + combined["ScreenPorch"]
        + combined["WoodDeckSF"]
    )

    # [STRIPPED FOR REGULARIZATION] combined['Quality_SF_Score'] = combined['OverallQual'] * combined['TotalSF']

    combined["House_Age"] = (combined["YrSold"] - combined["YearBuilt"]).clip(lower=0)
    combined["Remod_Age"] = (combined["YrSold"] - combined["YearRemodAdd"]).clip(
        lower=0
    )
    combined["Is_New_House"] = (combined["YearBuilt"] == combined["YrSold"]).astype(int)

    # 6. Skewness Correction
    num_features = [c for c in combined.select_dtypes(include=[np.number]).columns]
    skewed = (
        combined[num_features].apply(lambda x: skew(x)).sort_values(ascending=False)
    )
    high_skew = skewed[abs(skewed) > 0.75].index
    for col in high_skew:
        combined[col] = np.log1p(combined[col])

    # Extract Raw versions for CatBoost (before one-hot encoding, retaining raw string labels)
    X_train_raw = combined.iloc[: len(train)].copy()
    X_test_raw = combined.iloc[len(train) :].copy()

    # Restore the raw, unencoded Neighborhood string labels in the persisted raw artifacts
    X_train_raw["Neighborhood"] = train["Neighborhood"].values
    X_test_raw["Neighborhood"] = test["Neighborhood"].values

    # One-hot encode for other models
    encoded = pd.get_dummies(combined, drop_first=True)

    X_train_proc = encoded.iloc[: len(train)].copy()
    X_test_proc = encoded.iloc[len(train) :].copy()

    # Save processed data to directory
    os.makedirs("./processed_data", exist_ok=True)
    X_train_proc.to_csv("./processed_data/X_train.csv", index=False)
    X_test_proc.to_csv("./processed_data/X_test.csv", index=False)
    X_train_raw.to_csv("./processed_data/X_train_raw.csv", index=False)
    X_test_raw.to_csv("./processed_data/X_test_raw.csv", index=False)
    pd.DataFrame({"SalePrice": train["SalePrice"]}).to_csv(
        "./processed_data/y_train.csv", index=False
    )
    pd.DataFrame({"SalePrice": y_train_log}).to_csv(
        "./processed_data/y_train_log.csv", index=False
    )

    print(
        f"✨ Top 1% Processed Shapes: X_train={X_train_proc.shape}, X_test={X_test_proc.shape}"
    )
    return X_train_proc, X_test_proc, y_train_log, test["Id"].values


if __name__ == "__main__":
    preprocess_house_prices_data()
