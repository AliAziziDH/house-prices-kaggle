import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AmesDataTransformer(BaseEstimator, TransformerMixin):
    """
    Stateful Scikit-Learn transformer for Ames Housing data.
    Fits all statistics (medians, modes, target ranks, one-hot schema) strictly on training data
    and applies them without data leakage during transform.
    """

    def __init__(self):
        self.lot_frontage_neighborhood_medians_ = {}
        self.lot_frontage_global_median_ = 0.0
        self.categorical_modes_ = {}
        self.feature_columns_ = []

    def fit(self, X, y=None):
        X = X.copy()

        # 1. LotFrontage statistics
        if "LotFrontage" in X.columns:
            if "Neighborhood" in X.columns:
                self.lot_frontage_neighborhood_medians_ = (
                    X.groupby("Neighborhood")["LotFrontage"].median().to_dict()
                )
            med_val = X["LotFrontage"].median()
            self.lot_frontage_global_median_ = (
                float(med_val) if pd.notna(med_val) else 0.0
            )

        # 2. Categorical modes
        cat_cols_with_missing = [
            "Electrical",
            "MSZoning",
            "Utilities",
            "Exterior1st",
            "Exterior2nd",
            "KitchenQual",
            "Functional",
            "SaleType",
        ]
        for col in cat_cols_with_missing:
            if col in X.columns:
                mode_val = X[col].mode()
                if not mode_val.empty:
                    self.categorical_modes_[col] = mode_val[0]

        # 4. Transform training data to learn final column schema
        X_trans = self._transform_df(X)
        self.feature_columns_ = X_trans.columns.tolist()

        return self

    def _transform_df(self, df):
        df = df.copy()

        # 1. Garage features
        garage_cat_cols = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]
        for col in garage_cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna("No Garage")

        for col in ["GarageYrBlt", "GarageCars", "GarageArea"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # 2. Basement features
        bsmt_cat_cols = [
            "BsmtQual",
            "BsmtCond",
            "BsmtExposure",
            "BsmtFinType1",
            "BsmtFinType2",
        ]
        for col in bsmt_cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna("No Basement")

        bsmt_num_cols = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF"]
        for col in bsmt_num_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # 3. Masonry veneer features
        if "MasVnrType" in df.columns:
            df["MasVnrType"] = df["MasVnrType"].fillna("None")
        if "MasVnrArea" in df.columns:
            df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

        # 4. Optional features
        opt_cols = {
            "Alley": "No Alley",
            "PoolQC": "No Pool",
            "Fence": "No Fence",
            "FireplaceQu": "No Fireplace",
            "MiscFeature": "None",
        }
        for col, val in opt_cols.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)

        # 5. LotFrontage imputation using fitted medians
        if "LotFrontage" in df.columns:
            if "Neighborhood" in df.columns and self.lot_frontage_neighborhood_medians_:
                neigh_series = df["Neighborhood"].map(
                    self.lot_frontage_neighborhood_medians_
                )
                df["LotFrontage"] = df["LotFrontage"].fillna(neigh_series)
            df["LotFrontage"] = df["LotFrontage"].fillna(
                self.lot_frontage_global_median_
            )

        # 6. Categorical mode imputation using fitted modes
        for col, mode_val in self.categorical_modes_.items():
            if col in df.columns:
                df[col] = df[col].fillna(mode_val)

        bsmt_bath_cols = ["BsmtFullBath", "BsmtHalfBath"]
        for col in bsmt_bath_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # 7. Feature engineering
        if all(c in df.columns for c in ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]):
            df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

        if all(
            c in df.columns
            for c in ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
        ):
            df["TotalPorchSF"] = (
                df["OpenPorchSF"]
                + df["EnclosedPorch"]
                + df["3SsnPorch"]
                + df["ScreenPorch"]
            )

        if all(
            c in df.columns
            for c in ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]
        ):
            df["TotalBathrooms"] = (
                df["FullBath"]
                + 0.5 * df["HalfBath"]
                + df["BsmtFullBath"]
                + 0.5 * df["BsmtHalfBath"]
            )

        if all(c in df.columns for c in ["YrSold", "YearBuilt"]):
            df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
            df["IsNew"] = (df["YearBuilt"] == df["YrSold"]).astype(int)

        if all(c in df.columns for c in ["YrSold", "YearRemodAdd"]):
            df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

        if all(c in df.columns for c in ["OverallQual", "OverallCond"]):
            df["QualityScore"] = df["OverallQual"] * df["OverallCond"]

        if all(c in df.columns for c in ["YrSold", "GarageYrBlt"]):
            df["GarageAge"] = np.where(
                df["GarageYrBlt"] == 0, 0, df["YrSold"] - df["GarageYrBlt"]
            )

        # 8. Ordinal encoding
        quality_map = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
        bsmt_qual_map = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
        bsmt_exposure_map = {"No": 1, "Mn": 2, "Av": 3, "Gd": 4}
        bsmt_fin_map = {"Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
        functional_map = {
            "Sal": 1,
            "Sev": 2,
            "Maj2": 3,
            "Maj1": 4,
            "Mod": 5,
            "Min2": 6,
            "Min1": 7,
            "Typ": 8,
        }
        lot_shape_map = {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4}
        land_contour_map = {"Low": 1, "Bnk": 2, "HLS": 3, "Lvl": 4}
        utilities_map = {"NoSeWa": 1, "NoSewr": 2, "AllPub": 3}
        land_slope_map = {"Sev": 1, "Mod": 2, "Gtl": 3}

        ordinal_mappings = {
            "ExterQual": quality_map,
            "ExterCond": quality_map,
            "BsmtQual": bsmt_qual_map,
            "BsmtCond": bsmt_qual_map,
            "HeatingQC": quality_map,
            "KitchenQual": quality_map,
            "FireplaceQu": quality_map,
            "GarageQual": quality_map,
            "GarageCond": quality_map,
            "PoolQC": quality_map,
            "BsmtExposure": bsmt_exposure_map,
            "BsmtFinType1": bsmt_fin_map,
            "BsmtFinType2": bsmt_fin_map,
            "Functional": functional_map,
            "LotShape": lot_shape_map,
            "LandContour": land_contour_map,
            "Utilities": utilities_map,
            "LandSlope": land_slope_map,
        }

        for col, mapping in ordinal_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0)

        # 9. One-hot encoding
        nominal_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        if nominal_cols:
            df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

        # 10. VIF Enforcer: Drop highly collinear features
        vif_drops = ["GarageArea", "TotRmsAbvGrd", "1stFlrSF"]
        df = df.drop(columns=[col for col in vif_drops if col in df.columns])

        return df

    def transform(self, X):
        X_trans = self._transform_df(X)
        if self.feature_columns_:
            X_trans = X_trans.reindex(columns=self.feature_columns_, fill_value=0)
        return X_trans


def preprocess_data(df, is_training=True):
    """
    Backward-compatible preprocessing function using stateful AmesDataTransformer.
    """
    transformer = AmesDataTransformer()
    if "SalePrice" in df.columns:
        y = df["SalePrice"]
        X = df.drop(columns=["SalePrice"])
    else:
        y = None
        X = df

    transformer.fit(X, y)
    return transformer.transform(X)


if __name__ == "__main__":
    print("=" * 60)
    print("LOADING RAW DATA")
    print("=" * 60)

    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")

    # --- OUTLIER REMOVAL (BEFORE SPLIT) ---
    outlier_mask = (train["GrLivArea"] > 4000) & (train["SalePrice"] < 200000)
    train_full = train[~outlier_mask].reset_index(drop=True)

    print(f"Full Train shape after outlier removal: {train_full.shape}")
    print(f"Test shape: {test.shape}")

    # --- 90/10 SPLIT ---
    from sklearn.model_selection import train_test_split

    train_proper, calib_set = train_test_split(
        train_full, test_size=0.1, random_state=42
    )

    train_proper = train_proper.reset_index(drop=True)
    calib_set = calib_set.reset_index(drop=True)

    print(f"Proper Train shape (90%): {train_proper.shape}")
    print(f"Calibration shape (10%): {calib_set.shape}")

    y_train_full = train_full["SalePrice"]
    X_train_full_df = train_full.drop(["Id", "SalePrice"], axis=1)

    y_train = train_proper["SalePrice"]
    X_train_df = train_proper.drop(["Id", "SalePrice"], axis=1)

    y_calib = calib_set["SalePrice"]
    X_calib_df = calib_set.drop(["Id", "SalePrice"], axis=1)

    X_test_df = test.drop(["Id"], axis=1)

    # 100% data transformer
    transformer_full = AmesDataTransformer()
    transformer_full.fit(X_train_full_df, y_train_full)
    X_train_full = transformer_full.transform(X_train_full_df)

    # 90% data transformer
    transformer = AmesDataTransformer()
    # Fit ONLY on proper train for 90/10 models and test predictions
    transformer.fit(X_train_df, y_train)

    X_train = transformer.transform(X_train_df)
    X_calib = transformer.transform(X_calib_df)

    # We predict the test set using the 100% data transformer for final point prediction
    # Wait, the 90% models will need test predictions?
    # Actually, 90% models only need to predict on X_calib to find the quantile.
    # The final predictions use 100% models predicting on X_test transformed by transformer_full.
    X_test = transformer_full.transform(X_test_df)

    import os


    os.makedirs("./processed_data", exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    # save full datasets
    X_train_full.to_csv("./processed_data/X_train_full.csv", index=False)
    y_train_full.to_csv("./processed_data/y_train_full.csv", index=False)
    train_full.drop("SalePrice", axis=1).to_csv(
        "./processed_data/X_train_full_raw.csv", index=False
    )

    # save 90% and 10%
    X_train.to_csv("./processed_data/X_train.csv", index=False)
    y_train.to_csv("./processed_data/y_train.csv", index=False)

    X_calib.to_csv("./processed_data/X_calib.csv", index=False)
    y_calib.to_csv("./processed_data/y_calib.csv", index=False)

    X_test.to_csv("./processed_data/X_test.csv", index=False)

    train_proper.drop("SalePrice", axis=1).to_csv(
        "./processed_data/X_train_raw.csv", index=False
    )
    calib_set.drop("SalePrice", axis=1).to_csv(
        "./processed_data/X_calib_raw.csv", index=False
    )
    test.to_csv("./processed_data/X_test_raw.csv", index=False)

    # Instead of picking one transformer to save as boxcox_transformer, we use PowerTransformer in models.
    # Actually we just save X_test transformed by transformer_full, so that's okay.

    print("✅ Processed data saved successfully.")
