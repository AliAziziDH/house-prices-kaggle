import os

import numpy as np
import pandas as pd

from src.preprocess import AmesDataTransformer, preprocess_data


def test_data_ingestion():
    if os.path.exists("data/train.csv"):
        df = pd.read_csv("data/train.csv")
        assert df.shape == (1460, 81), f"Unexpected data shape: {df.shape}"


def test_preprocessing_transforms():
    # Create a fully populated, valid mock dataframe
    mock_data = pd.DataFrame(
        {
            "Id": [1, 2],
            "GrLivArea": [1710, 1262],
            "ExterQual": ["Ex", "TA"],
            "KitchenQual": ["Gd", "Fa"],
            "Neighborhood": ["CollgCr", "Veenker"],
            "SalePrice": [208500, 181500],
        }
    )

    # Run production preprocess function
    processed_df = preprocess_data(mock_data, is_training=True)

    # Assert that ordinal quality map successfully mapped string values to numeric types
    assert pd.api.types.is_numeric_dtype(processed_df["ExterQual"]), (
        "ExterQual was not converted to a numeric type!"
    )
    assert pd.api.types.is_numeric_dtype(processed_df["KitchenQual"]), (
        "KitchenQual was not converted to a numeric type!"
    )
    assert processed_df["ExterQual"].iloc[0] == 5
    assert processed_df["ExterQual"].iloc[1] == 3


def test_stateful_transformer_leak_free():
    train_data = pd.DataFrame(
        {
            "Id": [1, 2, 3],
            "LotFrontage": [65.0, np.nan, 80.0],
            "ExterQual": ["Ex", "TA", "Gd"],
            "KitchenQual": ["Gd", "Fa", "TA"],
            "Neighborhood": ["CollgCr", "CollgCr", "Veenker"],
            "Electrical": ["SBrkr", "SBrkr", np.nan],
        }
    )
    y_train = pd.Series([200000, 150000, 300000])

    test_data = pd.DataFrame(
        {
            "Id": [4, 5],
            "LotFrontage": [np.nan, 70.0],
            "ExterQual": ["Gd", "Fa"],
            "KitchenQual": ["Ex", "TA"],
            "Neighborhood": ["UnseenNeigh", "Veenker"],
            "Electrical": [np.nan, "FuseA"],
        }
    )

    transformer = AmesDataTransformer()
    transformer.fit(train_data, y_train)

    train_trans = transformer.transform(train_data)
    test_trans = transformer.transform(test_data)

    # Columns must match exactly between train and test
    assert list(train_trans.columns) == list(test_trans.columns)

    # Check missing LotFrontage filled using fitted training statistics
    assert not test_trans["LotFrontage"].isna().any()


def test_conformal_intervals():
    """
    Test conformal interval boundaries are sound.
    """
    if os.path.exists("./submissions/submission_with_intervals.csv"):
        df = pd.read_csv("./submissions/submission_with_intervals.csv")
        # Lower bound < Point < Upper bound
        assert (df["SalePrice_Lower"] <= df["SalePrice"]).all()
        assert (df["SalePrice"] <= df["SalePrice_Upper"]).all()

        # Lower bound clamped to min $42000
        assert (df["SalePrice_Lower"] >= 42000.0).all()
        # Upper bound clamped to max $525000
        assert (df["SalePrice_Upper"] <= 525000.0).all()
