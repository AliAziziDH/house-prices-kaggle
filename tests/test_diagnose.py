from unittest.mock import patch

import pandas as pd

from src.diagnose_pipeline import diagnose


def test_diagnose_missing_files(capsys):
    with patch("os.path.exists", return_value=False):
        diagnose()

    captured = capsys.readouterr()
    assert "❌ Raw data files missing in data/" in captured.out


def test_diagnose_success(capsys):
    mock_train_df = pd.DataFrame(
        {
            "Id": [1, 2],
            "SalePrice": [200000, 150000],
            "GrLivArea": [1500, 1200],
            "ExterQual": ["Ex", "TA"],
            "KitchenQual": ["Gd", "Fa"],
            "Neighborhood": ["CollgCr", "Veenker"],
        }
    )

    mock_test_df = pd.DataFrame(
        {
            "Id": [3, 4],
            "GrLivArea": [1300, 1400],
            "ExterQual": ["TA", "Gd"],
            "KitchenQual": ["TA", "Ex"],
            "Neighborhood": ["CollgCr", "CollgCr"],
        }
    )

    def mock_read_csv(filepath):
        if "train.csv" in filepath:
            return mock_train_df
        return mock_test_df

    with patch("os.path.exists", return_value=True), patch("pandas.read_csv", side_effect=mock_read_csv):
        diagnose()

    captured = capsys.readouterr()
    assert "✅ Pipeline diagnostics passed: No missing values found." in captured.out


def test_diagnose_with_nans(capsys):
    mock_train_df = pd.DataFrame(
        {
            "Id": [1],
            "SalePrice": [200000],
            "GrLivArea": [1500],
            "ExterQual": ["Ex"],
            "KitchenQual": ["Gd"],
            "Neighborhood": ["CollgCr"],
        }
    )

    mock_test_df = pd.DataFrame(
        {"Id": [2], "GrLivArea": [1300], "ExterQual": ["TA"], "KitchenQual": ["TA"], "Neighborhood": ["CollgCr"]}
    )

    def mock_read_csv(filepath):
        if "train.csv" in filepath:
            return mock_train_df
        return mock_test_df

    # We mock AmesDataTransformer.transform to return a DataFrame with NaNs
    mock_transformed_df = pd.DataFrame(
        {
            "Feature1": [None, 2]  # Contains NaN
        }
    )

    with (
        patch("os.path.exists", return_value=True),
        patch("pandas.read_csv", side_effect=mock_read_csv),
        patch("src.diagnose_pipeline.AmesDataTransformer.transform", return_value=mock_transformed_df),
    ):
        diagnose()

    captured = capsys.readouterr()
    assert "⚠️ Warning: Missing values remain in processed data." in captured.out
