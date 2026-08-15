import os
import pandas as pd
import pytest
from src.portfolio_utils import load_prediction_data

def test_load_prediction_data_file_not_found(mocker):
    """Test that a FileNotFoundError is raised when both submission files are missing."""
    # Mock os.path.exists to always return False
    mocker.patch('os.path.exists', return_value=False)

    with pytest.raises(FileNotFoundError, match="Could not find submission files."):
        load_prediction_data()

def test_load_prediction_data_fallback_success(mocker):
    """Test that it successfully falls back to the secondary file if the primary is missing."""

    # Custom mock for os.path.exists to return False for the first path, True for the second
    def mock_exists(path):
        if path == "./submissions/submission_with_intervals.csv":
            return False
        if path == "./submissions/submission_ensemble_final.csv":
            return True
        return False

    mocker.patch('os.path.exists', side_effect=mock_exists)

    # Mock pd.read_csv to return a sample DataFrame
    mock_df = pd.DataFrame({
        "Id": [1, 2],
        "SalePrice": [200000, 300000]
    })
    mocker.patch('pandas.read_csv', return_value=mock_df)

    df = load_prediction_data()

    # Check that SalePrice got renamed to SalePrice_pred
    assert "SalePrice_pred" in df.columns
    assert "SalePrice" not in df.columns

    # Check that fallback bounds were added because "LowerBound" was missing
    assert "LowerBound" in df.columns
    assert "UpperBound" in df.columns

    # Check the fallback calculations
    assert df["LowerBound"].iloc[0] == 200000 * 0.95
    assert df["UpperBound"].iloc[0] == 200000 * 1.05
