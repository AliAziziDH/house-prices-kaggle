"""
Automated Pytest Suite for Small-Dataset House Prices Preprocessing.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
from pathlib import Path
from src.preprocess import preprocess_house_prices_data, QUALITY_MAP
from src.find_ensemble_weights import rmsle_dollars

def test_quality_mapping_values():
    """Verify Ordinal Quality Map values."""
    assert QUALITY_MAP['Ex'] == 5
    assert QUALITY_MAP['Gd'] == 4
    assert QUALITY_MAP['TA'] == 3
    assert QUALITY_MAP['None'] == 0

def create_mock_data():
    columns = [
        'Id', 'SalePrice', 'GrLivArea', 'Neighborhood', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
        'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu',
        'GarageQual', 'GarageCond', 'PoolQC', 'BsmtFinType1', 'BsmtFinType2', 'BsmtExposure',
        'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath', 'OpenPorchSF', '3SsnPorch',
        'EnclosedPorch', 'ScreenPorch', 'WoodDeckSF', 'YrSold', 'YearBuilt', 'YearRemodAdd'
    ]
    train_data = {col: np.zeros(1460) for col in columns}
    train_data['Id'] = np.arange(1, 1461)
    train_data['GrLivArea'] = np.ones(1460) * 1000
    train_data['SalePrice'] = np.ones(1460) * 200000
    train_data['Neighborhood'] = ['CollgCr'] * 1460
    # Create 2 outliers (GrLivArea > 4000 & SalePrice < 300,000)
    train_data['GrLivArea'][0] = 4001
    train_data['SalePrice'][0] = 200000
    train_data['GrLivArea'][1] = 4001
    train_data['SalePrice'][1] = 200000
    train_df = pd.DataFrame(train_data)

    test_columns = [col for col in columns if col != 'SalePrice']
    test_data = {col: np.zeros(1459) for col in test_columns}
    test_data['Id'] = np.arange(1461, 2920)
    test_data['GrLivArea'] = np.ones(1459) * 1000
    test_data['Neighborhood'] = ['CollgCr'] * 1459
    test_df = pd.DataFrame(test_data)

    return train_df, test_df

@patch("os.makedirs")
@patch("pandas.DataFrame.to_csv")
@patch("pandas.read_csv")
def test_preprocess_data_shapes(mock_read_csv, mock_to_csv, mock_makedirs):
    """Verify data preprocessing pipeline outputs valid non-null datasets."""
    data_dir = "./data"

    # Skip test if actual data files are missing during testing environments
    if not (Path(data_dir) / "train.csv").exists():
        pytest.skip("Kaggle data files missing in ./data")

    X_tr, X_te, y_tr, test_ids = preprocess_house_prices_data(data_dir)
    assert len(X_tr) == 1458 # 1460 - 2 outliers
    assert len(X_te) == 1459


def test_rmsle_dollars_perfect_match():
    """Verify exact match returns RMSLE of 0."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([100.0, 200.0, 300.0])
    assert np.isclose(rmsle_dollars(y_true, y_pred), 0.0)

def test_rmsle_dollars_negative_clamping():
    """Verify negative values are correctly clamped to 0.0."""
    y_true = np.array([0.0, 5.0])
    y_pred = np.array([-10.0, 5.0])
    # -10 gets clamped to 0.0. RMSLE should be 0 since both arrays effectively become [0.0, 5.0]
    assert np.isclose(rmsle_dollars(y_true, y_pred), 0.0)

def test_rmsle_dollars_known_values():
    """Verify RMSLE calculation yields expected mathematical results."""
    # log1p(e-1) = 1, log1p(0) = 0
    y_true = np.array([np.exp(1) - 1])
    y_pred = np.array([0.0])
    # log difference is 1 - 0 = 1, squared is 1, mean is 1, sqrt is 1
    assert np.isclose(rmsle_dollars(y_true, y_pred), 1.0)

def test_rmsle_dollars_zero_values():
    """Verify behavior when both inputs are exactly zero."""
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([0.0, 0.0])
    assert np.isclose(rmsle_dollars(y_true, y_pred), 0.0)

if __name__ == '__main__':
    pytest.main(['-v', __file__])
