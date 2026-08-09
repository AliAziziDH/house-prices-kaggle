"""
Automated Pytest Suite for Small-Dataset House Prices Preprocessing.
"""


import numpy as np
import pandas as pd
import pytest

from src.preprocess import QUALITY_MAP, preprocess_house_prices_data


def test_quality_mapping_values():
    """Verify Ordinal Quality Map values."""
    assert QUALITY_MAP['Ex'] == 5
    assert QUALITY_MAP['Gd'] == 4
    assert QUALITY_MAP['TA'] == 3
    assert QUALITY_MAP['None'] == 0

def test_preprocess_data_shapes(tmp_path):
    """Verify data preprocessing pipeline outputs valid non-null datasets."""

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create dummy data
    np.random.seed(42)
    train_data = pd.DataFrame({
        'Id': range(1, 1461),
        'MSSubClass': [60] * 1460,
        'MSZoning': ['RL'] * 1460,
        'LotFrontage': [65.0] * 1460,
        'LotArea': [8450] * 1460,
        'Street': ['Pave'] * 1460,
        'Alley': ['None'] * 1460,
        'LotShape': ['Reg'] * 1460,
        'LandContour': ['Lvl'] * 1460,
        'Utilities': ['AllPub'] * 1460,
        'LotConfig': ['Inside'] * 1460,
        'LandSlope': ['Gtl'] * 1460,
        'Neighborhood': ['CollgCr'] * 1460,
        'Condition1': ['Norm'] * 1460,
        'Condition2': ['Norm'] * 1460,
        'BldgType': ['1Fam'] * 1460,
        'HouseStyle': ['2Story'] * 1460,
        'OverallQual': [7] * 1460,
        'OverallCond': [5] * 1460,
        'YearBuilt': [2003] * 1460,
        'YearRemodAdd': [2003] * 1460,
        'RoofStyle': ['Gable'] * 1460,
        'RoofMatl': ['CompShg'] * 1460,
        'Exterior1st': ['VinylSd'] * 1460,
        'Exterior2nd': ['VinylSd'] * 1460,
        'MasVnrType': ['BrkFace'] * 1460,
        'MasVnrArea': [196.0] * 1460,
        'ExterQual': ['Gd'] * 1460,
        'ExterCond': ['TA'] * 1460,
        'Foundation': ['PConc'] * 1460,
        'BsmtQual': ['Gd'] * 1460,
        'BsmtCond': ['TA'] * 1460,
        'BsmtExposure': ['No'] * 1460,
        'BsmtFinType1': ['GLQ'] * 1460,
        'BsmtFinSF1': [706] * 1460,
        'BsmtFinType2': ['Unf'] * 1460,
        'BsmtFinSF2': [0] * 1460,
        'BsmtUnfSF': [150] * 1460,
        'TotalBsmtSF': [856] * 1460,
        'Heating': ['GasA'] * 1460,
        'HeatingQC': ['Ex'] * 1460,
        'CentralAir': ['Y'] * 1460,
        'Electrical': ['SBrkr'] * 1460,
        '1stFlrSF': [856] * 1460,
        '2ndFlrSF': [854] * 1460,
        'LowQualFinSF': [0] * 1460,
        'GrLivArea': [1710] * 1460,
        'BsmtFullBath': [1] * 1460,
        'BsmtHalfBath': [0] * 1460,
        'FullBath': [2] * 1460,
        'HalfBath': [1] * 1460,
        'BedroomAbvGr': [3] * 1460,
        'KitchenAbvGr': [1] * 1460,
        'KitchenQual': ['Gd'] * 1460,
        'TotRmsAbvGrd': [8] * 1460,
        'Functional': ['Typ'] * 1460,
        'Fireplaces': [0] * 1460,
        'FireplaceQu': ['None'] * 1460,
        'GarageType': ['Attchd'] * 1460,
        'GarageYrBlt': [2003.0] * 1460,
        'GarageFinish': ['RFn'] * 1460,
        'GarageCars': [2] * 1460,
        'GarageArea': [548] * 1460,
        'GarageQual': ['TA'] * 1460,
        'GarageCond': ['TA'] * 1460,
        'PavedDrive': ['Y'] * 1460,
        'WoodDeckSF': [0] * 1460,
        'OpenPorchSF': [61] * 1460,
        'EnclosedPorch': [0] * 1460,
        '3SsnPorch': [0] * 1460,
        'ScreenPorch': [0] * 1460,
        'PoolArea': [0] * 1460,
        'PoolQC': ['None'] * 1460,
        'Fence': ['None'] * 1460,
        'MiscFeature': ['None'] * 1460,
        'MiscVal': [0] * 1460,
        'MoSold': [2] * 1460,
        'YrSold': [2008] * 1460,
        'SaleType': ['WD'] * 1460,
        'SaleCondition': ['Normal'] * 1460,
        'SalePrice': np.random.randint(100000, 300000, size=1460)
    })

    # Introduce 2 outliers
    train_data.loc[0, 'GrLivArea'] = 4500
    train_data.loc[0, 'SalePrice'] = 200000
    train_data.loc[1, 'GrLivArea'] = 4100
    train_data.loc[1, 'SalePrice'] = 150000

    # Ensure none for Alley, Fence, MiscFeature to prevent NaNs when get_dummies doesn't apply to numericals that are all None string
    train_data['Alley'] = 'None'
    train_data['Fence'] = 'None'
    train_data['MiscFeature'] = 'None'

    test_data = train_data.copy().drop('SalePrice', axis=1).iloc[:1459]
    test_data['Id'] = range(1461, 1461 + 1459)

    train_data.to_csv(data_dir / "train.csv", index=False)
    test_data.to_csv(data_dir / "test.csv", index=False)

    X_tr, X_te, _y_tr, _test_ids = preprocess_house_prices_data(str(data_dir))
    assert len(X_tr) == 1458 # 1460 - 2 outliers
    assert len(X_te) == 1459

if __name__ == '__main__':
    pytest.main(['-v', __file__])
