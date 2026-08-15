import pytest
import os


import numpy as np
import pandas as pd

from src.preprocess import AmesDataTransformer, preprocess_data


def test_data_ingestion():
    if os.path.exists("data/train.csv"):
        df = pd.read_csv("data/train.csv")
        assert df.shape[0] > 0, f"Unexpected data shape: {df.shape}"


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
    assert pd.api.types.is_numeric_dtype(processed_df["ExterQual"]), "ExterQual was not converted to a numeric type!"
    assert pd.api.types.is_numeric_dtype(processed_df["KitchenQual"]), "KitchenQual was not converted to a numeric type!"
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
        assert (df["LowerBound"] <= df["SalePrice"]).all()
        assert (df["SalePrice"] <= df["UpperBound"]).all()

        # Lower bound clamped to min $42000
        assert (df["LowerBound"] >= 42000.0).all()
        # Upper bound clamped to max $525000
        assert (df["UpperBound"] <= 525000.0).all()


def test_pyomo_portfolio_solver():
    """
    Test the Pyomo MILP portfolio solver to ensure budget and conformal risk constraints
    are strictly satisfied with fractional_mode=False.
    """
    from src.portfolio_solver import solve_pyomo

    # Create mock test data
    mock_data = pd.DataFrame(
        {
            "Id": [101, 102, 103, 104, 105],
            "SalePrice_pred": [200000, 250000, 150000, 300000, 180000],
            "LowerBound": [180000, 210000, 140000, 270000, 160000],
            "UpperBound": [220000, 290000, 160000, 330000, 200000],
        }
    )

    # Generate asking prices: 90% of predicted price
    mock_data["Asking_Price"] = mock_data["SalePrice_pred"] * 0.90

    budget = 400000
    theta = 0.15  # 15% max downside risk

    # Run pyomo solver with binary decision variables (fractional_mode=False)
    res_df = solve_pyomo(mock_data.copy(), budget=budget, theta=theta, fractional_mode=False)

    # Assert output has the required columns
    assert "Selected_Fraction" in res_df.columns
    assert "Expected_Profit" in res_df.columns
    assert "Conformal_Downside" in res_df.columns

    # Filter selected properties
    selected = res_df[res_df["Selected_Fraction"] > 0.5]

    # 1. Binary check
    assert res_df["Selected_Fraction"].isin([0.0, 1.0]).all()

    # 2. Budget constraint check
    total_spend = selected["Asking_Price"].sum()
    assert total_spend <= budget

    # 3. Conformal downside risk check
    total_risk = selected["Conformal_Downside"].sum()
    # allow for a tiny floating point tolerance
    assert total_risk <= (theta * total_spend) + 1e-6

def test_get_neighborhood_ranks():
    from src.bifurcated_pipeline import get_neighborhood_ranks
    train_data = pd.DataFrame(
        {
            "Neighborhood": ["A", "B", "C"],
            "SalePrice": [100000, 200000, 300000],
            "GrLivArea": [1000, 1000, 1000]
        }
    )
    test_data = pd.DataFrame(
        {
            "Neighborhood": ["A", "B", "D"], # 'D' is unseen
            "GrLivArea": [1000, 1000, 1000]
        }
    )

    transformed_test = get_neighborhood_ranks(train_data, test_data)

    assert "Neighborhood" in transformed_test.columns
    assert pd.api.types.is_numeric_dtype(transformed_test["Neighborhood"])
    assert not transformed_test["Neighborhood"].isna().any()

    # Check that unseen neighborhood gets mapped to 13
    assert transformed_test.loc[2, "Neighborhood"] == 13

def test_bifurcated_pipeline_end_to_end(monkeypatch):
    if not os.path.exists("./data/train.csv"):
        pytest.skip("Dataset ./data/train.csv not found. Skipping test.")

    import src.bifurcated_pipeline
    monkeypatch.setattr(src.bifurcated_pipeline, "N_FOLDS", 2)
    monkeypatch.setattr(src.bifurcated_pipeline, "N_TRIALS", 1)

    try:
        src.bifurcated_pipeline.main()
    except Exception as e:
        assert False, f"bifurcated_pipeline.py failed with exception: {e}"

def test_bifurcated_predictions():
    assert os.path.exists("./submissions/submission_with_intervals_bifurcated.csv"), "CSV artifact not found, run pipeline first!"
    df = pd.read_csv("./submissions/submission_with_intervals_bifurcated.csv")
    assert (df["SalePrice"] > 0).all()
    assert (df["SalePrice_Lower"] >= 42000.0).all()
    assert (df["SalePrice_Upper"] <= 525000.0).all()
    assert (df["SalePrice_Lower"] <= df["SalePrice"]).all()
    assert (df["SalePrice"] <= df["SalePrice_Upper"]).all()
    assert df.shape[1] == 4
    assert "Id" in df.columns
    assert "SalePrice" in df.columns
    assert "SalePrice_Lower" in df.columns
    assert "SalePrice_Upper" in df.columns

def test_transform_neighborhood_rank_edge_cases():
    from src.preprocess import transform_neighborhood_rank

    rank_mapping = {'A': 1, 'B': 25, 'C': 10}

    # Note: median rank logic applies to the unseen neighborhoods when implemented dynamically
    # For testing, we ensure whatever logic (e.g., hardcoded 13) is used is covered correctly
    # transform_neighborhood_rank returns a Series in src/preprocess.py, returning 13 for unseen.

    # 1. Normal neighborhood mapping
    df_normal = pd.DataFrame({'Neighborhood': ['A', 'C']})
    res_normal = transform_neighborhood_rank(df_normal, rank_mapping)
    assert len(res_normal) == 2
    assert res_normal.iloc[0] == 1
    assert res_normal.iloc[1] == 10

    # 2. Unseen neighborhood defaulting to rank 13
    df_unseen = pd.DataFrame({'Neighborhood': ['A', 'D', 'C']})
    res_unseen = transform_neighborhood_rank(df_unseen, rank_mapping)
    assert len(res_unseen) == 3
    assert res_unseen.iloc[1] == 13

    # 3. NaN in the Neighborhood column defaulting to rank 13
    df_nan = pd.DataFrame({'Neighborhood': ['A', np.nan, 'C']})
    res_nan = transform_neighborhood_rank(df_nan, rank_mapping)
    assert len(res_nan) == 3
    assert res_nan.iloc[1] == 13

    # 4. DataFrame with no Neighborhood column yielding rank 13 for all rows
    df_no_col = pd.DataFrame({'OtherCol': [1, 2, 3]})
    res_no_col = transform_neighborhood_rank(df_no_col, rank_mapping)
    assert len(res_no_col) == 3
    assert (res_no_col == 13).all()

    # 5. Empty dataframe returns empty Series
    df_empty = pd.DataFrame()
    res_empty = transform_neighborhood_rank(df_empty, rank_mapping)
    assert len(res_empty) == 0

    # 6. Empty dataframe with Neighborhood column returns empty Series
    df_empty_col = pd.DataFrame({'Neighborhood': []})
    res_empty_col = transform_neighborhood_rank(df_empty_col, rank_mapping)
    assert len(res_empty_col) == 0
