import numpy as np
import pandas as pd
import pytest

from src.metrics import rmsle


def test_rmsle_identical():
    """Test that identical arrays yield an RMSLE of 0.0."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    result = rmsle(y_true, y_pred)
    assert np.isclose(result, 0.0)


def test_rmsle_known_values():
    """Test RMSLE with known values."""
    # log1p(x) = log(1+x)
    # y_true = e^1 - 1, log1p(y_true) = 1
    # y_pred = e^2 - 1, log1p(y_pred) = 2
    # error = (1 - 2)^2 = 1
    # mean squared error = 1, root = 1
    y_true = np.array([np.exp(1) - 1, np.exp(1) - 1])
    y_pred = np.array([np.exp(2) - 1, np.exp(2) - 1])
    result = rmsle(y_true, y_pred)
    assert np.isclose(result, 1.0)


def test_rmsle_negative_values():
    """Test that negative values are clamped to 0."""
    # y_true = [-1, -5] -> clamped to [0, 0], log1p = [0, 0]
    # y_pred = [np.exp(1)-1, np.exp(1)-1] -> [1.718, 1.718], log1p = [1, 1]
    # mse = (0-1)^2 = 1, rmsle = 1
    y_true = np.array([-1.0, -5.0])
    y_pred = np.array([np.exp(1) - 1, np.exp(1) - 1])
    result = rmsle(y_true, y_pred)
    assert np.isclose(result, 1.0)


def test_rmsle_zero_values():
    """Test RMSLE with zero values."""
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([0.0, np.exp(2) - 1])
    # log1p(y_true) = [0, 0]
    # log1p(y_pred) = [0, 2]
    # error_sq = [0, 4], mean = 2, root = sqrt(2)
    result = rmsle(y_true, y_pred)
    assert np.isclose(result, np.sqrt(2.0))


def test_rmsle_lists():
    """Test RMSLE works with standard python lists."""
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.0, 2.0, 3.0, 4.0]
    result = rmsle(y_true, y_pred)
    assert np.isclose(result, 0.0)


def test_rmsle_pandas_series():
    """Test RMSLE works with pandas Series."""
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0])
    y_pred = pd.Series([1.0, 2.0, 3.0, 4.0])
    result = rmsle(y_true, y_pred)
    assert np.isclose(result, 0.0)


def test_rmsle_mismatched_lengths():
    """Test RMSLE raises ValueError on mismatched lengths."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        rmsle(y_true, y_pred)


def test_rmsle_large_values():
    """Test RMSLE with very large values."""
    y_true = np.array([1e6, 1e7])
    y_pred = np.array([1e6, 1e7])
    result = rmsle(y_true, y_pred)
    assert np.isclose(result, 0.0)
