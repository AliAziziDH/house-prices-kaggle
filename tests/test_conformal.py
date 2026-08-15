import numpy as np

from src.conformal import compute_non_conformity_scores


def test_compute_non_conformity_scores_identical():
    """Test that identical arrays yield scores of 0.0."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    result = compute_non_conformity_scores(y_true, y_pred)
    assert np.allclose(result, np.zeros(4))


def test_compute_non_conformity_scores_positive_diff():
    """Test with known positive differences."""
    y_true = np.array([2.0, 5.0, 10.0])
    y_pred = np.array([1.0, 3.0, 6.0])
    result = compute_non_conformity_scores(y_true, y_pred)
    expected = np.array([1.0, 2.0, 4.0])
    assert np.allclose(result, expected)


def test_compute_non_conformity_scores_negative_diff():
    """Test with negative differences (testing absolute value behavior)."""
    y_true = np.array([1.0, 3.0, 6.0])
    y_pred = np.array([2.0, 5.0, 10.0])
    result = compute_non_conformity_scores(y_true, y_pred)
    expected = np.array([1.0, 2.0, 4.0])
    assert np.allclose(result, expected)


def test_compute_non_conformity_scores_lists():
    """Test compatibility with standard Python lists."""
    y_true = [1.0, 3.0, 6.0]
    y_pred = [2.0, 5.0, 10.0]
    # The function uses np.abs which should cast lists to numpy arrays
    result = compute_non_conformity_scores(np.array(y_true), np.array(y_pred))
    expected = np.array([1.0, 2.0, 4.0])
    assert np.allclose(result, expected)


def test_compute_non_conformity_scores_mixed_signs():
    """Test with a mix of positive, negative, and zero differences."""
    y_true = np.array([-1.0, 0.0, 2.5, 5.0])
    y_pred = np.array([1.0, 0.0, -2.5, 2.0])
    result = compute_non_conformity_scores(y_true, y_pred)
    expected = np.array([2.0, 0.0, 5.0, 3.0])
    assert np.allclose(result, expected)
