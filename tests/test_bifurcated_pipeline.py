import numpy as np
import pandas as pd


# Simple tests to ensure our extracted functions don't crash
def test_bifurcated_pipeline_import():
    import src.bifurcated_pipeline

    assert src.bifurcated_pipeline is not None


def test_calculate_ensemble_weights():
    from src.bifurcated_pipeline import calculate_ensemble_weights

    # 3 models, 10 samples
    preds = np.random.rand(10, 3)
    y_true = np.random.rand(10)

    weights = calculate_ensemble_weights(preds, y_true)

    assert len(weights) == 3
    assert np.isclose(np.sum(weights), 1.0)
    assert all(w >= 0.0 for w in weights)
    assert all(w <= 1.0 for w in weights)


def test_save_submissions(tmp_path):
    import os

    from src.bifurcated_pipeline import save_submissions

    test_ids = pd.Series([1, 2, 3])
    y_pred = np.array([150000.0, 200000.0, 250000.0])
    lower = y_pred * 0.9
    upper = y_pred * 1.1

    # Monkey patch os.makedirs and pd.DataFrame.to_csv to write to temp dir
    original_makedirs = os.makedirs

    def mock_makedirs(path, *args, **kwargs):
        if path == "./submissions":
            return original_makedirs(tmp_path / "submissions", *args, **kwargs)
        return original_makedirs(path, *args, **kwargs)

    os.makedirs = mock_makedirs

    original_to_csv = pd.DataFrame.to_csv

    def mock_to_csv(self, path, *args, **kwargs):
        if path.startswith("submissions/"):
            filename = os.path.basename(path)
            new_path = tmp_path / "submissions" / filename
            return original_to_csv(self, new_path, *args, **kwargs)
        return original_to_csv(self, path, *args, **kwargs)

    pd.DataFrame.to_csv = mock_to_csv

    try:
        save_submissions(test_ids, y_pred, lower, upper)

        # Verify files were created
        assert os.path.exists(tmp_path / "submissions" / "submission_bifurcated.csv")
        assert os.path.exists(tmp_path / "submissions" / "submission_with_intervals_bifurcated.csv")

        # Verify content
        sub1 = pd.read_csv(tmp_path / "submissions" / "submission_bifurcated.csv")
        assert list(sub1.columns) == ["Id", "SalePrice"]
        assert len(sub1) == 3

        sub2 = pd.read_csv(tmp_path / "submissions" / "submission_with_intervals_bifurcated.csv")
        assert list(sub2.columns) == ["Id", "SalePrice", "SalePrice_Lower", "SalePrice_Upper"]
        assert len(sub2) == 3
    finally:
        os.makedirs = original_makedirs
        pd.DataFrame.to_csv = original_to_csv
