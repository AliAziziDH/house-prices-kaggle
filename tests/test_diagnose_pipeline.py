import pytest
import pandas as pd
from src.diagnose_pipeline import print_stats

def test_print_stats(capsys):
    """Verify that print_stats correctly calculates and formats statistics for a given pandas Series."""
    # Create a deterministic pandas Series
    s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])

    # Call the function
    print_stats("Test Stats", s)

    # Capture standard output
    captured = capsys.readouterr()

    # Assert formatting and correct values are printed
    assert "--- Test Stats ---" in captured.out
    assert "Count:  5" in captured.out
    assert "Min:    $10.00" in captured.out
    assert "25%:    $20.00" in captured.out
    assert "Median: $30.00" in captured.out
    assert "Mean:   $30.00" in captured.out
    assert "75%:    $40.00" in captured.out
    assert "Max:    $50.00" in captured.out

    # std and skew can be checked for approximate output format
    # For [10, 20, 30, 40, 50], std is ~ 15.811
    # skew is 0.0000
    assert "Std:    $15.81" in captured.out
    assert "Skew:   0.0000" in captured.out

if __name__ == '__main__':
    pytest.main(['-v', __file__])
