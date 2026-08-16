import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def solve_greedy(df: pd.DataFrame, budget: float, theta: float) -> pd.DataFrame:
    """
    Vectorized Greedy Heuristic Solver.
    Sorts by Expected Profit per dollar (ROI) and buys top properties until budget or risk is hit.
    """
    logger.warning("Pyomo solver failed or unavailable. Falling back to Vectorized Greedy Heuristic Solver.")

    # Calculate Expected Profit and Downside Risk
    df["Expected_Profit"] = df["SalePrice_pred"] - df["Asking_Price"]
    df["ROI"] = df["Expected_Profit"] / df["Asking_Price"]
    df["Downside_Risk"] = df["Asking_Price"] - df["LowerBound"]

    # Sort by ROI descending
    df_sorted = df.sort_values("ROI", ascending=False).reset_index()

    selected_idx = []
    current_spend = 0.0
    current_risk = 0.0

    for row in df_sorted.itertuples(index=False):
        # Check budget
        if current_spend + row.Asking_Price > budget:
            continue

        # Check downside conformal risk constraint:
        # Sum(A_i - L_i) <= theta * Sum(A_i)
        if current_risk + row.Downside_Risk > theta * (current_spend + row.Asking_Price):
            continue

        selected_idx.append(row.index)
        current_spend += row.Asking_Price
        current_risk += row.Downside_Risk

    df["Selected_Fraction"] = 0.0
    df.loc[selected_idx, "Selected_Fraction"] = 1.0
    df["Expected_Profit"] = df["SalePrice_pred"] - df["Asking_Price"]
    df["Conformal_Downside"] = df["Asking_Price"] - df["LowerBound"]
    return df


def generate_simulated_market_prices(df: pd.DataFrame, rng_seed: int = 42) -> pd.DataFrame:
    """Generates simulated market asking prices."""
    rng = np.random.default_rng(rng_seed)
    noise = rng.uniform(-4000, 4000, size=len(df))
    df["Asking_Price"] = 0.92 * df["SalePrice_pred"] + noise
    return df


def load_prediction_data() -> pd.DataFrame:
    """Loads prediction data with intervals, with fallback."""
    input_path = "./submissions/submission_with_intervals.csv"
    if not os.path.exists(input_path):
        input_path = "./submissions/submission_ensemble_final.csv"
        if not os.path.exists(input_path):
            raise FileNotFoundError("Could not find submission files.")

    df = pd.read_csv(input_path)

    if "LowerBound" not in df.columns:
        logger.warning("Conformal bounds missing. Simulating 5% bounds for optimization.")
        df["LowerBound"] = df["SalePrice"] * 0.95
        df["UpperBound"] = df["SalePrice"] * 1.05

    return df.rename(columns={"SalePrice": "SalePrice_pred"})
