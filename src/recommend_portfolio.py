import logging
import os

import numpy as np
import pandas as pd

try:
    import pyomo.environ as pyo

    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def solve_greedy(df, budget, theta):
    """
    Vectorized Greedy Heuristic Solver.
    Sorts by Expected Profit per dollar (ROI) and buys top properties until budget or risk is hit.
    """
    logger.warning("Pyomo solver failed or unavailable. Falling back to Vectorized Greedy Heuristic Solver.")

    # Calculate Expected Profit and Downside Risk
    df["Expected_Profit"] = df["SalePrice_pred"] - df["Asking_Price"]
    df["ROI"] = df["Expected_Profit"] / df["Asking_Price"]
    df["Downside_Risk"] = df["Asking_Price"] - df["SalePrice_Lower"]

    # Sort by ROI descending
    df_sorted = df.sort_values("ROI", ascending=False).reset_index()

    selected_idx = []
    current_spend = 0.0
    current_risk = 0.0

    for i, row in df_sorted.iterrows():
        # Check budget
        if current_spend + row["Asking_Price"] > budget:
            continue

        # Check downside conformal risk constraint:
        # Sum(A_i - L_i) <= theta * Sum(A_i)
        # current_risk + row_risk <= theta * (current_spend + row_spend)
        if current_risk + row["Downside_Risk"] > theta * (current_spend + row["Asking_Price"]):
            continue

        selected_idx.append(row["index"])
        current_spend += row["Asking_Price"]
        current_risk += row["Downside_Risk"]

    df["Selected_Fraction"] = 0.0
    df.loc[selected_idx, "Selected_Fraction"] = 1.0
    df["Expected_Profit"] = df["SalePrice_pred"] - df["Asking_Price"]
    df["Conformal_Downside"] = df["Asking_Price"] - df["SalePrice_Lower"]
    return df


def solve_pyomo(df, budget, theta, fractional_mode=True):
    """
    Pyomo optimization solver.
    """
    # Check if pyomo is available
    if not PYOMO_AVAILABLE:
        raise ImportError("Pyomo is not installed.")

    # Calculate pre-requisites
    N = len(df)
    asking_prices = df["Asking_Price"].values
    expected_profits = (df["SalePrice_pred"] - df["Asking_Price"]).values
    lower_bounds = df["SalePrice_Lower"].values

    # Create model
    m = pyo.ConcreteModel()

    # Indices
    m.I = pyo.RangeSet(0, N - 1)

    # Variables
    if fractional_mode:
        m.x = pyo.Var(m.I, domain=pyo.NonNegativeReals, bounds=(0.0, 1.0))
    else:
        m.x = pyo.Var(m.I, domain=pyo.Binary)

    # Objective
    m.obj = pyo.Objective(expr=sum(expected_profits[i] * m.x[i] for i in m.I), sense=pyo.maximize)

    # Budget Constraint
    m.budget_cons = pyo.Constraint(expr=sum(asking_prices[i] * m.x[i] for i in m.I) <= budget)

    # Risk Constraint: Sum ( (1 - theta) * A_i - L_i ) * x_i <= 0
    m.risk_cons = pyo.Constraint(expr=sum(((1.0 - theta) * asking_prices[i] - lower_bounds[i]) * m.x[i] for i in m.I) <= 0)

    # Solve
    solver = pyo.SolverFactory("glpk")

    if not solver.available():
        solver = pyo.SolverFactory("cbc")
        if not solver.available():
            raise RuntimeError("No suitable Pyomo solver found.")

    results = solver.solve(m, tee=False)

    if (results.solver.status == pyo.SolverStatus.ok) and (
        results.solver.termination_condition == pyo.TerminationCondition.optimal
    ):
        logger.info("Pyomo optimization successful.")
        df["Selected_Fraction"] = [pyo.value(m.x[i]) for i in m.I]
        df["Expected_Profit"] = expected_profits
        df["Conformal_Downside"] = asking_prices - lower_bounds
        return df
    else:
        raise RuntimeError("Pyomo optimization did not converge optimally.")


def recommend_portfolio(budget=1500000.0, theta=0.10, fractional_mode=True):
    print("=" * 60)
    print("STARTING PORTFOLIO OPTIMIZATION")
    print("=" * 60)

    # 1. Load Data
    input_path = "./submissions/submission_with_intervals.csv"
    if not os.path.exists(input_path):
        input_path = "./submissions/submission_ensemble_final.csv"  # Fallback if intervals missing
        if not os.path.exists(input_path):
            raise FileNotFoundError("Could not find submission files.")

    df = pd.read_csv(input_path)

    # Ensure interval bounds exist
    if "SalePrice_Lower" not in df.columns:
        logger.warning("Conformal bounds missing. Simulating 5% bounds for optimization.")
        df["SalePrice_Lower"] = df["SalePrice"] * 0.95
        df["SalePrice_Upper"] = df["SalePrice"] * 1.05

    df = df.rename(columns={"SalePrice": "SalePrice_pred"})

    # 2. Simulated Market Asking Prices
    rng = np.random.default_rng(42)
    noise = rng.uniform(-4000, 4000, size=len(df))
    df["Asking_Price"] = 0.92 * df["SalePrice_pred"] + noise

    # 3. Optimize
    try:
        res_df = solve_pyomo(df.copy(), budget, theta, fractional_mode)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Optimization exception: {e}")
        res_df = solve_greedy(df.copy(), budget, theta)

    # 4. Save
    output_cols = [
        "Id",
        "SalePrice_pred",
        "Asking_Price",
        "SalePrice_Lower",
        "Expected_Profit",
        "Conformal_Downside",
        "Selected_Fraction",
    ]

    # Filter for selected
    res_df = res_df[res_df["Selected_Fraction"] > 0.001].copy()

    os.makedirs("./submissions", exist_ok=True)
    res_df[output_cols].to_csv("./submissions/portfolio_recommendation.csv", index=False)

    total_spend = (res_df["Asking_Price"] * res_df["Selected_Fraction"]).sum()
    total_profit = (res_df["Expected_Profit"] * res_df["Selected_Fraction"]).sum()
    total_downside = (res_df["Conformal_Downside"] * res_df["Selected_Fraction"]).sum()

    print(f"✅ Portfolio optimization complete. Selected {len(res_df)} properties.")
    print(f"   Total Budget Spent: ${total_spend:,.2f} / ${budget:,.2f}")
    print(f"   Total Expected Profit: ${total_profit:,.2f}")
    print(f"   Max Downside Risk: ${total_downside:,.2f} ({(total_downside / total_spend) * 100:.1f}% of spend)")
    print("✅ Results saved to './submissions/portfolio_recommendation.csv'")


if __name__ == "__main__":
    recommend_portfolio()
