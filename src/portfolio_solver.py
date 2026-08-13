import logging
import os

import pandas as pd

from src.portfolio_utils import generate_simulated_market_prices, load_prediction_data, solve_greedy

try:
    import pyomo.environ as pyo

    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False
    pyo = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def solve_pyomo(df: pd.DataFrame, budget: float, theta: float, fractional_mode: bool = False) -> pd.DataFrame:
    """Pyomo optimization solver."""
    if not PYOMO_AVAILABLE or pyo is None:
        raise ImportError("Pyomo is not installed.")

    N = len(df)
    asking_prices = df["Asking_Price"].values
    expected_profits = (df["SalePrice_pred"] - df["Asking_Price"]).values
    lower_bounds = df["SalePrice_Lower"].values

    m = pyo.ConcreteModel(doc="Portfolio Optimization Model")
    m.I = pyo.RangeSet(0, N - 1, doc="Set of available properties")

    m.expected_profit = pyo.Param(m.I, initialize=lambda m, i: expected_profits[i], mutable=True)
    m.asking_price = pyo.Param(m.I, initialize=lambda m, i: asking_prices[i], mutable=True)
    m.lower_bound = pyo.Param(m.I, initialize=lambda m, i: lower_bounds[i], mutable=True)
    m.budget_param = pyo.Param(initialize=budget, mutable=True)
    m.theta_param = pyo.Param(initialize=theta, mutable=True)

    if fractional_mode:
        m.x = pyo.Var(m.I, domain=pyo.NonNegativeReals, bounds=(0.0, 1.0))
    else:
        m.x = pyo.Var(m.I, domain=pyo.Binary)

    m.obj = pyo.Objective(expr=sum(m.expected_profit[i] * m.x[i] for i in m.I), sense=pyo.maximize)
    m.budget_cons = pyo.Constraint(expr=sum(m.asking_price[i] * m.x[i] for i in m.I) <= m.budget_param)
    m.risk_cons = pyo.Constraint(
        expr=sum(((1.0 - m.theta_param) * m.asking_price[i] - m.lower_bound[i]) * m.x[i] for i in m.I) <= 0
    )

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

    raise RuntimeError("Pyomo optimization did not converge optimally.")


def recommend_portfolio(budget=1500000.0, theta=0.10, fractional_mode=False):
    print("=" * 60)
    print("STARTING PORTFOLIO OPTIMIZATION")
    print("=" * 60)

    df = load_prediction_data()
    df = generate_simulated_market_prices(df)

    try:
        res_df = solve_pyomo(df.copy(), budget, theta, fractional_mode)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Optimization exception: {e}")
        res_df = solve_greedy(df.copy(), budget, theta)

    output_cols = [
        "Id",
        "SalePrice_pred",
        "Asking_Price",
        "SalePrice_Lower",
        "Expected_Profit",
        "Conformal_Downside",
        "Selected_Fraction",
    ]

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
