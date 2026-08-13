import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.portfolio_solver import solve_pyomo
from src.portfolio_utils import solve_greedy

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(page_title="Portfolio Optimization Dashboard", layout="wide")
st.title("🏡 Ames Housing: Prescriptive Decision Intelligence")


# ============================================
# DATA LOADING
# ============================================
@st.cache_data
def load_data():
    mode = "Production Mode"
    try:
        df = pd.read_csv("./submissions/submission_with_intervals.csv")
    except Exception:  # noqa: BLE001
        try:
            df = pd.read_csv("./submission.csv")
        except Exception:  # noqa: BLE001
            # Generate synthetic sample predictions for headless/cloud environment
            mode = "Simulation Mode"
            rng = np.random.default_rng(42)
            n_samples = 1459
            preds = rng.normal(180000, 50000, size=n_samples).clip(50000, 750000)
            df = pd.DataFrame(
                {
                    "Id": np.arange(1461, 1461 + n_samples),
                    "SalePrice": preds,
                    "SalePrice_Lower": preds * 0.9,
                    "SalePrice_Upper": preds * 1.1,
                    "Neighborhood": rng.choice(
                        ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst"],
                        size=n_samples,
                    ),
                }
            )

    if "SalePrice" in df.columns and "SalePrice_pred" not in df.columns:
        df = df.rename(columns={"SalePrice": "SalePrice_pred"})

    # Generate deterministic simulated Asking Prices matching pipeline
    rng = np.random.default_rng(42)
    noise = rng.uniform(-4000, 4000, size=len(df))
    df["Asking_Price"] = 0.92 * df["SalePrice_pred"] + noise
    return df, mode


df, app_mode = load_data()
st.sidebar.info(f"Mode: {app_mode}")

# ============================================
# SIDEBAR CONTROLS
# ============================================
st.sidebar.header("Optimization Constraints")

budget = st.sidebar.slider(
    "Investment Budget ($)",
    min_value=500_000,
    max_value=5_000_000,
    value=1_500_000,
    step=100_000,
)

theta = st.sidebar.slider("Max Downside Conformal Risk (%)", min_value=1, max_value=30, value=10, step=1) / 100.0

opt_mode = st.sidebar.radio(
    "Optimization Mode",
    ["Fractional Portfolio (Pure LP)", "Physical Acquisition (Binary MILP)"],
)
fractional_mode = "Fractional" in opt_mode

# ============================================
# PORTFOLIO OPTIMIZATION ENGINE
# ============================================
with st.spinner("Optimizing Portfolio..."):
    # Run the imported optimization logic
    try:
        res_df = solve_pyomo(df.copy(), budget, theta, fractional_mode)
        solver_used = "Pyomo (glpk/cbc)"
    except Exception as e:  # noqa: BLE001
        st.sidebar.warning(f"Pyomo failed/unavailable. Using Vectorized Greedy Heuristic. Error: {e}")
        res_df = solve_greedy(df.copy(), budget, theta)
        solver_used = "Vectorized Greedy Heuristic"

st.sidebar.markdown(f"**Solver Used:** {solver_used}")

# Process selected properties
selected_df = res_df[res_df["Selected_Fraction"] > 0.001].copy()

total_spend = (selected_df["Asking_Price"] * selected_df["Selected_Fraction"]).sum()
total_profit = (selected_df["Expected_Profit"] * selected_df["Selected_Fraction"]).sum()
total_downside = (selected_df["Conformal_Downside"] * selected_df["Selected_Fraction"]).sum()

# ============================================
# METRICS DASHBOARD
# ============================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Properties Selected", f"{len(selected_df)}")
col2.metric(
    "Total Invested",
    f"${total_spend:,.0f}",
    f"{total_spend / budget * 100:.1f}% of Budget",
)
col3.metric("Expected Profit", f"${total_profit:,.0f}")
col4.metric(
    "Downside Risk",
    f"${total_downside:,.0f}",
    f"-{(total_downside / total_spend * 100) if total_spend > 0 else 0:.1f}%",
)

# ============================================
# VISUALIZATION TABS
# ============================================
tab1, tab2, tab3 = st.tabs(["Conformal Pricing Envelopes", "Risk-Return Frontier", "Selected Properties"])

with tab1:
    st.subheader("Asset Valuation with 95% Conformal Limits")

    # Sort for visual envelope mapping
    viz_df = res_df.sort_values("SalePrice_pred").reset_index(drop=True)

    fig1 = go.Figure()

    # Upper Bound
    fig1.add_trace(
        go.Scatter(
            x=viz_df.index,
            y=viz_df["SalePrice_Upper"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            name="Upper Bound",
        )
    )

    # Lower Bound
    fig1.add_trace(
        go.Scatter(
            x=viz_df.index,
            y=viz_df["SalePrice_Lower"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(173, 216, 230, 0.3)",
            name="95% Conformal Envelope",
        )
    )

    # All Predictions
    fig1.add_trace(
        go.Scatter(
            x=viz_df.index,
            y=viz_df["SalePrice_pred"],
            mode="markers",
            marker={"size": 4, "color": "gray", "opacity": 0.5},
            name="Market Properties",
        )
    )

    # Highlight Selected
    selected_mask = viz_df["Selected_Fraction"] > 0.001
    fig1.add_trace(
        go.Scatter(
            x=viz_df[selected_mask].index,
            y=viz_df[selected_mask]["SalePrice_pred"],
            mode="markers",
            marker={"size": 8, "color": "red", "symbol": "diamond"},
            name="Selected Portfolio",
        )
    )

    fig1.update_layout(xaxis_title="Properties (Sorted by Predicted Price)", yaxis_title="Price ($USD)")
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.subheader("Risk vs. Reward Positioning")

    fig2 = px.scatter(
        res_df,
        x="Conformal_Downside",
        y="Expected_Profit",
        color=res_df["Selected_Fraction"] > 0.001,
        color_discrete_map={True: "red", False: "gray"},
        labels={"color": "Selected in Portfolio"},
        hover_data=["Id", "Asking_Price", "Selected_Fraction"],
        opacity=0.6,
    )

    fig2.update_layout(xaxis_title="Conformal Downside Risk ($)", yaxis_title="Expected Profit ($)")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Portfolio Breakdown")
    display_cols = [
        "Id",
        "Selected_Fraction",
        "Asking_Price",
        "SalePrice_pred",
        "Expected_Profit",
        "Conformal_Downside",
    ]
    st.dataframe(
        selected_df[display_cols]
        .sort_values("Expected_Profit", ascending=False)
        .style.format(
            {
                "Selected_Fraction": "{:.2%}",
                "Asking_Price": "${:,.0f}",
                "SalePrice_pred": "${:,.0f}",
                "Expected_Profit": "${:,.0f}",
                "Conformal_Downside": "${:,.0f}",
            }
        ),
        use_container_width=True,
    )
