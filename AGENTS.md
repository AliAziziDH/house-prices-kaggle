# AGENTS.md — Google Jules & AI Agents Configuration Reference

This document serves as the repository-level instruction reference (the "USB-C Port") for Google Jules, Claude Code, and other agentic coding assistants. It enforces strict engineering standards, directory layouts, and validation gates for the Ames Housing Decision Intelligence pipeline.

---

## 1. Project Overview & Tech Stack
- **Project Name:** Ames Housing Decision Intelligence Platform
- **Objective:** Bridge predictive machine learning (XGBoost, CatBoost, LightGBM) with prescriptive optimization models (Pyomo + SLSQP solver) to support robust, risk-controlled property acquisition decisions under uncertainty.
- **Key Technologies:**
  - **Predictive ML:** Python 3.12, scikit-learn, XGBoost, CatBoost, LightGBM, Optuna
  - **Prescriptive OR:** Pyomo, SciPy (SLSQP solver), COIN-OR CBC, GLPK
  - **Testing & Quality:** Pytest, Ruff (linter/formatter)

---

## 2. Directory Layout
Agentic code generations must conform strictly to this modular structure:
```text
house-prices-kaggle/
├── data/                            # Raw data (train.csv, test.csv)
├── src/                             # Core Python Source Code
│   ├── preprocess.py                # Ordinal quality mappings & VIF pruning
│   ├── train_linear_models.py       # LassoCV, RidgeCV, ElasticNetCV pipelines
│   ├── train_catboost.py            # Highly regularized CatBoost regressor
│   ├── find_ensemble_weights.py     # SLSQP convex optimization solver
│   ├── ensemble.py                  # Predict pipeline and Conformal Calibration
│   └── ui/app.py       # Streamlit UI & interactive visualization
├── tests/                           # Verification Suite
│   └── test_pipeline.py             # Pytest automated test scripts
├── requirements.txt                 # Project dependencies
└── AGENTS.md                        # This configuration file
```

---

## 3. Strict Coding Conventions (The Rules)

### 3.1 File Size & Complexity Limit (The Rule of 125)
- No single Python file under `/src/` should exceed **125 lines** of code. 
- If a file exceeds this limit, you must refactor the helper functions into separate utility modules to preserve high readability and SHAP-based feature interpretability.

### 3.2 Dependency Discipline
- Do **NOT** install additional third-party Python libraries unless explicitly approved.
- Always use the pre-installed **`uv`** package manager within your VM for installation to optimize start-up speeds.

### 3.3 Target Variable Engineering & Leakage Prevention
- Always log-transform the target price variable using `np.log1p` before fitting to resolve right-skewness and stabilize residual variance.
- Target encoding (e.g., neighborhood price encoding) must be computed **strictly fold-locally** inside the cross-validation loops to prevent target leakage. 
- Low-density categories (< 10 samples) must dynamically fallback to global fold-level quantiles.

### 3.4 Multi-Model Stacking Convex Constraints
- Stacking weights ($w_j$) must represent a true convex combination:
  $$\sum w_j = 1.0, \quad w_j \ge 0$$
- Stacking optimization must minimize the Sum of Squared Errors (SSE) **plus a covariance penalty** of base model prediction errors to solve the Optimizer's Curse:
  $$\lambda \sum_{j \neq k} w_j w_k \operatorname{Cov}(e_j, e_k)$$

---

## 4. Automated Verification & PR Gate

Every time you modify files, you must execute the following validation steps locally in your sandbox:
1. **Linting and Auto-formatting:**
   ```bash
   python3 -m ruff check src/ --fix
   ```
2. **Execute Pytest Suite:**
   ```bash
   PYTHONPATH=. pytest tests/ -v
   ```

Do **NOT** open a Pull Request (PR) or push changes unless all checks and unit tests pass with an exit code of `0`. Open all Pull Requests as **Draft PRs** to allow for peer mathematical verification before merging.
