# gem_ith_conformal_stacking.py
"""
================================================================================
          DECISION INTELLIGENCE ENSEMBLE: GEM-ITH & SEMI-CONFORMAL PSEUDO-LABELING
================================================================================
Architect: Decision Intelligence Architect (for Ali Azizi)
Target: Ames Housing Price Prediction (Kaggle)
Metric: Root Mean Squared Logarithmic Error (RMSLE)

This script implements:
1. Leakage-Free Fold-Local Target Encoding for high-cardinality categoricals.
2. Advanced Target Variable Transformation (Quantile Transformer / Log1p).
3. GEM-ITH (Generalized Weighted Ensemble with Internally Tuned Hyperparameters):
   Bi-level joint optimization of tree hyperparameters and SLSQP stacking weights.
4. Conformal Pseudo-Labeling (SemiCP): Using Inductive Conformal Prediction (ICP)
   uncertainty intervals to safely pseudo-label high-confidence test instances.
================================================================================
"""

import os

import catboost as cb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler

# Ensure outputs folder exists
os.makedirs("submissions", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ------------------------------------------------------------------------------
# 1. LEAKAGE-FREE NEIGHBORHOOD TARGET ENCODING
# ------------------------------------------------------------------------------
class FoldLocalTargetEncoder:
    """
    Calculates target statistics strictly on the active training folds
    to prevent target leakage, with fallback for unseen categories.
    """

    def __init__(self, category_col="Neighborhood", target_col="SalePrice", noise_level=0.01):
        self.category_col = category_col
        self.target_col = target_col
        self.noise_level = noise_level
        self.mapping_ = {}
        self.global_median_ = 0.0

    def fit(self, X, y):
        df_temp = pd.DataFrame({"cat": X[self.category_col], "target": y})
        group = df_temp.groupby("cat")["target"].median()
        self.mapping_ = group.to_dict()
        self.global_median_ = np.median(y)

        sorted_cats = sorted(self.mapping_.items(), key=lambda x: x[1])
        self.ranks_ = {cat: i + 1 for i, (cat, _) in enumerate(sorted_cats)}
        return self

    def transform(self, X, random_state=42):
        X_out = X.copy()
        mapped_ranks = X_out[self.category_col].map(self.ranks_).fillna(13.0).values.astype(np.float64)

        if self.noise_level > 0 and random_state is not None:
            np.random.seed(random_state)
            noise = np.random.normal(0, self.noise_level, size=len(mapped_ranks))
            mapped_ranks += noise

        X_out[self.category_col + "_Rank"] = mapped_ranks
        return X_out


# ------------------------------------------------------------------------------
# 2. DATA PREPROCESSING PIPELINE
# ------------------------------------------------------------------------------
def load_and_preprocess(train_path="data/train.csv", test_path="data/test.csv"):
    """
    Loads raw Ames housing dataset, imputes missing values cleanly,
    and handles spatial feature alignment without pruning the tree models.
    """
    train_df = pd.read_csv(train_path, index_col="Id")
    test_df = pd.read_csv(test_path, index_col="Id")

    train_df.columns = [c.replace(" ", "") for c in train_df.columns]
    test_df.columns = [c.replace(" ", "") for c in test_df.columns]

    y = train_df["SalePrice"].values
    X_train_raw = train_df.drop(columns=["SalePrice"])
    X_test_raw = test_df.copy()

    for col in X_train_raw.columns:
        if X_train_raw[col].dtype in [np.float64, np.int64]:
            median_val = X_train_raw[col].median()
            X_train_raw[col] = X_train_raw[col].fillna(median_val)
            X_test_raw[col] = X_test_raw[col].fillna(median_val)
        else:
            X_train_raw[col] = X_train_raw[col].fillna("None").astype(str)
            X_test_raw[col] = X_test_raw[col].fillna("None").astype(str)

    return X_train_raw, y, X_test_raw


# ------------------------------------------------------------------------------
# 3. SLSQP CONVEX COVARIANCE-REGULARIZED SOLVER
# ------------------------------------------------------------------------------
def solve_slsqp_weights(OOF_preds, y_true, lmbda=0.1):
    """
    Solves a Sequential Least Squares Programming (SLSQP) quadratic program
    to calculate optimal stacking weights that sum to 1.0 and are non-negative,
    incorporating a covariance penalty to punish correlated model errors.
    """
    num_models = OOF_preds.shape[1]
    cov_matrix = np.cov(OOF_preds.T)

    def objective(w):
        ensemble_pred = np.dot(OOF_preds, w)
        sse = np.sum((y_true - ensemble_pred) ** 2)
        penalty = lmbda * np.dot(w, np.dot(cov_matrix, w))
        return sse + penalty

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0) for _ in range(num_models)]

    res = minimize(
        fun=objective, x0=np.ones(num_models) / num_models, method="SLSQP", bounds=bounds, constraints=constraints
    )
    return res.x


# ------------------------------------------------------------------------------
# 4. BI-LEVEL GEM-ITH OPTIMIZATION
# ------------------------------------------------------------------------------
def run_gem_ith_optimization(X_train, y, n_splits=5, n_trials=30):
    """
    Generalized Ensemble Method with Internally Tuned Hyperparameters (GEM-ITH).
    Uses a stochastic search over the continuous hyperparameter space of GBDT models,
    jointly solving for the optimal SLSQP stacking weights inside the loop.
    """
    print(f"[*] Starting GEM-ITH Optimization (CV Splits: {n_splits}, Search Trials: {n_trials})...")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    numeric_cols = [col for col in X_train.columns if X_train[col].dtype in [np.float64, np.int64]]

    best_rmsle = float("inf")
    best_params = {}
    best_weights = None

    for trial in range(1, n_trials + 1):
        xgb_lr = np.random.uniform(0.01, 0.1)
        xgb_subsample = np.random.uniform(0.6, 0.9)
        xgb_colsample = np.random.uniform(0.5, 0.8)
        xgb_reg_lambda = np.random.uniform(1.0, 10.0)

        cb_lr = np.random.uniform(0.01, 0.1)
        cb_l2 = np.random.uniform(1.0, 10.0)

        # Keep tree depth strictly clamped to [2, 3] for high regularization
        xgb_depth = np.random.choice([2, 3])
        cb_depth = np.random.choice([2, 3])

        oof_preds = np.zeros((len(X_train), 3))

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, _y_val = y[train_idx], y[val_idx]

            encoder = FoldLocalTargetEncoder()
            encoder.fit(X_tr, y_tr)

            X_tr_enc = encoder.transform(X_tr)
            X_val_enc = encoder.transform(X_val)

            all_num_cols = numeric_cols + ["Neighborhood_Rank"]
            X_tr_num = X_tr_enc[all_num_cols]
            X_val_num = X_val_enc[all_num_cols]

            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr_num)
            X_val_scaled = scaler.transform(X_val_num)

            # --- Model 1: XGBoost (Log-Space Target) ---
            model_xgb = xgb.XGBRegressor(
                learning_rate=xgb_lr,
                max_depth=xgb_depth,
                subsample=xgb_subsample,
                colsample_bytree=xgb_colsample,
                reg_lambda=xgb_reg_lambda,
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            )
            transformed_xgb = TransformedTargetRegressor(regressor=model_xgb, func=np.log1p, inverse_func=np.expm1)
            transformed_xgb.fit(X_tr_scaled, y_tr)
            oof_preds[val_idx, 0] = transformed_xgb.predict(X_val_scaled)

            # --- Model 2: CatBoost (Log-Space Target) ---
            model_cb = cb.CatBoostRegressor(
                learning_rate=cb_lr, depth=cb_depth, l2_leaf_reg=cb_l2, iterations=100, verbose=False, random_state=42
            )
            transformed_cb = TransformedTargetRegressor(regressor=model_cb, func=np.log1p, inverse_func=np.expm1)
            transformed_cb.fit(X_tr_scaled, y_tr)
            oof_preds[val_idx, 1] = transformed_cb.predict(X_val_scaled)

            # --- Model 3: Ridge (Normal Quantile Target Transform) ---
            target_transformer = QuantileTransformer(
                n_quantiles=min(len(X_tr_scaled) - 1, 50), output_distribution="normal", random_state=42
            )
            transformed_ridge = TransformedTargetRegressor(regressor=Ridge(alpha=10.0), transformer=target_transformer)
            transformed_ridge.fit(X_tr_scaled, y_tr)
            oof_preds[val_idx, 2] = transformed_ridge.predict(X_val_scaled)

        weights = solve_slsqp_weights(oof_preds, y, lmbda=0.1)
        ensemble_oof_pred = np.dot(oof_preds, weights)
        rmsle = np.sqrt(np.mean((np.log1p(ensemble_oof_pred) - np.log1p(y)) ** 2))

        if rmsle < best_rmsle:
            best_rmsle = rmsle
            best_weights = weights
            best_params = {
                "xgb_lr": xgb_lr,
                "xgb_depth": xgb_depth,
                "xgb_subsample": xgb_subsample,
                "xgb_colsample": xgb_colsample,
                "xgb_reg_lambda": xgb_reg_lambda,
                "cb_lr": cb_lr,
                "cb_depth": cb_depth,
                "cb_l2": cb_l2,
            }
            print(f" [Trial {trial:02d}/{n_trials}] Found Better Params! RMSLE: {best_rmsle:.5f} | Weights: {best_weights}")

    print(f"\n[+] GEM-ITH Optimization Complete! Best OOF RMSLE: {best_rmsle:.5f}")
    return best_params, best_weights


# ------------------------------------------------------------------------------
# 5. CALIBRATION & CONFORMAL PSEUDO-LABELING (SemiCP)
# ------------------------------------------------------------------------------
def train_and_conformal_pseudolabel(X_train, y, X_test, best_params, best_weights, alpha=0.05, pseudo_ratio=0.10):
    """
    Executes Conformal Pseudo-Labeling (SemiCP) under strict coverage bounds.
    """
    print(f"\n[*] Launching Conformal Pseudo-Labeling (SemiCP) at alpha={alpha}...")

    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(X_train))
    split_point = int(len(X_train) * 0.90)

    train_idx = shuffled_indices[:split_point]
    cal_idx = shuffled_indices[split_point:]

    X_prop, y_prop = X_train.iloc[train_idx], y[train_idx]
    X_cal, y_cal = X_train.iloc[cal_idx], y[cal_idx]

    encoder = FoldLocalTargetEncoder()
    encoder.fit(X_prop, y_prop)

    X_prop_enc = encoder.transform(X_prop)
    X_cal_enc = encoder.transform(X_cal)
    X_test_enc = encoder.transform(X_test)

    numeric_cols = [col for col in X_train.columns if X_train[col].dtype in [np.float64, np.int64]]
    all_num_cols = numeric_cols + ["Neighborhood_Rank"]

    scaler = StandardScaler()
    X_prop_scaled = scaler.fit_transform(X_prop_enc[all_num_cols])
    X_cal_scaled = scaler.transform(X_cal_enc[all_num_cols])
    X_test_scaled = scaler.transform(X_test_enc[all_num_cols])

    # --- XGBoost ---
    model_xgb = xgb.XGBRegressor(
        learning_rate=best_params["xgb_lr"],
        max_depth=best_params["xgb_depth"],
        subsample=best_params["xgb_subsample"],
        colsample_bytree=best_params["xgb_colsample"],
        reg_lambda=best_params["xgb_reg_lambda"],
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    xgb_reg = TransformedTargetRegressor(regressor=model_xgb, func=np.log1p, inverse_func=np.expm1)
    xgb_reg.fit(X_prop_scaled, y_prop)

    # --- CatBoost ---
    model_cb = cb.CatBoostRegressor(
        learning_rate=best_params["cb_lr"],
        depth=best_params["cb_depth"],
        l2_leaf_reg=best_params["cb_l2"],
        iterations=100,
        verbose=False,
        random_state=42,
    )
    cb_reg = TransformedTargetRegressor(regressor=model_cb, func=np.log1p, inverse_func=np.expm1)
    cb_reg.fit(X_prop_scaled, y_prop)

    # --- Ridge ---
    target_transformer = QuantileTransformer(
        n_quantiles=min(len(X_prop_scaled) - 1, 50), output_distribution="normal", random_state=42
    )
    ridge_reg = TransformedTargetRegressor(regressor=Ridge(alpha=10.0), transformer=target_transformer)
    ridge_reg.fit(X_prop_scaled, y_prop)

    cal_preds = np.column_stack(
        [xgb_reg.predict(X_cal_scaled), cb_reg.predict(X_cal_scaled), ridge_reg.predict(X_cal_scaled)]
    )
    cal_ensemble = np.dot(cal_preds, best_weights)

    nonconformity_scores = np.abs(np.log1p(y_cal) - np.log1p(cal_ensemble))
    n_cal = len(y_cal)
    quantile_val = np.clip((1.0 - alpha) * (1.0 + 1.0 / n_cal), 0.0, 1.0)
    q_threshold = np.quantile(nonconformity_scores, quantile_val)
    print(f" [+] Calibrated Conformal Log-Residual Threshold at 95% Confidence: {q_threshold:.5f}")

    test_preds = np.column_stack(
        [xgb_reg.predict(X_test_scaled), cb_reg.predict(X_test_scaled), ridge_reg.predict(X_test_scaled)]
    )
    test_ensemble = np.dot(test_preds, best_weights)

    test_lower_bound = np.clip(np.expm1(np.clip(np.log1p(test_ensemble) - q_threshold, 0, None)), 42000.0, 525000.0)
    test_upper_bound = np.clip(np.expm1(np.log1p(test_ensemble) + q_threshold), 42000.0, 525000.0)
    interval_widths = test_upper_bound - test_lower_bound

    n_pseudo = int(len(X_test) * pseudo_ratio)
    sorted_test_indices = np.argsort(interval_widths)
    pseudo_indices = sorted_test_indices[:n_pseudo]

    print(f" [+] SemiCP Selected {n_pseudo} low-uncertainty test instances for pseudo-labeling.")

    pseudo_X = X_test.iloc[pseudo_indices]
    pseudo_y = test_ensemble[pseudo_indices]

    X_enriched = pd.concat([X_train, pseudo_X], axis=0)
    y_enriched = np.concatenate([y, pseudo_y])

    print(" [*] Re-fitting finalized 3-model stacking blend on the enriched training dataset...")
    final_encoder = FoldLocalTargetEncoder()
    final_encoder.fit(X_enriched, y_enriched)

    X_enriched_enc = final_encoder.transform(X_enriched)
    X_test_final_enc = final_encoder.transform(X_test)

    final_scaler = StandardScaler()
    X_enriched_scaled = final_scaler.fit_transform(X_enriched_enc[all_num_cols])
    X_test_final_scaled = final_scaler.transform(X_test_final_enc[all_num_cols])

    final_xgb = TransformedTargetRegressor(regressor=model_xgb, func=np.log1p, inverse_func=np.expm1)
    final_xgb.fit(X_enriched_scaled, y_enriched)

    final_cb = TransformedTargetRegressor(regressor=model_cb, func=np.log1p, inverse_func=np.expm1)
    final_cb.fit(X_enriched_scaled, y_enriched)

    final_ridge = TransformedTargetRegressor(regressor=Ridge(alpha=10.0), transformer=target_transformer)
    final_ridge.fit(X_enriched_scaled, y_enriched)

    final_test_preds = np.column_stack(
        [
            final_xgb.predict(X_test_final_scaled),
            final_cb.predict(X_test_final_scaled),
            final_ridge.predict(X_test_final_scaled),
        ]
    )
    final_submissions_pred = np.dot(final_test_preds, best_weights)
    final_submissions_pred = np.clip(final_submissions_pred, 42000.0, 525000.0)

    sub_df = pd.DataFrame({"Id": np.arange(1461, 1461 + len(final_submissions_pred)), "SalePrice": final_submissions_pred})
    sub_df.to_csv("submissions/submission.csv", index=False)
    print("[+] Successfully wrote finalized, un-leaked predictions to 'submissions/submission.csv'!")

    intervals_df = pd.DataFrame(
        {
            "Id": sub_df["Id"],
            "SalePrice": final_submissions_pred,
            "LowerBound": test_lower_bound,
            "UpperBound": test_upper_bound,
            "IntervalWidth": interval_widths,
        }
    )
    intervals_df.to_csv("submissions/submission_with_intervals.csv", index=False)
    print("[+] Wrote calibrated conformal bounds to 'submissions/submission_with_intervals.csv'.")


# ------------------------------------------------------------------------------
# MAIN EXECUTION ROUTINE
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    X_train, y, X_test = load_and_preprocess()
    best_params, best_weights = run_gem_ith_optimization(X_train, y, n_splits=5, n_trials=10)
    train_and_conformal_pseudolabel(X_train, y, X_test, best_params, best_weights)
