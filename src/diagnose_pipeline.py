"""
Comprehensive Pipeline & Submission Diagnostic Script
Audits prediction distributions, feature alignment, dataset drift, and model weights.
"""

import os

import numpy as np
import pandas as pd

def print_stats(name, s):
    print(f"--- {name} ---")
    print(f"  Count:  {len(s)}")
    print(f"  Min:    ${s.min():,.2f}")
    print(f"  25%:    ${s.quantile(0.25):,.2f}")
    print(f"  Median: ${s.median():,.2f}")
    print(f"  Mean:   ${s.mean():,.2f}")
    print(f"  75%:    ${s.quantile(0.75):,.2f}")
    print(f"  Max:    ${s.max():,.2f}")
    print(f"  Std:    ${s.std():,.2f}")
    print(f"  Skew:   {s.skew():.4f}")

def main():
    print("=" * 70)
    print("COMPREHENSIVE PIPELINE DIAGNOSTIC AUDIT")
    print("=" * 70)

    # 1. LOAD DATA & SUBMISSION
    sub_path = './submissions/submission_ensemble_final.csv'
    train_path = './processed_data/y_train.csv'
    train_log_path = './processed_data/y_train_log.csv'
    X_train_path = './processed_data/X_train.csv'
    X_test_path = './processed_data/X_test.csv'

    sub = pd.read_csv(sub_path)
    y_train = pd.read_csv(train_path).squeeze()
    y_train_log = pd.read_csv(train_log_path).squeeze()
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)

    # ============================================
    # AUDIT 1: SUBMISSION DISTRIBUTION STATS
    # ============================================
    print("\n" + "=" * 70)
    print("AUDIT 1: SUBMISSION VS TRAIN DISTRIBUTION STATS")
    print("=" * 70)

    sub_prices = sub['SalePrice']

    print_stats("TRAIN SalePrice", y_train)
    print("\n")
    print_stats("SUBMISSION SalePrice", sub_prices)

    # Check nulls or abnormal values
    null_count = sub_prices.isnull().sum()
    inf_count = np.isinf(sub_prices).sum()
    neg_count = (sub_prices <= 0).sum()
    low_count = (sub_prices < 30000).sum()
    high_count = (sub_prices > 600000).sum()

    print("\n--- ANOMALY COUNTS IN SUBMISSION ---")
    print(f"  Null / NaN values:  {null_count}")
    print(f"  Infinite values:    {inf_count}")
    print(f"  <= 0 prices:        {neg_count}")
    print(f"  < $30,000 prices:   {low_count}")
    print(f"  > $600,000 prices:  {high_count}")

    # Log-scale stats
    sub_log = np.log1p(sub_prices)
    print("\n--- LOG-SCALE COMPARISON ---")
    print(f"  Train Log Mean: {y_train_log.mean():.4f} | Sub Log Mean: {sub_log.mean():.4f}")
    print(f"  Train Log Std:  {y_train_log.std():.4f}  | Sub Log Std:  {sub_log.std():.4f}")

    # ============================================
    # AUDIT 2: FEATURE ALIGNMENT & SHIFT CHECK
    # ============================================
    print("\n" + "=" * 70)
    print("AUDIT 2: FEATURE ALIGNMENT & TEST DATA DRIFT")
    print("=" * 70)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")

    train_nans = X_train.isnull().sum().sum()
    test_nans = X_test.isnull().sum().sum()

    print(f"Total NaNs in X_train: {train_nans}")
    print(f"Total NaNs in X_test:  {test_nans}")

    # Check columns with zero variance in test set
    zero_var_test = X_test.columns[X_test.std() == 0].tolist()
    print(f"\nFeatures with 0 variance in X_test ({len(zero_var_test)} features):")
    if len(zero_var_test) > 0:
        print(zero_var_test[:15])

    # Check features in X_test with values outside X_train min/max bounds
    out_of_bounds = []
    for col in X_train.select_dtypes(include=['float64', 'int64']).columns:
        tr_min, tr_max = X_train[col].min(), X_train[col].max()
        te_min, te_max = X_test[col].min(), X_test[col].max()
        if te_min < tr_min or te_max > tr_max:
            out_of_bounds.append((col, tr_min, tr_max, te_min, te_max))

    print(f"\nFeatures in X_test exceeding X_train bounds ({len(out_of_bounds)} features):")
    for col, tr_min, tr_max, te_min, te_max in out_of_bounds[:15]:
        print(f"  {col:20s}: Train [{tr_min:.2f}, {tr_max:.2f}] | Test [{te_min:.2f}, {te_max:.2f}]")

    # ============================================
    # AUDIT 3: LINEAR MODEL COEFFICIENT MAGNITUDES
    # ============================================
    print("\n" + "=" * 70)
    print("AUDIT 3: LINEAR MODEL COEFFICIENT AUDIT")
    print("=" * 70)

    from sklearn.linear_model import LassoCV, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler, StandardScaler

    # Fit simple models to inspect max coefficient weights
    pipe_ridge = make_pipeline(StandardScaler(), Ridge(alpha=15.0)).fit(X_train, y_train_log)
    coefs_ridge = pipe_ridge.named_steps['ridge'].coef_

    pipe_lasso = make_pipeline(RobustScaler(), LassoCV(cv=5, random_state=42)).fit(X_train, y_train_log)
    coefs_lasso = pipe_lasso.named_steps['lassocv'].coef_

    print(f"Ridge Coefs  -> Max: {coefs_ridge.max():.4f}, Min: {coefs_ridge.min():.4f}, Abs Mean: {np.abs(coefs_ridge).mean():.4f}")
    print(f"Lasso Coefs  -> Max: {coefs_lasso.max():.4f}, Min: {coefs_lasso.min():.4f}, Non-zero: {(coefs_lasso != 0).sum()}/{len(coefs_lasso)}")

    # Top 5 most influential features in Lasso
    top_lasso_idx = np.argsort(np.abs(coefs_lasso))[-10:][::-1]
    print("\nTop 10 Most Influential Features in Lasso:")
    for idx in top_lasso_idx:
        print(f"  {X_train.columns[idx]:25s}: {coefs_lasso[idx]:.4f}")

    # ============================================
    # AUDIT 4: MODEL CORRELATION MATRIX (INDIVIDUAL TEST PREDS)
    # ============================================
    print("\n" + "=" * 70)
    print("AUDIT 4: INDIVIDUAL SUBMISSION FILE COMPARISON")
    print("=" * 70)

    sub_files = {
        'CatBoost': './submissions/submission_catboost_log.csv',
        'XGBoost': './submissions/submission_xgboost_log.csv',
        'LightGBM': './submissions/submission_lightgbm_log.csv',
        'Final Ensemble': './submissions/submission_ensemble_final.csv'
    }

    preds_dict = {}
    for name, path in sub_files.items():
        if os.path.exists(path):
            preds_dict[name] = pd.read_csv(path)['SalePrice']

    if len(preds_dict) > 1:
        preds_df = pd.DataFrame(preds_dict)
        print("Correlation between individual model test predictions:")
        print(preds_df.corr().round(5))
        print("\nMean absolute percentage difference with Final Ensemble:")
        for name in preds_dict:
            if name != 'Final Ensemble':
                mape = np.mean(np.abs(preds_dict[name] - preds_dict['Final Ensemble']) / preds_dict['Final Ensemble']) * 100
                print(f"  {name:15s} vs Final Ensemble: {mape:.2f}%")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC AUDIT COMPLETED")
    print("=" * 70)

if __name__ == '__main__':
    main()
