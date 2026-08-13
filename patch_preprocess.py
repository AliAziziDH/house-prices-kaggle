import re

with open('src/preprocess.py', 'r') as f:
    content = f.read()

main_code = """
if __name__ == "__main__":
    print("=" * 60)
    print("LOADING RAW DATA")
    print("=" * 60)

    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")

    # --- OUTLIER REMOVAL (BEFORE SPLIT) ---
    outlier_mask = (train["GrLivArea"] > 4000) & (train["SalePrice"] < 200000)
    train_full = train[~outlier_mask].reset_index(drop=True)

    print(f"Full Train shape after outlier removal: {train_full.shape}")
    print(f"Test shape: {test.shape}")

    # --- 90/10 SPLIT ---
    from sklearn.model_selection import train_test_split
    train_proper, calib_set = train_test_split(train_full, test_size=0.1, random_state=42)

    train_proper = train_proper.reset_index(drop=True)
    calib_set = calib_set.reset_index(drop=True)

    print(f"Proper Train shape (90%): {train_proper.shape}")
    print(f"Calibration shape (10%): {calib_set.shape}")

    y_train_full = train_full["SalePrice"]
    X_train_full_df = train_full.drop(["Id", "SalePrice"], axis=1)

    y_train = train_proper["SalePrice"]
    X_train_df = train_proper.drop(["Id", "SalePrice"], axis=1)

    y_calib = calib_set["SalePrice"]
    X_calib_df = calib_set.drop(["Id", "SalePrice"], axis=1)

    X_test_df = test.drop(["Id"], axis=1)

    # 100% data transformer
    transformer_full = AmesDataTransformer()
    transformer_full.fit(X_train_full_df, y_train_full)
    X_train_full = transformer_full.transform(X_train_full_df)

    # 90% data transformer
    transformer = AmesDataTransformer()
    # Fit ONLY on proper train for 90/10 models and test predictions
    transformer.fit(X_train_df, y_train)

    X_train = transformer.transform(X_train_df)
    X_calib = transformer.transform(X_calib_df)

    # We predict the test set using the 100% data transformer for final point prediction
    # Wait, the 90% models will need test predictions?
    # Actually, 90% models only need to predict on X_calib to find the quantile.
    # The final predictions use 100% models predicting on X_test transformed by transformer_full.
    X_test = transformer_full.transform(X_test_df)

    import os
    import joblib
    os.makedirs("./processed_data", exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    # save full datasets
    X_train_full.to_csv("./processed_data/X_train_full.csv", index=False)
    y_train_full.to_csv("./processed_data/y_train_full.csv", index=False)
    train_full.drop("SalePrice", axis=1).to_csv("./processed_data/X_train_full_raw.csv", index=False)

    # save 90% and 10%
    X_train.to_csv("./processed_data/X_train.csv", index=False)
    y_train.to_csv("./processed_data/y_train.csv", index=False)

    X_calib.to_csv("./processed_data/X_calib.csv", index=False)
    y_calib.to_csv("./processed_data/y_calib.csv", index=False)

    X_test.to_csv("./processed_data/X_test.csv", index=False)

    train_proper.drop("SalePrice", axis=1).to_csv("./processed_data/X_train_raw.csv", index=False)
    calib_set.drop("SalePrice", axis=1).to_csv("./processed_data/X_calib_raw.csv", index=False)
    test.to_csv("./processed_data/X_test_raw.csv", index=False)

    # Instead of picking one transformer to save as boxcox_transformer, we use PowerTransformer in models.
    # Actually we just save X_test transformed by transformer_full, so that's okay.

    print("✅ Processed data saved successfully.")
"""

new_content = re.sub(r'if __name__ == "__main__":.*', main_code.strip(), content, flags=re.DOTALL)
with open('src/preprocess.py', 'w') as f:
    f.write(new_content)
