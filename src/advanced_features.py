import os

import pandas as pd

from src.preprocess import AmesDataTransformer


def generate_advanced_features():
    train_path = "./data/train.csv"
    if not os.path.exists(train_path):
        print("Data file not found.")
        return

    df = pd.read_csv(train_path)
    y = df["SalePrice"] if "SalePrice" in df.columns else None
    X = df.drop(columns=["Id", "SalePrice"], errors="ignore")

    transformer = AmesDataTransformer()
    transformer.fit(X, y)
    features = transformer.transform(X)

    os.makedirs("./processed_data", exist_ok=True)
    features.to_csv("./processed_data/advanced_features.csv", index=False)
    print("✅ Advanced features saved.")


if __name__ == "__main__":
    generate_advanced_features()
