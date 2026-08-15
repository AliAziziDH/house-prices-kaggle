import numpy as np
import pandas as pd


def load_processed_data():
    """
    Loads full (100%) and 90% processed datasets for training, along with their log1p-transformed targets.

    Returns:
        tuple: (X_train_full, y_train_full, y_full_transformed, X_train_90, y_train_90, y_90_transformed)
    """
    X_train_full = pd.read_csv("./processed_data/X_train_full.csv")
    y_train_full = pd.read_csv("./processed_data/y_train_full.csv").squeeze()
    y_full_transformed = np.log1p(y_train_full.values)

    X_train_90 = pd.read_csv("./processed_data/X_train.csv")
    y_train_90 = pd.read_csv("./processed_data/y_train.csv").squeeze()
    y_90_transformed = np.log1p(y_train_90.values)

    return X_train_full, y_train_full, y_full_transformed, X_train_90, y_train_90, y_90_transformed
