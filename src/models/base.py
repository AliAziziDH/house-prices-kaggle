import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.metrics import rmsle
from src.preprocess import AmesDataTransformer


def fit_neighborhood_rank(X_train, y_train):
    if "GrLivArea" not in X_train.columns:
        medians = y_train.groupby(X_train["Neighborhood"]).median()
    else:
        price_per_sqft = y_train / X_train["GrLivArea"]
        medians = price_per_sqft.groupby(X_train["Neighborhood"]).median()

    sorted_neighborhoods = medians.sort_values().index
    rank_mapping = {neigh: i + 1 for i, neigh in enumerate(sorted_neighborhoods)}

    if len(rank_mapping) > 1:
        min_rank = 1
        max_rank = len(rank_mapping)
        for k, v in rank_mapping.items():
            scaled = 1 + ((v - min_rank) / (max_rank - min_rank)) * 24
            rank_mapping[k] = int(round(scaled))

    return rank_mapping


def transform_neighborhood_rank(X, rank_mapping):
    if "Neighborhood" in X.columns:
        return X["Neighborhood"].map(rank_mapping).fillna(13).astype(int)
    return pd.Series(13, index=X.index)


RANDOM_STATE = 42
N_FOLDS = 5


@dataclass
class CVConfig:
    n_folds: int = 5
    seed: int = 42
    use_raw: bool = False
    params: Optional[Dict[str, Any]] = None
    model_type: Optional[str] = None


def run_cv_experiment(
    model_factory,
    X_raw,
    y_raw,
    config: Optional[CVConfig] = None,
    cat_features=None,
):
    config = config or CVConfig()
    n_folds = config.n_folds
    seed = config.seed
    use_raw_features = config.use_raw
    """
    Leak-free cross-validation runner.
    Fits AmesDataTransformer strictly on training folds and evaluates on validation folds.
    Applies log1p target transformation and expm1 inversion.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y_log = np.log1p(y_raw)

    oof_preds = np.zeros(len(X_raw))
    fold_scores = []
    models = []
    transformers = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_raw)):
        X_tr_raw, y_tr_raw = X_raw.iloc[train_idx], y_raw.iloc[train_idx]
        X_val_raw, y_val_raw = X_raw.iloc[val_idx], y_raw.iloc[val_idx]

        y_tr_log = y_log.iloc[train_idx]
        if not use_raw_features:
            if "Neighborhood" in X_tr_raw.columns:
                rank_mapping = fit_neighborhood_rank(X_tr_raw, y_tr_raw)
                X_tr_raw = X_tr_raw.copy()
                X_val_raw = X_val_raw.copy()
                X_tr_raw["Neighborhood_Rank"] = transform_neighborhood_rank(X_tr_raw, rank_mapping)
                X_val_raw["Neighborhood_Rank"] = transform_neighborhood_rank(X_val_raw, rank_mapping)
                X_tr_raw = X_tr_raw.drop(columns=["Neighborhood"])
                X_val_raw = X_val_raw.drop(columns=["Neighborhood"])

            transformer = AmesDataTransformer()
            transformer.fit(X_tr_raw, y_tr_raw)
            X_tr = transformer.transform(X_tr_raw)
            X_val = transformer.transform(X_val_raw)
            transformers.append(transformer)
        else:
            X_tr = X_tr_raw.copy()
            X_val = X_val_raw.copy()

            if "Neighborhood" in X_tr_raw.columns:
                rank_mapping = fit_neighborhood_rank(X_tr_raw, y_tr_raw)
                X_tr["Neighborhood_Rank"] = transform_neighborhood_rank(X_tr_raw, rank_mapping)
                X_val["Neighborhood_Rank"] = transform_neighborhood_rank(X_val_raw, rank_mapping)

            if cat_features:
                for col in cat_features:
                    if col in X_tr.columns:
                        X_tr[col] = X_tr[col].fillna("Missing").astype(str)
                        X_val[col] = X_val[col].fillna("Missing").astype(str)
        model = model_factory(fold)
        if use_raw_features and cat_features:
            model.fit(X_tr, y_tr_log, cat_features=cat_features, verbose=False)
        else:
            model.fit(X_tr, y_tr_log)

        val_preds_log = model.predict(X_val)
        val_preds_orig = np.expm1(val_preds_log)
        val_preds_orig = np.maximum(val_preds_orig, 0)

        oof_preds[val_idx] = val_preds_orig
        fold_score = rmsle(y_val_raw.values, val_preds_orig)
        fold_scores.append(fold_score)
        models.append(model)

    overall_rmsle = rmsle(y_raw.values, oof_preds)
    return {
        "oof_preds": oof_preds,
        "fold_scores": fold_scores,
        "overall_rmsle": overall_rmsle,
        "models": models,
        "transformers": transformers,
    }


def save_oof_predictions(model_name, oof_preds, train_ids):
    """Save aligned OOF predictions to disk."""
    os.makedirs("./processed_data", exist_ok=True)
    df_oof = pd.DataFrame({"Id": train_ids, "OOF_SalePrice": oof_preds})
    out_path = f"./processed_data/oof_{model_name}.csv"
    df_oof.to_csv(out_path, index=False)
    print(f"✅ Saved OOF predictions to '{out_path}'")
