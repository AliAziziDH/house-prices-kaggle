import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.metrics import rmsle
from src.preprocess import AmesDataTransformer

RANDOM_STATE = 42
N_FOLDS = 5


def run_cv_experiment(
    model_factory,
    X_raw,
    y_raw,
    n_folds=N_FOLDS,
    seed=RANDOM_STATE,
    use_raw_features=False,
    cat_features=None,
):
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
            transformer = AmesDataTransformer()
            transformer.fit(X_tr_raw, y_tr_raw)
            X_tr = transformer.transform(X_tr_raw)
            X_val = transformer.transform(X_val_raw)
            transformers.append(transformer)
        else:
            X_tr = X_tr_raw.copy()
            X_val = X_val_raw.copy()
            if cat_features:
                for col in cat_features:
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
