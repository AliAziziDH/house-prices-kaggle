from catboost import CatBoostRegressor

from src.models.base import RANDOM_STATE, run_cv_experiment

DEFAULT_CAT_PARAMS = {
    "iterations": 1000,
    "depth": 7,
    "learning_rate": 0.0394,
    "l2_leaf_reg": 2.415,
    "subsample": 0.938,
    "colsample_bylevel": 0.940,
    "random_seed": RANDOM_STATE,
    "verbose": False,
}


def train_catboost_cv(X_raw, y_raw, params=None, n_folds=5, seed=RANDOM_STATE, use_raw=False):
    """Train CatBoost using leak-free CV."""
    model_params = params or DEFAULT_CAT_PARAMS

    cat_features = None
    if use_raw:
        cat_features = X_raw.select_dtypes(include=["object", "string"]).columns.tolist()

    def model_factory(fold_idx):
        p = model_params.copy()
        p["random_seed"] = seed + fold_idx
        return CatBoostRegressor(**p)

    return run_cv_experiment(
        model_factory=model_factory,
        X_raw=X_raw,
        y_raw=y_raw,
        n_folds=n_folds,
        seed=seed,
        use_raw_features=use_raw,
        cat_features=cat_features,
    )
