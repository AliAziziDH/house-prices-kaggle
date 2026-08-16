from typing import Optional

from catboost import CatBoostRegressor

from src.models.base import RANDOM_STATE, CVConfig, run_cv_experiment

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


def train_catboost_cv(X_raw, y_raw, config: Optional[CVConfig] = None):
    """Train CatBoost using leak-free CV."""
    config = config or CVConfig()
    model_params = config.params or DEFAULT_CAT_PARAMS

    cat_features = None
    if config.use_raw:
        cat_features = X_raw.select_dtypes(include=["object", "string"]).columns.tolist()

    def model_factory(fold_idx):
        p = model_params.copy()
        p["random_seed"] = config.seed + fold_idx
        return CatBoostRegressor(**p)

    return run_cv_experiment(
        model_factory=model_factory,
        X_raw=X_raw,
        y_raw=y_raw,
        n_folds=config.n_folds,
        seed=config.seed,
        use_raw_features=config.use_raw,
        cat_features=cat_features,
    )
