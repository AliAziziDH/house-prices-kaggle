from xgboost import XGBRegressor

from src.models.base import RANDOM_STATE, run_cv_experiment

DEFAULT_XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.0256,
    "subsample": 0.70,
    "colsample_bytree": 0.67,
    "min_child_weight": 2,
    "random_state": RANDOM_STATE,
    "verbosity": 0,
}


def train_xgboost_cv(X_raw, y_raw, params=None, n_folds=5, seed=RANDOM_STATE):
    """Train XGBoost using leak-free CV."""
    model_params = params or DEFAULT_XGB_PARAMS

    def model_factory(fold_idx):
        return XGBRegressor(**model_params)

    return run_cv_experiment(
        model_factory=model_factory,
        X_raw=X_raw,
        y_raw=y_raw,
        n_folds=n_folds,
        seed=seed,
        use_raw_features=False,
    )
