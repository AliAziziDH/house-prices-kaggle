from typing import Optional

from xgboost import XGBRegressor

from src.models.base import RANDOM_STATE, CVConfig, run_cv_experiment

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


def train_xgboost_cv(X_raw, y_raw, config: Optional[CVConfig] = None):
    """Train XGBoost using leak-free CV."""
    config = config or CVConfig()
    model_params = config.params or DEFAULT_XGB_PARAMS

    def model_factory(fold_idx):
        # XGBoost handles randomness internally via random_state in params, but let's override with fold seed if needed
        p = model_params.copy()
        if "random_state" in p:
            p["random_state"] = config.seed + fold_idx
        return XGBRegressor(**p)

    return run_cv_experiment(
        model_factory=model_factory,
        X_raw=X_raw,
        y_raw=y_raw,
        n_folds=config.n_folds,
        seed=config.seed,
        use_raw_features=config.use_raw,
    )
