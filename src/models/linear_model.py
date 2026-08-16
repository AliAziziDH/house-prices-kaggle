from typing import Optional

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from src.models.base import CVConfig, run_cv_experiment


def train_linear_cv(X_raw, y_raw, config: Optional[CVConfig] = None):
    """Train regularized linear model using leak-free CV with RobustScaler."""
    config = config or CVConfig()
    model_type = config.model_type or "lasso"

    def model_factory(fold_idx):
        scaler = RobustScaler()
        if model_type == "lasso":
            estimator = LassoCV(max_iter=10000, random_state=config.seed + fold_idx, cv=5)
        elif model_type == "ridge":
            estimator = RidgeCV(cv=5)
        elif model_type == "elasticnet":
            estimator = ElasticNetCV(max_iter=10000, random_state=config.seed + fold_idx, cv=5)
        else:
            raise ValueError(f"Unknown linear model type: {model_type}")

        return make_pipeline(scaler, estimator)

    return run_cv_experiment(
        model_factory=model_factory,
        X_raw=X_raw,
        y_raw=y_raw,
        n_folds=config.n_folds,
        seed=config.seed,
        use_raw_features=config.use_raw,
    )
