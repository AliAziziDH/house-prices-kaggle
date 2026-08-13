### Architectural Consultation Findings & Trade-off Analysis:

1.  **The Scale Conflict & Target Transformations:**
    *   **Current State:** The current ensemble script (`src/ensemble.py`) and stacking weight optimization (`src/find_ensemble_weights.py`) optimize SLSQP stacking weights in the **raw USD space** using Sum of Squared Errors (SSE), but Kaggle evaluates on **RMSLE** (which is effectively RMSE in the log-space). Furthermore, the targets for GBDTs are transformed using `PowerTransformer(method="box-cox")`, while the project guidelines strongly recommend `np.log1p()`.
    *   **Trade-off ($\Delta \text{CV}$):**
        *   Baseline BoxCox + Raw SSE Stacking (Current): **0.11337** RMSLE.
        *   Option 1: BoxCox + Log SSE Stacking: **0.11331** RMSLE ($\Delta$ -0.00006).
        *   Option 2: Log1p Target + Log SSE Stacking: **0.11363** RMSLE. (Slightly worse CV individually for GBDTs, but better aligned with evaluation metric).
        *   **Recommendation:** Switch SLSQP stacking optimization to operate in the **log-space** (i.e. `np.log1p(preds)`) to directly minimize RMSLE, aligning with the Kaggle metric. This also mathematically improves the covariance penalty calculation by preventing high-price outliers from dominating the variance structure.

2.  **Feature Engineering & Multicollinearity:**
    *   **Current State:** The codebase enforces strict VIF pruning and 1-25 Neighborhood ranks. This produces stable, low-variance models (good for linear models) but may restrict the GBDTs from finding complex, non-linear interactions without strong feature engineering.

3.  **Hyperparameter Tuning & Diversity:**
    *   **Current State:** The ensemble is currently composed only of XGBoost and CatBoost.
    *   **Experiment:** I introduced Linear Estimators (Lasso, Ridge, ElasticNet) back into the stacking mix. Because these linear models use robust scaling and capture global linear trends well (whereas GBDTs are better at local step-functions), their predictions have low correlation with the tree models, providing excellent diversity.
    *   **Trade-off ($\Delta \text{CV}$):**
        *   GDBTs only (Log Space Stacking): **0.11331**
        *   GBDTs + ElasticNet/Lasso/Ridge (Log Space Stacking): **0.10928** ($\Delta$ -0.00403)
        *   *Weight Distribution:* ElasticNet absorbed ~48% of the weight, XGBoost ~31%, CatBoost ~21%.
        *   **Recommendation:** Integrate an ElasticNet (or a Ridge/Lasso pipeline) into the stacking ensemble alongside XGBoost and CatBoost. The blending of linear and tree-based models yields a significantly lower local RMSLE.

### Proposed Next Steps (The Action Plan):

1.  *Fix the Linear Models Script and Generate OOF Predictions.*
    *   `src/train_linear_models.py` currently crashes due to feature mismatch (it was using an older `processed_data/y_train_log.csv` and has a bug with `X_te`). We need to rewrite `src/train_linear_models.py` to properly align with `X_train_full` and output the correct OOF predictions.
2.  *Update `src/find_ensemble_weights.py` to optimize weights in Log-Space.*
    *   Modify the SLSQP objective function to compute SSE on `np.log1p(y_true)` and `np.log1p(preds)`.
3.  *Update `src/find_ensemble_weights.py` to include Linear Models in the stacking ensemble.*
    *   Since `src/train_linear_models.py` produces `models/oof_elasticnet.pkl` and `models/oof_lasso.pkl`, load these instead of retraining them in `find_ensemble_weights.py`.
4.  *Update `src/ensemble.py` to load and blend the linear models during final prediction.*
    *   Apply the new stacking logic and log-space objective.
5.  *Update pipeline scripts (`Makefile`, etc.) to ensure the linear models are trained and saved correctly.*
