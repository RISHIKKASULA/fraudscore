"""Model training: plain logistic-regression baseline and HistGradientBoosting main model.

No resampling and no class weights anywhere — imbalance is handled at the decision layer
(calibration + expected cost), not by lying to the model about the base rate. The baseline
is calibrated-and-costed, not a strawman: it goes through the identical calibration and
decision layers and stays in the eval report permanently.

HistGradientBoosting over XGBoost deliberately: one fewer dependency, native sklearn
integration, trains in seconds — the differentiator here is the decision layer, not the
model brand.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from fraudscore.features import RAW_FEATURE_COLUMNS, TARGET_COLUMN, build_preprocessor

RANDOM_STATE = 42

MAIN_PARAM_GRID = {
    "model__max_iter": [200, 400],
    "model__learning_rate": [0.05, 0.1],
    "model__max_leaf_nodes": [31, 63],
}


def train_baseline(train_frame: pd.DataFrame) -> Pipeline:
    """Fit preprocessor + plain LogisticRegression on the train split."""
    model = Pipeline(
        [
            ("features", build_preprocessor()),
            ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(train_frame[RAW_FEATURE_COLUMNS], train_frame[TARGET_COLUMN])
    return model


@dataclass(frozen=True)
class MainModelResult:
    estimator: Pipeline
    best_params: dict
    cv_pr_auc: float


def train_main(train_frame: pd.DataFrame) -> MainModelResult:
    """Fit HistGradientBoosting, selecting hyperparameters by PR-AUC under a 3-fold
    TimeSeriesSplit within the train window only.

    The train split is already time-ordered (chronological_split sorts by `Time`), so
    each CV fold trains on the past and validates on the future — the selection never
    peeks across time, mirroring how the service runs in production.
    """
    pipe = Pipeline(
        [
            ("features", build_preprocessor()),
            ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]
    )
    search = GridSearchCV(
        pipe,
        MAIN_PARAM_GRID,
        scoring="average_precision",  # PR-AUC, the primary metric under imbalance
        cv=TimeSeriesSplit(n_splits=3),
        n_jobs=-1,
        refit=True,
    )
    search.fit(train_frame[RAW_FEATURE_COLUMNS], train_frame[TARGET_COLUMN])
    return MainModelResult(
        estimator=search.best_estimator_,
        best_params={k.removeprefix("model__"): v for k, v in search.best_params_.items()},
        cv_pr_auc=float(search.best_score_),
    )
