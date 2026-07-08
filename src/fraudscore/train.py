"""Model training: plain logistic-regression baseline (main model lands separately).

No resampling and no class weights anywhere — imbalance is handled at the decision layer
(calibration + expected cost), not by lying to the model about the base rate. The baseline
is calibrated-and-costed, not a strawman: it goes through the identical calibration and
decision layers and stays in the eval report permanently.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fraudscore.features import RAW_FEATURE_COLUMNS, TARGET_COLUMN, build_preprocessor

RANDOM_STATE = 42


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
