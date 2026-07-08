"""Feature pipeline: V1..V28 passthrough, Amount log1p+robust-scale, Time -> cycle_phase.

The preprocessor is an sklearn ColumnTransformer serialized inside the model artifact,
so serve-time transforms are identical to train-time by construction.

Naming honesty: `Time` is seconds since the *dataset's first transaction*, not clock time.
`cycle_phase` is therefore phase within a 24-hour cycle relative to dataset start — it can
capture daily periodicity but is NOT time-of-day. Raw `Time` is dropped: it is a sequence
index, and using it raw would leak the chronological split.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler

SECONDS_PER_DAY = 86_400

V_COLUMNS = [f"V{i}" for i in range(1, 29)]
RAW_FEATURE_COLUMNS = ["Time", *V_COLUMNS, "Amount"]
TARGET_COLUMN = "Class"


def encode_cycle_phase(time_column: np.ndarray) -> np.ndarray:
    """Map seconds-since-first-transaction to (sin, cos) of phase in a 24-hour cycle.

    phase = (Time mod 86400) / 86400, relative to dataset start — NOT time-of-day.
    The sin/cos pair makes the encoding continuous across the day boundary
    (Time = 86400 encodes identically to Time = 0).
    """
    time_s = np.asarray(time_column, dtype=float).reshape(-1)
    phase = (time_s % SECONDS_PER_DAY) / SECONDS_PER_DAY
    angle = 2.0 * np.pi * phase
    return np.column_stack([np.sin(angle), np.cos(angle)])


def _cycle_phase_names(transformer: object, input_features: object) -> np.ndarray:
    return np.asarray(["phase_sin", "phase_cos"], dtype=object)


def build_preprocessor() -> ColumnTransformer:
    """Preprocessor over raw dataset columns.

    Output columns, in order: V1..V28 (passthrough — already PCA components),
    Amount (log1p then RobustScaler for the heavy right tail), phase_sin, phase_cos.
    Everything else (raw Time, Class if present) is dropped.
    """
    amount = Pipeline(
        [
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", RobustScaler()),
        ]
    )
    cycle = FunctionTransformer(encode_cycle_phase, feature_names_out=_cycle_phase_names)
    return ColumnTransformer(
        [
            ("v", "passthrough", V_COLUMNS),
            ("amount", amount, ["Amount"]),
            ("cycle_phase", cycle, ["Time"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
