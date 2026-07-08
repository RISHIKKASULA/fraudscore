"""Data loading and the chronological train/calibration/test split.

Split rationale: leakage avoidance — train on the past, decide on the future, exactly as
the service would run in production. The dataset spans only ~2 days, so no concept-drift
claims; a random split would simply let the model peek at the future side of transaction
sequences it will be judged on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fraudscore.features import RAW_FEATURE_COLUMNS, TARGET_COLUMN

EXPECTED_COLUMNS = [*RAW_FEATURE_COLUMNS, TARGET_COLUMN]

TRAIN_FRAC = 0.6
CALIBRATION_FRAC = 0.2  # remainder (0.2) is test


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame

    def __iter__(self):
        return iter((self.train, self.calibration, self.test))


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV with the ULB schema (Time, V1..V28, Amount, Class) and validate columns."""
    frame = pd.read_csv(path)
    missing = [c for c in EXPECTED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing expected columns {missing}")
    return frame[EXPECTED_COLUMNS]


def chronological_split(
    frame: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    calibration_frac: float = CALIBRATION_FRAC,
) -> Splits:
    """Sort by `Time` and cut train / calibration / test by position (60/20/20 default).

    Stable sort so ties keep their original file order; boundaries are floor(frac * n),
    every row lands in exactly one split.
    """
    ordered = frame.sort_values("Time", kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_frac)
    calibration_end = int(n * (train_frac + calibration_frac))
    return Splits(
        train=ordered.iloc[:train_end].reset_index(drop=True),
        calibration=ordered.iloc[train_end:calibration_end].reset_index(drop=True),
        test=ordered.iloc[calibration_end:].reset_index(drop=True),
    )
