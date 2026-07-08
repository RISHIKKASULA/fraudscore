"""Probability calibration on the calibration split only (prefit pattern).

Both isotonic and sigmoid (Platt) are fitted on the base model's raw probabilities.
~98 fraud cases is thin for isotonic (tail overfit risk); sigmoid is the expected
winner, but it's an empirical question.

Selection rule: lower Brier score under 5-fold CV within the calibration split;
tie -> better reliability fit in the p < 0.1 region. Both curves ship in the eval report.

Calibration matters doubly here: the primary decision rule multiplies p-hat by dollars,
so a 2x miscalibration is a 2x mispricing of risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from fraudscore.features import RAW_FEATURE_COLUMNS

LOW_REGION = 0.1  # tie-breaker region: most transactions live here
N_CV_FOLDS = 5
_EPS = 1e-12


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


class SigmoidCalibrator:
    """Platt scaling: logistic fit on the log-odds of the raw probability.

    Near-unregularized (C=1e10) so it is a maximum-likelihood sigmoid fit, not a
    shrunk one. Strictly monotone, so ranking is exactly preserved.
    """

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> SigmoidCalibrator:
        self._lr = LogisticRegression(C=1e10, max_iter=10_000)
        self._lr.fit(_logit(p_raw).reshape(-1, 1), y)
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(_logit(p_raw).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Isotonic regression on the raw probability: monotone non-decreasing, clipped."""

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
        self._iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self._iso.fit(p_raw, y)
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        return self._iso.predict(p_raw)


CALIBRATOR_FACTORIES = {
    "sigmoid": SigmoidCalibrator,
    "isotonic": IsotonicCalibrator,
}


class CalibratedModel:
    """Base pipeline + fitted calibrator, exposing the predict_proba contract."""

    def __init__(self, base: Pipeline, calibrator, method: str):
        self.base = base
        self.calibrator = calibrator
        self.method = method

    def predict_proba_raw(self, x: pd.DataFrame) -> np.ndarray:
        return self.base.predict_proba(x[RAW_FEATURE_COLUMNS])[:, 1]

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Calibrated P(fraud) for each row (1-D, unlike sklearn's (n, 2) convention)."""
        return self.calibrator.transform(self.predict_proba_raw(x))


@dataclass
class CalibrationResult:
    model: CalibratedModel
    method: str
    cv_brier: dict[str, float]
    low_region_error: dict[str, float]
    # Both calibrators fitted on the full calibration split, for the report's overlay.
    candidates: dict[str, CalibratedModel] = field(default_factory=dict)


def low_region_reliability_error(p: np.ndarray, y: np.ndarray, cutoff: float = LOW_REGION,
                                 n_bins: int = 5) -> float:
    """Count-weighted |mean predicted - observed rate| over equal-width bins below `cutoff`."""
    mask = p < cutoff
    if not mask.any():
        return float("inf")
    p_low, y_low = p[mask], y[mask]
    bins = np.minimum((p_low / (cutoff / n_bins)).astype(int), n_bins - 1)
    total_err = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            total_err += in_bin.sum() * abs(p_low[in_bin].mean() - y_low[in_bin].mean())
    return float(total_err / mask.sum())


def select_method(cv_brier: dict[str, float], low_region_error: dict[str, float],
                  tie_atol: float = 1e-9) -> str:
    """Lower CV Brier wins; a tie (within tie_atol) goes to the better p<0.1 reliability."""
    sig, iso = cv_brier["sigmoid"], cv_brier["isotonic"]
    if abs(sig - iso) <= tie_atol:
        return min(low_region_error, key=low_region_error.get)
    return "sigmoid" if sig < iso else "isotonic"


def _cross_validated(p_raw: np.ndarray, y: np.ndarray, method: str) -> tuple[float, np.ndarray]:
    """Mean held-out Brier and out-of-fold predictions for one calibration method."""
    oof = np.empty_like(p_raw)
    briers = []
    folds = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=False)
    for fit_idx, val_idx in folds.split(p_raw.reshape(-1, 1), y):
        cal = CALIBRATOR_FACTORIES[method]().fit(p_raw[fit_idx], y[fit_idx])
        oof[val_idx] = cal.transform(p_raw[val_idx])
        briers.append(brier_score_loss(y[val_idx], oof[val_idx]))
    return float(np.mean(briers)), oof


def calibrate(base: Pipeline, calibration_frame: pd.DataFrame,
              target_column: str = "Class") -> CalibrationResult:
    """Fit both calibrators on the calibration split and select per the frozen rule.

    The base model is already fitted (prefit pattern) and is never refitted here;
    only the calibration mapping is learned, and only from the calibration split.
    """
    p_raw = base.predict_proba(calibration_frame[RAW_FEATURE_COLUMNS])[:, 1]
    y = calibration_frame[target_column].to_numpy()

    cv_brier: dict[str, float] = {}
    low_err: dict[str, float] = {}
    for method in CALIBRATOR_FACTORIES:
        cv_brier[method], oof = _cross_validated(p_raw, y, method)
        low_err[method] = low_region_reliability_error(oof, y)

    chosen = select_method(cv_brier, low_err)
    candidates = {
        method: CalibratedModel(base, factory().fit(p_raw, y), method)
        for method, factory in CALIBRATOR_FACTORIES.items()
    }
    return CalibrationResult(
        model=candidates[chosen],
        method=chosen,
        cv_brier=cv_brier,
        low_region_error=low_err,
        candidates=candidates,
    )
