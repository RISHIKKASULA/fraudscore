"""Unit tests for calibration (architecture.md §8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudscore.calibrate import (
    IsotonicCalibrator,
    SigmoidCalibrator,
    calibrate,
    select_method,
)
from fraudscore.data import chronological_split
from fraudscore.features import RAW_FEATURE_COLUMNS, TARGET_COLUMN
from fraudscore.train import train_baseline


@pytest.fixture(scope="module")
def calibrated_on_fixture(request):
    df = request.getfixturevalue("synthetic_df")
    splits = chronological_split(df)
    base = train_baseline(splits.train)
    return calibrate(base, splits.calibration), splits


class _StubModel:
    """Deterministic 'fitted model' emitting predefined raw probabilities."""

    def __init__(self, p_raw: np.ndarray):
        self._p = np.asarray(p_raw, dtype=float)

    def predict_proba(self, x) -> np.ndarray:
        p = self._p[: len(x)]
        return np.column_stack([1.0 - p, p])


def _frame_for(p_raw: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(0.0, index=range(len(p_raw)), columns=RAW_FEATURE_COLUMNS)
    frame[TARGET_COLUMN] = y
    return frame


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def test_probabilities_in_unit_interval(calibrated_on_fixture):
    result, splits = calibrated_on_fixture
    for model in result.candidates.values():
        p = model.predict_proba(splits.test)
        assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_sigmoid_preserves_ranking_exactly(calibrated_on_fixture):
    result, splits = calibrated_on_fixture
    sigmoid = result.candidates["sigmoid"]
    p_raw = sigmoid.predict_proba_raw(splits.test)
    p_cal = sigmoid.predict_proba(splits.test)
    # Monotone everywhere (ties only where float64 saturates at exactly 0.0/1.0) ...
    order = np.argsort(p_raw, kind="stable")
    assert np.all(np.diff(p_cal[order]) >= -1e-15)
    # ... and spearman rho = 1 wherever the sigmoid hasn't saturated.
    interior = (p_cal > 1e-9) & (p_cal < 1.0 - 1e-9)
    assert interior.sum() > 20
    assert _spearman(p_raw[interior], p_cal[interior]) == pytest.approx(1.0)


def test_isotonic_is_monotone_non_decreasing(calibrated_on_fixture):
    result, splits = calibrated_on_fixture
    isotonic = result.candidates["isotonic"]
    order = np.argsort(isotonic.predict_proba_raw(splits.test), kind="stable")
    p_cal = isotonic.predict_proba(splits.test)[order]
    assert np.all(np.diff(p_cal) >= -1e-12)


def test_cv_brier_reported_for_both_methods(calibrated_on_fixture):
    result, _ = calibrated_on_fixture
    assert set(result.cv_brier) == {"sigmoid", "isotonic"}
    assert all(0.0 <= v <= 1.0 for v in result.cv_brier.values())
    assert result.method in result.candidates


class TestSelectionRule:
    def test_lower_brier_wins(self):
        assert select_method({"sigmoid": 0.10, "isotonic": 0.20}, {}) == "sigmoid"
        assert select_method({"sigmoid": 0.20, "isotonic": 0.10}, {}) == "isotonic"

    def test_tie_broken_by_low_region_reliability(self):
        briers = {"sigmoid": 0.1, "isotonic": 0.1}
        assert select_method(briers, {"sigmoid": 0.01, "isotonic": 0.05}) == "sigmoid"
        assert select_method(briers, {"sigmoid": 0.05, "isotonic": 0.01}) == "isotonic"

    def test_rigged_step_relation_selects_isotonic(self):
        """True P(y|p_raw) is a step — isotonic fits it, a sigmoid cannot."""
        rng = np.random.default_rng(42)
        p_raw = rng.uniform(0.0, 1.0, size=1500)
        y = rng.binomial(1, np.where(p_raw < 0.5, 0.05, 0.95))
        result = calibrate(_StubModel(p_raw), _frame_for(p_raw, y))
        assert result.method == "isotonic"
        assert result.cv_brier["isotonic"] < result.cv_brier["sigmoid"]


def test_calibrators_fit_transform_roundtrip():
    rng = np.random.default_rng(0)
    p_raw = rng.uniform(0.0, 1.0, size=400)
    y = rng.binomial(1, p_raw)
    for cls in (SigmoidCalibrator, IsotonicCalibrator):
        p_cal = cls().fit(p_raw, y).transform(p_raw)
        assert p_cal.shape == p_raw.shape
        assert np.all((p_cal >= 0.0) & (p_cal <= 1.0))
