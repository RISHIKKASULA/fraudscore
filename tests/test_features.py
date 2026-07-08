"""Unit tests for the feature pipeline (architecture.md §8)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudscore.features import (
    RAW_FEATURE_COLUMNS,
    SECONDS_PER_DAY,
    build_preprocessor,
    encode_cycle_phase,
)

from .conftest import FIXTURE_CSV, load_make_fixture_module


class TestCyclePhase:
    def test_time_zero(self):
        out = encode_cycle_phase(np.array([0.0]))
        np.testing.assert_allclose(out, [[0.0, 1.0]], atol=1e-12)

    def test_quarter_cycle(self):
        # Time = 21600 s -> phase 0.25 -> sin 1, cos ~ 0
        out = encode_cycle_phase(np.array([21_600.0]))
        np.testing.assert_allclose(out, [[1.0, 0.0]], atol=1e-12)

    def test_full_cycle_wraps(self):
        # Time = 86400 encodes identically to Time = 0
        np.testing.assert_allclose(
            encode_cycle_phase(np.array([float(SECONDS_PER_DAY)])),
            encode_cycle_phase(np.array([0.0])),
            atol=1e-12,
        )

    def test_output_shape_and_range(self):
        out = encode_cycle_phase(np.linspace(0, 200_000, 50))
        assert out.shape == (50, 2)
        assert np.all(np.abs(out) <= 1.0 + 1e-12)


class TestPreprocessor:
    def test_amount_log1p_robust_scale_known_vector(self):
        # Amounts chosen so log1p is exactly [0, 1, 2, 3, 4]:
        # median 2, IQR 2 -> scaled exactly [-1, -0.5, 0, 0.5, 1].
        amounts = np.expm1([0.0, 1.0, 2.0, 3.0, 4.0])
        frame = pd.DataFrame({c: 0.0 for c in RAW_FEATURE_COLUMNS}, index=range(5))
        frame["Amount"] = amounts
        out = build_preprocessor().fit_transform(frame)
        np.testing.assert_allclose(out[:, 28], [-1.0, -0.5, 0.0, 0.5, 1.0], atol=1e-12)

    def test_column_order_and_names(self):
        frame = pd.DataFrame(
            np.random.default_rng(0).normal(size=(10, len(RAW_FEATURE_COLUMNS))),
            columns=RAW_FEATURE_COLUMNS,
        ).abs()
        pre = build_preprocessor()
        out = pre.fit_transform(frame)
        assert out.shape == (10, 31)  # 28 V + Amount + phase_sin + phase_cos
        names = list(pre.get_feature_names_out())
        assert names[:28] == [f"V{i}" for i in range(1, 29)]
        assert names[28:] == ["Amount", "phase_sin", "phase_cos"]

    def test_transform_determinism(self, synthetic_df):
        x = synthetic_df[RAW_FEATURE_COLUMNS]
        a = build_preprocessor().fit_transform(x)
        b = build_preprocessor().fit_transform(x)
        np.testing.assert_array_equal(a, b)

    def test_no_nan_on_fixture(self, synthetic_df):
        out = build_preprocessor().fit_transform(synthetic_df[RAW_FEATURE_COLUMNS])
        assert np.isfinite(out).all()

    def test_drops_raw_time_and_class(self, synthetic_df):
        pre = build_preprocessor()
        pre.fit(synthetic_df)  # Class present; must be dropped, not used
        names = list(pre.get_feature_names_out())
        assert "Time" not in names
        assert "Class" not in names


class TestCommittedFixture:
    def test_generator_reproduces_committed_csv(self, tmp_path):
        """The committed fixture is exactly what make_fixture.py generates (seed 42)."""
        module = load_make_fixture_module()
        regenerated = tmp_path / "synthetic.csv"
        module.make_fixture().to_csv(regenerated, index=False)
        assert regenerated.read_bytes() == FIXTURE_CSV.read_bytes()
