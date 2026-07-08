"""Unit tests for the chronological split."""

from __future__ import annotations

import pandas as pd

from fraudscore.data import chronological_split


def test_split_fractions_and_no_row_loss(synthetic_df):
    splits = chronological_split(synthetic_df)
    n = len(synthetic_df)
    assert len(splits.train) == int(n * 0.6)
    assert len(splits.calibration) == int(n * 0.8) - int(n * 0.6)
    assert len(splits.train) + len(splits.calibration) + len(splits.test) == n


def test_split_is_chronological(synthetic_df):
    splits = chronological_split(synthetic_df)
    assert splits.train["Time"].max() <= splits.calibration["Time"].min()
    assert splits.calibration["Time"].max() <= splits.test["Time"].min()
    for part in splits:
        assert part["Time"].is_monotonic_increasing


def test_split_preserves_rows(synthetic_df):
    splits = chronological_split(synthetic_df)
    recombined = pd.concat(list(splits), ignore_index=True)
    original = synthetic_df.sort_values("Time", kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(recombined, original)
