"""Unit tests for the bootstrap CIs (architecture.md §8)."""

from __future__ import annotations

import numpy as np

from fraudscore.cost import (
    CIResult,
    bootstrap_ci,
    cost_per_10k_ci,
    savings_pct_ci,
    savings_per_10k_ci,
)


def test_seeded_determinism():
    rng = np.random.default_rng(7)
    rows = rng.lognormal(2.0, 1.0, size=500)
    a = cost_per_10k_ci(rows, b=2000, seed=42)
    b = cost_per_10k_ci(rows, b=2000, seed=42)
    assert a == b
    c = cost_per_10k_ci(rows, b=2000, seed=43)
    assert c != a  # different seed, different resamples


def test_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    rows = rng.lognormal(2.0, 1.0, size=500)
    ci = cost_per_10k_ci(rows, b=2000, seed=42)
    assert ci.low <= ci.point <= ci.high


def test_rigged_two_value_case_hand_checkable():
    """Rows [0, 10]: replicate sums are 0/10/20 with probs 1/4, 1/2, 1/4.

    The 2.5th percentile must be 0 and the 97.5th must be 20 (each tail has
    probability 1/4 >> 2.5%); the point estimate is the actual sum, 10.
    """
    ci = bootstrap_ci([np.array([0.0, 10.0])], lambda s: s, b=10_000, seed=42)
    assert ci.point == 10.0
    assert ci.low == 0.0
    assert ci.high == 20.0


def test_paired_savings_of_identical_rules_is_zero_width():
    rows = np.random.default_rng(1).uniform(0, 100, size=300)
    ci = savings_per_10k_ci(rows, rows.copy(), b=1000, seed=42)
    assert ci.point == 0.0 and ci.low == 0.0 and ci.high == 0.0
    assert ci.includes_zero()


def test_savings_pct_detects_strict_improvement():
    worse = np.full(400, 10.0)
    better = np.full(400, 8.0)  # every row exactly 20% cheaper
    ci = savings_pct_ci(worse, better, b=1000, seed=42)
    assert ci.point == 20.0
    assert ci.low == 20.0 and ci.high == 20.0
    assert not ci.includes_zero()


def test_format_is_point_then_interval():
    assert f"{CIResult(1234.5, 1000.0, 1500.25):,.2f}" == "1,234.50 [1,000.00, 1,500.25]"
    assert f"{CIResult(5.0, -1.0, 11.0):.1f}" == "5.0 [-1.0, 11.0]"


def test_misaligned_rows_raise():
    import pytest

    with pytest.raises(ValueError):
        bootstrap_ci([np.zeros(3), np.zeros(4)], lambda a, b: a - b, b=10, seed=0)
