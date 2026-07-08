"""Unit tests for the cost model — hand-computed literals throughout (architecture.md §8)."""

from __future__ import annotations

import numpy as np
import pytest

from fraudscore.cost import (
    approve_all_cost,
    cost_curve,
    expected_cost_decisions,
    fit_threshold,
    load_cost_params,
    per_10k,
    realized_cost,
    threshold_decisions,
)

C_REVIEW = 10.0


class TestAmountAwareRule:
    """5-transaction toy case; expected decisions as literals."""

    P = np.array([0.9, 0.02, 0.001, 0.5, 0.3])
    A = np.array([5.0, 800.0, 100.0, 20.0, 30.0])
    Y = np.array([1, 1, 0, 0, 0])

    def test_hand_computed_decisions(self):
        # p*a:      4.5     16.0    0.1     10.0    9.0
        # decision: approve review approve review  approve
        expected = np.array([False, True, False, True, False])
        np.testing.assert_array_equal(
            expected_cost_decisions(self.P, self.A, C_REVIEW), expected
        )

    def test_high_p_small_amount_never_reviewed(self):
        # p=0.9 on $5: expected fraud cost 4.50 < 10 -> economically not worth a review
        assert not expected_cost_decisions(self.P, self.A, C_REVIEW)[0]

    def test_low_p_large_amount_reviewed(self):
        # p=0.02 on $800: expected fraud cost 16 >= 10 -> review
        assert expected_cost_decisions(self.P, self.A, C_REVIEW)[1]

    def test_boundary_is_review(self):
        # p*a == c_review exactly -> review (>= in the rule)
        assert expected_cost_decisions(self.P, self.A, C_REVIEW)[3]

    def test_realized_cost_hand_computed(self):
        # approve fraud $5 -> 5; review -> 10; approve legit -> 0; review -> 10; approve -> 0
        decisions = expected_cost_decisions(self.P, self.A, C_REVIEW)
        assert realized_cost(decisions, self.Y, self.A, C_REVIEW) == 25.0


class TestCostCurve:
    """4-transaction toy table; curve values as literals."""

    P = np.array([0.1, 0.4, 0.6, 0.9])
    Y = np.array([0, 0, 1, 1])
    A = np.array([100.0, 50.0, 200.0, 25.0])
    GRID = np.array([0.05, 0.3, 0.5, 0.7, 1.0])

    def test_curve_hand_computed(self):
        # t=0.05: review all 4                    -> 40
        # t=0.3:  review {.4,.6,.9}               -> 30
        # t=0.5:  review {.6,.9}, both frauds hit -> 20
        # t=0.7:  review {.9}, miss $200 fraud    -> 210
        # t=1.0:  review none, miss both frauds   -> 225
        curve = cost_curve(self.P, self.Y, self.A, C_REVIEW, self.GRID)
        np.testing.assert_allclose(curve, [40.0, 30.0, 20.0, 210.0, 225.0])

    def test_fit_threshold_picks_known_argmin(self):
        t_star, curve = fit_threshold(self.P, self.Y, self.A, C_REVIEW, self.GRID)
        assert t_star == 0.5
        assert curve.min() == 20.0

    def test_curve_matches_bruteforce_realized_cost(self):
        rng = np.random.default_rng(42)
        p = rng.uniform(0, 1, 300)
        y = rng.binomial(1, 0.1, 300)
        a = rng.lognormal(3.0, 1.0, 300)
        grid = np.linspace(0, 1, 101)
        curve = cost_curve(p, y, a, C_REVIEW, grid)
        brute = [realized_cost(threshold_decisions(p, t), y, a, C_REVIEW) for t in grid]
        np.testing.assert_allclose(curve, brute)


class TestDegenerateCases:
    GRID = np.linspace(0, 1, 11)

    def test_all_legit_optimum_reviews_nothing(self):
        p = np.array([0.2, 0.6, 0.9])
        y = np.zeros(3, dtype=int)
        a = np.array([50.0, 50.0, 50.0])
        t_star, curve = fit_threshold(p, y, a, C_REVIEW, self.GRID)
        assert realized_cost(threshold_decisions(p, t_star), y, a, C_REVIEW) == curve.min()
        assert curve.min() == 0.0  # some t reviews nothing and misses nothing

    def test_all_fraud_above_review_cost_reviews_everything(self):
        p = np.array([0.3, 0.5, 0.8])
        y = np.ones(3, dtype=int)
        a = np.array([100.0, 100.0, 100.0])
        t_star, curve = fit_threshold(p, y, a, C_REVIEW, self.GRID)
        assert curve.min() == 3 * C_REVIEW  # review all beats eating any $100 fraud
        assert t_star == 0.0

    def test_approve_all_floor(self):
        assert approve_all_cost(np.array([0, 1, 1]), np.array([5.0, 7.5, 2.5])) == 10.0


def test_per_10k():
    assert per_10k(50.0, 5000) == 100.0


def test_load_cost_params_from_repo_yaml():
    params = load_cost_params()
    assert params.c_review == 10.0
    assert len(params.threshold_grid) == 1001
    assert params.threshold_grid[0] == 0.0 and params.threshold_grid[-1] == 1.0
    assert params.bootstrap_b == 10_000
    assert params.bootstrap_seed == 42
    assert params.ci_level == pytest.approx(0.95)
