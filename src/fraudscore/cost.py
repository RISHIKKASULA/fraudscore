"""Cost model and decision policy — the heart of the project.

Cost matrix per transaction i with amount a_i:
  TN (approve legit)  0
  FP (review legit)   c_review
  TP (review fraud)   c_review   (fraud caught, review still costs)
  FN (approve fraud)  a_i        (charge-back eaten)

PRIMARY — amount-aware expected-cost rule (Bayes-optimal under this matrix):

    review  <=>  p_hat_i * a_i >= c_review

Expected cost of approving = p_hat_i * a_i; of reviewing = c_review; pick the cheaper
action per transaction. No fitted threshold — the "threshold" is the cost ratio itself,
which is why the rule stands or falls with calibration quality. Consequence worth owning:
a transaction under c_review is never reviewed even at p_hat = 1 — economically correct
under this matrix; see the README limitations section.

BASELINE — single global threshold t*: argmin of the empirical cost curve on the
calibration split, frozen before touching test. Kept because it's what most shops
actually deploy — the dollar gap between it and the amount-aware rule is the finding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

DEFAULT_PARAMS_PATH = Path(__file__).resolve().parents[2] / "cost_params.yaml"


@dataclass(frozen=True)
class CostParams:
    c_review: float
    threshold_grid: np.ndarray
    bootstrap_b: int
    bootstrap_seed: int
    ci_level: float


def load_cost_params(path: str | Path = DEFAULT_PARAMS_PATH) -> CostParams:
    with Path(path).open() as fh:
        raw = yaml.safe_load(fh)
    grid_cfg = raw["threshold_grid"]
    return CostParams(
        c_review=float(raw["c_review"]),
        threshold_grid=np.linspace(grid_cfg["start"], grid_cfg["stop"], grid_cfg["num"]),
        bootstrap_b=int(raw["bootstrap"]["B"]),
        bootstrap_seed=int(raw["bootstrap"]["seed"]),
        ci_level=float(raw["bootstrap"]["ci_level"]),
    )


def expected_cost_decisions(p_hat: np.ndarray, amounts: np.ndarray,
                            c_review: float) -> np.ndarray:
    """Amount-aware rule: True (review) where p_hat * amount >= c_review."""
    return np.asarray(p_hat) * np.asarray(amounts) >= c_review


def threshold_decisions(p_hat: np.ndarray, threshold: float) -> np.ndarray:
    """Single-global-threshold rule: True (review) where p_hat >= t."""
    return np.asarray(p_hat) >= threshold


def decision_row_costs(review: np.ndarray, y: np.ndarray, amounts: np.ndarray,
                       c_review: float) -> np.ndarray:
    """Per-row realized dollars of a decision vector under the cost matrix.

    Reviewed rows cost c_review regardless of label (TP and FP alike); approved fraud
    costs its amount; approved legit is free. Per-row form so the bootstrap can
    resample rows.
    """
    review = np.asarray(review, dtype=bool)
    y = np.asarray(y)
    amounts = np.asarray(amounts, dtype=float)
    return np.where(review, c_review, np.where(y == 1, amounts, 0.0))


def realized_cost(review: np.ndarray, y: np.ndarray, amounts: np.ndarray,
                  c_review: float) -> float:
    """Total realized dollars of a decision vector under the cost matrix."""
    return float(decision_row_costs(review, y, amounts, c_review).sum())


def cost_curve(p_hat: np.ndarray, y: np.ndarray, amounts: np.ndarray,
               c_review: float, grid: np.ndarray) -> np.ndarray:
    """Empirical cost(t) = sum_i [p_i >= t]*c_review + [p_i < t]*y_i*a_i over the grid.

    Vectorized via a single sort: for each t, the rows with p < t contribute their
    fraud amounts (prefix sum), the rest contribute c_review each.
    """
    p_hat = np.asarray(p_hat, dtype=float)
    fn_cost = np.asarray(y, dtype=float) * np.asarray(amounts, dtype=float)

    order = np.argsort(p_hat, kind="stable")
    p_sorted = p_hat[order]
    fn_prefix = np.concatenate([[0.0], np.cumsum(fn_cost[order])])

    n_below = np.searchsorted(p_sorted, np.asarray(grid), side="left")
    n_reviewed = len(p_hat) - n_below
    return c_review * n_reviewed + fn_prefix[n_below]


def fit_threshold(p_hat: np.ndarray, y: np.ndarray, amounts: np.ndarray,
                  c_review: float, grid: np.ndarray) -> tuple[float, np.ndarray]:
    """t* = argmin of the empirical cost curve (first argmin on ties — deterministic).

    Fit on the calibration split only, frozen before touching test.
    """
    curve = cost_curve(p_hat, y, amounts, c_review, grid)
    return float(np.asarray(grid)[int(np.argmin(curve))]), curve


def approve_all_cost(y: np.ndarray, amounts: np.ndarray) -> float:
    """The do-nothing floor: every fraud's amount is eaten, nothing is reviewed."""
    y = np.asarray(y)
    return float(np.asarray(amounts, dtype=float)[y == 1].sum())


def per_10k(total_cost: float, n: int) -> float:
    """Normalize a total dollar cost to dollars per 10,000 transactions."""
    return total_cost / n * 10_000


# --- Bootstrap 95% CIs -------------------------------------------------------------
#
# ~98 test frauds make point estimates noisy, so every reported cost and every
# improvement carries a percentile bootstrap CI: resample test rows with replacement,
# B = 10,000, seeded. Report format everywhere: point [CI_low, CI_high]. If an
# improvement's CI includes zero, we say so plainly.

_BOOTSTRAP_CHUNK = 512  # replicates per chunk; bounds the index-matrix memory


@dataclass(frozen=True)
class CIResult:
    point: float
    low: float
    high: float

    def includes_zero(self) -> bool:
        return self.low <= 0.0 <= self.high

    def __format__(self, spec: str) -> str:
        spec = spec or ",.2f"
        return f"{self.point:{spec}} [{self.low:{spec}}, {self.high:{spec}}]"


def bootstrap_ci(row_arrays: list[np.ndarray], stat_of_sums, b: int, seed: int,
                 ci_level: float = 0.95) -> CIResult:
    """Percentile-bootstrap CI for a statistic of per-row sums.

    `row_arrays` are k aligned per-row value vectors (e.g. one per decision rule);
    each replicate resamples the same n row indices for all k (paired bootstrap) and
    passes the k resampled sums to `stat_of_sums`, which must be numpy-vectorized
    (it receives scalars for the point estimate and (B,) arrays for replicates).

    Deterministic for a given seed: replicates are drawn in fixed-size chunks so the
    RNG stream never depends on available memory.
    """
    rows = [np.asarray(r, dtype=float) for r in row_arrays]
    n = len(rows[0])
    if any(len(r) != n for r in rows):
        raise ValueError("row arrays must be aligned (same length)")

    point = float(stat_of_sums(*(r.sum() for r in rows)))

    rng = np.random.default_rng(seed)
    replicate_sums = np.empty((b, len(rows)))
    for start in range(0, b, _BOOTSTRAP_CHUNK):
        stop = min(start + _BOOTSTRAP_CHUNK, b)
        idx = rng.integers(0, n, size=(stop - start, n))
        for j, r in enumerate(rows):
            replicate_sums[start:stop, j] = r[idx].sum(axis=1)

    replicates = stat_of_sums(*(replicate_sums[:, j] for j in range(len(rows))))
    alpha = (1.0 - ci_level) / 2.0
    low, high = np.percentile(replicates, [100 * alpha, 100 * (1 - alpha)])
    return CIResult(point=point, low=float(low), high=float(high))


def cost_per_10k_ci(row_costs: np.ndarray, b: int, seed: int,
                    ci_level: float = 0.95) -> CIResult:
    """CI for one rule's cost in dollars per 10k transactions."""
    n = len(row_costs)
    return bootstrap_ci([row_costs], lambda s: per_10k(s, n), b, seed, ci_level)


def savings_per_10k_ci(row_costs_worse: np.ndarray, row_costs_better: np.ndarray,
                       b: int, seed: int, ci_level: float = 0.95) -> CIResult:
    """CI for the paired dollar gap (worse - better) per 10k transactions."""
    n = len(row_costs_worse)
    return bootstrap_ci(
        [row_costs_worse, row_costs_better],
        lambda sw, sb: per_10k(sw - sb, n),
        b, seed, ci_level,
    )


def savings_pct_ci(row_costs_worse: np.ndarray, row_costs_better: np.ndarray,
                   b: int, seed: int, ci_level: float = 0.95) -> CIResult:
    """CI for the paired % improvement of `better` over `worse`."""
    return bootstrap_ci(
        [row_costs_worse, row_costs_better],
        lambda sw, sb: 100.0 * (sw - sb) / sw,
        b, seed, ci_level,
    )
