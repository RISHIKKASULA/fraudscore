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


def realized_cost(review: np.ndarray, y: np.ndarray, amounts: np.ndarray,
                  c_review: float) -> float:
    """Total realized dollars of a decision vector under the cost matrix.

    Reviewed rows cost c_review regardless of label (TP and FP alike); approved fraud
    costs its amount; approved legit is free.
    """
    review = np.asarray(review, dtype=bool)
    y = np.asarray(y)
    amounts = np.asarray(amounts, dtype=float)
    return float(c_review * review.sum() + amounts[~review & (y == 1)].sum())


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
