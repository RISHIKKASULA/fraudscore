# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/).

## [1.0.0] — 2026-07-08

First release: the full calibrated, cost-based fraud-scoring service.

### Added
- Chronological 60/20/20 split (train / calibration / test) with leakage-avoidance
  rationale; data fetch with row-count + SHA-256 integrity checks; committed synthetic
  fixture (seeded, deterministic) so CI never touches the real data.
- Feature pipeline: V1–V28 passthrough, Amount log1p + robust scaling, `Time` →
  `cycle_phase` sin/cos encoding (phase relative to dataset start, not time-of-day).
- Models trained plain — no resampling, no class weights: logistic-regression baseline
  and HistGradientBoosting with PR-AUC-selected hyperparameters under 3-fold
  time-series CV.
- Probability calibration on the calibration split only (isotonic + sigmoid, selection
  by 5-fold CV Brier, tie broken by p < 0.1 reliability).
- Expected-cost decision layer: amount-aware rule `review ⟺ p̂·amount ≥ c_review`
  (primary) and frozen single-threshold baseline t\*; configurable economics in
  `cost_params.yaml`.
- Bootstrap 95% CIs (B = 10,000, seeded, percentile, paired) on every reported dollar
  figure; evaluation report with reliability diagrams, the cost-vs-threshold signature
  chart, four dollar comparisons, confusion matrices, and a data card.
- Champion selection by calibration-split expected cost (ADR-002) after the honest
  finding that the logistic baseline beats gradient boosting under chronological
  evaluation.
- FastAPI service (`/score`, `/health`, `/model-info`) with a strict pydantic v2
  contract; batch scoring to a DuckDB `scores` table.
- CI (ruff + pytest on the synthetic fixture), Dockerfile (python:3.12-slim, non-root),
  63-test suite: hand-computed toy cases, property tests, API contract, end-to-end
  integration, and golden-metric regression pins.

## [0.1.0]

### Added
- Package scaffold and tooling (uv, ruff, pytest).
