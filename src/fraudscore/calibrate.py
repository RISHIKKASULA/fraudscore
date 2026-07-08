"""Probability calibration on the calibration split only (prefit pattern).

Isotonic and sigmoid both fitted; selection by 5-fold CV Brier score within the
calibration split, ties broken by reliability fit in the p < 0.1 region.
"""
