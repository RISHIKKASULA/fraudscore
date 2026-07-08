"""Data loading and the chronological train/calibration/test split.

Split rationale: leakage avoidance — train on the past, decide on the future,
exactly as the service would run in production.
"""
