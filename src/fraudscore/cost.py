"""Cost model and decision policy — the heart of the project.

Primary rule (Bayes-optimal under the cost matrix): review  <=>  p_hat * amount >= c_review.
Baseline: single global threshold t* fitted on the calibration split's empirical cost curve.
"""
