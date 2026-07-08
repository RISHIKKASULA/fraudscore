"""Model training: plain logistic-regression baseline and HistGradientBoosting main model.

No resampling and no class weights anywhere — imbalance is handled at the decision
layer (calibration + expected cost), not by distorting the base rate.
"""
