"""Model training: plain logistic-regression baseline and HistGradientBoosting main model.

No resampling and no class weights anywhere — imbalance is handled at the decision layer
(calibration + expected cost), not by lying to the model about the base rate. The baseline
is calibrated-and-costed, not a strawman: it goes through the identical calibration and
decision layers and stays in the eval report permanently.

HistGradientBoosting over XGBoost deliberately: one fewer dependency, native sklearn
integration, trains in seconds — the differentiator here is the decision layer, not the
model brand.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from fraudscore import __version__
from fraudscore.calibrate import CalibrationResult, calibrate
from fraudscore.cost import (
    expected_cost_decisions,
    fit_threshold,
    load_cost_params,
    realized_cost,
)
from fraudscore.data import chronological_split, load_dataset
from fraudscore.features import RAW_FEATURE_COLUMNS, TARGET_COLUMN, build_preprocessor

RANDOM_STATE = 42

MAIN_PARAM_GRID = {
    "model__max_iter": [200, 400],
    "model__learning_rate": [0.05, 0.1],
    "model__max_leaf_nodes": [31, 63],
}


def train_baseline(train_frame: pd.DataFrame) -> Pipeline:
    """Fit preprocessor + plain LogisticRegression on the train split."""
    model = Pipeline(
        [
            ("features", build_preprocessor()),
            ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(train_frame[RAW_FEATURE_COLUMNS], train_frame[TARGET_COLUMN])
    return model


@dataclass(frozen=True)
class MainModelResult:
    estimator: Pipeline
    best_params: dict
    cv_pr_auc: float


def train_main(train_frame: pd.DataFrame) -> MainModelResult:
    """Fit HistGradientBoosting, selecting hyperparameters by PR-AUC under a 3-fold
    TimeSeriesSplit within the train window only.

    The train split is already time-ordered (chronological_split sorts by `Time`), so
    each CV fold trains on the past and validates on the future — the selection never
    peeks across time, mirroring how the service runs in production.
    """
    pipe = Pipeline(
        [
            ("features", build_preprocessor()),
            ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]
    )
    search = GridSearchCV(
        pipe,
        MAIN_PARAM_GRID,
        scoring="average_precision",  # PR-AUC, the primary metric under imbalance
        cv=TimeSeriesSplit(n_splits=3),
        n_jobs=-1,
        refit=True,
    )
    search.fit(train_frame[RAW_FEATURE_COLUMNS], train_frame[TARGET_COLUMN])
    return MainModelResult(
        estimator=search.best_estimator_,
        best_params={k.removeprefix("model__"): v for k, v in search.best_params_.items()},
        cv_pr_auc=float(search.best_score_),
    )


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def amount_aware_calibration_cost(cal_result: CalibrationResult,
                                  calibration_frame: pd.DataFrame,
                                  c_review: float) -> float:
    """Realized amount-aware cost of a calibrated model on the calibration split."""
    p = cal_result.model.predict_proba(calibration_frame)
    y = calibration_frame[TARGET_COLUMN].to_numpy()
    amounts = calibration_frame["Amount"].to_numpy(dtype=float)
    review = expected_cost_decisions(p, amounts, c_review)
    return realized_cost(review, y, amounts, c_review)


def select_champion(costs: dict[str, float]) -> str:
    """Champion = lowest calibration-split amount-aware cost; ties go to 'baseline'.

    Selection uses the calibration split only (ADR-002) — the same split that fits the
    calibration mapping and t* — so the test split stays untouched by any selection.
    """
    if costs["baseline"] <= costs["main"]:
        return "baseline"
    return "main"


def run_training(data_path: str | Path, cost_params_path: str | Path,
                 out_dir: str | Path) -> Path:
    """Full training run: split, fit both models, calibrate both, pick the champion,
    freeze t*, save artifact.

    Writes out_dir/model.joblib (both calibrated models + champion key + t* + cost
    params) and an initial out_dir/model-card.json; `fraudscore evaluate` fills in
    test-split metrics. Returns the artifact path.
    """
    params = load_cost_params(cost_params_path)
    splits = chronological_split(load_dataset(data_path))

    baseline_cal = calibrate(train_baseline(splits.train), splits.calibration)
    main = train_main(splits.train)
    main_cal = calibrate(main.estimator, splits.calibration)

    # Champion selection on the calibration split (ADR-002).
    calibration_costs = {
        "main": amount_aware_calibration_cost(main_cal, splits.calibration, params.c_review),
        "baseline": amount_aware_calibration_cost(baseline_cal, splits.calibration,
                                                  params.c_review),
    }
    champion_key = select_champion(calibration_costs)
    champion = {"main": main_cal, "baseline": baseline_cal}[champion_key]

    # Freeze the single-threshold baseline t* for the champion on the calibration
    # split, before test.
    p_cal = champion.model.predict_proba(splits.calibration)
    t_star, _ = fit_threshold(
        p_cal,
        splits.calibration[TARGET_COLUMN].to_numpy(),
        splits.calibration["Amount"].to_numpy(),
        params.c_review,
        params.threshold_grid,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifact_path = out / "model.joblib"
    joblib.dump(
        {
            "version": __version__,
            "main": main_cal,
            "baseline": baseline_cal,
            "champion": champion_key,
            "t_star": t_star,
            "c_review": params.c_review,
        },
        artifact_path,
    )

    card = {
        "version": __version__,
        "git_sha": _git_sha(),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "champion": champion_key,
        "champion_model": {"main": "HistGradientBoostingClassifier",
                           "baseline": "LogisticRegression"}[champion_key],
        "calibration_split_amount_aware_cost": {
            k: round(v, 2) for k, v in calibration_costs.items()
        },
        "main_model": "HistGradientBoostingClassifier",
        "main_best_params": main.best_params,
        "main_cv_pr_auc_train_window": round(main.cv_pr_auc, 6),
        "calibration_method_main": main_cal.method,
        "calibration_method_baseline": baseline_cal.method,
        "t_star": t_star,
        "cost_params": {
            "c_review": params.c_review,
            "bootstrap_b": params.bootstrap_b,
            "bootstrap_seed": params.bootstrap_seed,
            "ci_level": params.ci_level,
        },
        "metrics": None,  # filled by `fraudscore evaluate`
    }
    (out / "model-card.json").write_text(json.dumps(card, indent=2) + "\n")
    return artifact_path
