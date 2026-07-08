"""Evaluation report generation: metrics, reliability diagrams, cost curves, bootstrap CIs.

Writes docs/eval-report.md + PNGs; seeded end-to-end so fetch -> train -> evaluate
reproduces the report bit-for-bit (no timestamps in the report body).

Everything is reported in dollars per 10k transactions with a bootstrap 95% CI,
format `point [CI_low, CI_high]`. If an improvement's CI includes zero, the report
says so plainly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from fraudscore.cost import (
    CIResult,
    cost_curve,
    cost_per_10k_ci,
    decision_row_costs,
    expected_cost_decisions,
    load_cost_params,
    savings_pct_ci,
    savings_per_10k_ci,
    threshold_decisions,
)
from fraudscore.data import chronological_split, load_dataset
from fraudscore.features import TARGET_COLUMN

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

NAIVE_THRESHOLD = 0.5  # what a notebook would ship: t = 0.5 on the uncalibrated model


@dataclass(frozen=True)
class Policy:
    key: str
    label: str
    review: np.ndarray  # decision vector on the test split
    row_costs: np.ndarray


def _confusion(review: np.ndarray, y: np.ndarray) -> dict[str, int]:
    review = np.asarray(review, dtype=bool)
    y = np.asarray(y).astype(bool)
    return {
        "tn": int((~review & ~y).sum()),
        "fp": int((review & ~y).sum()),
        "fn": int((~review & y).sum()),
        "tp": int((review & y).sum()),
    }


def _fmt_ci(ci: CIResult, unit: str = "$") -> str:
    if unit == "$":
        return f"${ci.point:,.2f} [${ci.low:,.2f}, ${ci.high:,.2f}]"
    return f"{ci.point:.1f}% [{ci.low:.1f}%, {ci.high:.1f}%]"


def run_evaluation(artifact_path: str | Path, data_path: str | Path,
                   cost_params_path: str | Path, report_path: str | Path,
                   bootstrap_b: int | None = None) -> dict:
    """Evaluate the trained artifact on the test split and write the report.

    `bootstrap_b` overrides cost_params.yaml's B (CI uses a reduced value to stay fast).
    Returns the summary dict (also used by the regression tests).
    """
    params = load_cost_params(cost_params_path)
    b = bootstrap_b if bootstrap_b is not None else params.bootstrap_b
    seed, ci_level, c_review = params.bootstrap_seed, params.ci_level, params.c_review

    artifact = joblib.load(artifact_path)
    t_star = artifact["t_star"]
    champion_key = artifact["champion"]
    challenger_key = "baseline" if champion_key == "main" else "main"
    champion, challenger = artifact[champion_key], artifact[challenger_key]
    model_names = {"main": "gradient boosting", "baseline": "logistic"}
    champ_name, chall_name = model_names[champion_key], model_names[challenger_key]

    splits = chronological_split(load_dataset(data_path))
    test = splits.test
    y = test[TARGET_COLUMN].to_numpy()
    amounts = test["Amount"].to_numpy(dtype=float)
    n = len(test)

    # Probabilities on the test split (headline comparisons run on the champion).
    p_raw = champion.model.predict_proba_raw(test)
    p_cal = champion.model.predict_proba(test)
    p_sigmoid = champion.candidates["sigmoid"].predict_proba(test)
    p_isotonic = champion.candidates["isotonic"].predict_proba(test)
    p_chall_cal = challenger.model.predict_proba(test)

    # Decision policies under comparison.
    def make_policy(key: str, label: str, review: np.ndarray) -> Policy:
        return Policy(key, label, review, decision_row_costs(review, y, amounts, c_review))

    policies = {
        p.key: p
        for p in [
            make_policy("aa_cal", "amount-aware, calibrated champion (primary)",
                        expected_cost_decisions(p_cal, amounts, c_review)),
            make_policy("tstar_cal", f"single threshold t* = {t_star:.3f}, calibrated champion",
                        threshold_decisions(p_cal, t_star)),
            make_policy("naive_raw", "naive t = 0.5, uncalibrated champion",
                        threshold_decisions(p_raw, NAIVE_THRESHOLD)),
            make_policy("aa_raw", "amount-aware, uncalibrated champion",
                        expected_cost_decisions(p_raw, amounts, c_review)),
            make_policy("aa_chall", f"amount-aware, calibrated challenger ({chall_name})",
                        expected_cost_decisions(p_chall_cal, amounts, c_review)),
            make_policy("approve_all", "approve all (do-nothing floor)",
                        np.zeros(n, dtype=bool)),
        ]
    }

    costs = {k: cost_per_10k_ci(p.row_costs, b, seed, ci_level) for k, p in policies.items()}

    def compare(worse: str, better: str) -> dict:
        return {
            "savings": savings_per_10k_ci(policies[worse].row_costs,
                                          policies[better].row_costs, b, seed, ci_level),
            "pct": savings_pct_ci(policies[worse].row_costs,
                                  policies[better].row_costs, b, seed, ci_level),
        }

    comparisons = {
        "aa_vs_tstar": compare("tstar_cal", "aa_cal"),
        "tstar_vs_naive": compare("naive_raw", "tstar_cal"),
        "cal_vs_uncal_aa": compare("aa_raw", "aa_cal"),
        "aa_vs_approve_all": compare("approve_all", "aa_cal"),
    }

    def _model_metrics(p: np.ndarray) -> dict[str, float]:
        return {
            "pr_auc": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
        }

    metrics = {
        "champion_calibrated": _model_metrics(p_cal),
        "champion_uncalibrated": _model_metrics(p_raw),
        "challenger_calibrated": _model_metrics(p_chall_cal),
    }

    report_path = Path(report_path)
    report_dir = report_path.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    _plot_reliability(report_dir / "reliability.png", y,
                      {"uncalibrated": p_raw, "sigmoid": p_sigmoid, "isotonic": p_isotonic},
                      chosen=champion.method)
    _plot_cost_curve(report_dir / "cost-curve.png", p_cal, y, amounts, c_review,
                     params.threshold_grid, t_star, costs["aa_cal"].point)

    summary = {
        "n_test": n,
        "t_star": t_star,
        "champion": champion_key,
        "calibration_method": champion.method,
        "metrics": metrics,
        "costs_per_10k": {k: (ci.point, ci.low, ci.high) for k, ci in costs.items()},
        "comparisons": {
            k: {
                "savings": (v["savings"].point, v["savings"].low, v["savings"].high),
                "pct": (v["pct"].point, v["pct"].low, v["pct"].high),
            }
            for k, v in comparisons.items()
        },
        "confusion_t_star": _confusion(policies["tstar_cal"].review, y),
        "confusion_amount_aware": _confusion(policies["aa_cal"].review, y),
    }

    report_path.write_text(_render_report(
        splits=splits, policies=policies, costs=costs, comparisons=comparisons,
        metrics=metrics, champion=champion, champ_name=champ_name, chall_name=chall_name,
        challenger=challenger, t_star=t_star, c_review=c_review, b=b, seed=seed,
        ci_level=ci_level, y=y, summary=summary,
    ))

    _update_model_card(Path(artifact_path).parent / "model-card.json", summary)
    return summary


def _plot_reliability(path: Path, y: np.ndarray, curves: dict[str, np.ndarray],
                      chosen: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, p in curves.items():
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        marker = "o" if name == chosen else "s"
        label = f"{name} (selected)" if name == chosen else name
        ax.plot(mean_pred, frac_pos, marker=marker, ms=4, lw=1.5, label=label)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed fraud rate")
    ax.set_title("Reliability (test split)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_cost_curve(path: Path, p_cal: np.ndarray, y: np.ndarray, amounts: np.ndarray,
                     c_review: float, grid: np.ndarray, t_star: float,
                     aa_cost_per_10k: float) -> None:
    """The signature chart: threshold cost curve vs the amount-aware horizontal line."""
    n = len(p_cal)
    curve = cost_curve(p_cal, y, amounts, c_review, grid) / n * 10_000
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(grid, curve, lw=1.8, label="single global threshold t (calibrated)")
    ax.axvline(t_star, color="gray", ls=":", lw=1.2,
               label=f"t* = {t_star:.3f} (frozen on calibration split)")
    ax.axhline(aa_cost_per_10k, color="crimson", ls="--", lw=1.5,
               label=f"amount-aware rule: ${aa_cost_per_10k:,.0f} / 10k")
    ax.set_xlabel("threshold t")
    ax.set_ylabel("expected cost ($ per 10k transactions)")
    ax.set_title("Cost vs threshold — no single threshold reaches the amount-aware rule")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _confusion_table(c: dict[str, int]) -> str:
    return (
        "|  | predicted approve | predicted review |\n"
        "|---|---|---|\n"
        f"| **legit** | {c['tn']:,} (TN) | {c['fp']:,} (FP) |\n"
        f"| **fraud** | {c['fn']:,} (FN) | {c['tp']:,} (TP) |\n"
    )


def _render_report(*, splits, policies, costs, comparisons, metrics, champion, champ_name,
                   chall_name, challenger, t_star, c_review, b, seed, ci_level, y,
                   summary) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Evaluation report")
    add("")
    add("Generated by `fraudscore evaluate`; seeded end-to-end, so a rerun reproduces this")
    add(f"file bit-for-bit. All intervals are bootstrap {ci_level:.0%} CIs "
        f"(B = {b:,}, seed = {seed}, percentile), format `point [low, high]`.")
    add("")

    add("## Data card")
    add("")
    add("Chronological split (train on the past, decide on the future — leakage avoidance).")
    add("")
    add("| split | rows | frauds | base rate | Time range (s) |")
    add("|---|---|---|---|---|")
    for name, part in [("train", splits.train), ("calibration", splits.calibration),
                       ("test", splits.test)]:
        frauds = int(part[TARGET_COLUMN].sum())
        add(f"| {name} | {len(part):,} | {frauds:,} | {frauds / len(part):.4%} "
            f"| {part['Time'].min():,.0f} – {part['Time'].max():,.0f} |")
    add("")

    add("## Models under comparison")
    add("")
    add(f"Champion (served, headlines below): **{champ_name}** — selected by amount-aware "
        "expected cost on the calibration split only (see decisions.md ADR-002). "
        f"Challenger: {chall_name}. Both stay in this report permanently.")
    add("")
    add("| model | PR-AUC | ROC-AUC¹ | Brier |")
    add("|---|---|---|---|")
    rows = [
        (f"champion: {champ_name} (calibrated, {champion.method})",
         metrics["champion_calibrated"]),
        (f"champion: {champ_name} (uncalibrated)", metrics["champion_uncalibrated"]),
        (f"challenger: {chall_name} (calibrated, {challenger.method})",
         metrics["challenger_calibrated"]),
    ]
    for label, m in rows:
        add(f"| {label} | {m['pr_auc']:.4f} | {m['roc_auc']:.4f} | {m['brier']:.6f} |")
    add("")
    add("¹ ROC-AUC is inflated under heavy class imbalance; PR-AUC is the primary metric.")
    add("")

    add("## Calibration (champion)")
    add("")
    add(f"Selected method: **{champion.method}** (lower Brier under 5-fold CV within the "
        "calibration split; tie broken by reliability fit in the p < 0.1 region).")
    add("")
    add("| method | CV Brier (calibration split) | p < 0.1 reliability error |")
    add("|---|---|---|")
    for method in ("sigmoid", "isotonic"):
        add(f"| {method} | {champion.cv_brier[method]:.6f} "
            f"| {champion.low_region_error[method]:.6f} |")
    add("")
    add("![Reliability diagram](reliability.png)")
    add("")

    add("## Decision policy costs (test split, $ per 10k transactions)")
    add("")
    add(f"Cost matrix: review costs ${c_review:,.2f} (fraud or not); approved fraud costs "
        "its amount; approved legit is free.")
    add("")
    add("| policy | cost per 10k |")
    add("|---|---|")
    for key in ("aa_cal", "tstar_cal", "naive_raw", "aa_raw", "aa_chall", "approve_all"):
        add(f"| {policies[key].label} | {_fmt_ci(costs[key])} |")
    add("")

    add("### The four comparisons")
    add("")
    comparison_rows = [
        ("aa_vs_tstar",
         "**Amount-aware vs single threshold t\\*** (both calibrated) — the headline"),
        ("tstar_vs_naive",
         "**t\\* vs naive t = 0.5 on the uncalibrated model** — what a notebook would ship"),
        ("cal_vs_uncal_aa",
         "**Calibrated vs uncalibrated under the amount-aware rule** — what calibration "
         "buys in dollars"),
        ("aa_vs_approve_all",
         "**Amount-aware vs approve-all** — distance from the do-nothing floor"),
    ]
    for i, (key, title) in enumerate(comparison_rows, start=1):
        savings, pct = comparisons[key]["savings"], comparisons[key]["pct"]
        add(f"{i}. {title}")
        add(f"   Savings: {_fmt_ci(savings)} per 10k ({_fmt_ci(pct, unit='%')}).")
        if savings.includes_zero():
            add("   **This CI includes zero** — the improvement is not distinguishable "
                "from noise at this sample size, and we say so plainly.")
        add("")

    add("## Signature chart")
    add("")
    add("![Cost vs threshold](cost-curve.png)")
    add("")
    add("The curve is the best any single global threshold can do on the test split; the "
        "dashed line is the amount-aware rule. The gap between the curve's minimum and the "
        "line is the value of pricing each transaction individually.")
    add("")

    add("## Confusion matrices (test split)")
    add("")
    add(f"### At t\\* = {t_star:.3f} (calibrated)")
    add("")
    add(_confusion_table(summary["confusion_t_star"]))
    add(f"### Amount-aware rule (review ⟺ p̂ · amount ≥ ${c_review:,.0f})")
    add("")
    add(_confusion_table(summary["confusion_amount_aware"]))
    add("Note: under the amount-aware rule a 'FN' can be economically correct — a small "
        "fraud not worth a review. The dollar tables above, not raw counts, are the "
        "measure of the policy.")
    add("")
    return "\n".join(lines)


def _update_model_card(card_path: Path, summary: dict) -> None:
    if not card_path.exists():
        return
    card = json.loads(card_path.read_text())
    card["metrics"] = {
        "test": summary["metrics"],
        "costs_per_10k": summary["costs_per_10k"],
        "comparisons": summary["comparisons"],
    }
    card_path.write_text(json.dumps(card, indent=2) + "\n")
