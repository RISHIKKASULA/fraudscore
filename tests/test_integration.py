"""Integration + regression tests: full pipeline on the committed synthetic fixture.

The regression goldens pin the fixture run (seed 42, bootstrap B = 500). Real-data
metrics live only in the committed eval report and are asserted nowhere.
"""

from __future__ import annotations

import json

import pytest

from fraudscore.cli import main as cli_main

from .conftest import FIXTURE_CSV, REPO_ROOT

BOOTSTRAP_B = 500  # reduced for test speed; cost_params.yaml keeps the real 10,000
TOL = 1e-6


@pytest.fixture(scope="session")
def pipeline_run(tmp_path_factory):
    """fixture -> train -> evaluate, via the CLI exactly as a user would run it."""
    out = tmp_path_factory.mktemp("pipeline")
    artifact_dir = out / "artifacts"
    report = out / "report" / "eval-report.md"
    cost_params = str(REPO_ROOT / "cost_params.yaml")

    assert cli_main(["train", "--data", str(FIXTURE_CSV), "--cost-params", cost_params,
                     "--out", str(artifact_dir)]) == 0
    assert cli_main(["evaluate", "--data", str(FIXTURE_CSV), "--cost-params", cost_params,
                     "--artifact", str(artifact_dir / "model.joblib"),
                     "--report", str(report), "--bootstrap-b", str(BOOTSTRAP_B)]) == 0
    return artifact_dir, report


class TestEndToEnd:
    def test_report_exists_and_parses(self, pipeline_run):
        _, report = pipeline_run
        text = report.read_text()
        assert "# Evaluation report" in text
        assert "## Data card" in text
        assert "### The four comparisons" in text
        assert "point [low, high]" in text
        assert (report.parent / "reliability.png").exists()
        assert (report.parent / "cost-curve.png").exists()

    def test_model_card_filled_by_evaluate(self, pipeline_run):
        artifact_dir, _ = pipeline_run
        card = json.loads((artifact_dir / "model-card.json").read_text())
        assert card["metrics"] is not None
        assert 0.0 <= card["t_star"] <= 1.0
        assert card["cost_params"]["c_review"] == 10.0
        assert card["calibration_method_main"] in {"sigmoid", "isotonic"}

    def test_rerun_evaluate_is_deterministic(self, pipeline_run, tmp_path):
        artifact_dir, report = pipeline_run
        report2 = tmp_path / "eval-report.md"
        cli_main(["evaluate", "--data", str(FIXTURE_CSV),
                  "--cost-params", str(REPO_ROOT / "cost_params.yaml"),
                  "--artifact", str(artifact_dir / "model.joblib"),
                  "--report", str(report2), "--bootstrap-b", str(BOOTSTRAP_B)])
        assert report2.read_text() == report.read_text()


class TestRegressionGoldens:
    """Golden metrics from the synthetic-fixture run; tolerance ±1e-6.

    These pin end-to-end determinism (data -> split -> train -> calibrate -> t* ->
    bootstrap). If a dependency bump shifts them, re-derive deliberately and record
    the change in docs/decisions.md.
    """

    def test_golden_metrics(self, pipeline_run):
        artifact_dir, _ = pipeline_run
        card = json.loads((artifact_dir / "model-card.json").read_text())
        m = card["metrics"]["test"]

        assert m["main_calibrated"]["pr_auc"] == pytest.approx(GOLDEN["pr_auc"], abs=TOL)
        assert m["main_calibrated"]["brier"] == pytest.approx(GOLDEN["brier"], abs=TOL)
        assert card["t_star"] == pytest.approx(GOLDEN["t_star"], abs=TOL)

        low, high = card["metrics"]["comparisons"]["aa_vs_approve_all"]["savings"][1:]
        assert low == pytest.approx(GOLDEN["aa_vs_approve_all_ci_low"], abs=TOL)
        assert high == pytest.approx(GOLDEN["aa_vs_approve_all_ci_high"], abs=TOL)


GOLDEN = {
    "pr_auc": 0.9955679241393527,
    "brier": 0.007502352801728797,
    "t_star": 0.197,
    "aa_vs_approve_all_ci_low": 32528.950000000004,
    "aa_vs_approve_all_ci_high": 280953.4437499998,
}
