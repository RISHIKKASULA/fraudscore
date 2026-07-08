"""Batch scoring tests: CSV in, DuckDB scores table out, appends across runs."""

from __future__ import annotations

import duckdb
import pytest

from fraudscore.cli import main as cli_main
from fraudscore.serve import score_batch


@pytest.fixture()
def input_csv(synthetic_df, tmp_path):
    path = tmp_path / "in.csv"
    synthetic_df.head(50).to_csv(path, index=False)
    return path


def test_scores_written_with_contract_columns(trained_artifact_dir, input_csv, tmp_path):
    db = tmp_path / "scores.duckdb"
    assert score_batch(input_csv, db, trained_artifact_dir) == 50

    with duckdb.connect(str(db)) as con:
        rows = con.execute(
            "SELECT id, p_hat, expected_fraud_cost, decision, model_version FROM scores"
        ).fetchall()
    assert len(rows) == 50
    for row_id, p_hat, efc, decision, _version in rows:
        assert 0 <= row_id < 50
        assert 0.0 <= p_hat <= 1.0
        assert decision == ("review" if efc >= 10.0 else "approve")


def test_second_run_appends(trained_artifact_dir, input_csv, tmp_path):
    db = tmp_path / "scores.duckdb"
    score_batch(input_csv, db, trained_artifact_dir)
    score_batch(input_csv, db, trained_artifact_dir)
    with duckdb.connect(str(db)) as con:
        assert con.execute("SELECT count(*) FROM scores").fetchone()[0] == 100


def test_missing_columns_rejected(trained_artifact_dir, synthetic_df, tmp_path):
    bad = tmp_path / "bad.csv"
    synthetic_df.head(5).drop(columns=["V7"]).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="V7"):
        score_batch(bad, tmp_path / "scores.duckdb", trained_artifact_dir)


def test_cli_entrypoint(trained_artifact_dir, input_csv, tmp_path, capsys):
    db = tmp_path / "scores.duckdb"
    assert cli_main(["score-batch", str(input_csv), "--out", str(db),
                     "--artifact-dir", str(trained_artifact_dir)]) == 0
    assert "scored 50 rows" in capsys.readouterr().out
