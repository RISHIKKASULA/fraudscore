# STATE

Build status against [docs/architecture.md](docs/architecture.md). Updated as milestones land.

## Repo layout

```
fraudscore/
├── src/fraudscore/           __init__.py data.py features.py train.py calibrate.py
│                             cost.py evaluate.py serve.py cli.py
├── tests/                    unit + integration mirroring src; conftest.py builds fixture
├── scripts/                  fetch_data.py  make_fixture.py
├── data/fixtures/synthetic.csv          (committed; raw/ gitignored)
├── docs/                     architecture.md  decisions.md  eval-report.md (generated)
├── .github/workflows/ci.yml  ruff check → pytest (3.12, uv, pip cache)
├── Dockerfile                python:3.12-slim, non-root, uvicorn entrypoint
├── cost_params.yaml          c_review, threshold grid, bootstrap {B, seed, ci_level}
├── pyproject.toml            uv-managed
├── README.md CHANGELOG.md LICENSE STATE.md .gitignore
```

## Build order

- [x] 1. Scaffold package layout and tooling
- [x] 2. Data fetch with integrity checks + synthetic fixture generator
- [x] 3. Feature pipeline with cyclical phase encoding (+ unit tests)
- [x] 4. Baseline logistic regression
- [x] 5. Gradient boosting with time-series CV selection
- [x] 6. Probability calibration with method selection (+ tests)
- [x] 7. Expected-cost decision rule and threshold baseline (+ toy-case tests)
- [x] 8. Bootstrap confidence intervals for cost metrics
- [x] 9. Evaluation report with cost curves and CIs
- [x] 10. Serve expected-cost decisions over FastAPI (+ contract tests)
- [x] 11. Batch scoring to DuckDB
- [x] 12. CI workflow + Dockerfile
- [x] 13. README with measured results (champion selection per ADR-002)
- [ ] 14. v1.0.0 release
