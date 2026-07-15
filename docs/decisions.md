# Decisions (ADR log)

Deviations from [architecture.md](architecture.md) land here. Simplest defensible choice wins.

## ADR-003 — Docker quickstart verification deferred (2026-07-08) — RESOLVED 2026-07-15

architecture.md §9 (acceptance) calls for the README quickstart to be verified once on a
clean machine via Docker. Docker is not available on the build machine at release time, so
v1.0.0 ships with the Dockerfile untested end-to-end. Mitigations: the image is a plain
`pip install .` of the tested package on `python:3.12-slim`, and the quickstart commands it
wraps are exercised directly by CI and the contract tests on every push. The verification
will run when a Docker environment is next available; this entry gets closed out then.

**Resolved 2026-07-15.** Docker quickstart verified end-to-end on Docker 29.6.1
(desktop-linux). `docker build -t fraudscore .` builds clean; the container runs under the
documented `-v "$PWD/artifacts:/app/artifacts:ro"` mount and serves correctly: `GET /health`
→ `{"status":"ok","model_loaded":true}`, `POST /score` on the README payload → 200 with a
valid `approve` decision, `GET /model-info` → the champion model card (`baseline`, per
ADR-002), and the strict contract returns 422 with field-level detail on a malformed `v`
vector. The container's `/score` output is byte-for-byte identical to the same request
against a local (non-Docker) run of the artifact, confirming the image faithfully wraps the
tested package. One observation, unrelated to Docker: the README's illustrative `/score`
example JSON shows `fraud_probability 0.0009 / expected_fraud_cost 0.13`, whereas the
committed artifact returns `0.00017 / 0.025` for that input (same locally and in-container) —
the example numbers are stale relative to the shipped model; the response *schema* and
decision are correct. Tracked separately from this ADR.

## ADR-002 — Served model chosen by calibration-split expected cost (2026-07-08)

architecture.md §3 designates HistGradientBoosting as the served model. On the real data,
under the honest chronological split, the plain logistic baseline beats it — PR-AUC 0.64 vs
0.36 on test, and it is cheaper under the amount-aware rule. Cause (verified by ablation, not
a pipeline bug): with only ~360 train frauds, the boosted trees overfit the train time-window
and do not transfer across the temporal shift; the linear model does.

Decision: the artifact carries both calibrated models and the service scores with a
**champion** selected by amount-aware expected cost on the **calibration split only** —
the same split that already fits the calibration mapping and t*, so the test split stays
untouched by any selection. The eval report keeps both models permanently (champion and
challenger); the headline comparisons are computed on the champion. Ties go to the simpler
model. This changes which model serves; it changes nothing about the decision layer, which
is the point of the project.

## ADR-001 — Direct commits to `main` pre-publication (2026-07-08)

Until the repository has a remote, the feature-branch + self-reviewed-PR flow has nothing to
target, so the initial build arc lands as direct conventional commits on `main`. PR flow
begins once the repo is published. Simplest defensible choice; no code impact.
