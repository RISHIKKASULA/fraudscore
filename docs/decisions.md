# Decisions (ADR log)

Deviations from [architecture.md](architecture.md) land here. Simplest defensible choice wins.

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
