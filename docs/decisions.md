# Decisions (ADR log)

Deviations from [architecture.md](architecture.md) land here. Simplest defensible choice wins.

## ADR-001 — Direct commits to `main` pre-publication (2026-07-08)

Until the repository has a remote, the feature-branch + self-reviewed-PR flow has nothing to
target, so the initial build arc lands as direct conventional commits on `main`. PR flow
begins once the repo is published. Simplest defensible choice; no code impact.
