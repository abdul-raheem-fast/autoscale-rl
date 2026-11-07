# Submission PDF Outline (3–4 pages)

Title: **AutoScale-RL — Core ML Pipeline Milestone (MLOps, Fall 2026)**
Team: BCS-7A — M. Abdul Raheem (23L-0588), M. Hayyan Haider (23L-0727), Bilal Kashif (23L-0757)

---

## Page 1 — Overview + links
- One-paragraph summary: safe RL autoscaler trained in a container-traffic simulator
  from scratch; milestone 1 = reproducible core pipeline (no Docker/cloud/API/monitoring).
- **GitHub**: https://github.com/abdul-raheem-fast/autoscale-rl
- **DagsHub (DVC + MLflow)**: https://dagshub.com/abdul-raheem-fast/autoscale-rl
- Architecture sketch / pipeline diagram → [*Screenshot: `dvc dag` *]

## Page 2 — Data pipeline & versioning (DVC)
- META: NASA-HTTP July-1995 trace, pinned Wayback snapshot, sha256 recorded.
- Stage list: ingest → extract → validate (schema ✓, nulls ✓, ranges ✓, continuity ✓)
  → env-config (simulator parameters derived from data = DVC-versioned, RL track) →
  baseline → train_dqn → evaluate.
- Write-up bullets: why 'dvc repro' reproduces from scratch; dvc.lock code/param md5s;
  DagsHub remote (`dvc push`); fail-fast validation gate; synthetic-fallback honesty
  note; what is versioned (trace, per-minute dataset, env_params.json, checkpoint, reports).

## Page 3 — Models + evaluation
- Threshold baseline vs random floor vs Double-DQN (from scratch, PyTorch; hyperparams
  from params.yaml; rewards = cost + SLA-violation + scaled-stability).
- Held-out evaluation table: mean reward / cost units / SLA-violation minutes /
  scaling actions / p95 → [*Screenshot: comparison.png artifact*]

## Page 4 — MLflow + write-up + verdict
- **Screenshots:** (1) DagsHub MLflow experiment run list (≥2 runs — we have 8),
  (2) `dqn-train` run detail with per-episode reward curve, (3) Model Registry:
  AutoScaleRLPolicy in **Staging**, (4) `dvc dag`.
- Per-component short write-up: Data/DVC · Baselines & model · MLflow tracking &
  registry · Reproducibility (seeds + dvc.lock; `pytest` determinism tests).
- Next-milestones teaser: PPO + safety layer + shadow-mode → FastAPI/Cloud Run → monitoring.

---

### Screenshot checklist (from docs/milestone1_checklist.md §4)
- [ ] MLflow experiment list (minimum 2 runs)
- [ ] dqn-train episode metrics (reward curve)
- [ ] Registry Staging table
- [ ] `dvc dag`
