# Milestone 1 Checklist — Core ML Pipeline (RL track)

Goal: core ML pipeline running **end-to-end, fully reproducible** — no Docker, cloud,
REST APIs or monitoring yet.

---

## 0. One-time setup

1. **GitHub repo**: https://github.com/abdul-raheem-fast/autoscale-rl
   - either clone it and keep working there, or push the local repo from your machine.
2. **DagsHub repo** (same name): https://dagshub.com/abdul-raheem-fast/autoscale-rl
   - create it, then grab an access token (User Settings → Token).
3. Local setup (per machine):
   ```bash
   python -m venv .venv && source .venv/bin/activate
   python -m pip install -r requirements.txt && python -m pip install -e . --no-deps
   cp .env.example .env        # fill in DAGSHUB_USER / DAGSHUB_TOKEN
   python scripts/setup_dvc_remote.py
   ```

## 1. Environment variables (needed by MLflow + DVC)

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/abdul-raheem-fast/autoscale-rl.mlflow
export MLFLOW_TRACKING_USERNAME=abdul-raheem-fast
export MLFLOW_TRACKING_PASSWORD=<dagshub-token>
# (scripts/setup_dvc_remote.py sets the DVC remote credentials in .dvc/config.local)
```

## 2. Run the pipeline

```bash
dvc repro         # ~10 min first run: trace download → … → registry registration
dvc push          # versioned data + model -> DagsHub
git add dvc.lock .dvc && git commit -m "pipeline: reproducible core build"
```

## 3. Verify

- [ ] `dvc dag` prints the full DAG with all named stages
- [ ] `reports/data_validation.json` says `"passed": true`
- [ ] `reports/comparison.json` has threshold / random / dqn metrics on held-out days
- [ ] DagsHub MLflow: ≥ 2 experiment runs visible (8 expected: ingest, extract,
      validation, env-config, baseline ×2, train, eval)
- [ ] DagsHub MLflow: `dqn-train` run shows per-episode reward curve metrics
- [ ] DagsHub MLflow Model Registry: **AutoScaleRLPolicy** registered, stage = **Staging**
- [ ] `dvc remote list` shows DagsHub; `dvc status` clean after `dvc push`
- [ ] `pytest` green
- [ ] git log shows commits from all three group members

## 4. Screenshots needed for the PDF submission

1. DagsHub MLflow — experiment page listing ALL runs (min. 2 required).
2. DagsHub MLflow — `dqn-train` run page: parameters + per-episode reward metrics plot.
3. DagsHub MLflow — Model Registry: `AutoScaleRLPolicy` · version · **Staging**.
4. `dvc dag` output (terminal).
5. (Bonus) `dvc plots show reports/episode_log.csv -x episode -y episode_reward` CLI rendering.

## 5. Commit plan (every member needs commits)

1. **Abdul Raheem (repo lead)** — repo scaffold, README, .gitignore, params/DVC wiring, ingestion + validation modules, docs.
2. **Hayyan Haider** — simulator + env-config derivation + policy baseline modules + tests (`test_simulator.py`, `test_policies.py`).
3. **Bilal Kashif** — DQN implementation, training/evaluation stages, MLflow logging & registry integration, smoke-run verification.

Each member: commit your own module(s) with meaningful messages (e.g.
`feat(sim): container-sim with cost/latency/stability reward`), push today.

## 6. Submission doc (3–4 pages) — outline in docs/submission_outline.md
