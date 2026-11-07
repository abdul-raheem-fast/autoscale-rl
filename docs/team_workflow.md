# Team workflow — commit plan (every member must have commits)

Rule: **each member authors and pushes their own commits** from their own machine
(or GitHub web UI). Set up git identity once:

```bash
git config --local user.name  "<Your Name>"
git config --local user.email "<the email you use on GitHub>"
```

## Commit ownership (Milestone 1)

| Member | Owns (files) | Suggested commits |
|---|---|---|
| **M. Abdul Raheem** (23L-0588, repo lead) | scaffold, README, .gitignore, requirements/pyproject, params.yaml, dvc.yaml, ingest/extract/validate, scripts/, docs/ | `feat(repo): bootstrap autoscale-rl` · `feat(data): pinned-wayback trace ingest + schema/range validation gate` · `docs: milestone checklist + submission outline` |
| **M. Hayyan Haider** (23L-0727) | simulator.py, env_config.py, policies.py, evaluation.py, tests/test_simulator.py, tests/test_policies.py | `feat(sim): container-sim with cost/latency/stability reward` · `feat(env): derive simulator config from trace stats` · `test(sim): determinism + policy behaviour` |
| **Bilal Kashif** (23L-0757) | dqn.py, train.py, evaluate.py, registry.py, mlflow_utils.py, context.py, tests/test_dqn.py | `feat(dqn): double-DQN from scratch (Q-net, replay, targets)` · `feat(mlflow): per-episode logging, wanted registry Staging flow` · `feat(eval): held-out comparison + model registration` |

## Practical tips

- Do NOT force-push over each other; rebase on `main` before pushing
  (`git pull --rebase origin main`).
- Each member runs `pytest` before their commits; green tests only.
- Each member can make ONE extra real commit tonight by:
  1. running the full `dvc repro` on their machine and committing the resulting
     `dvc.lock` fixes (`chore: lock pipeline vN`), or
  2. improving a doc/report bullet with real numbers from their run, or
  3. adding one extra unit test for their module.
- Never commit `data/`, `models/`, `reports/`, `mlruns/`, `.env` (see .gitignore);
  artifacts travel through DVC (`dvc push`).
