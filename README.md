# AutoScale-RL

**Safe reinforcement-learning autoscaling for containerized services — trained from scratch in a traffic simulator, engineered with a full MLOps pipeline.**

MLOps course project (Fall 2026) · BCS-7A · Group members: **M. Abdul Raheem (23L-0588)**, **M. Hayyan Haider (23L-0727)**, **Bilal Kashif (23L-0757)**

> Milestone 1 scope: **core ML pipeline, fully reproducible** — no Docker, no cloud deployment, no REST APIs, no monitoring yet. Those come in later milestones.

---

## 1. What this repo does

Real autoscalers are rule-based ("add a replica when CPU > 70%"). They react late and cannot trade cost against latency. **AutoScale-RL** treats scaling as a *sequential decision problem*: a Double-DQN agent, implemented from scratch in PyTorch (no RL libraries, no pretrained models), decides every minute whether a containerized web service should **scale down / hold / scale up**. It is trained inside a container traffic simulator that replays the **NASA HTTP July-1995 trace** (a classic public web-trace dataset from the Internet Traffic Archive), with a reward that balances:

- ☁️ **cost** — number of active instances over time
- ⏱️ **SLA violations** — minutes with p95 latency above budget
- 🔄 **instability** — number of scale-up/down actions (flapping)

The agent is compared against **two baselines** (a rule-based threshold autoscaler — the "Cloud-Run-default"-style behaviour — and a random floor) on **held-out calendar days** it never saw.

### Pipeline at a glance

```
                 params.yaml (single source of truth)
                        │
   ingest ──► extract ──► validate ──► env-config ──► ┬─► baseline (MLflow ×2)
   (NASA trace  log →     schema/nulls/   derived sim  │
    pinned   per-min      ranges/gaps  parameters     └─► train_dqn (MLflow, per-episode
    snapshot) RDDs            │        + splits            reward curves) ──► evaluate
                              ▼                                   (held-out days,
                 reports/data_validation.json                register model → Staging)
```

- **DVC** versions everything: the ingested trace, the per-minute dataset, the derived simulator config, the trained checkpoint, and the reports. `dvc repro` rebuilds all of it from zero.
- **MLflow (hosted on DagsHub)** logs every run — ingestion, extraction, validation, baselines, training (per-episode reward curves), evaluation — and stores the registered policy in the **Model Registry (Staging)**.

---

## 2. Repository hosting

| What | Where |
|---|---|
| GitHub | <https://github.com/abdul-raheem-fast/autoscale-rl> |
| DagsHub (DVC remote + MLflow tracking) | <https://dagshub.com/abdul-raheem-fast/autoscale-rl> |

> NOTE: raw data and model weights are **never committed to git** — they are DVC-managed and pushed to the DagsHub remote (git history is clean).

---

## 3. Setup

Prereqs: **Python 3.10–3.12**, **Git**, a local clone of this repo, and a DagsHub access token (DagsHub → Settings → Tokens).

```bash
# 1. environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

# 2. credentials — create a .env from the template (never committed)
cp .env.example .env                 # Windows: copy .env.example .env
#   edit .env: DAGSHUB_USER, DAGSHUB_TOKEN, DAGSHUB_REPO, MLFLOW_TRACKING_*

# 3. point DVC at DagsHub (creds go to .dvc/config.local — gitignored)
python scripts/setup_dvc_remote.py
```

 PowerShell users: `$events = Get-Content .env` … or simply export the variables manually:

```powershell
$env:DAGSHUB_USER="abdul-raheem-fast"; $env:DAGSHUB_TOKEN="<token>"
$env:MLFLOW_TRACKING_URI="https://dagshub.com/abdul-raheem-fast/autoscale-rl.mlflow"
$env:MLFLOW_TRACKING_USERNAME=$env:DAGSHUB_USER; $env:MLFLOW_TRACKING_PASSWORD=$env:DAGSHUB_TOKEN
```

---

## 4. How to run

```bash
# full pipeline from scratch (trace download → validate → baseline → DQN → evaluate → registry)
dvc repro

# view the DAG / show metrics & plots
dvc dag
dvc metrics show reports/comparison.json -f metrics     # or: dvc experiments show
dvc plots show reports/episode_log.csv -x episode -y episode_reward   # rl reward curve

# push versioned data + models to DagsHub
dvc push
```

Sanity checks:

```bash
pytest            # unit tests: simulator determinism, DQN, validation, policies
```

**Smoke run (seconds instead of minutes)** — for quick CI-style iteration:

```bash
AUTOSCALE_SMOKE=1 dvc repro     # Windows: $env:AUTOSCALE_SMOKE="1"; dvc repro
```

> Note: without `MLFLOW_TRACKING_URI` the code logs to a local `mlruns/` fallback (tagged `tracking.server=fallback-local`) and **skips model registration** — for the milestone submission always run with the DagsHub tracking URI so the hosted runs/registry are populated.

---

## 5. Folder structure

```
autoscale-rl/
├── params.yaml               # every stage's config; the "single source of truth"
├── dvc.yaml                  # named DVC stages; dvc repro runs the whole DAG
├── dvc.lock                  # md5s of data/params/code per stage (committed)
├── .dvc/config               # DVC remote (DagsHub) — machine-specific creds in config.local
│
├── src/autoscale_rl/         # installed as an editable package
│   ├── ingest.py             # S1: pinned NASA trace download (+ honest synthetic fallback)
│   ├── extract.py            # S2: raw log → per-minute request counts
│   ├── validate.py           # S3: schema/null/range/continuity gates (fails the DAG)
│   ├── env_config.py         # S4: derive simulator params from the trace
│   ├── baseline.py           # S5: threshold + random baselines (2 MLflow runs)
│   ├── dqn.py                # Double DQN from scratch (Q-net, replay, targets)
│   ├── train.py              # S6: training; per-episode MLflow metrics/artifacts
│   ├── evaluate.py           # S7: held-out comparison; registry registration
│   ├── registry.py           # MLflow Model Registry helper (Staging transition)
│   ├── simulator.py          # container-sim: arrivals, latency curve, queue, reward
│   ├── policies.py           # threshold / random / greedy-RL policies
│   ├── evaluation.py         # shared episode-evaluation harness
│   ├── context.py            # one loader for trace+config used by all stages
│   └── mlflow_utils.py       # DagsHub tracking, param/metric/artifact logging
│
├── tests/                    # pytest: simulator determinism, DQN, validation, policies
├── scripts/setup_dvc_remote.py
├── docs/                     # milestone checklist + submission PDF outline
├── data/  ├── models/  └── reports/   # DVC-outs (gitignored, DVC-versioned)
```

---

## 6. Data

- **Source**: *NASA-HTTP* server access log, July 1995 (Internet Traffic Archive) — pinned to an immutable Wayback Machine snapshot so every machine downloads byte-identical data (sha256 recorded in `data/raw/trace_meta.json`).
- **Extraction**: ~1.9 M log lines → **39,531 minute rows over 28 days** (requests per minute), aggregated by minute of day + calendar day index.
- **Validation** happens **before** any training: schema, nulls, non-negative bounded rates, monotonic unique timestamps, minute-continuity, day coverage, and a calendar-ordered train/eval split. Any failure stops `dvc repro` (**fail fast**).
- **Splits**: first 80% of days → training episodes; last 7 days → held-out evaluation (never seen in training).
- Fully offline? `ingest.py` falls back to a seeded synthetic trace **in the same format**, honestly labelled `source=synthetic-fallback`, so the pipeline stays runnable anywhere.

---

## 7. MLflow (DagsHub-hosted)

- Tracking URI: `https://dagshub.com/abdul-raheem-fast/autoscale-rl.mlflow`
- Experiment: **`autoscale-rl`** — every stage logs a run (`data-ingest`, `trace-extract`, `data-validation`, `env-config-build`, `baseline-threshold`, `baseline-random`, `dqn-train`, `rl-eval-heldout`).
- **Per-episode**: `episode_reward`, `episode_cost_units`, `episode_violations_min`, `episode_p95_mean_ms`, `episode_scale_actions`, `episode_epsilon`, `episode_loss` … plus hyperparameters for every run and artifacts (rendered reward curve, episode log, checkpoint).
- **Model Registry**: the trained policy checkpoint is registered as [`AutoScaleRLPolicy`](https://dagshub.com/abdul-raheem-fast/autoscale-rl.mlflow/#/models) with the stage set to **Staging** (+ tags incl. held-out evaluation metrics).

---

## 8. Roadmap (later milestones — intentionally out of scope here)

| Milestone | Items |
|---|---|
| 2 | PPO as the second main model; safety-governor wrapper (rule-based fallback on risky actions); shadow-mode promotion gate |
| 3 | FastAPI policy-decision service, LoCust load tests, demo service, Docker + Cloud Run deployment (CI/CD with eval gate) |
| 4 | Monitoring dashboards/alerts; retraining on logged real decisions; final report comparing sim vs cloud results |

---

## 9. License

MIT (course-project code).
