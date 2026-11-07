"""Stage 4 — build the simulator configuration from the (validated) trace.

This is the RL-equivalent of "schema inference": the container-sim needs its
traffic, capacity and latency budget derived from data instead of magic
numbers. Everything it emits lands in configs/env_params.json, which is
DVC-tracked — i.e. the simulator parameters are versioned exactly like data.

Derived:
  p99/mean traffic stats, burstiness index,
  max_instances  = ceil(p99_rate / service_rate * 1.25), clamped,
  train / eval day indices (calendar-ordered split),
  cost unit definitions (documented in params.yaml>env_config).
"""
from __future__ import annotations

import math

import pandas as pd

from .config import load_params, out_path, set_seeds, save_json
from .mlflow_utils import (configure_mlflow, log_artifact, log_params_flat,
                            log_summary_metrics, start_run)
from .validate import train_eval_split


def build_env_params(df: pd.DataFrame, params: dict) -> dict:
    cfg = params["env_config"]
    svc = float(cfg["service_rate_per_instance"])
    req = df["requests"]
    mean_rate = float(req.mean())
    p99_rate = float(req.quantile(0.99))
    max_rate = int(req.max())

    derived_max_instances = int(
        math.ceil(min(cfg["max_instances_cap"],
                      max(cfg["min_instances"] + 1, p99_rate * 1.25 / svc))))

    train_days_raw, eval_days_raw = train_eval_split(df, float(params["validate"]["train_frac"]))
    # cap episode count if someone eval-days > available
    eval_days = [d for d in eval_days_raw[: int(cfg["eval_days"])]]
    train_days = [d for d in train_days_raw]

    return {
        "name": "autoscale-rl-sim",
        "source_trace": params["ingest"]["trace_name"],
        "service_rate_per_instance": svc,
        "min_instances": int(cfg["min_instances"]),
        "max_instances": derived_max_instances,
        "sla_latency_ms": float(cfg["sla_latency_ms"]),
        "base_latency_ms": float(cfg["base_latency_ms"]),
        "traffic": {
            "mean_rate_per_minute": mean_rate,
            "p99_rate_per_minute": p99_rate,
            "max_rate_per_minute": max_rate,
            "burstiness_p99_over_mean": float(p99_rate / mean_rate),
        },
        "episodes": {
            "steps_per_episode": 1440,     # one replayed day
            "train_days": [int(d) for d in train_days],
            "eval_days": [int(d) for d in eval_days],
        },
        "note": "Derived from the trace's actual peak demand so the environment "
                "asks the agent for realistic scale range.",
    }


def main() -> None:
    params = load_params()
    set_seeds(int(params["seed"]))
    df = pd.read_csv(out_path(params, "processed"), parse_dates=["timestamp"])
    envp = build_env_params(df, params)

    out = out_path(params, "env_params")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_json(out, envp)
    print(f"[env-config] max_instances={envp['max_instances']} "
          f"p99={envp['traffic']['p99_rate_per_minute']:.0f}/min "
          f"mean={envp['traffic']['mean_rate_per_minute']:.1f}/min "
          f"train_days={len(envp['episodes']['train_days'])} "
          f"eval_days={len(envp['episodes']['eval_days'])}")

    mlflow, _ = configure_mlflow(params)
    with start_run(mlflow, run_name="env-config-build") as run:
        log_params_flat(mlflow, "env_config.", params["env_config"])
        log_summary_metrics(mlflow, {
            "max_instances": envp["max_instances"],
            "p99_rate": envp["traffic"]["p99_rate_per_minute"],
            "mean_rate": envp["traffic"]["mean_rate_per_minute"],
            "burstiness": envp["traffic"]["burstiness_p99_over_mean"],
            "train_days": len(envp["episodes"]["train_days"]),
            "eval_days": len(envp["episodes"]["eval_days"]),
        })
        log_artifact(mlflow, str(out), artifact_dir="configs")
    print(f"[env-config] MLflow run: {run.info.run_id}")


if __name__ == "__main__":
    main()
