"""Stage 5 — run_baseline: evaluate the rule-based + random policies.

Two MLflow runs (baseline-threshold, baseline-random) are produced here on
the TRAIN replay window with per-episode metrics + artifacts. This is the
baseline half of the model requirement; stage 7 compares the DQN against
exactly these metrics on held-out days.
"""
from __future__ import annotations

import pandas as pd

from .config import load_params, out_path, save_json, set_seeds, smoke
from .context import load_context, training_days
from .evaluation import evaluate_policy, make_env_factory
from .mlflow_utils import (configure_mlflow, log_artifact, log_params_flat,
                            log_summary_metrics, start_run)
from .policies import RandomPolicy, ThresholdPolicy


def run_baselines(params, trace, envp, sim_cfg, factory) -> dict:
    bl = params["baseline"]
    df_frames = pd.DataFrame()
    metrics_out = {}
    policies = {
        bl["name"]: ThresholdPolicy(
            scale_up_util=float(bl["scale_up_util"]),
            scale_down_util=float(bl["scale_down_util"]),
            cooldown_minutes=int(bl["cooldown_minutes"]),
            min_instances=envp["min_instances"],
            max_instances=envp["max_instances"]),
        # unscaled floor: what 'never autoscale with discipline' looks like
        bl["random_name"]: RandomPolicy(seed=int(params["seed"])),
    }
    mlflow, _ = configure_mlflow(params)
    for name, policy in policies.items():
        metrics, per_ep, per_min = evaluate_policy(factory, policy, factory.n_episodes)
        metrics_out[name] = metrics
        with start_run(mlflow, run_name=f"baseline-{name}") as run:
            log_params_flat(mlflow, "config.", {
                "baseline.name": name,
                "baseline.scale_up_util": bl["scale_up_util"],
                "baseline.scale_down_util": bl["scale_down_util"],
                "baseline.cooldown_minutes": bl["cooldown_minutes"],
                "sim.cost_weight": params["sim"]["cost_weight"],
                "sim.sla_weight": params["sim"]["sla_weight"],
                "sim.change_weight": params["sim"]["change_weight"],
                "env.max_instances": envp["max_instances"],
                "env.service_rate": envp["service_rate_per_instance"],
                "env.sla_latency_ms": envp["sla_latency_ms"],
                "seed": params["seed"],
            })
            log_summary_metrics(mlflow, metrics)
            ep_csv = out_path(params, "baseline_episodes")
            per_ep.to_csv(ep_csv, index=False)
            log_artifact(mlflow, str(ep_csv), artifact_dir="baselines")
        print(f"[baseline] {name}: reward={metrics['mean_reward']:.1f} "
              f"cost={metrics['total_cost_units']:.1f} "
              f"viol_min={metrics['total_violations_min']} "
              f"actions={metrics['scale_actions_total']} (run {run.info.run_id})")
    return metrics_out


def main() -> None:
    params, trace, envp, sim_cfg = load_context()
    set_seeds(int(params["seed"]))
    bl = params["baseline"]
    days = training_days(envp, params)[: (2 if smoke() else int(bl["max_train_episodes"]))]
    if not days:
        raise RuntimeError("no training days available for baseline")

    from .evaluation import make_env_factory as _f  # see eval module limits
    factory = _f(trace, sim_cfg, days)
    # deterministic episode count for the baseline window
    factory.n_episodes = len(days)

    metrics = run_baselines(params, trace, envp, sim_cfg, factory)
    metrics_path = out_path(params, "baseline_metrics")
    save_json(metrics_path, metrics)
    print(f"[baseline] metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
