"""Stage 7 — evaluate: held-out day comparison + MLflow Model Registry.

Replays the held-out (eval) days NEVER seen during training and evaluates:
  threshold (rule-based baseline)
  random    (no autoscaling floor)
  dqn       (trained Double-DQN, greedy)

Produces reports/comparison.json (+ plot), an eval-scoped RL metrics JSON,
per-minute CSVs, and registers the policy checkpoint in the MLflow Model
Registry transitioning it to `Staging` (params.mlflow.stage_after_register).
"""
from __future__ import annotations

import json

import pandas as pd

from .config import load_params, out_path, save_json, set_seeds
from .context import eval_days, load_context
from .dqn import DQNAgent
from .evaluation import evaluate_policy, make_env_factory
from .mlflow_utils import (configure_mlflow, log_artifact, log_params_flat,
                            log_summary_metrics, start_run)
from .policies import GreedyPolicy, RandomPolicy, ThresholdPolicy
from .registry import register_and_stage


def relative_vs(policy_metrics: dict) -> dict:
    """RL savings vs each baseline (positive = better)."""
    rl = policy_metrics["dqn"]
    out = {}
    for name, m in policy_metrics.items():
        if name == "dqn":
            continue
        out[f"vs_{name}"] = {
            "reward_delta_pct": 100.0 * (rl["mean_reward"] - m["mean_reward"])
            / max(1e-9, abs(m["mean_reward"])),
            "cost_saved_pct": 100.0 * (m["total_cost_units"] - rl["total_cost_units"])
            / max(1e-9, m["total_cost_units"]),
            "violations_added": rl["total_violations_min"] - m["total_violations_min"],
            "actions_saved": m["scale_actions_total"] - rl["scale_actions_total"],
        }
    return out


def build_comparison_plot(policy_metrics: dict, out_png) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(policy_metrics.keys())
    fields = [("mean_reward", "mean episode reward"),
              ("total_cost_units", "total cost units"),
              ("total_violations_min", "SLA violations (min)"),
              ("scale_actions_total", "scaling actions")]
    fig, axes = plt.subplots(1, len(fields), figsize=(4 * len(fields), 3.4))
    for ax, (field, title) in zip(axes, fields):
        vals = [policy_metrics[n][field] for n in names]
        ax.bar(names, vals, color=["#888", "#c33", "#2b7"])
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Held-out evaluation — DQN vs baselines")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> None:
    params, trace, envp, sim_cfg = load_context()
    set_seeds(int(params["seed"]))
    days = eval_days(envp, params)
    if not days:
        raise RuntimeError("no evaluation days available")
    print(f"[evaluate] held-out days: {days}")

    factory = make_env_factory(trace, sim_cfg, days,
                               steps_per_episode=sim_cfg.steps_per_episode)
    bl = params["baseline"]
    threshold = ThresholdPolicy(float(bl["scale_up_util"]), float(bl["scale_down_util"]),
                                int(bl["cooldown_minutes"]), envp["min_instances"],
                                envp["max_instances"])
    random_pol = RandomPolicy(seed=int(params["seed"]))

    blob = json.loads(out_path(params, "train_meta").read_text(encoding="utf-8"))
    train_run_id = str(blob.get("mlflow_run_id", ""))
    model_path = out_path(params, "model")
    dqn_net = DQNAgent.load_policy_net(model_path)
    dqn_policy = GreedyPolicy(dqn_net, eps=0.0, seed=int(params["seed"]))

    policy_map = {bl["name"]: threshold, bl["random_name"]: random_pol, "dqn": dqn_policy}
    policy_metrics, per_minute_frames = {}, []
    for name, policy in policy_map.items():
        metrics, _, per_min = evaluate_policy(factory, policy, len(days))
        policy_metrics[name] = metrics
        per_min["policy"] = name
        per_minute_frames.append(per_min)
        print(f"[evaluate] {name}: reward={metrics['mean_reward']:.1f} "
              f"cost={metrics['total_cost_units']:.1f} "
              f"viol={metrics['total_violations_min']} "
              f"actions={metrics['scale_actions_total']}")

    comparison = {
        "held_out_days": days,
        "metrics": policy_metrics,
        "dqn_vs_baselines": relative_vs(policy_metrics),
        "train_run_id": train_run_id,
    }
    comparison_path = out_path(params, "comparison")
    save_json(comparison_path, comparison)
    plot_path = out_path(params, "comparison_plot")
    build_comparison_plot(policy_metrics, plot_path)
    csv_combo = out_path(params, "rl_episodes")
    pd.concat(per_minute_frames).to_csv(csv_combo, index=False)
    rl_metrics_path = out_path(params, "rl_metrics")
    save_json(rl_metrics_path, {k: v for k, v in policy_metrics["dqn"].items()})

    mlflow, _ = configure_mlflow(params)
    with start_run(mlflow, run_name="rl-eval-heldout") as run:
        log_params_flat(mlflow, "config.", {
            "eval_days": days, "train_run_id": train_run_id,
            "sim.sla_latency_ms": envp["sla_latency_ms"],
            "seed": params["seed"]})
        for name, metrics in policy_metrics.items():
            log_summary_metrics(mlflow, {f"eval_{name}.{k}": v for k, v in metrics.items()})
        log_summary_metrics(mlflow, {f"dqn_savings.{k}": v["cost_saved_pct"]
                                     for k, v in comparison["dqn_vs_baselines"].items()})
        log_artifact(mlflow, str(comparison_path), artifact_dir="evaluation")
        log_artifact(mlflow, str(plot_path), artifact_dir="evaluation")
        log_artifact(mlflow, str(csv_combo), artifact_dir="evaluation")
        log_artifact(mlflow, str(rl_metrics_path), artifact_dir="evaluation")

        # ---- registry: checkpoint from the TRAIN run ----
        if train_run_id:
            status = register_and_stage(
                model_uri=f"runs:/{train_run_id}/policy/dqn_policy.pt",
                model_name=params["mlflow"]["registry_model_name"],
                stage=params["mlflow"]["stage_after_register"],
                extra_tags={"milestone": "core-pipeline", "model_type": "double-dqn",
                            "eval_mean_reward": str(policy_metrics["dqn"]["mean_reward"])})
        else:
            status = "skipped (train run id missing)"
        comparison["registration_status"] = status
        save_json(comparison_path, comparison)
    print(f"[evaluate] registry: {comparison.get('registration_status')}")
    print(f"[evaluate] comparison -> {comparison_path}")


if __name__ == "__main__":
    main()
