"""Stage 6 — train_dqn: Double DQN training loop (from scratch, PyTorch).

Per-episode metrics are streamed to MLflow (reward curve, cost, SLA
violations, epsilon, loss, ...) for the whole run — satisfying "reward curves
and key metrics logged per episode". Final checkpoint + episode log + reward
plot are logged as artifacts. The checkpoint is the artifact that gets
REGISTERED in the MLflow Model Registry by stage 7.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_params, out_path, save_json, set_seeds, smoke
from .context import eval_days, load_context, training_days
from .dqn import DQNAgent, compute_eps
from .evaluation import evaluate_policy, make_env_factory
from .mlflow_utils import (configure_mlflow, log_artifact, log_int_metric,
                            log_params_flat, log_summary_metrics, start_run)
from .policies import GreedyPolicy

STATE_DIM = 7   # rate, p95, util, queue, instances, sin(tod), cos(tod)
ACTION_DIM = 3  # down / hold / up


def reward_plot(episode_log: pd.DataFrame, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rewards = episode_log["episode_reward"].to_numpy(dtype=float)
    window = max(1, len(rewards) // 8)
    acc = np.convolve(rewards, np.ones(window) / window, mode="valid")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, alpha=0.35, label="episode reward")
    ax.plot(range(window - 1, len(rewards)), acc, lw=2, label=f"moving avg({window})")
    ax.set_xlabel("episode")
    ax.set_ylabel("total reward")
    ax.set_title("Double-DQN training — reward curve")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def train_dqn(params, trace, envp, sim_cfg) -> dict:
    dqn_cfg = params["dqn"]
    seed = int(params["seed"])
    days = training_days(envp, params)
    if not days:
        raise RuntimeError("no training days")

    episodes = int(dqn_cfg["episodes"])
    if smoke():
        episodes = 3
    factory = make_env_factory(trace, sim_cfg, days)
    total_steps = episodes * sim_cfg.steps_per_episode

    agent = DQNAgent(STATE_DIM, ACTION_DIM, dqn_cfg, seed=seed)
    uevery = max(1, int(dqn_cfg["update_every"]))
    sync_every_learner = max(1, int(dqn_cfg["target_sync_steps"]) // uevery)

    rows, global_step = [], 0
    mlflow, exp_name = configure_mlflow(params)
    with start_run(mlflow, run_name="dqn-train") as run:
        log_params_flat(mlflow, "config.", {
            **{f"dqn.{k}": v for k, v in dqn_cfg.items()},
            **{f"sim.{k}": v for k, v in params["sim"].items()},
            "env.max_instances": envp["max_instances"],
            "env.sla_latency_ms": envp["sla_latency_ms"],
            "env.train_days": len(days),
            "seed": seed,
            "state_dim": STATE_DIM,
        })

        for ep in range(episodes):
            env, day, ep_seed = factory(ep)
            obs, info = env.reset(day=day, seed=ep_seed)
            losses, steps = [], 0
            done = False
            while not done:
                eps = compute_eps(global_step, total_steps, dqn_cfg["eps_start"],
                                  dqn_cfg["eps_end"], float(dqn_cfg["eps_decay_frac"]))
                action = agent.act(obs, eps)
                next_obs, reward, done, info = env.step(action)
                agent.remember(obs, action, reward, next_obs, done)
                obs = next_obs
                global_step += 1
                steps += 1
                if global_step % uevery == 0:
                    loss = agent.learn()
                    if loss is not None:
                        losses.append(loss)
                    if agent.learner_steps % sync_every_learner == 0:
                        agent.sync_target()

                if smoke() and steps >= 400:
                    break
            totals = env.totals
            tlat = env.telemetry.latency_p95
            eps_now = compute_eps(global_step, total_steps, dqn_cfg["eps_start"],
                                  dqn_cfg["eps_end"], float(dqn_cfg["eps_decay_frac"]))
            row = {
                "episode": ep, "day": int(day), "steps": steps,
                "episode_reward": totals["reward"],
                "episode_cost_units": totals["cost_units"],
                "episode_violations_min": totals["violations_min"],
                "episode_viol_excess": totals["viol_excess"],
                "episode_p95_mean_ms": float(np.mean(tlat)) if tlat else 0.0,
                "episode_scale_actions": totals["scale_actions"],
                "episode_avg_instances": float(np.mean(env.telemetry.instances)),
                "epsilon": eps_now,
                "episode_loss": float(np.mean(losses)) if losses else 0.0,
                "learner_steps": agent.learner_steps,
            }
            rows.append(row)
            for key, val in row.items():
                if key in ("episode", "day", "steps", "learner_steps"):
                    continue
                log_int_metric(mlflow, key, step=ep, value=val)

        episode_log = pd.DataFrame(rows)
        model_path = out_path(params, "model")
        ckpt_meta = {
            "mlflow_run_id": run.info.run_id,
            "episodes": episodes,
            "total_env_steps": global_step,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "seed": seed,
            "train_days": days,
            "final_epsilon": rows[-1]["epsilon"] if rows else dqn_cfg["eps_start"],
        }
        agent.save(model_path, {"mlflow_run_id": run.info.run_id, "seed": seed,
                                 "train_days": days})

        episode_csv = out_path(params, "episode_log")
        episode_csv.parent.mkdir(parents=True, exist_ok=True)
        episode_log.to_csv(episode_csv, index=False)
        curve_png = out_path(params, "reward_curve")
        reward_plot(episode_log, curve_png)
        train_meta_path = out_path(params, "train_meta")

        # held-out sanity: greedy policy on one eval day (preview of stage 7)
        eval_ids = eval_days(envp, params)
        heldout = {}
        if eval_ids:
            efactory = make_env_factory(trace, sim_cfg, eval_ids[:1], steps_per_episode=sim_cfg.steps_per_episode)
            greedy = GreedyPolicy(agent.net, eps=0.0, seed=seed)
            hmetrics, _, _ = evaluate_policy(efactory, greedy, 1)
            heldout = hmetrics
            log_summary_metrics(mlflow, {f"heldout_{k}": v for k, v in hmetrics.items()})

        save_json(train_meta_path, {**ckpt_meta, "heldout_ep0": heldout})
        log_artifact(mlflow, str(model_path), artifact_dir="policy")
        log_artifact(mlflow, str(episode_csv), artifact_dir="training")
        log_artifact(mlflow, str(curve_png), artifact_dir="training")
        log_artifact(mlflow, str(train_meta_path), artifact_dir="training")
        log_summary_metrics(mlflow, {k: v for k, v in rows[-1].items()
                                     if isinstance(v, (int, float))})
        return {"run_id": run.info.run_id, "episode_log": episode_log,
                "model_path": model_path, "train_meta_path": train_meta_path}


def main() -> None:
    params, trace, envp, sim_cfg = load_context()
    set_seeds(int(params["seed"]))
    out = train_dqn(params, trace, envp, sim_cfg)
    print(f"[train] done — MLflow run {out['run_id']}, model at {out['model_path']}")


if __name__ == "__main__":
    main()
