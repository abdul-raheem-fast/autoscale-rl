"""Shared episode evaluation: run any policy over the replay env and collect
per-episode metrics (reward, cost, SLA violations, p95 latency, actions).

This is what "both baseline and main model are evaluated" concretely means:
the same evaluate_policy() runs for every policy, so comparisons are fair.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .simulator import SimConfig, TrafficEnv, day_frames_from_df


def make_env_factory(df: pd.DataFrame, sim_cfg: SimConfig, day_indices,
                     steps_per_episode: int | None = None):
    """Return factory(ep_index) -> (env, day, seed).

    Calendar is deterministic: ep i replays day_indices[i % len], seeded with
    100003 + 7919*i so every replay is reproducible across runs and machines.
    """
    day_indices = [int(d) for d in day_indices]
    frames = day_frames_from_df(df, day_indices)
    if steps_per_episode:
        sim_cfg.steps_per_episode = int(steps_per_episode)

    def factory(ep_index: int) -> tuple:
        day = day_indices[ep_index % len(day_indices)]
        seed = 100003 + 7919 * ep_index
        env = TrafficEnv({day: frames[day]}, sim_cfg)
        return env, day, seed

    return factory


def evaluate_policy(factory, policy, n_episodes: int) -> tuple:
    """Run `policy` for n_episodes. Returns (metrics, per_episode_df, per_minute_df).

    Requires the policy to expose: act(obs, info), reset().
    """
    rows, minute_rows = [], []
    for ep in range(n_episodes):
        env, day, seed = factory(ep)
        obs, info = env.reset(day=day, seed=seed)
        policy.reset()
        total_reward, steps, lat_sum = 0.0, 0, 0.0
        while True:
            action = policy.act(obs, info)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            lat_sum += float(info["p95_latency"])
            steps += 1
            minute_rows.append({
                "policy": policy.name, "episode": ep, "day": int(day),
                "minute": int(info["minute"]), "arrivals": info["arrivals"],
                "p95_latency": info["p95_latency"],
                "utilization": info["utilization"], "queue": info["queue"],
                "instances": info["instances"], "reward": reward,
                "violation": info["violation"],
            })
            if done:
                break
        totals = env.totals
        rows.append({
            "policy": policy.name, "episode": ep, "day": int(day),
            "reward": total_reward,
            "cost_units": totals["cost_units"],
            "violations_min": totals["violations_min"],
            "viol_excess": totals["viol_excess"],
            "p95_mean_ms": lat_sum / max(1, steps),
            "p95_peak_ms": float(np.max(env.telemetry.latency_p95)) if steps else 0.0,
            "scale_actions": totals["scale_actions"],
            "abs_scale_delta": totals["abs_scale_delta"],
            "avg_instances": float(np.mean(env.telemetry.instances)),
            "steps": steps,
        })
    per_episode = pd.DataFrame(rows)
    per_minute = pd.DataFrame(minute_rows)

    metrics = {
        "n_episodes": int(n_episodes),
        "mean_reward": float(per_episode["reward"].mean()),
        "total_cost_units": float(per_episode["cost_units"].sum()),
        "mean_cost_units_per_episode": float(per_episode["cost_units"].mean()),
        "total_violations_min": int(per_episode["violations_min"].sum()),
        "total_viol_excess": float(per_episode["viol_excess"].sum()),
        "p95_mean_ms": float(per_minute["p95_latency"].mean()) if len(per_minute) else 0.0,
        "p95_peak_ms": float(per_episode["p95_peak_ms"].max()) if len(per_episode) else 0.0,
        "sla_compliance_pct": 100.0 * (1.0 - float(per_minute["violation"].mean())
                                       if len(per_minute) else 1.0),
        "scale_actions_total": int(per_episode["scale_actions"].sum()),
        "avg_instances": float(per_episode["avg_instances"].mean()),
    }
    return metrics, per_episode, per_minute
