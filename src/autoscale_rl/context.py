"""Shared context loader for pipeline stages: params, trace data, env config."""
from __future__ import annotations

import json

import pandas as pd

from .config import load_params, out_path, smoke
from .simulator import SimConfig


def load_context():
    """Load (params, trace_df, env_params_dict, sim_cfg). Applies smoke scaling."""
    params = load_params()
    trace = pd.read_csv(out_path(params, "processed"), parse_dates=["timestamp"])
    envp = json.loads(out_path(params, "env_params").read_text(encoding="utf-8"))
    sim_cfg = SimConfig.from_env_params(envp, params["sim"])

    if smoke():
        print("[smoke] AUTOSCALE_SMOKE=1 -> shrinking episode lengths & counts")
        sim_cfg.steps_per_episode = int(min(720, sim_cfg.steps_per_episode))
        envp["episodes"]["steps_per_episode"] = sim_cfg.steps_per_episode
    return params, trace, envp, sim_cfg


def training_days(envp: dict, params: dict) -> list:
    days = [int(d) for d in envp["episodes"]["train_days"]]
    if smoke():
        days = days[:3]
    return days


def eval_days(envp: dict, params: dict) -> list:
    days = [int(d) for d in envp["episodes"]["eval_days"]]
    limit = int(params["eval"]["episodes"])
    if smoke():
        limit = min(limit, 2)
    return days[:limit]
