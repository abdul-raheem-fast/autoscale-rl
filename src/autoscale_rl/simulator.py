"""Container-sim: a Gym-style environment that replays the public web trace.

One step = one minute of simulated service. The agent observes
(request rate, p95 latency, utilisation/queue pressure, queue length, active
instances, time-of-day) and chooses scale-down / hold / scale-up. The reward
balances cost, SLA violations and scaling instability (params.yaml>sim).

Dynamics (per step, deliberately simple but realistic in shape):
  arrivals  ~ Poisson(rate_t) where rate_t is the empirical per-minute rate
              from the ingested, DVC-versioned trace.
  capacity  existing instances * service_rate  +  cold instances * 50%
            (newly added containers come online partially warmed).
  latency   M/M-queue-style response curve; explodes near saturation,
            capped at 1800 ms.
  queue     backlog decays 10%/min, grows with unserved arrivals and inflates
            the p95 latency estimate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SimConfig:
    service_rate_per_instance: float
    min_instances: int
    max_instances: int
    sla_latency_ms: float
    base_latency_ms: float
    cost_per_instance_minute: float = 0.05
    cost_weight: float = 1.0
    sla_weight: float = 3.0
    change_weight: float = 0.2
    steps_per_episode: int = 1440
    p99_rate: float = 100.0            # normalisation constant (arrivals)
    queue_norm: float = 200.0

    @classmethod
    def from_env_params(cls, envp: dict, sim_weights: dict) -> "SimConfig":
        traffic = envp["traffic"]
        return cls(
            service_rate_per_instance=float(envp["service_rate_per_instance"]),
            min_instances=int(envp["min_instances"]),
            max_instances=int(envp["max_instances"]),
            sla_latency_ms=float(envp["sla_latency_ms"]),
            base_latency_ms=float(envp["base_latency_ms"]),
            cost_per_instance_minute=float(sim_weights.get("cost_per_instance_minute", 0.05)),
            cost_weight=float(sim_weights.get("cost_weight", 1.0)),
            sla_weight=float(sim_weights.get("sla_weight", 3.0)),
            change_weight=float(sim_weights.get("change_weight", 0.2)),
            steps_per_episode=int(envp["episodes"]["steps_per_episode"]),
            p99_rate=max(1.0, float(traffic["p99_rate_per_minute"])),
            queue_norm=max(1.0, float(traffic["p99_rate_per_minute"])),
        )


@dataclass
class EpisodeTelemetry:
    arrivals: list = field(default_factory=list)
    latency_p95: list = field(default_factory=list)
    utilization: list = field(default_factory=list)
    queue: list = field(default_factory=list)
    instances: list = field(default_factory=list)
    reward: list = field(default_factory=list)

    def push(self, **vals) -> None:
        for k, v in vals.items():
            getattr(self, k).append(v)


class TrafficEnv:
    """Replays a set of calendar days from the trace, day by day."""

    def __init__(self, day_frames: dict, sim_cfg: SimConfig):
        self.days = sorted(day_frames.keys())
        self.frames = {d: np.asarray(rates, dtype=np.float64)
                       for d, rates in day_frames.items()}
        self.cfg = sim_cfg
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(self, day: int | None = None, seed: int = 0):
        """Start an episode. `day` selects the trace day (None -> first day);
        `seed` feeds the Poisson RNG — same seed & day = same episode."""
        self.day = int(day if day is not None else self.days[0])
        self.rates = self.frames[self.day]
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.instances = self.cfg.min_instances
        self.queue = 0.0
        self.telemetry = EpisodeTelemetry()
        self.totals = {"cost_units": 0.0, "violations_min": 0, "viol_excess": 0.0,
                       "scale_actions": 0, "abs_scale_delta": 0, "reward": 0.0}
        rate0 = float(self.rates[0]) if len(self.rates) else 0.0
        util = rate0 / max(1.0, self.cfg.min_instances * self.cfg.service_rate_per_instance)
        info = {"minute": 0, "instances": self.instances, "queue": self.queue,
                "rate": rate0, "utilization": min(1.0, util),
                "p95_latency": self.cfg.base_latency_ms}
        return self._obs(info), info

    def step(self, action: int):
        cfg = self.cfg
        if action not in (0, 1, 2):
            raise ValueError(f"invalid action {action}")
        target = int(min(cfg.max_instances,
                         max(cfg.min_instances, self.instances + (action - 1))))
        if target != self.instances:
            self.totals["scale_actions"] += 1
            self.totals["abs_scale_delta"] += abs(target - self.instances)

        cold = max(0, target - self.instances)      # newly adopted, half-warm
        warm = target - cold
        capacity = (warm * cfg.service_rate_per_instance
                    + cold * 0.5 * cfg.service_rate_per_instance)

        rate = float(self.rates[self.t])
        arrivals = int(self.rng.poisson(max(0.0, rate)))

        rho = arrivals / max(1.0, capacity)
        denom = max(1e-3, 1.0 - min(rho, 0.9999))
        resp = cfg.base_latency_ms * (1.0 + 2.5 * rho * rho / denom)
        resp = float(min(1800.0, max(cfg.base_latency_ms, resp)))

        p95 = resp * (1.0 + 0.3 * min(3.0, self.queue / max(1.0, arrivals)))
        self.queue = max(0.0, self.queue * 0.9 + max(0.0, arrivals - capacity))

        violation = p95 > cfg.sla_latency_ms
        excess = max(0.0, (p95 - cfg.sla_latency_ms) / cfg.sla_latency_ms)
        cost = target * cfg.cost_per_instance_minute
        reward = -(cfg.cost_weight * cost
                   + cfg.sla_weight * excess
                   + cfg.change_weight * abs(target - self.instances))

        self.instances = target
        self.totals["cost_units"] += cost
        self.totals["violations_min"] += int(violation)
        self.totals["viol_excess"] += excess
        self.totals["reward"] += reward

        self.telemetry.push(arrivals=arrivals, latency_p95=p95,
                            utilization=min(1.0, rho), queue=self.queue,
                            instances=target, reward=reward)

        info = {"minute": self.t, "arrivals": arrivals, "rate": rate,
                "p95_latency": p95, "utilization": float(min(1.0, rho)),
                "queue": self.queue, "instances": target,
                "cost_units": cost, "violation": bool(violation)}
        self.t += 1
        done = self.t >= cfg.steps_per_episode or self.t >= len(self.rates)
        return self._obs(info), float(reward), bool(done), info

    # -------------------------------------------------------- observations
    def _obs(self, info: dict) -> np.ndarray:
        """7-d observation, all bounded ~[0,1] except sin/cos time-of-day."""
        cfg = self.cfg
        rate = info.get("rate", 0.0)
        p95 = info.get("p95_latency", cfg.base_latency_ms)
        util = info.get("utilization", 0.0)
        queue = info.get("queue", 0.0)
        instances = info.get("instances", cfg.min_instances)
        tod = info.get("minute", 0) / 1440.0
        return np.array([
            min(1.0, max(0.0, rate) / cfg.p99_rate),
            min(1.0, max(0.0, p95) / (3.0 * cfg.sla_latency_ms)),
            min(1.0, max(0.0, util)),
            min(1.0, max(0.0, queue) / cfg.queue_norm),
            instances / max(1, cfg.max_instances),
            math.sin(2.0 * math.pi * tod),
            math.cos(2.0 * math.pi * tod),
        ], dtype=np.float32)


def day_frames_from_df(df: pd.DataFrame, day_indices) -> dict:
    """minute-of-day -> requests, for the requested calendar days."""
    out = {}
    for d in day_indices:
        sub = df[df["day_index"] == d].sort_values("minute_of_day")
        out[int(d)] = sub["requests"].to_numpy(dtype=np.float64)
    return out
