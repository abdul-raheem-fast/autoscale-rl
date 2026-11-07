"""Scalable policies: rule-based baseline, random floor, greedy RL policy.

All policies share the same interface: `act(obs, info) -> action` with
action in {0=scale down, 1=hold, 2=scale up}, mirroring the proposal's
action space. The ~Cloud Run default behaviour is the threshold policy:
Google's default autoscaler also reacts to per-instance load thresholds.
"""
from __future__ import annotations

import random

import numpy as np


class ThresholdPolicy:
    """Rule-based autoscaler: utilisation thresholds with a cooldown.

    Matches the proposal's baseline ("add an instance when pH load exceeds
    65%, drop below 25%, don't flap faster than `cooldown` minutes").
    """

    name = "threshold"

    def __init__(self, scale_up_util: float = 0.65, scale_down_util: float = 0.25,
                 cooldown_minutes: int = 10, min_instances: int = 1,
                 max_instances: int = 24):
        self.up = scale_up_util
        self.down = scale_down_util
        self.cooldown = cooldown_minutes
        self.min_instances = min_instances
        self.max_instances = max_instances
        self._cool_until = -1
        self._t = -1

    def reset(self) -> None:
        self._cool_until = -1
        self._t = -1

    def act(self, obs: np.ndarray, info: dict) -> int:
        self._t += 1
        util = info.get("utilization")
        if util is None:
            util = float(obs[2]) if obs is not None and len(obs) > 2 else 0.0
        instances = int(info.get("instances", 1))
        if self._t < self._cool_until:
            return 1
        if util > self.up and instances < self.max_instances:
            self._cool_until = self._t + self.cooldown
            return 2
        if util < self.down and instances > self.min_instances:
            self._cool_until = self._t + self.cooldown
            return 0
        return 1


class RandomPolicy:
    """Unscaled floor: shows what happens with no autoscaling discipline.
    Seeded so results are reproducible."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, info: dict) -> int:
        return self.rng.choice([0, 1, 2])


class GreedyPolicy:
    """Wraps a DQN-style Q-network for deterministic (greedy) evaluation."""

    name = "dqn-greedy"

    def __init__(self, net, eps: float = 0.0, seed: int = 0):
        self.net = net.eval()
        self.eps = eps
        self.rng = random.Random(seed)

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, info: dict) -> int:
        import torch
        if self.eps > 0 and self.rng.random() < self.eps:
            return self.rng.randrange(self.net.action_dim)
        with torch.no_grad():
            q = self.net(torch.from_numpy(obs.astype(np.float32)).unsqueeze(0))
            return int(torch.argmax(q, dim=1).item())
