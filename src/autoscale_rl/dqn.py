"""Double DQN — implemented from scratch in PyTorch (no RL libraries).

Components:
  QNet          MLP state->Q-values (per action: 0=down 1=hold 2=up)
  ReplayBuffer  fixed-size circular numpy buffer
  DQNAgent      ε-greedy acting, Double-DQN targets
                  y = r + γ · Q_target(s', argmax_a' Q_online(s', a'))
                AdamW + gradient clipping, periodic target sync.

Deterministic given (seed, data): seeds cover torch, numpy and python RNG.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes):
        super().__init__()
        layers, prev = [], state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)
        self.action_dim = int(action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.pos, self.size = 0, 0

    def __len__(self):
        return self.size

    def add(self, obs, action, reward, next_obs, done) -> None:
        i = self.pos
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = 1.0 if done else 0.0
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int):
        idx = np.random.randint(0, self.size, size=batch)
        return (torch.from_numpy(self.obs[idx]), torch.from_numpy(self.actions[idx]),
                torch.from_numpy(self.rewards[idx]), torch.from_numpy(self.next_obs[idx]),
                torch.from_numpy(self.dones[idx]))


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: dict,
                 device: str = "cpu", seed: int = 0):
        self.device = torch.device(device)
        self.cfg = cfg
        self.action_dim = action_dim
        hidden = cfg.get("hidden_sizes", [128, 128])
        self.net = QNet(state_dim, action_dim, hidden).to(self.device)
        self.target = QNet(state_dim, action_dim, hidden).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=float(cfg["lr"]))
        self.buffer = ReplayBuffer(int(cfg["replay_capacity"]), state_dim)
        self.batch_size = int(cfg["batch_size"])
        self.gamma = float(cfg["gamma"])
        self.learner_steps = 0
        torch.manual_seed(seed)
        random.seed(seed)

    # ---------------------------------------------------------------- act
    def act(self, obs: np.ndarray, eps: float) -> int:
        if eps > 0 and random.random() < eps:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            q = self.net(torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device))
            return int(torch.argmax(q, dim=1).item())

    # -------------------------------------------------------------- learn
    def remember(self, obs, action, reward, next_obs, done) -> None:
        self.buffer.add(obs, action, reward, next_obs, done)

    def learn(self) -> float | None:
        if len(self.buffer) < max(self.batch_size, int(self.cfg.get("learn_start", 2000))):
            return None
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
        obs, next_obs = obs.to(self.device), next_obs.to(self.device)
        actions, rewards, dones = (actions.to(self.device), rewards.to(self.device),
                                   dones.to(self.device))

        q = self.net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        # Double-DQN: online net picks a', target net evaluates it
        with torch.no_grad():
            next_actions = self.net(next_obs).argmax(dim=1, keepdim=True)
            next_q = self.target(next_obs).gather(1, next_actions).squeeze(1)
            target = rewards + self.gamma * (1.0 - dones) * next_q

        loss = nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.optimizer.step()
        self.learner_steps += 1
        return float(loss.item())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.net.state_dict())
        self.target.eval()

    # ---------------------------------------------------------------- io
    def save(self, path: str | Path, meta: dict) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "state_dim": self.net.net[0].in_features,
            "action_dim": self.action_dim,
            "hidden_sizes": self.cfg.get("hidden_sizes", [128, 128]),
            "meta": meta,
        }, path)

    @classmethod
    def load_policy_net(cls, path: str | Path, device: str = "cpu") -> QNet:
        blob = torch.load(path, map_location=device, weights_only=False)
        net = QNet(int(blob["state_dim"]), int(blob["action_dim"]),
                   blob.get("hidden_sizes", [128, 128]))
        net.load_state_dict(blob["state_dict"])
        net.eval()
        return net


def compute_eps(step: int, total_steps: int, start: float, end: float, decay_frac: float) -> float:
    """Linear ε schedule over the first `decay_frac` of the training steps."""
    cut = max(1, int(total_steps * decay_frac))
    if step >= cut:
        return end
    return float(start + (end - start) * (step / cut))
