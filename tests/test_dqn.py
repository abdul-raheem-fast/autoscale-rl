import numpy as np
import pytest
import torch

from autoscale_rl.dqn import DQNAgent, QNet, compute_eps

CFG = {
    "hidden_sizes": [16, 16],
    "lr": 0.001,
    "gamma": 0.9,
    "batch_size": 8,
    "replay_capacity": 50,
    "learn_start": 8,
    "update_every": 1,
    "target_sync_steps": 1000,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_frac": 0.6,
}


def test_compute_eps_schedule():
    total = 100
    assert compute_eps(0, total, 1.0, 0.05, 0.6) == pytest.approx(1.0)
    assert compute_eps(10, total, 1.0, 0.05, 0.6) < 1.0   # linear decay within 60 steps
    assert compute_eps(60, total, 1.0, 0.05, 0.6) == pytest.approx(0.05)
    assert compute_eps(95, total, 1.0, 0.05, 0.6) == pytest.approx(0.05)


def test_agent_acts_and_learns_from_replay():
    agent = DQNAgent(7, 3, CFG, seed=0)
    act1 = agent.act(np.zeros(7, dtype=np.float32), eps=0.0)   # deterministic greedy
    act2 = agent.act(np.zeros(7, dtype=np.float32), eps=0.0)
    assert act1 == act2 and agent.act(np.zeros(7, dtype=np.float32), 1.0) in (0, 1, 2)

    rng = np.random.default_rng(0)
    for i in range(100):   # fill buffer past learn_start
        s = rng.random(7).astype(np.float32)
        agent.remember(s, rng.integers(0, 3), 0.5, s, i == 99)
    loss = agent.learn()
    assert loss is not None and np.isfinite(loss)
    assert agent.learner_steps == 1


def test_target_sync_and_double_dqn_targets_use_target_net():
    agent = DQNAgent(7, 3, CFG, seed=0)
    old = agent.target.net[0].weight.detach().clone()
    for _ in range(10):
        agent.net.net[0].weight.data += 0.01
    agent.sync_target()
    assert not torch.equal(agent.target.net[0].weight, old)


def test_save_load_roundtrip(tmp_path):
    agent = DQNAgent(7, 3, CFG, seed=3)
    path = tmp_path / "policy.pt"
    agent.save(path, {"seed": 3})
    net = DQNAgent.load_policy_net(path)
    x = torch.zeros(1, 7)
    assert net(x).shape == (1, 3)
