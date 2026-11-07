import numpy as np
import torch

from autoscale_rl.policies import GreedyPolicy, RandomPolicy, ThresholdPolicy
from autoscale_rl.dqn import QNet


def test_threshold_scales_up_then_holds_through_cooldown():
    p = ThresholdPolicy(scale_up_util=0.6, scale_down_util=0.3, cooldown_minutes=10,
                        min_instances=1, max_instances=8)
    assert p.act(None, {"utilization": 0.9, "instances": 1}) == 2      # scale up
    assert p.act(None, {"utilization": 0.9, "instances": 2}) == 1      # cooldown
    assert p.act(None, {"utilization": 0.9, "instances": 2}) == 1


def test_threshold_scales_down_only_below_floor():
    p = ThresholdPolicy(scale_up_util=0.6, scale_down_util=0.3, cooldown_minutes=0,
                        min_instances=1, max_instances=4)
    assert p.act(None, {"utilization": 0.05, "instances": 2}) == 0     # scale down
    p2 = ThresholdPolicy(scale_up_util=0.6, scale_down_util=0.3, cooldown_minutes=0,
                         min_instances=2, max_instances=4)
    assert p2.act(None, {"utilization": 0.05, "instances": 2}) == 1    # at min


def test_threshold_clamps_at_max_instances():
    p = ThresholdPolicy(scale_up_util=0.6, scale_down_util=0.3, cooldown_minutes=0,
                        min_instances=1, max_instances=2)
    assert p.act(None, {"utilization": 0.99, "instances": 2}) == 1
    assert p.act(None, {"utilization": 0.99, "instances": 2}) == 1


def test_random_policy_deterministic_by_seed():
    r1 = [RandomPolicy(seed=99).act(None, {}) for _ in range(10)]
    r2 = [RandomPolicy(seed=99).act(None, {}) for _ in range(10)]
    assert r1 == r2


def test_greedy_policy_uses_network_argmax():
    torch.manual_seed(0)
    net = QNet(7, 3, [16, 16])
    net.eval()
    greedy = GreedyPolicy(net, eps=0.0, seed=0)
    obs = np.zeros(7, dtype=np.float32)
    with torch.no_grad():
        expected = int(torch.argmax(net(torch.from_numpy(obs).unsqueeze(0)), dim=1).item())
    assert greedy.act(obs, {}) == expected
    # with eps=1 it must randomize but stay in range
    prov = GreedyPolicy(net, eps=1.0, seed=5)
    for _ in range(20):
        assert prov.act(obs, {}) in (0, 1, 2)
