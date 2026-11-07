import numpy as np
import pandas as pd
import pytest

from autoscale_rl.simulator import SimConfig, TrafficEnv, day_frames_from_df


def make_env(seed=7, steps=120, max_instances=8):
    cfg = SimConfig(service_rate_per_instance=100.0, min_instances=1,
                    max_instances=max_instances, sla_latency_ms=200.0,
                    base_latency_ms=40.0, steps_per_episode=steps,
                    p99_rate=600.0, queue_norm=600.0)
    rates = {0: np.full(steps, 200.0)}      # 2x capacity of one instance, day 0
    return TrafficEnv(rates, cfg)


def test_episode_deterministic_for_same_seed():
    env1 = make_env()
    env2 = make_env()
    for _ in range(50):
        r1 = env1.step(1); r2 = env2.step(1)
        assert r1[1] == r2[1]
        assert np.allclose(r1[0], r2[0])


def test_steps_and_termination():
    env = make_env(steps=10)
    env.reset(day=0, seed=1)
    done = False
    n = 0
    while not done:
        _, _, done, _ = env.step(1)
        n += 1
    assert n == 10


def test_actions_clamped_to_bounds():
    env = make_env(max_instances=3)
    env.reset(day=0, seed=1)
    for _ in range(5):
        env.step(2)   # scale up repeatedly
    assert env.instances <= 3
    for _ in range(5):
        env.step(0)   # scale down repeatedly
    assert env.instances >= 1


def test_reward_components_finite_environment_telemetry():
    env = make_env()
    env.reset(day=0, seed=3)
    for i in range(30):
        obs, reward, done, info = env.step(1 if i % 3 else 2)
        assert np.isfinite(reward)
        assert np.isfinite(obs).all()
        assert "p95_latency" in info
    assert env.totals["cost_units"] > 0


def test_day_frames_and_observation_shape():
    df = pd.DataFrame({
        "timestamp": pd.date_range("1995-07-01", periods=2880, freq="min"),
        "requests": np.random.default_rng(0).integers(20, 60, size=2880),
        "minute_of_day": list(range(1440)) * 2,
        "day_index": [0] * 1440 + [1] * 1440,
    })
    frames = day_frames_from_df(df, [0, 1])
    assert set(frames) == {0, 1}
    assert len(frames[0]) == 1440
    env = TrafficEnv(frames, SimConfig(service_rate_per_instance=50.0, min_instances=1,
                                       max_instances=4, sla_latency_ms=200.0,
                                       base_latency_ms=40.0, p99_rate=60.0, queue_norm=60.0))
    obs, info = env.reset(day=1, seed=5)
    assert obs.shape == (7,)
