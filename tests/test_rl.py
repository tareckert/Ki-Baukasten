"""Tests des Reinforcement Learnings: Umgebungen, Belohnungsterme, Algorithmen."""

import pytest
import torch

from ki_baukasten import Trainer, TrainingConfig, build_model, registry
from ki_baukasten.core import ComponentSpec
from ki_baukasten.training.rl import RewardSystem, Transition


@pytest.mark.parametrize("name", [n for n in registry("environment").names() if n != "gymnasium"])
def test_environments(name):
    env = registry("environment").build(name)
    obs = env.reset(seed=0)
    assert obs.shape == (env.obs_dim,)
    action = torch.zeros(env.action_dim) if env.action_type == "continuous" else 0
    o, r, term, trunc, info = env.step(action)
    assert isinstance(info, dict) and o.shape == (env.obs_dim,)


def test_reward_system_combines_terms():
    specs = [ComponentSpec.from_value("reward", s) for s in [
        "env", {"name": "constant", "value": -0.1}, {"name": "event", "key": "hit", "value": 2.0},
        {"name": "potential", "key": "distance"}, {"name": "action_penalty", "costs": [0.0, 1.0]},
        {"name": "count"}, {"name": "rnd"}, {"name": "terminal", "value": -5.0},
        {"name": "info", "key": "distance", "transform": "neg", "weight": 0.1, "start_step": 10}]]
    rs = RewardSystem(specs, obs_dim=3, num_envs=2, gamma=0.99, normalization="none", clip=None)
    tr = Transition(torch.zeros(2, 3), torch.tensor([0, 1]), torch.ones(2, 3), torch.tensor([1.0, 0.0]),
                    torch.tensor([False, True]), torch.tensor([False, False]),
                    [{"hit": True, "distance": 3.0}, {"hit": False, "distance": 1.0}], None, 0, 0.99)
    total = rs(tr)
    assert total.shape == (2,)
    assert "ertrag_gesamt" in rs.pop_breakdown()
    assert "rnd_loss" in rs.learn(torch.randn(8, 3))


@pytest.mark.parametrize("algo,model,env", [
    ("ppo", {"name": "actor_critic"}, "cartpole"),
    ("a2c", {"name": "actor_critic", "args": {"obs_dim": 4, "action_dim": 4}}, "gridworld"),
    ("reinforce", {"name": "actor_critic", "args": {"obs_dim": 1, "action_dim": 10}}, "bandit"),
    ("dqn", {"name": "q_network"}, "cartpole"),
    ("ppo", {"name": "actor_critic", "args": {"obs_dim": 3, "action_dim": 1, "action_type": "continuous"}}, "pendulum"),
])
def test_algorithms_run(algo, model, env):
    rl = {"algorithm": {"name": algo, "args": {"n_steps": 32} if algo != "dqn" else {"learning_starts": 64}},
          "environment": env, "total_timesteps": 512, "num_envs": 2, "eval_episodes": 1, "log_every_env_steps": 10 ** 6}
    tr = Trainer(build_model(model), TrainingConfig(paradigm={"name": "reinforcement", "args": rl}, verbose=False))
    hist = tr.fit()
    assert any("val_return" in h for h in hist)


def test_ppo_solves_cartpole():
    rl = {"algorithm": {"name": "ppo", "args": {"n_steps": 128}}, "environment": "cartpole",
          "total_timesteps": 25000, "log_every_env_steps": 10 ** 6}
    tr = Trainer(build_model("actor_critic"), TrainingConfig(
        paradigm={"name": "reinforcement", "args": rl}, verbose=False, scheduler="constant",
        optimizer={"name": "adamw", "args": {"lr": 3e-3, "weight_decay": 0.0}},
        gradient_clipping={"max_norm": 0.5}))
    hist = tr.fit()
    assert hist[-1]["val_return"] > 120
