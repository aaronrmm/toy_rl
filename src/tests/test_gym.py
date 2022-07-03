import unittest

import gym
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.utils import check_env


class TestCase_GymBlackjack(unittest.TestCase):
    env_name = "Blackjack-v1"

    def test_blackjack_runs(self):
        env = gym.make(self.env_name)
        check_env(env)
        # env is created, now we can use it:
        for episode in range(10):
            obs = env.reset()
            for step in range(50):
                action = (
                    env.action_space.sample()
                )  # or given a custom model, action = policy(observation)
                nobs, reward, done, info = env.step(action)

    def test_blackjack_is_learnable(self):
        from ray import tune

        tune.run(
            DQNTrainer,
            config={
                "env": self.env_name,
                "framework": "torch",
                "log_level": "INFO",
                "num_gpus": 0,
            },
            time_budget_s=10,
        )
