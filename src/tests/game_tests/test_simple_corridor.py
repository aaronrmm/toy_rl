import unittest

import gym
from ray.rllib.agents.ppo import PPOTrainer

from games.gym import SimpleCorridor
from tests.test_framework.rllib import IsLearnableTest


class TestSimpleCorridor(unittest.TestCase):
    game: gym.Env = SimpleCorridor()

    def test_is_playable(self):
        assert self.game is not None
        assert self.game.action_space is not None
        print("Done")

    def test_is_learnable(self):
        t = IsLearnableTest(
            self.game,
            max_time_s=150,
            max_training_iterations=2000,
            minimum_successful_reward=-2,
        )
