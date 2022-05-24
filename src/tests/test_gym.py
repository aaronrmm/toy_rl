import unittest

import gym


class TestCase_GymCartPole(unittest.TestCase):
    def test_cartpole_runs(self):
        env = gym.make("CartPole-v1")
        observation, info = env.reset(seed=42, return_info=True)

        for _ in range(100):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            if done:
                observation, info = env.reset(return_info=True)
        env.close()
        assert len(observation) > 0
