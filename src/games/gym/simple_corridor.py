# From ray rllib tutorial

import gym
from ray.rllib.agents.ppo import PPOTrainer


# Define your problem using python and openAI's gym API:
class SimpleCorridor(gym.Env):
    """Corridor in which an agent must learn to move right to reach the exit.

    ---------------------
    | S | 1 | 2 | 3 | G |   S=start; G=goal; corridor_length=5
    ---------------------

    Possible actions to chose from are: 0=left; 1=right
    Observations are floats indicating the current field index, e.g. 0.0 for
    starting position, 1.0 for the field next to the starting position, etc..
    Rewards are -0.1 for all steps, except when reaching the goal (+1.0).
    """

    def __init__(self, config=None):
        self.end_pos = (config or {}).get("corridor_length", 20)
        self.cur_pos = 0
        self.action_space = gym.spaces.Discrete(2)  # left and right
        self.observation_space = gym.spaces.Box(0.0, self.end_pos, shape=(1,))

    def reset(self):
        """Resets the episode and returns the initial observation of the new one."""
        self.cur_pos = 0
        # Return initial observation.
        return [self.cur_pos]

    def step(self, action):
        """Takes a single step in the episode given `action`

        Returns:
            New observation, reward, done-flag, info-dict (empty).
        """
        # Walk left.
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        # Walk right.
        elif action == 1:
            self.cur_pos += 1
        # Set `done` flag when end of corridor (goal) reached.
        done = self.cur_pos >= self.end_pos
        # +1 when goal reached, otherwise -1.
        reward = 1.0 if done else -0.1
        return [self.cur_pos], reward, done, {}
