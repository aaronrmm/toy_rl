import random

import gym
from gym import spaces


class RockPaperScissorsTournament(gym.Env):
    def __init__(self):
        self.players = {
            "Rocco": [1.0, 0.0, 0.0],  # likes rock
            "XGrizzlyX": [0.0, 1.0, 0],  # loves paper
            "Sissero": [0.0, 0.0, 1.0],  # likes scissors
        }
        self.current_opponent_probabilities = self.players["Rocco"]
        self.last_player_index = 0

        # Observations are the opponent player's probabilities
        self.observation_space = spaces.Box(0, 1, shape=(3,), dtype=float)

        # We have 4 actions, corresponding to "rock","paper", and "scissors"
        self.action_space = spaces.Discrete(3)

    def reset(self):
        """Resets the episode and returns the initial observation of the new one."""
        self.last_player_index = (self.last_player_index + 1) % 3
        self.current_opponent_probabilities = list(self.players.values())[
            self.last_player_index
        ]
        # Return initial observation.
        return self.current_opponent_probabilities

    def step(self, action):
        """Play a single round of tic-tac-toe

        Returns:
            New observation, reward, done-flag, info-dict (empty).
        """
        choices = random.choices(
            population=[0, 1, 2], weights=self.current_opponent_probabilities, k=10
        )
        reward = 0
        for hand in choices:
            if (
                (action == 0 and hand == 1)
                or (action == 1 and hand == 2)
                or (action == 2 and hand == 0)
            ):
                reward -= 0.1
            if (
                (action == 0 and hand == 2)
                or (action == 1 and hand == 0)
                or (action == 2 and hand == 1)
            ):
                reward += 0.1
            else:
                reward -= 0.05

        done = True
        return self.current_opponent_probabilities, reward, done, {}
