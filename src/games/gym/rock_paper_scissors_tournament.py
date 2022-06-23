import random

import gym
from gym import spaces


class RockPaperScissorsTournament(gym.Env):
    def __init__(self):
        self.player_names = ["Rocco", "XGrizzlyX", "Sissero"]

        self.player_probabilities = [
            [1.0, 0.0, 0.0],  # Rocco likes rock
            [0.0, 1.0, 0],  # XGrizzlyX loves paper
            [0.0, 0.0, 1.0],  # Sissero likes scissors
        ]
        self.current_player_index: int = 0

        # Observations are the opponent player's probabilities
        self.observation_space = spaces.Discrete(3)

        # We have 4 actions, corresponding to "rock","paper", and "scissors"
        self.action_space = spaces.Discrete(3)

    def reset(self):
        """Resets the episode and returns the initial observation of the new one."""
        self.current_player_index = random.choice([0, 1, 2])

        # one-hot encode current player for use by the AI
        observation = [0.0, 0.0, 0.0]
        observation[self.current_player_index] += 1.0
        return observation

    def step(self, action):
        """Play a single round of tic-tac-toe

        Returns:
            New observation, reward, done-flag, info-dict (empty).
        """

        # Opponent chooses rock, paper, or scissors depending on their tendencies
        opponent_probabilities = self.player_probabilities[self.current_player_index]
        opponent_hand = random.choices(
            population=[0, 1, 2],
            weights=opponent_probabilities,
            k=1,  # play 1 round
        )[0]

        if (
            (action == 0 and opponent_hand == 1)  # their paper beats your rock
            or (action == 1 and opponent_hand == 2)  # their scissors beat your paper
            or (action == 2 and opponent_hand == 0)  # their rock beats your scissors
        ):
            reward = -1  # penalty for losing this toss
        elif (
            (action == 0 and opponent_hand == 2)  # your rock beats their scissors
            or (action == 1 and opponent_hand == 0)  # your paper beats their rock
            or (action == 2 and opponent_hand == 1)  # your scissors beat their paper
        ):
            reward = 1  # reward for winning this toss
        else:
            reward = -0.5  # small penalty to encourage AI not to tie

        done = True

        # one-hot encode current player for use by the AI
        observation = [0.0, 0.0, 0.0]
        observation[self.current_player_index] += 1.0

        return observation, reward, done, {}
