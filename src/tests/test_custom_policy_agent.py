import unittest

import gym
import torch
import torch.nn as nn
from gym import Space
from gym.spaces import Discrete
from ray.rllib.utils import check_env
from torch import optim
from tqdm.auto import tqdm

# from https://docs.ray.io/en/master/ray-overview/index.html
import games.gym.rock_paper_scissors_tournament


class TestCase_Custom_Gym_Agent(unittest.TestCase):
    num_samples = 20
    input_size = 10
    layer_size = 15
    output_size = 5
    # In this example we use a randomly generated dataset.
    input = torch.randn(num_samples, input_size)
    labels = torch.randn(num_samples, output_size)

    class PolicyNetwork(nn.Module):
        def __init__(self, observation_space: Space, layer_size, action_space: Space):
            super().__init__()
            self.input_size = observation_space.n
            self.layer_size = layer_size
            self.output_size = (
                action_space.n
                if type(action_space) is Discrete
                else action_space.shape[0]
            )
            self.layers = nn.Sequential(
                nn.Linear(self.input_size, self.output_size),
            )

        def forward(self, input):
            input = torch.tensor(input) if type(input) is list else input
            return self.layers(input)

    def test_is_valid_env(self):
        env = games.gym.rock_paper_scissors_tournament.RockPaperScissorsTournament()
        check_env(env)

    def test_is_learnable(self):
        env: gym.Env = (
            games.gym.rock_paper_scissors_tournament.RockPaperScissorsTournament()
        )
        brain = TestCase_Custom_Gym_Agent.PolicyNetwork(
            observation_space=env.observation_space,
            action_space=env.action_space,
            layer_size=16,
        )
        optimizer = optim.SGD(lr=0.4, params=brain.parameters())

        print("Starting behavior:")
        state = [1.0, 0.0, 0.0]
        action_logits: torch.Tensor = brain(state)
        print(state, action_logits)

        state = [0.0, 1.0, 0.0]
        action_logits: torch.Tensor = brain(state)
        print(state, action_logits)

        state = [0.0, 0.0, 1.0]
        action_logits: torch.Tensor = brain(state)
        print(state, action_logits)

        num_games_to_play = 1000
        diffs = [0, 0, 0]
        for _ in tqdm(range(num_games_to_play), total=num_games_to_play):
            state = env.reset()
            action_logits: torch.Tensor = brain(state)
            action = torch.argmax(action_logits)

            # return self.current_opponent_probabilities, reward, done, {}
            state, reward, done, _ = env.step(action)
            optimizer.zero_grad()
            action_mask = torch.zeros_like(action_logits)
            action_mask[action] = 1
            if True:
                probabilities = torch.softmax(action_logits, dim=0)
                log_probabilities = torch.log(probabilities)
                rewarded_action = action_mask * reward * (-1)
                masked_probabilities = log_probabilities * rewarded_action
                policy_loss = torch.sum(masked_probabilities)
                policy_loss.backward()
                optimizer.step()
            action_logits_new: torch.Tensor = brain(state)
            diff = action_logits_new - action_logits
            diff = diff.detach()
            if reward >= 0.1:
                if torch.argmax(diff) != action:
                    print("?")
                assert torch.argmax(diff) == action, (
                    "state",
                    state,
                    "action",
                    action,
                    "reward",
                    reward,
                    "grad",
                    diff,
                )
            elif reward <= -0.1:
                if torch.argmin(diff) != action:
                    print("?")
                assert torch.argmin(diff) == action, (
                    "state",
                    state,
                    "action",
                    action,
                    "reward",
                    reward,
                    "grad",
                    diff,
                )
            diffs[0] += diff[0]
            diffs[1] += diff[1]
            diffs[2] += diff[2]

        for difft in diffs:
            diff: torch.Tensor = difft
            print(diff.item())

        print("End behavior")
        state = [1.0, 0.0, 0.0]
        action_logits: torch.Tensor = brain(state)
        print(state, torch.argmax(action_logits))
        assert torch.argmax(action_logits) == 1

        state = [0.0, 1.0, 0.0]
        action_logits: torch.Tensor = brain(state)
        print(state, torch.argmax(action_logits))
        assert torch.argmax(action_logits) == 2

        state = [0.0, 0.0, 1.0]
        action_logits: torch.Tensor = brain(state)
        print(state, torch.argmax(action_logits))
        assert torch.argmax(action_logits) == 0


if __name__ == "__main__":
    unittest.main()
