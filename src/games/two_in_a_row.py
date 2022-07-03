import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F


class TwoInARow:
    """
    Given a list of random integers, pick integers in sequential order

    Parameters:
    ensure_goal (int): if True will ensure the game can be won
    steps (int): number of steps agent will take and length of goal sequence
    num_choices (int): length of list to populate
    """

    def __init__(
        self, max: int = 10, num_choices: int = 5, steps: int = 2, ensure_goal=True
    ):
        self.max = max
        self.num_choices = num_choices
        self.steps = steps
        self.ensure_goal = ensure_goal
        self.choices: list = None
        self.chosen_ints = []
        self.restart()

    def restart(self):
        if self.ensure_goal:
            sequence_start: int = random.randint(0, self.max - self.steps)
            goal_sequence = [sequence_start + s for s in range(self.steps)]
            self.choices = goal_sequence
        else:
            self.choices = []
        self.fill_with_random_numbers_randomly()

    def fill_with_random_numbers_randomly(self):
        while len(self.choices) < self.num_choices:
            self.choices.insert(
                random.randint(0, len(self.choices) - 1),
                random.randint(0, self.max - 1),
            )

    def act(self, input: torch.Tensor) -> int:
        chosen_index = torch.argmax(input)
        self.chosen_ints.append(self.choices[chosen_index])

        if len(self.chosen_ints) >= self.steps:  # game end
            for i in range(self.steps - 1):
                if self.chosen_ints[i + 1] - self.chosen_ints[i] != 1:
                    return -1  # sequence is not in order - game lost
            return 1  # sequence in order. Good job!
        else:  # continue game
            self.choices.pop(chosen_index)
            self.fill_with_random_numbers_randomly()
            return 0

    def get_action_space(self):
        return self.num_choices

    def get_observation_space(self):
        """
        Return dimensions of observation state
        :return: observation space is the presented choices in order and the last int selected
        """
        return [self.num_choices + 1, self.max]

    def get_state(self) -> torch.Tensor:
        """
        :return:
        """
        choices: torch.Tensor = torch.Tensor(self.choices)
        chosen_ints: torch.Tensor = torch.Tensor(self.chosen_ints)
        choices_t: torch.Tensor = F.one_hot(
            choices.to(torch.int64), num_classes=self.max
        )
        if len(chosen_ints) > 0:
            try:
                last_int_t: torch.Tensor = F.one_hot(
                    chosen_ints[-1:].to(torch.int64), num_classes=self.max
                )
            except Exception as e:
                print(e)
        else:
            last_int_t: torch.Tensor = torch.zeros(size=[1, self.max])
        state = torch.concat([choices_t, last_int_t], dim=0).to(torch.float32)
        state = state[None, :, :]  # add a batch dim of 1 at axis 0
        return state


if __name__ == "__main__":
    game = TwoInARow(max=3, num_choices=100)
    state = game.get_state()
    action = torch.randn(size=[game.get_action_space()])
    print(f"Chosen ints: {game.chosen_ints}")
    print(f"Action: {action}")
    score = game.act(input=action)
    print(f"Score: {score}")
