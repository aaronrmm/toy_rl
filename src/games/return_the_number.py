import random

import torch


class ReturnTheNumber:
    """
    Returns win when the number given is returned # TODO change the number and give it as the state
    """

    def __init__(self, max=10):
        self.max = max
        self.number = random.randint(a=0, b=self.max)

    def act(self, input: torch.Tensor):
        guess = torch.argmax(input)
        if guess == self.number:
            score = 1
        else:
            score = 0
        self.number = random.randint(a=0, b=self.max)
        return score

    def get_action_space(self):
        return self.max + 1

    def get_observation_space(self):
        return self.max + 1

    def get_state(self):
        space = torch.zeros(size=[self.get_observation_space()])
        space[self.number] += 1
        return space
