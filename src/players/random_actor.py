from typing import List

import torch


class RandomActor:
    def __init__(self, action_space: List[int]):
        self.action_space = action_space

    def get_action(self, observations):
        return torch.rand(size=self.action_space)
