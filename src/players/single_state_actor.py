import random
from typing import List

import torch

from players.networks.smolnn import TinyValueNN


class SingleStateActorValueBandit:
    def __init__(self, action_space: List[int]):
        self.action_space = action_space
        self.memory = []
        self.max_memory_size = 500
        self.learning_frequency = 20
        self.training_steps = 5
        self.value_estimator = TinyValueNN(self.action_space[0])
        self.last_move = None
        self.best_move: torch.Tensor = None
        self.total_memories_learned = 0
        self.randomize = 0.5

    def get_action(self, observations) -> torch.Tensor:
        do_random = random.random() < self.randomize
        if do_random or self.best_move is None:
            self.last_move = torch.rand(size=self.action_space)
        else:
            self.last_move = self.best_move
        return self.last_move

    def give_feedback(self, score):
        if self.last_move is not None:
            memory = (self.last_move, score)
            self._store_memory(memory)
            self.total_memories_learned += 1
            if self.total_memories_learned % self.learning_frequency == 0:
                self.learn()
        else:
            print("Error - could not find last move to give feedback for")

    def learn(self):
        acts, scores = zip(*self.memory)
        input: torch.Tensor = torch.stack(acts, dim=0)
        scores = torch.Tensor(scores)
        scores = torch.reshape(scores, shape=(-1, 1))
        self.value_estimator.train(
            input=input, target=scores, steps=self.training_steps
        )
        test = torch.eye(self.action_space[0], requires_grad=False)
        test_scores = self.value_estimator.model(test)
        self.best_move = test_scores
        self.best_move = torch.reshape(self.best_move, shape=[-1])
        self.best_move = self.best_move.detach()

    def reset_memory(self):
        self.memory = []
        self.total_memories_learned = 0

    def _store_memory(self, memory):
        if len(self.memory) > self.max_memory_size:
            self.memory[random.randint(0, len(self.memory) - 1)] = memory
        else:
            self.memory.append(memory)
