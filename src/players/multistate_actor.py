import random
from typing import List

import torch

from players.networks.bodies.TinyFeedForwardBody import TinyFeedForwardBody
from players.networks.heads import ValueHead
from players.networks.network import Network
from players.networks.smolnn import TinyValueNN


class MultiStateActorValueBandit:
    def __init__(self, action_dims: int, observation_dims: int):
        self.action_dims = action_dims
        self.memory = []
        self.max_memory_size = 500
        self.learning_frequency = 20
        self.training_steps = 5
        self.value_estimator = Network(
            body=TinyFeedForwardBody(
                input_len=observation_dims + action_dims, output_len=4
            ),
            head=ValueHead(input_len=4),
        )
        self.last_move = None
        self.last_state: torch.Tensor = None
        self.total_memories_learned = 0
        self.randomize = 0.25

    def get_action(self, observations) -> torch.Tensor:
        self.last_state = observations
        do_random = random.random() < self.randomize
        # TODO have self.randomize chance be a function of observations - e.g. randomize less in well-known states
        if do_random:
            self.last_move = torch.rand(size=[self.action_dims])
        else:
            self.last_move = self._get_best_move(observations)
        return self.last_move

    def give_feedback(self, score):
        if self.last_move is not None:
            memory = (
                self._concatenate_turns(action=self.last_move, state=self.last_state),
                score,
            )
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

    def reset_memory(self):
        self.memory = []
        self.total_memories_learned = 0

    def _store_memory(self, memory):
        if len(self.memory) > self.max_memory_size:
            self.memory[random.randint(0, len(self.memory) - 1)] = memory
        else:
            self.memory.append(memory)

    def _get_best_move(self, observations: torch.Tensor):
        test = torch.eye(self.action_dims, requires_grad=False)
        if observations is not None:
            observations = torch.broadcast_to(
                observations, size=(len(observations), len(test))
            )
            test = self._concatenate_turns(action=test, state=observations)
        test_scores = self.value_estimator.model(test)
        best_move = test_scores
        best_move = torch.reshape(best_move, shape=[-1])
        best_move = best_move.detach()
        return best_move

    def _concatenate_turns(self, state, action):
        return torch.concat([state, action], dim=-1)
