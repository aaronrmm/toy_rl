import torch
import torch.nn as nn


class Network:
    def __init__(
        self,
        body: nn.Module,
        head: nn.Module,
        embedding_layer: nn.Module = None,
        starting_learning_rate=5e-2,
        weight_decay=1e-5,
        loss_function=torch.nn.BCELoss(),
    ):
        if embedding_layer:
            self.model: nn.Module = nn.Sequential(embedding_layer, body, head)
        else:
            self.model: nn.Module = nn.Sequential(body, head)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=starting_learning_rate,
            weight_decay=weight_decay,
        )
        self.loss_fn = loss_function

    def train(self, input: torch.Tensor, target: torch.Tensor, steps=1):
        for _ in range(steps):
            prediction = self.model(input)
            loss = self.loss_fn(prediction, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
