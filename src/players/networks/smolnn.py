import torch


class TinyValueNN:
    def __init__(self, input_len: int, starting_learning_rate=5e-2, weight_decay=1e-5):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_len, out_features=4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=4, out_features=4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=4, out_features=1),
        )
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=starting_learning_rate,
            weight_decay=weight_decay,
        )

    def train(self, input: torch.Tensor, target: torch.Tensor, steps=1):
        for _ in range(steps):
            prediction = self.model(input)
            loss = self.loss_fn(prediction, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
