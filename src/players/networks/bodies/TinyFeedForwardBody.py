import torch
import torch.nn as nn

from players.networks.heads import ValueHead


def TinyFeedForwardBody(input_len: int, output_len: int) -> nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(in_features=input_len, out_features=4),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(in_features=4, out_features=output_len),
        torch.nn.LeakyReLU(),
    )
