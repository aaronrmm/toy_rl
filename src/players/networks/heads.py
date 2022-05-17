import torch


def ValueHead(input_len) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(in_features=input_len, out_features=1),
        torch.nn.Sigmoid(),
    )
