import torch


class RememberTheNumber:
    """
    Returns win when the number is guessed (within a range)
    """

    def __init__(self, number=0.7, range=0.05):
        self.number = number
        self.range = range

    def act(self, input: torch.Tensor):
        guess = input[0]
        if guess < self.number + self.range and guess > self.number - self.range:
            return 1
        else:
            return 0
