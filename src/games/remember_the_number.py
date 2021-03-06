import torch


class RememberTheNumber:
    """
    Returns win when the number is guessed (within a range)
    """

    def __init__(self, number=7, max=10):
        self.number = number
        self.max = max

    def act(self, input: torch.Tensor):
        guess = torch.argmax(input)
        if guess == self.number:
            return 1
        else:
            return 0

    def get_action_space(self):
        return [self.max + 1]
