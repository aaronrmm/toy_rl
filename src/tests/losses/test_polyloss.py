import unittest

from players.networks.losses import Poly1CrossEntropyLoss


class TestPolyloss(unittest.TestCase):
    def test_library_example(self):
        import torch

        # Poly1 Cross-Entropy Loss
        # classification task
        batch_size = 10
        num_classes = 5
        logits = torch.rand([batch_size, num_classes], requires_grad=True)
        labels = torch.randint(high=num_classes, size=[batch_size], requires_grad=False)
        loss = Poly1CrossEntropyLoss(num_classes=num_classes, reduction="mean")
        out = loss(logits, labels)
        out.backward()


if __name__ == "__main__":
    unittest.main()
