import unittest

import torch

from games.remember_the_number import RememberTheNumber
from players.random_actor import RandomActor


class TestCase_RandomPlayer_Plays_RememberTheNumber(unittest.TestCase):
    def test_actions_are_in_bounds(self):
        """
        Assert all the RandomActor's action values are between 0 and 1
        """
        player = RandomActor(action_space=[10, 10])
        act: torch.Tensor = player.get_action(observations=None)
        act_max = torch.max(act)
        act_min = torch.min(act)
        assert act_max <= 1
        assert act_min >= 0

    def test_scoring_is_possible(self):
        """
        Assert the game can be won, but is not won every time by a random actor
        """
        game = RememberTheNumber()
        player = RandomActor(action_space=(1,))
        games_to_play = 1000
        scores = [
            game.act(player.get_action(observations=None)) for _ in range(games_to_play)
        ]
        assert sum(scores) > 1
        assert sum(scores) < games_to_play


if __name__ == "__main__":
    unittest.main()
