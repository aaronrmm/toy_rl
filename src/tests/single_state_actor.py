import unittest
from tqdm.auto import tqdm
import torch

from games.remember_the_number import RememberTheNumber
from players.random_actor import RandomActor
from players.single_state_actor import SingleStateActorValueBandit


class TestCase_SingleStateActorValueBandit_Plays_RememberTheNumber(unittest.TestCase):
    def test_actions_are_in_bounds(self):
        """
        Assert all the SingleStateActorValueBandit's action values are between 0 and 1
        """
        player = SingleStateActorValueBandit(action_space=[10, 10])
        act: torch.Tensor = player.get_action(observations=None)
        act_max = torch.max(act)
        act_min = torch.min(act)
        assert act_max <= 1
        assert act_min >= 0

    def test_scoring_is_possible(self):
        """
        Assert the game can be won, but is not won every time by the actor
        """
        game = RememberTheNumber()
        player = SingleStateActorValueBandit(action_space=game.get_action_space())
        games_to_play = 100
        scores = []
        for _ in tqdm(range(games_to_play)):
            score = game.act(player.get_action(observations=None))
            scores.append(score)
            player.give_feedback(score)
        print("Scored: " + str(sum(scores)))
        assert sum(scores) > 1, "Game may not be winnable"
        assert sum(scores) < games_to_play, "Game may not be losable"

    def test_game_is_learnable(self):
        """
        Assert the game can be learned by an AI
        """
        game = RememberTheNumber()
        trials = 10
        winners = 0
        losers = 0
        for _ in range(trials):
            player = SingleStateActorValueBandit(action_space=game.get_action_space())
            games_to_play = 1000
            scores = []
            for _ in tqdm(range(games_to_play)):
                score = game.act(player.get_action(observations=None))
                scores.append(score)
                player.give_feedback(score)
            print("Scored: " + str(sum(scores)))
            if sum(scores) > games_to_play / 3:
                winners += 1
            else:
                losers += 1
        assert winners > 0


if __name__ == "__main__":
    unittest.main()
