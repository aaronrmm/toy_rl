import unittest
from tqdm.auto import tqdm
import torch

from games.remember_the_number import RememberTheNumber
from games.return_the_number import ReturnTheNumber
from players.multistate_actor import MultiStateActorValueBandit
from players.random_actor import RandomActor
from players.single_state_actor import SingleStateActorValueBandit


class TestCase_MultiStateActorValueBandit_Plays_ReturnTheNumber(unittest.TestCase):
    def test_actions_can_be_taken(self):
        """
        Assert the actor can act on the game
        """
        player = MultiStateActorValueBandit(action_dims=10, observation_dims=0)
        act: torch.Tensor = player.get_action(observations=None)
        assert act is not None

    def test_scoring_is_possible(self):
        """
        Assert the game can be won, but is not won every time by the actor
        """
        game = ReturnTheNumber()
        player = MultiStateActorValueBandit(
            action_dims=game.get_action_space(),
            observation_dims=game.get_observation_space(),
        )
        games_to_play = 100
        scores = []
        for _ in tqdm(range(games_to_play)):
            action = player.get_action(observations=game.get_state())
            score = game.act(action)
            player.give_feedback(score)
            scores.append(score)
        print("Scored: " + str(sum(scores)))
        assert sum(scores) > 1, "Game may not be winnable"
        assert sum(scores) < games_to_play, "Game may not be losable"

    def test_game_is_learnable(self):
        """
        Assert the game can be learned by an AI
        """
        game = ReturnTheNumber()
        trials = 10
        winners = 0
        losers = 0
        for _ in range(trials):
            player = MultiStateActorValueBandit(
                action_dims=game.get_action_space(),
                observation_dims=game.get_observation_space(),
            )
            games_to_play = 1000
            scores = []
            for _ in tqdm(range(games_to_play)):
                action = player.get_action(observations=game.get_state())
                score = game.act(action)
                player.give_feedback(score)
                scores.append(score)
            print(f"Scored: {sum(scores)}/{games_to_play}")
            if sum(scores) > games_to_play / 3:
                winners += 1
            else:
                losers += 1
        assert winners > 0


if __name__ == "__main__":
    unittest.main()
