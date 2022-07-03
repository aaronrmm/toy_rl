import unittest
from tqdm.auto import tqdm
import torch

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
        game = ReturnTheNumber(max=2)
        trials = 10
        winners = 0
        losers = 0
        for _ in range(trials):
            player = MultiStateActorValueBandit(
                action_dims=game.get_action_space(),
                observation_dims=game.get_observation_space(),
            )
            games_to_play = 3000
            scores = []
            for _ in tqdm(range(games_to_play)):
                action = player.get_action(observations=game.get_state())
                score = game.act(action)
                player.give_feedback(score)
                scores.append(score)
            print(f"Scored: {sum(scores)}/{games_to_play}")
            if sum(scores) > games_to_play * 3 / 4:
                winners += 1
            else:
                losers += 1
        assert winners > 0

    def test_return_the_number_game_scores_1_when_correct(self):
        game = ReturnTheNumber(max=4)
        for number in range(4):
            game.number = number
            observation = game.get_state()
            assert torch.argmax(observation) == number
            correct_action = torch.zeros(size=[game.get_action_space()])
            correct_action[number] += 1
            score = game.act(correct_action)
            assert (
                score == 1
            ), f"Scoring failed for number={number}, state={observation}, correct action={correct_action}, score={score}"

    def test_return_the_number_game_scores_0_when_incorrect(self):
        game = ReturnTheNumber(max=4)
        for number in range(4):
            game.number = number
            observation = game.get_state()
            assert torch.argmax(observation) == number
            incorrect_action = torch.zeros(size=[game.get_action_space()])
            incorrect_action[(number + 1) % 4] += 1
            score = game.act(incorrect_action)
            assert (
                score == 0
            ), f"Scoring failed for number={number}, state={observation}, incorrect action={incorrect_action}, score={score}"


if __name__ == "__main__":
    unittest.main()
