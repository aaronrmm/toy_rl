import unittest

import torch
from tqdm.auto import tqdm

import players.multistate_policy_actor
from games.return_the_number import ReturnTheNumber


class TestCase_MultiStatePolicyActor_Plays_ReturnTheNumber(unittest.TestCase):
    def test_actions_can_be_taken(self):
        """
        Assert the actor can act on the game
        """
        player = players.multistate_policy_actor.MultiStatePolicyActor(
            action_dims=10, observation_dims=1
        )
        act: torch.Tensor = player.get_action(observations=torch.zeros(size=[1]))
        assert act is not None

    def test_scoring_is_possible(self):
        """
        Assert the game can be won, but is not won every time by the actor
        """
        game = ReturnTheNumber()
        player = players.multistate_policy_actor.MultiStatePolicyActor(
            action_dims=game.get_action_space(),
            observation_dims=game.get_observation_space(),
        )
        games_to_play = 100
        scores = []
        for _ in tqdm(range(games_to_play)):
            state = game.get_state()
            action = player.get_action(observations=state)
            score = game.act(action)
            best_action = state
            player.give_feedback(best_action.data)
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
            player = players.multistate_policy_actor.MultiStatePolicyActor(
                action_dims=game.get_action_space(),
                observation_dims=game.get_observation_space(),
            )
            games_to_play = 3000
            scores = []
            for _ in tqdm(range(games_to_play)):
                action = player.get_action(observations=game.get_state())
                score = game.act(action)
                player.give_feedback(game.get_state())
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
