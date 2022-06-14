import unittest

import gym
from ray.rllib.agents.ppo import PPOTrainer

from games.gym import SimpleCorridor


class TestSimpleCorridor(unittest.TestSuite):
    game: gym.Env = SimpleCorridor()

    def test_is_playable(self):
        assert self.game is not None
        assert self.game.action_space is not None
        print("Done")

    def test_is_learnable(self):
        # Create an RLlib Trainer instance.
        trainer = PPOTrainer(
            config={
                # Env class to use (here: our gym.Env sub-class from above).
                "env": SimpleCorridor,
                # Config dict to be passed to our custom env's constructor.
                "env_config": {
                    # Use corridor with 20 fields (including S and G).
                    "corridor_length": 20
                },
                # Parallelize environment rollouts.
                "num_workers": 8,
                "framework": "torch",
                "num_gpus_per_worker": 0.0,
            },
        )

        # Train for n iterations and report results (mean episode rewards).
        # Since we have to move at least 19 times in the env to reach the goal and
        # each move gives us -0.1 reward (except the last move at the end: +1.0),
        # we can expect to reach an optimal episode reward of -0.1*18 + 1.0 = -0.8
        for i in range(5):
            results = trainer.train()
            print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")

        # Perform inference (action computations) based on given env observations.
        # Note that we are using a slightly different env here (len 10 instead of 20),
        # however, this should still work as the agent has (hopefully) learned
        # to "just always walk right!"
        env = SimpleCorridor({"corridor_length": 10})
        # Get the initial observation (should be: [0.0] for the starting position).
        obs = env.reset()
        done = False
        total_reward = 0.0
        # Play one episode.
        while not done:
            # Compute a single action, given the current observation
            # from the environment.
            action = trainer.compute_single_action(obs)
            # Apply the computed action in the environment.
            obs, reward, done, info = env.step(action)
            # Sum up rewards for reporting purposes.
            total_reward += reward
        # Report results.
        print(f"Played 1 episode; total-reward={total_reward}")

    # def test_scoring_is_possible(self): #TODO
    #     """
    #     Assert the game can be won, but is not won every time by the actor
    #     """
    #     games_to_play = 100
    #     scores = []
    #     for _ in tqdm(range(games_to_play)):
    #         self.game.action_space
    #         score = game.act(player.get_action(observations=None))
    #         scores.append(score)
    #         player.give_feedback(score)
    #     print("Scored: " + str(sum(scores)))
    #     assert sum(scores) > 1, "Game may not be winnable"
    #     assert sum(scores) < games_to_play, "Game may not be losable"
