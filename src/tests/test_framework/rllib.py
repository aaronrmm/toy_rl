import time

import gym
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer


class IsLearnableTest:
    def __init__(
        self,
        env: gym.Env,
        max_time_s: int = 10,
        minimum_successful_reward: int = 0,
        max_training_iterations: int = 1000,
    ):
        trainer = DQNTrainer(
            config={
                # Env class to use (here: our gym.Env sub-class from above).
                "env": type(env),
                # Config dict to be passed to our custom env's constructor.
                # Parallelize environment rollouts.
                "num_workers": 1,
                "framework": "torch",
                "num_gpus_per_worker": 0.0,
            },
        )

        start_time_s = time.time()
        for i in range(max_training_iterations):
            results = trainer.train()
            print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")
            if time.time() - start_time_s >= max_time_s:
                break

        inference_rewards = []
        for i in range(5):
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
            inference_rewards.append(total_reward)
        assert (
            max(inference_rewards) > minimum_successful_reward
        ), f"Max reward {max(inference_rewards)} was less than required reward {minimum_successful_reward}"
