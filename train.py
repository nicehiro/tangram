import math

from ravens import tasks as ravens_tasks
from stable_baselines3 import PPO
from models.feature_extractor import CustomCombinedExtractor

import tasks
from environment.env_wrapper import WrappedTangramEnv


def train(
    assets_root="./assets",
    task="tangram",
    disp=False,
    mode="train",
    data_dir=".",
    n=100,
):
    task = ravens_tasks.names[task]()
    task.mode = mode
    env = WrappedTangramEnv(assets_root, task, disp)

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(),
    )
    model = PPO(
        "MultiInputPolicy",
        env,
        n_steps=20,
        verbose=1,
        tensorboard_log="./log/",
        policy_kwargs=policy_kwargs,
    )

    model.learn(total_timesteps=10_000)

    for i in range(n):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(reward)
        if done:
            obs = env.reset()

    env.close()


if __name__ == "__main__":
    train(disp=False)
