from stable_baselines3 import PPO
from ravens.environments.environment import Environment
from environment.env_wrapper import WrappedEnv
from ravens import tasks as ravens_tasks
import tasks


def train(
    assets_root="./assets",
    task="tangram",
    disp=False,
    mode="train",
    data_dir=".",
    n=100,
):
    env = WrappedEnv(assets_root, disp)
    task = ravens_tasks.names[task]()
    task.mode = mode

    model = PPO("MlpPolicy", env, n_steps=20, verbose=1, tensorboard_log="./log/")
    env.set_task(task)

    model.learn(total_timesteps=10_000)


    for i in range(n):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(reward)
        if done:
            obs = env.reset()

    env.close()


if __name__ == '__main__':
    train(disp=False)