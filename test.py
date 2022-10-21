import os

import numpy as np
from absl import app, flags
from ravens import tasks
from ravens.dataset import Dataset
from ravens.environments.environment import Environment

from tasks.tangram import Tangram
from utils import registe_task

flags.DEFINE_string("assets_root", "./assets", "")
flags.DEFINE_string("data_dir", ".", "")
flags.DEFINE_bool("disp", True, "")
flags.DEFINE_string("task", "hanoi", "")
flags.DEFINE_string("mode", "train", "")
flags.DEFINE_integer("n", 1000, "")

FLAGS = flags.FLAGS


def demos(
    assets_root="./assets", task="tangram", disp=True, mode="train", data_dir=".", n=100
):
    # Initialize environment and task.
    env = Environment(assets_root, disp, hz=480)
    task = tasks.names[task]()
    task.mode = mode

    # Initialize scripted oracle agent and dataset.
    agent = task.oracle(env)
    dataset = Dataset(os.path.join(data_dir, f"{task}-{task.mode}"))

    # Train seeds are even and test seeds are odd.
    seed = dataset.max_seed
    if seed < 0:
        seed = -1 if (task.mode == "test") else -2

    # Collect training data from oracle demonstrations.
    while dataset.n_episodes < n:
        print(f"Oracle demonstration: {dataset.n_episodes + 1}/{n}")
        episode, total_reward = [], 0
        seed += 2
        np.random.seed(seed)
        env.set_task(task)
        obs = env.reset()
        info = None
        reward = 0
        for _ in range(task.max_steps):
            act = agent.act(obs, info)
            episode.append((obs, act, reward, info))
            obs, reward, done, info = env.step(act)
            total_reward += reward
            print(f"{done} {total_reward}")
            if done:
                break
        episode.append((obs, None, reward, info))

        # Only save completed demonstrations.
        # TODO(andyzeng): add back deformable logic.
        if total_reward > 0.99:
            dataset.add(seed, episode)


if __name__ == "__main__":
    # app.run(demos)
    registe_task("tangram", Tangram)
    demos()
