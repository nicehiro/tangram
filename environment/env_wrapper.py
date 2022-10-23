import gym
from ravens.environments.environment import Environment
import numpy as np


class WrappedEnv(Environment):
    """Wrap the ravens' Environment to fit sb3.

    Cause sb3 doesn't support nested observation space.
    """

    def __init__(
        self,
        assets_root,
        task=None,
        disp=False,
        shared_memory=False,
        hz=240,
        use_egl=False,
    ):
        super().__init__(assets_root, task, disp, shared_memory, hz, use_egl)
        self.observation_space = gym.spaces.flatten_space(self.observation_space)
        self.action_space = gym.spaces.flatten_space(self.action_space)

    def _get_obs(self):
        colors, depths = [], []
        for config in self.agent_cams:
            color, depth, _ = self.render_camera(config)
            # flatten
            colors.append(color)
            depths.append(depth)
        colors = np.concatenate(colors, axis=None)
        depths = np.concatenate(depths, axis=None)
        obs = np.concatenate((colors, depths), axis=None)
        return obs

    def step(self, action=None):
        # input a flattened action
        # and turn it to nest space
        action_ = None
        if action is not None:
            actions = np.split(action, 2)
            action_ = {
                "pose0": (actions[0][:3].tolist(), actions[0][3:].tolist()),
                "pose1": (actions[1][:3].tolist(), actions[1][3:].tolist()),
            }
        return super().step(action_)
