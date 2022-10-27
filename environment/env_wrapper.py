import gym
import numpy as np
import pybullet as p
import ravens.utils.utils as ravens_utils
from ravens.environments.environment import Environment
import utils


class WrappedTangramEnv(Environment):
    """Wrap the ravens' Environment for tangram to fit sb3.

    Cause sb3 doesn't support nested observation space.

    action_space:
        original: {'pose0': tuple(7,), 'pose1': tuple(7)}
        wrapped: tuple(14)
                 (:7) one-hot used for check which block to pick
                 (8:) pose1
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
        color_tuple = [
            gym.spaces.Box(0, 255, config["image_size"] + (3,), dtype=np.uint8)
            for config in self.agent_cams
        ]
        depth_tuple = [
            gym.spaces.Box(0.0, 20.0, config["image_size"], dtype=np.float32)
            for config in self.agent_cams
        ]
        # goal_space = gym.spaces.Box(0.0, 1, [64, 64], dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                # environment camera
                "color_front": color_tuple[0],
                "color_left": color_tuple[1],
                "color_right": color_tuple[2],
                "depth_front": depth_tuple[0],
                "depth_left": depth_tuple[1],
                "depth_right": depth_tuple[2],
                # goal image
                # "goal": goal_space,
            }
        )
        # self.observation_space = gym.spaces.flatten_space(self.observation_space)
        if not task:
            raise "Tangram environment must be initialized first."
        # build action space for tangram
        self.position_bounds = gym.spaces.Box(
            low=np.array([0.25, 0, 0.0], dtype=np.float32),
            high=np.array([0.75, 0.5, 0.28], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )
        pick_action = gym.spaces.Box(0.0, 1.0, (self.task.blocks_n,), dtype=np.float32)
        place_action = gym.spaces.Tuple(
            (
                self.position_bounds,
                # only z-axis
                gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            )
        )
        action_space = gym.spaces.Dict({"pose0": pick_action, "pose1": place_action})
        self.action_space = gym.spaces.flatten_space(action_space)

    def reset(self):
        # TODO: randomly reset goal for each trajectory
        self.task.set_goals(utils.goals)
        return super().reset()

    def _get_obs(self):
        """Return tangram observation.

        1. Image of current environment.
        2. Image of goal shape.
        """
        # env observation
        obs = {}
        obs["color_front"], obs["depth_front"], _ = self.render_camera(self.agent_cams[0])
        obs["color_left"], obs["depth_left"], _ = self.render_camera(self.agent_cams[1])
        obs["color_right"], obs["depth_right"], _ = self.render_camera(self.agent_cams[2])
        # TODO: goal observation
        return obs

    def step(self, action=None):
        """Turn the flattened action to nest space.

        action: (11,)
            (:7): one-hot data of pose0
            (7:10): position of pose1
            (11:): z-axis orientation of pose1
        """
        action_ = None
        if action is not None:
            # transfer one-hot data to pose0
            block_id = np.argmax(action[:7])
            pose0 = p.getBasePositionAndOrientation(self.task.blocks[block_id].id)
            # the orientation of pose1 only contains z-axis
            pose1_ori = p.getQuaternionFromEuler((0, 0, action[-1].item()))
            pose1 = (action[7:-1].tolist(), pose1_ori)
            # pose1 = ravens_utils.multiply(
            #     self.task.tangram_pose, (action[7:-1].tolist(), pose1_ori)
            # )
            action_ = {
                "pose0": pose0,
                "pose1": pose1,
            }
        return super().step(action_)
