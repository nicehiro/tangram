import math
from enum import Enum

import cv2
import numpy as np
import pybullet as p
import ravens.utils.utils as ravens_utils
import utils
from ravens.tasks import Task

sqrt_2 = math.sqrt(2)


class BlockType(Enum):
    BIG_TRIANGLE = 1
    MID_TRIANGLE = 2
    SML_TRIANGLE = 3
    SQUARE = 4
    PARALLEL = 5


# properties of 3D blocks
properties = {
    # type: (obj, vhacd, scale, base_size)
    BlockType.BIG_TRIANGLE: ("triangle", "triangle_vhacd", 2, [sqrt_2, sqrt_2, 0.01]),
    BlockType.MID_TRIANGLE: ("triangle", "triangle_vhacd", math.sqrt(2), [1, 1, 0.01]),
    BlockType.SML_TRIANGLE: (
        "triangle",
        "triangle_vhacd",
        1,
        [sqrt_2 / 2, sqrt_2 / 2, 0.01],
    ),
    BlockType.SQUARE: ("square", "square_vhacd", 1, [sqrt_2 / 2, sqrt_2 / 2, 0.01]),
    BlockType.PARALLEL: (
        "parallel",
        "parallel_vhacd",
        1,
        [sqrt_2 / 2, 3 * sqrt_2 / 2, 0.01],
    ),
}


class Block:
    def __init__(self, name, type, color, scale=1, base_dir="tangram/") -> None:
        """Block of tangram game.

        Args:
            name (str): block name
            type (BlockType): block type
            color (list): color list
            scale (int, optional): position scale. Defaults to 1.
            base_dir (str, optional): objects file base dir. Defaults to 'tangram/'.
        """
        self.name = name
        self.obj = base_dir + properties[type][0]
        self.vhacd = base_dir + properties[type][1]
        self.color = color
        # size scale for x and y axis
        s = properties[type][2]
        self.scale = [scale * s, scale * s, 0.1]
        # base size
        bs = properties[type][3]
        self.base_size = [x * scale for x in bs]


class Tangram(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocks_n = 7
        self.max_steps = 10
        # mesh position adjustments
        self.posi_scale = 0.1 / sqrt_2
        # mesh size adjustments
        self.size_scale = self.posi_scale
        # path of tangram blocks
        self.blocks = [
            Block("big_tri_0", BlockType.BIG_TRIANGLE, [1, 0, 0, 1], self.size_scale),
            Block("big_tri_1", BlockType.BIG_TRIANGLE, [0, 1, 0, 1], self.size_scale),
            Block("mid_tri", BlockType.MID_TRIANGLE, [0, 0, 1, 1], self.size_scale),
            Block("sml_tri_0", BlockType.SML_TRIANGLE, [1, 1, 0, 1], self.size_scale),
            Block("sml_tri_1", BlockType.SML_TRIANGLE, [1, 0, 1, 1], self.size_scale),
            Block("square", BlockType.SQUARE, [0, 1, 1, 1], self.size_scale),
            Block("parallel", BlockType.PARALLEL, [1, 1, 1, 1], self.size_scale),
        ]

    def reset(self, env):
        super().reset(env)
        # add blocks
        objs = []
        for block in self.blocks:
            pose = self.get_random_pose(env, block.base_size)
            obj_id = self.add_object(env, block, pose)
            objs.append((obj_id, (np.pi / 2, None)))
            # utils.mark_point(pose[0])
            # text_to_block = ((0, 0, 0.05), p.getQuaternionFromEuler((0, 0, 0)))
            # p.addUserDebugText(block.name, ravens_utils.multiply(pose, text_to_block)[0])
        block_to_tangram = [
            [[0, -2 / 3 * sqrt_2, 0.02], [0, 0, -1 * math.pi / 4]],
            [[2 / 3 * sqrt_2, 0, 0.02], [0, 0, math.pi / 4]],
            [[-2 / 3 * sqrt_2, 2 / 3 * sqrt_2, 0.02], [0, 0, 0]],
            [[-sqrt_2 / 3, 0, 0.02], [0, 0, -3 * math.pi / 4]],
            [[sqrt_2 / 2, 5 / 6 * sqrt_2, 0.02], [0, 0, 3 * math.pi / 4]],
            [[0, sqrt_2 / 2, 0.02], [0, 0, math.pi / 4]],
            [[-3 / 4 * sqrt_2, -1 / 4 * sqrt_2, 0.02], [0, math.pi, 3 * math.pi / 4]],
        ]
        tangram_to_world = [[0.5, 0.25, 0], p.getQuaternionFromEuler([0, 0, 0])]
        block_to_world = []
        for p_t, o_t in block_to_tangram:
            p_ = [x * self.posi_scale for x in p_t]
            o_ = p.getQuaternionFromEuler(o_t)
            pose = ravens_utils.multiply(tangram_to_world, (p_, o_))
            block_to_world.append(pose)
            utils.mark_point(pose[0])
        # goals structure:
        # obj, 1-matrix, goal, False, True,, 'pose', None
        for i in range(self.blocks_n):
            self.goals.append(
                (
                    objs[i : i + 1],
                    np.ones((1, 1)),
                    block_to_world[i : i + 1],
                    False,
                    True,
                    "pose",
                    None,
                    1 / self.blocks_n,
                )
            )

        p.addUserDebugLine([0.25, 0, 0.05], [0.75, 0, 0.05], [0, 1, 0])
        p.addUserDebugLine([0.4, 0.15, 0.05], [0.4, 0.35, 0.05], [0, 0, 1])
        p.addUserDebugLine([0.4, 0.35, 0.05], [0.6, 0.35, 0.05], [0, 0, 1])
        p.addUserDebugLine([0.6, 0.35, 0.05], [0.6, 0.15, 0.05], [0, 0, 1])
        p.addUserDebugLine([0.6, 0.15, 0.05], [0.4, 0.15, 0.05], [0, 0, 1])

    def add_object(self, env, block, pose, category="rigid"):
        # create objects and add it into env
        shift = [0, 0, 0]
        obj = block.obj
        vhacd = block.vhacd
        color = block.color
        scale = block.scale
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=obj + ".obj",
            rgbaColor=color,
            specularColor=[0.4, 0.4, 0.4],  # reflection color
            visualFramePosition=shift,  # offset with respect to the link
            meshScale=scale,
        )
        # create collision shape from mesh or shape
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=vhacd + ".obj",
            collisionFramePosition=shift,
            meshScale=scale,
        )
        id = p.createMultiBody(
            baseMass=1,  # mass of the base, in kg
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=pose[0],
            baseOrientation=pose[1],
            useMaximalCoordinates=True,
        )
        env.obj_ids[category].append(id)
        return id

    def get_random_pose(self, env, obj_size):
        """Get random collision-free object pose within workspace bounds.

        Copied from ravens project.
        """

        # Get erosion size of object in pixels.
        max_size = np.sqrt(obj_size[0] ** 2 + obj_size[1] ** 2)
        erode_size = int(np.round(max_size / self.pix_size))

        _, hmap, obj_mask = self.get_true_image(env)

        # Randomly sample an object pose within free-space pixels.
        free = np.ones(obj_mask.shape, dtype=np.uint8)
        for obj_ids in env.obj_ids.values():
            for obj_id in obj_ids:
                free[obj_mask == obj_id] = 0
        free[0, :], free[:, 0], free[-1, :], free[:, -1] = 0, 0, 0, 0
        # DIFFERENT between ravens
        # we use half plane to initial objects and another half for placing
        free = free[:160, :]
        free = cv2.erode(free, np.ones((erode_size, erode_size), np.uint8))
        if np.sum(free) == 0:
            return None, None
        pix = ravens_utils.sample_distribution(np.float32(free))
        pos = ravens_utils.pix_to_xyz(pix, hmap, self.bounds, self.pix_size)
        pos = (pos[0], pos[1], obj_size[2] / 2)
        theta = np.random.rand() * 2 * np.pi
        rot = ravens_utils.eulerXYZ_to_quatXYZW((0, 0, theta))
        return pos, rot
