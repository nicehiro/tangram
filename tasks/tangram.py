import numpy as np
import ravens.utils.utils as ravens_utils
from ravens import tasks
from ravens.tasks import Task
from utils import TANGRAM_SQUARE, registe_task
import pybullet as p


class Block:
    
    def __init__(self, name, obj, vhacd, color) -> None:
        self.name = name
        self.obj = obj
        self.vhacd = vhacd
        self.color = color


class Tangram(Task):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_steps = 10
        # mesh position adjustments
        self.scale = 0.01
        self.shift = [0.5, 0, 0]
        # path of tangram blocks
        base_dir = "tangram/"
        self.blocks = [
            Block("big_tri_0", base_dir+"big_tri", base_dir+"big_tri_vhacd", [1, 0, 0, 1]),
            Block("big_tri_1", base_dir+"big_tri", base_dir+"big_tri_vhacd", [0, 1, 0, 1]),
            Block("mid_tri", base_dir+"mid_tri", base_dir+"mid_tri_vhacd", [0, 0, 1, 1]),
            Block("sml_tri_0", base_dir+"sml_tri", base_dir+"sml_tri_vhacd", [1, 1, 0, 1]),
            Block("sml_tri_1", base_dir+"sml_tri", base_dir+"sml_tri_vhacd", [1, 0, 1, 1]),
            Block("square", base_dir+"square", base_dir+"square", [0, 1, 1, 1]),
            Block("parallel", base_dir+"parallel", base_dir+"parallel", [1, 1, 1, 1]),
        ]

    def add_object(self, env, obj, vhacd, color, pose, category='rigid'):
        shift = [0, -0.02, 0]
        scale = [0.05, 0.05, 0.05]
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=obj+".obj",
            rgbaColor=color,
            specularColor=[0.4, 0.4, 0.4], # reflection color
            visualFramePosition=shift, # offset with respect to the link
            meshScale=scale
        )
        # create collision shape from mesh or shape
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=vhacd+".obj",
            collisionFramePosition=shift,
            meshScale=scale
        )
        id = p.createMultiBody(
            baseMass=1, # mass of the base, in kg
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=pose[0],
            baseOrientation=pose[1],
            useMaximalCoordinates=True
        )
        env.obj_ids[category].append(id)
        return id

    def reset(self, env):
        super().reset(env)

        # add base
        base_size = (0.05, 0.15, 0.005)
        base_urdf = 'stacking/stand.urdf'
        base_pose = self.get_random_pose(env, base_size)
        env.add_object(base_urdf, base_pose, 'fixed')

        # add blocks
        objs = []
        start_pose = []

        for position, orientation in TANGRAM_SQUARE:
            posi = tuple(position[i] * self.scale + self.shift[i] for i in range(len(position)))
            start_pose.append((posi, orientation))

        for i, block in enumerate(self.blocks):
            pose = start_pose[i]
            obj_id = self.add_object(
                env, block.obj, block.vhacd, block.color, pose
            )
            objs.append((obj_id, (np.pi / 2, None)))

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
                     (0, 0.05, 0.03), (0, -0.025, 0.08),
                     (0, 0.025, 0.08), (0, 0, 0.13),
                     (0, 0, 0.13)]
        targs = [(ravens_utils.apply(base_pose, i), base_pose[1]) for i in place_pos]

        # Goal: blocks are stacked in a pyramid (bottom row: green, blue, purple).
        self.goals.append((objs[:3], np.ones((3, 3)), targs[:3],
                           False, True, 'pose', None, 1 / 2))

        # Goal: blocks are stacked in a pyramid (middle row: yellow, orange).
        self.goals.append((objs[3:5], np.ones((2, 2)), targs[3:5],
                           False, True, 'pose', None, 1 / 3))

        # Goal: blocks are stacked in a pyramid (top row: red).
        self.goals.append((objs[5:], np.ones((1, 1)), targs[5:],
                           False, True, 'pose', None, 1 / 6))


if __name__ == '__main__':
    from ravens import demos

    registe_task('tangram', Tangram)
    demos.main(assets_root="./assets", task='tangram')
