import math
import time

import pybullet as p
import pybullet_data
from ravens import tasks

sqrt_2 = math.sqrt(2)

goals = [
    [[0, -2 / 3 * sqrt_2, 0.02], [0, 0, -1 * math.pi / 4]],
    [[2 / 3 * sqrt_2, 0, 0.02], [0, 0, math.pi / 4]],
    [[-2 / 3 * sqrt_2, 2 / 3 * sqrt_2, 0.02], [0, 0, 0]],
    [[-sqrt_2 / 3, 0, 0.02], [0, 0, -3 * math.pi / 4]],
    [[sqrt_2 / 2, 5 / 6 * sqrt_2, 0.02], [0, 0, 3 * math.pi / 4]],
    [[0, sqrt_2 / 2, 0.02], [0, 0, math.pi / 4]],
    [[-3 / 4 * sqrt_2, -1 / 4 * sqrt_2, 0.02], [0, math.pi, 3 * math.pi / 4]],
]


def testit(func):
    """PyBullet basic environment settings before and after real motion."""

    def wrap(*args, **kwargs):
        p.connect(p.GUI)
        # search path for models
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # solver iteration number of engine
        # small iteration means quick but unaccurate rendering/calculation
        p.setPhysicsEngineParameter(numSolverIterations=10)
        # load plane
        # maximal coordinates for a non-joint object will shorten load time
        p.loadURDF("plane100.urdf", useMaximalCoordinates=True)
        # disable rendering during object creating
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        # disable GUI
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # disable tiny rederer, like embeded video card
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        p.setGravity(0, 0, -10)
        func(*args, **kwargs)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        while True:
            p.stepSimulation()
            time.sleep(1 / 240)

    return wrap


def create_multi_body(
    obj, vhacd, color, position, orientation=[0, 0, 0], scale=[1, 1, 1]
):
    shift = [0, 0, 0]
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
        basePosition=position,
        baseOrientation=orientation,
        useMaximalCoordinates=True,
    )
    return id


def registe_task(task_name, task):
    tasks.names[task_name] = task


def v_hacd_generator(
    obj_path="./assets/mid_tri.obj", vhacd_path="./assets/mid_tri_vhacd.obj"
):
    p.connect(p.DIRECT)
    p.vhacd(obj_path, vhacd_path, "log.txt")


def mark_point(pose):
    p.addUserDebugLine(
        lineFromXYZ=[pose[0], pose[1], -1],
        lineToXYZ=[pose[0], pose[1], 1],
        lineColorRGB=[1, 0, 0],
        lineWidth=0.5,
    )


if __name__ == "__main__":
    v_hacd_generator(
        "./assets/tangram/parallel.obj", "./assets/tangram/parallel_vhacd.obj"
    )
