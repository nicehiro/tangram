import time

import pybullet as p
import pybullet_data

from ravens import tasks


TANGRAM_SQUARE = [
    ((-0.943, 0, 0.01), (0, 0, 0)),
    ((0, -0.943, 0.01), (0, 0, 0)),
    ((0.7643, 0.7643, 0.01), (0, 0, 0)),
    ((0, 0.333, 0.01), (0, 0, 0)),
    ((0.8333, -0.5, 0.01), (0, 0, 0)),
    ((0.5, 0, 0.01), (0, 0, 0)),
    ((-0.25, 0.75, 0.01), (0, 0, 0))
]


def testit(func):
    """PyBullet basic environment settings before and after real motion.
    """

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
            time.sleep(1/240)

    return wrap


def create_multi_body(obj, vhacd, color, position):
    shift = [0, -0.02, 0]
    scale = [0.1, 0.1, 0.1]
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
        basePosition=position,
        useMaximalCoordinates=True
    )
    return id


def registe_task(task_name, task):
    tasks.names[task_name] = task


def v_hacd_generator(obj_path="./assets/mid_tri.obj", vhacd_path="./assets/mid_tri_vhacd.obj"):
    p.connect(p.DIRECT)
    p.vhacd(obj_path, vhacd_path, "log.txt")